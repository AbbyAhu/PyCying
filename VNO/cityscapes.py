import sys
sys.path.append('..')
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from cying.nn import VisionModel
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from cityscapesscripts.helpers.labels import labels
from torch_linear_assignment import batch_linear_assignment
from torchinfo import summary

label_to_Id = {label.name: label.trainId for label in labels}

cityscapes_root = "/root/autodl-tmp/cityscapes"
target_size = (256, 512)
batch_size = 12
num_query = 291
num_classes = 19
num_heads = 4
decoder_layers = 3
hidden_width = 256

num_epochs = 1000
lr = 1e-3
num_workers = 4

device = torch.device("cuda:0")

class CityscapesDataset(Dataset):
    def __init__(self, root, split, target_size, num_query):
        self.root = Path(root)
        self.split = split
        self.size = target_size
        self.num_query = num_query

        gt_dir = self.root / 'gtFine' / split
        self.json_files = sorted(gt_dir.glob('*/*_gtFine_polygons.json'))

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)

        img_path = Path(str(json_path).replace('_gtFine_polygons.json', '_leftImg8bit.png').replace('gtFine', 'leftImg8bit'))
        image = np.array(Image.open(img_path).convert('RGB'))  

        h, w = image.shape[:2]
        target_h, target_w = self.size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        image_resized = torch.tensor(cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR), dtype=torch.float32).permute(2, 0, 1) / 255.0
        image = torch.tile(
            image_resized,
            (
                1,
                (target_h + new_h - 1) // new_h, 
                (target_w + new_w - 1) // new_w   
            ),
        )[:, :target_h, :target_w]
        padding_mask = torch.ones((target_h, target_w), dtype=bool)
        padding_mask[:new_h, :new_w] = False

        masks = torch.zeros((self.num_query, target_h, target_w), dtype=torch.uint8)
        labels = torch.full((self.num_query,), -1, dtype=torch.long)
        bboxes = torch.zeros((self.num_query, 4), dtype=torch.float32)

        valid_idx = 0
        for obj in data['objects']:
            label_str = obj['label']
            label_id = label_to_Id.get(label_str, -1)
            if label_id == -1 or label_id == 255:
                continue

            polygon = np.array(obj['polygon'], dtype=np.int32)
            x_coords = polygon[:, 0]
            y_coords = polygon[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            polygon_scaled = (polygon * scale).astype(np.int32)
            bbox_scaled = [x_min * scale/target_w, y_min * scale/target_h, x_max * scale/target_w, y_max * scale/target_h]

            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_scaled], 1)

            masks[valid_idx] = torch.as_tensor(mask, dtype=torch.uint8)
            labels[valid_idx] = torch.as_tensor(label_id, dtype=torch.long)
            bboxes[valid_idx] = torch.as_tensor(bbox_scaled, dtype=torch.float32)

            valid_idx += 1

        return image, labels, bboxes, masks, padding_mask
class SetCriterion(nn.Module):
    def __init__(self, num_classes, loss_weights, matcher_weights, eov_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.matcher_weights = matcher_weights
        self.eov_coef = eov_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eov_coef
        self.register_buffer("empty_weight", empty_weight)
    
    def generalized_box_iou(self, boxes1, boxes2):
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * \
                (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * \
                (boxes2[..., 3] - boxes2[..., 1])

        lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
        rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        union = area1[:, :, None] + area2[:, None, :] - inter

        iou = inter / union.clamp(min=1e-6)

        lt_c = torch.min(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
        rb_c = torch.max(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])
        wh_c = (rb_c - lt_c).clamp(min=0)
        area_c = wh_c[..., 0] * wh_c[..., 1]

        giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
        return giou

    @torch.no_grad()
    def hungarian_matcher(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        tgt_labels = targets["labels"]
        tgt_boxes = targets["boxes"]

        invalid_tgt = tgt_labels < 0

        pred_prob = pred_logits.softmax(-1)
        cost_class = - pred_prob.gather(
            2, tgt_labels[:,None,:].clamp(min=0).expand(-1, pred_logits.size(1), -1)
        )
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)
        cost_giou = - self.generalized_box_iou(pred_boxes, tgt_boxes)

        C = self.matcher_weights['class'] * cost_class \
            + self.matcher_weights['bbox'] * cost_bbox \
            + self.matcher_weights['giou'] * cost_giou

        C = C.masked_fill(invalid_tgt[:,None,:], 1e6)

        assignment = batch_linear_assignment(C)

        assignment = assignment.masked_fill(
            invalid_tgt.gather(1, assignment), -1
        )
        return assignment

    def forward(self, outputs, targets):
        tgt_idx = self.hungarian_matcher(outputs, targets)
        
        valid = tgt_idx >= 0

        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        pred_masks = outputs["pred_masks"]

        tgt_labels_pad = targets["labels"]
        tgt_boxes_pad = targets["boxes"]
        tgt_masks_pad = targets["masks"]
        padding_mask = targets["padding_mask"]

        H_t, W_t = tgt_masks_pad.shape[-2:]

        safe_idx = tgt_idx.clamp(min=0)
        matched_labels = tgt_labels_pad.gather(1, safe_idx)
        matched_boxes = tgt_boxes_pad.gather(
            1, safe_idx[..., None].expand(-1, -1, 4)
        )
        matched_masks = tgt_masks_pad.gather(
            1, safe_idx[:, :, None, None].expand(-1, -1, H_t, W_t)
        )

        target_classes = matched_labels.masked_fill(~valid, self.num_classes)

        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )

        num_matched = valid.sum().clamp(min=1.0)

        l1_per = F.l1_loss(pred_boxes, matched_boxes, reduction="none").sum(-1)
        giou = self.generalized_box_iou(pred_boxes, matched_boxes).diagonal(dim1=-2, dim2=-1)
        loss_bbox = (l1_per * valid).sum() / num_matched
        loss_giou = ((1.0 - giou) * valid).sum() / num_matched

        valid_pixel = (~padding_mask).unsqueeze(1)
        pred_m = pred_masks * valid_pixel
        tgt_m  = matched_masks * valid_pixel

        ys = (torch.arange(H_t, device=pred_masks.device, dtype=pred_masks.dtype) + 0.5) / H_t
        xs = (torch.arange(W_t, device=pred_masks.device, dtype=pred_masks.dtype) + 0.5) / W_t

        x1, y1, x2, y2 = pred_boxes.unbind(-1)
        in_box = ((xs >= x1[..., None, None]) & (xs < x2[..., None, None]) &
                  (ys[:, None] >= y1[..., None, None]) & (ys[:, None] < y2[..., None, None]))

        bce_mask = in_box & (~padding_mask).unsqueeze(1)

        bce = F.binary_cross_entropy(
            pred_masks.clamp(1e-6, 1 - 1e-6), matched_masks, reduction="none"
        ) * bce_mask

        loss_bce = (bce.sum((-1, -2)) / bce_mask.sum((-1, -2)).clamp(min=1) * valid).sum() / num_matched

        inter = (pred_m * tgt_m).sum((-1, -2))
        union = (pred_m * bce_mask).sum((-1, -2)) + tgt_m.sum((-1, -2))
        dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
        loss_dice = (dice * valid).sum() / num_matched

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_dice": loss_dice,
            "loss_bce": loss_bce,
        }
        loss_dict["total_loss"] = sum(
            self.loss_weights.get(k, 1.0) * v for k, v in loss_dict.items()
        )
        return loss_dict

train_dataset = CityscapesDataset(
    root = cityscapes_root,
    split = "train",
    target_size = target_size,
    num_query = num_query
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

valid_dataset = CityscapesDataset(
    root = cityscapes_root,
    split = "valid",
    target_size = target_size,
    num_query = num_query
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=num_workers
)

test_dataset = CityscapesDataset(
    root = cityscapes_root,
    split = "test",
    target_size = target_size,
    num_query = num_query
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

backbone_params = [
    {
        'size': target_size,
        'in_channels': 3,
        'out_channels': 8,
        'hidden_width': hidden_width,
        'spe_opt_size': (11, 11),
        'spa_opt_size': 11
    },
    {
        'size': target_size,
        'in_channels': 8,
        'out_channels': 16,
        'hidden_width': hidden_width,
        'spe_opt_size': (11, 11),
        'spa_opt_size': 11
    }
]

encoder_params = [
    {
        'size': (64, 128),
        'in_channels': 16,
        'out_channels': 32,
        'hidden_width': hidden_width,
        'spe_opt_size': (11, 11),
        'spa_opt_size': 11
    },
    {
        'size': (16, 32),
        'in_channels': 32,
        'out_channels': 64,
        'hidden_width': hidden_width,
        'spe_opt_size': (9, 9),
        'spa_opt_size': 7
    },
    {
        'size': (8, 16),
        'in_channels': 64,
        'out_channels': 128,
        'hidden_width': hidden_width,
        'spe_opt_size': (5, 5),
        'spa_opt_size': 5
    }
]

model = VisionModel(
    backbone_params = backbone_params,
    encoder_params = encoder_params,
    num_classes = num_classes,
    num_heads = num_heads,
    decoder_hidden_width = hidden_width,
    decoder_layers = decoder_layers,
    num_query = num_query
).to(device)

summary(model)

loss_weight = {
    "loss_ce":    1.0,
    "loss_bbox":  5.0,
    "loss_giou":  2.0,
    "loss_dice":  5.0,
    "loss_bce":   2.0,
}

matcher_weight = {
    "class": 1.0,
    "bbox": 5.0,
    "giou": 2.0
}

loss_fn = SetCriterion(
    num_classes=num_classes,
    loss_weights=loss_weight,
    matcher_weights=matcher_weight,
    eov_coef=0.1,
).to(device)

train_optim = torch.optim.Adam(model.parameters(), lr=lr)
valid_optim = torch.optim.Adam(model.get_mask_weight(), lr=1e-2)

for epoch in range(num_epochs):
    model.train()

    train_total = 0.0
    train_ce = 0.0
    train_bbox = 0.0
    train_giou = 0.0
    train_dice = 0.0
    train_bce = 0.0
    train_count = 0

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels, bboxes, masks, padding_mask in pbar:
        train_optim.zero_grad(set_to_none=True)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)

        targets = {
            "masks": masks.float(),
            "labels": labels,
            "boxes": bboxes,
            "padding_mask": padding_mask,
        }

        logits, boxes, masks = model(images, padding_mask, mode="train")

        outputs = {
            "pred_logits": logits,
            "pred_boxes": torch.cat(
                [boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], dim=-1
            ),
            "pred_masks": masks,
        }

        losses = loss_fn(outputs, targets)

        losses["total_loss"].backward()

        train_optim.step()

        train_total += losses["total_loss"].item()
        train_ce += losses["loss_ce"].item()
        train_bbox += losses["loss_bbox"].item()
        train_giou += losses["loss_giou"].item()
        train_dice += losses["loss_dice"].item()
        train_bce += losses["loss_bce"].item()
        train_count += 1

        pbar.set_postfix(
            {
                "total": f"{losses['total_loss'].item():.4f}",
                "ce": f"{losses['loss_ce'].item():.4f}",
                "bbox": f"{losses['loss_bbox'].item():.4f}",
                "giou": f"{losses['loss_giou'].item():.4f}",
                "dice": f"{losses['loss_dice'].item():.4f}",
                "bce": f"{losses['loss_bce'].item():.4f}",
            }
        )

    print(f"\n===== Epoch {epoch+1} Train 整体误差 (共 {train_count} 个 batch) =====")
    print(f"total = {train_total / train_count:.4f}")
    print(f"ce    = {train_ce / train_count:.4f}")
    print(f"bbox  = {train_bbox / train_count:.4f}")
    print(f"giou  = {train_giou / train_count:.4f}")
    print(f"dice  = {train_dice / train_count:.4f}")
    print(f"bce   = {train_bce / train_count:.4f}")

    valid_total = 0.0
    valid_ce = 0.0
    valid_bbox = 0.0
    valid_giou = 0.0
    valid_dice = 0.0
    valid_bce = 0.0
    valid_count = 0

    pbar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels, bboxes, masks, padding_mask in pbar:
        valid_optim.zero_grad(set_to_none=True)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)

        targets = {
            "masks": masks.float(),
            "labels": labels,
            "boxes": bboxes,
            "padding_mask": padding_mask,
        }

        logits, boxes, masks = model(images, padding_mask, mode="valid")

        outputs = {
            "pred_logits": logits,
            "pred_boxes": torch.cat(
                [boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], dim=-1
            ),
            "pred_masks": masks,
        }

        losses = loss_fn(outputs, targets)

        losses["total_loss"].backward()

        valid_optim.step()

        valid_total += losses["total_loss"].item()
        valid_ce += losses["loss_ce"].item()
        valid_bbox += losses["loss_bbox"].item()
        valid_giou += losses["loss_giou"].item()
        valid_dice += losses["loss_dice"].item()
        valid_bce += losses["loss_bce"].item()
        valid_count += 1

        pbar.set_postfix(
            {
                "total": f"{losses['total_loss'].item():.4f}",
                "ce": f"{losses['loss_ce'].item():.4f}",
                "bbox": f"{losses['loss_bbox'].item():.4f}",
                "giou": f"{losses['loss_giou'].item():.4f}",
                "dice": f"{losses['loss_dice'].item():.4f}",
                "bce": f"{losses['loss_bce'].item():.4f}",
            }
        )

    print(f"\n===== Epoch {epoch+1} Valid 整体误差 (共 {valid_count} 个 batch) =====")
    print(f"total = {valid_total / valid_count:.4f}")
    print(f"ce    = {valid_ce / valid_count:.4f}")
    print(f"bbox  = {valid_bbox / valid_count:.4f}")
    print(f"giou  = {valid_giou / valid_count:.4f}")
    print(f"dice  = {valid_dice / valid_count:.4f}")
    print(f"bce   = {valid_bce / valid_count:.4f}")

    model.eval()
    torch.save(model, "./cityscapes.pth")

    test_total = 0.0
    test_ce = 0.0
    test_bbox = 0.0
    test_giou = 0.0
    test_dice = 0.0
    test_bce = 0.0
    test_count = 0

    pbar = tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    with torch.no_grad():
        for images, labels, bboxes, masks, padding_mask in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)

            targets = {
                "masks": masks.float(),
                "labels": labels,
                "boxes": bboxes,
                "padding_mask": padding_mask,
            }

            logits, boxes, masks = model(images, padding_mask, mode="test")

            outputs = {
                "pred_logits": logits,
                "pred_boxes": torch.cat(
                    [boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], dim=-1
                ),
                "pred_masks": masks,
            }

            losses = loss_fn(outputs, targets)

            test_total += losses["total_loss"].item()
            test_ce += losses["loss_ce"].item()
            test_bbox += losses["loss_bbox"].item()
            test_giou += losses["loss_giou"].item()
            test_dice += losses["loss_dice"].item()
            test_bce += losses["loss_bce"].item()
            test_count += 1

            pbar.set_postfix(
                {
                    "total": f"{losses['total_loss'].item():.4f}",
                    "ce": f"{losses['loss_ce'].item():.4f}",
                    "bbox": f"{losses['loss_bbox'].item():.4f}",
                    "giou": f"{losses['loss_giou'].item():.4f}",
                    "dice": f"{losses['loss_dice'].item():.4f}",
                    "bce": f"{losses['loss_bce'].item():.4f}",
                }
            )

    print(f"\n===== Epoch {epoch+1} Test 整体误差 (共 {test_count} 个 batch) =====")
    print(f"total = {test_total / test_count:.4f}")
    print(f"ce    = {test_ce / test_count:.4f}")
    print(f"bbox  = {test_bbox / test_count:.4f}")
    print(f"giou  = {test_giou / test_count:.4f}")
    print(f"dice  = {test_dice / test_count:.4f}")
    print(f"bce   = {test_bce / test_count:.4f}")