import sys
sys.path.append('..')
import torch
import cying.nn as cynn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = torch.device('cuda:0')

size = 128
transform = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet RGB 均值
        std=[0.229, 0.224, 0.225]    # ImageNet RGB 标准差
    ),
])

train_dataset = datasets.ImageFolder(
    root='/root/autodl-tmp/datasets/ImageNet_20/train', 
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)
valid_dataset = datasets.ImageFolder(
    root='/root/autodl-tmp/datasets/ImageNet_20/valid',
    transform=transform
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

test_dataset = datasets.ImageFolder(
    root='/root/autodl-tmp/datasets/ImageNet_20/test', 
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    num_workers=4
)

from torchinfo import summary
params = [
    {
        'size': (128, 128),
        'in_channels': 3,
        'out_channels': 8,
        'hidden_width': 128,
        'spe_opt_size': (128,65),
        'spa_opt_size': 0
    },
    {
        'size': (128, 128),
        'in_channels': 8,
        'out_channels': 16,
        'hidden_width': 128,
        'spe_opt_size': (128,65),
        'spa_opt_size': 0
    },
    {
        'size': (128, 128),
        'in_channels': 16,
        'out_channels': 32,
        'hidden_width': 128,
        'spe_opt_size': (128,65),
        'spa_opt_size': 0
    }
    # {
    #     'size': (32, 32),
    #     'in_channels': 32,
    #     'out_channels': 64,
    #     'hidden_width': 256,
    #     'spe_opt_size': (32,16),
    #     'spa_opt_size': 0
    # },
    # {
    #     'size': (16, 16),
    #     'in_channels': 64,
    #     'out_channels': 128,
    #     'hidden_width': 512,
    #     'spe_opt_size': (16,8),
    #     'spa_opt_size': 0
    # },
    # {
    #     'size': (8, 8),
    #     'in_channels': 128,
    #     'out_channels': 256,
    #     'hidden_width': 1024,
    #     'spe_opt_size': (8,4),
    #     'spa_opt_size': 0
    # }
]
class ImageClassifier(cynn.OperatorModel2d):
    def __init__(self, params, num_classes=1000):
        super().__init__(params)
        self.classifier = torch.nn.Linear(params[-1]['out_channels'], num_classes)

    def forward(self, x, mode='train'):
        x = super().forward(x, mode)
        x = torch.mean(x, dim=(-2, -1))
        x = self.classifier(x)
        return x
classifier = ImageClassifier(
    params,
    num_classes=20
).to(device)
# classifier = torch.load('./imagenet_classifier.pth', weights_only=False)
summary(classifier)

from tqdm import tqdm
epochs = 100
lr = 1e-3

mask_params = list(classifier.get_mask_weight())
mask_param_ids = set(id(p) for p in mask_params)

other_params = [p for p in classifier.parameters() if id(p) not in mask_param_ids]

optim_train = torch.optim.Adam(other_params, lr=lr)
optim_valid = torch.optim.Adam(mask_params, lr=lr)

loss_fn = torch.nn.CrossEntropyLoss()
mini_epoch = 1
j = 0
for i in range(epochs):
    classifier.train()
    train_loss = 0.
    train_correct = 0.
    pbar = tqdm(train_loader, desc=f"Epoch {i+1}/{epochs} [Train]")
    j += 1
    for images, labels in pbar:
        optim_train.zero_grad()

        outputs = classifier(images.to(device), mode='train')
        loss = loss_fn(outputs, labels.to(device))
        loss.backward()

        optim_train.step()

        train_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels.to(device)).sum().item()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}",correct=(preds == labels.to(device)).float().mean().item())
    
    classifier.eval()
    valid_loss = 0.
    valid_correct = 0.
    if j % mini_epoch == 0:
        j = 0
        pbar = tqdm(valid_loader, desc=f"Epoch {i+1}/{epochs} [Valid]")
        for images, labels in pbar:
            optim_valid.zero_grad()
    
            outputs = classifier(images.to(device), mode='valid')
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
    
            optim_valid.step()
            
            valid_loss += loss.item() * images.size(0)
    
            preds = outputs.argmax(dim=1)
            valid_correct += (preds == labels.to(device)).sum().item()
    
            pbar.set_postfix(loss=f"{loss.item():.4f}",correct=(preds == labels.to(device)).float().mean().item())

    torch.save(classifier, './imagenet_classifier.pth')
    test_loss = 0.
    test_correct = 0.
    pbar = tqdm(test_loader, desc=f"Epoch {i+1}/{epochs} [Test]")
    with torch.no_grad():
        for images, labels in pbar:
            outputs = classifier(images.to(device), mode='test')
            loss = loss_fn(outputs, labels.to(device))
            
            test_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels.to(device)).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}",correct=(preds == labels.to(device)).float().mean().item())
    train_loss /= len(train_loader.dataset)
    valid_loss /= len(valid_loader.dataset)
    test_loss /= len(test_loader.dataset)
    train_correct /= len(train_loader.dataset)
    valid_correct /= len(valid_loader.dataset)
    test_correct /= len(test_loader.dataset)
    print(f"Epoch {i+1}/{epochs}, train: {train_loss:.4f}, {train_correct:.4f}, valid {valid_loss:.4f}, {valid_correct:.4f}, test: {test_loss:.4f}, {test_correct:.4f}")