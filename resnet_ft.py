import timm
import argparse
import torch 
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm


parser = argparse.ArgumentParser(description='LSQ ResNet fine-tuning')

parser.add_argument('--model',
                    choices=[
                        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
                    ],
                    help='model')
parser.add_argument('--seed', default=0, type=int, help='seed')

def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        pred = outputs.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, topk=(1, 5)):
    model.eval()

    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)

            # ---- TOP-K LOGIC ----
            _, pred = outputs.topk(max(topk), dim=1, largest=True, sorted=True)

            # Top-1
            correct1 += pred[:, 0].eq(targets).sum().item()

            # Top-5 (any of first 5 matches target)
            correct5 += pred[:, :5].eq(targets.unsqueeze(1)).any(dim=1).sum().item()

    epoch_loss = running_loss / total
    top1_acc = 100.0 * correct1 / total
    top5_acc = 100.0 * correct5 / total

    return epoch_loss, top1_acc, top5_acc

def main():
    args = parser.parse_args()
    seed(args.seed)
    
    model = timm.create_model(args.model, pretrained=True)
    num_classes = 100 # CIFAR-100 has 100 classes
    in_features = model.get_classifier().in_features
    model.reset_classifier(num_classes=num_classes)
    
    model.cuda()

    IMG_SIZE = 224

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_ds = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_ds = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        # weight_decay=0.05
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50
    )
    
    for epoch in range(50):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, t_top1, t_top5 = evaluate(model, test_loader, criterion)

        scheduler.step()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss {train_loss:.4f} | "
            f"Train {train_acc:.2f}% | "
            f"Test Loss {test_loss:.4f} | "
            f"Test Top-1: {t_top1:.2f}% | "
            f"Test Top-5: {t_top5:.2f}%"
        )
    
    # save the model
    torch.save(model.state_dict(), f"{args.model}_cifar100.pth")

if __name__ == "__main__":
    main()
    