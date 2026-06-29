"""LSQ quantization training script.
Thanks to https://github.com/zhutmost/lsq-net
"""

import os
from loguru import logger
import argparse

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import timm

from omegaconf import OmegaConf
from tqdm import tqdm

import quan
# import util

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

def parse_args():
    parser = argparse.ArgumentParser(description='LSQ quantization training script')
    parser.add_argument('--config_file', type=str, default='./quant_configs.yaml',
                        help='path to the config file')
    parser.add_argument('--model',
                    choices=[
                        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
                    ],
                    help='model')
    parser.add_argument('-- ', type=int, default=3, help='bit-width for quantization')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--eval', action='store_true', help='evaluate the model')
    parser.add_argument('--name', type=str, default=None, help='name of the experiment')
    parser.add_argument('--device', type=str, default='0', help='device to use for training/evaluation')
    args = parser.parse_args()
    return args

def main():
    # args, cfg
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    cfg = OmegaConf.load(args.config_file)
    
    # override bit
    cfg.quan.act.bit = args.bit
    cfg.quan.weight.bit = args.bit
    logger.info(f'Using bit-width: {args.bit} for both weights and activations')
    
    seed(args.seed)
    
    output_dir = './output/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Enable the cudnn built-in auto-tuner to accelerating training, but it
    # will introduce some fluctuations in a narrow range.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((cfg.imsize, cfg.imsize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(cfg.imsize, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((cfg.imsize, cfg.imsize)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])
    
    # Initialize data loader
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
    
    # logger.info('Dataset `%s` size:' % cfg.dataloader.dataset +
    #             '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
    #             # '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
    #             '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    # Create the model
    model = timm.create_model(args.model, pretrained=True)
    num_classes = 100 # CIFAR-100 has 100 classes
    # in_features = model.get_classifier().in_features
    model.reset_classifier(num_classes=num_classes)
    # model.load_state_dict(torch.load(f"{args.model}_cifar100.pth", map_location='cpu'), strict=False)

    modules_to_replace = quan.find_modules_to_quantize(model, cfg.quan)
    model = quan.replace_module_by_names(model, modules_to_replace)
    logger.info('Inserted quantizers into the original model')

    model.cuda()

    start_epoch = 0

    # Define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        # weight_decay=0.05
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs
    )
        
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    if args.eval:
        msg = model.load_state_dict(torch.load(f"./output/{args.model}_quant_model.pth", map_location='cuda:0'), strict=False)
        logger.success('Loaded model weights from %s' % f"{args.model}_quant_model.pth")
        logger.info(msg)
        test_loss, t_top1, t_top5 = validate_model(model, test_loader, criterion, topk=(1, 5))
        logger.success('Test evaluation results: Loss: %.4f | Top-1 Acc: %.2f%% | Top-5 Acc: %.2f%%' % (test_loss, t_top1, t_top5))
    else:  # training
        # best_val_top1 = 0.0
        for epoch in range(start_epoch, cfg.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, lr_scheduler, criterion)
            logger.info(f'>>>>>>>> Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Top-1 Acc: {train_acc:.2f}%')
            # val_loss, v_top1, v_top5 = validate_model(model, val_loader, criterion, topk=(1, 5))

            # if v_top1 > best_val_top1:
            #     best_val_top1 = v_top1
            #     torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_epoch_{epoch}.pth'))
            # ==> We instead save and evaluate on the last epoch model

        logger.info(f'>>>>>>>> Epoch -1 (final model evaluation)')
        test_loss, t_top1, t_top5 = validate_model(model, test_loader, criterion, topk=(1, 5))
        logger.success(f'Final Test Loss: {test_loss:.4f} | Test Top-1 Acc: {t_top1:.2f}% | Test Top-5 Acc: {t_top5:.2f}%')
    
    torch.save(model.state_dict(), os.path.join(output_dir, f'{args.model}_quant_{cfg.quan.act.bit}_model.pth'))
    logger.info('Program completed successfully ... exiting ...')

def train_one_epoch(model, train_loader, optimizer, lr_scheduler, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_model(model, val_loader, criterion, topk=(1, 5)):
    model.eval()

    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
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

if __name__ == "__main__":
    main()