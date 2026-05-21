import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputDir", type=str, default="benchmark/results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Not using standard normalization to match exactly JavaTorch raw pixel inputs [0..255] -> [0..1]
    # (assuming JavaTorch divides by 255 if it does, actually JavaTorch might use raw float values)
    # Looking at Cifar10Loader, it returns raw floats. We'll use ToTensor which makes it [0..1].
    transform = transforms.ToTensor()
    
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=False)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False)
    
    model = ResNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    run_id = f"resnet_cifar10_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}"
    os.makedirs(os.path.join(args.outputDir, "pytorch", "resnet_cifar10", run_id), exist_ok=True)
    
    best_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for b, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if (b + 1) == 1 or (b + 1) % 50 == 0 or (b + 1) == len(trainloader):
                print(f"[Benchmark][ResNet][Train] epoch={epoch+1}/{args.epochs} batch={b+1}/{len(trainloader)} loss={loss.item():.5f}")
                
        # Eval
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
                
        train_acc = correct_train / max(1, total_train)
        val_acc = correct_val / max(1, total_val)
        avg_loss = total_loss / max(1, len(trainloader))
        epoch_ms = int((time.time() - t0) * 1000)
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        print(f"[Benchmark][ResNet] epoch={epoch+1}/{args.epochs} loss={avg_loss:.5f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time_ms={epoch_ms}")

if __name__ == "__main__":
    main()
