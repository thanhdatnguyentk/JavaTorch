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

class LeNetCustom(nn.Module):
    def __init__(self):
        super(LeNetCustom, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputDir", type=str, default="benchmark/results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.ToTensor()
    
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=False)

    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
    
    model = LeNetCustom().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    run_id = f"lenet_mnist_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}"
    os.makedirs(os.path.join(args.outputDir, "pytorch", "lenet_mnist", run_id), exist_ok=True)
    
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
            
            if (b + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} batch {b+1}/{len(trainloader)}  loss={loss.item():.4f}")
                
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
            
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}  train_acc={train_acc:.4f}  test_acc={val_acc:.4f}")

if __name__ == "__main__":
    main()
