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

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=64, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_c=3, num_classes=10, embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_c, patch_size, embed_dim, img_size)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu', batch_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        
        x = self.norm(x)
        cls_out = x[:, 0]
        x = self.head(cls_out)
        return x

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

    transform = transforms.ToTensor()
    
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=False)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False)
    
    model = ViT(32, 4, 3, 10, 64, 4, 4, 128, 0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    run_id = f"vit_cifar10_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}"
    os.makedirs(os.path.join(args.outputDir, "pytorch", "vit_cifar10", run_id), exist_ok=True)
    
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
                print(f"[Benchmark][ViT][Train] epoch={epoch+1}/{args.epochs} batch={b+1}/{len(trainloader)} loss={loss.item():.5f}")
                
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
                
        scheduler.step()

        train_acc = correct_train / max(1, total_train)
        val_acc = correct_val / max(1, total_val)
        avg_loss = total_loss / max(1, len(trainloader))
        epoch_ms = int((time.time() - t0) * 1000)
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        print(f"[Benchmark][ViT] epoch={epoch+1}/{args.epochs} loss={avg_loss:.5f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time_ms={epoch_ms}")

if __name__ == "__main__":
    main()
