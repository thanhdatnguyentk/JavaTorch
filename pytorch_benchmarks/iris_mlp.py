import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime

class IrisMLP(nn.Module):
    def __init__(self):
        super(IrisMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)

def species_to_index(s):
    s = s.strip()
    if s.startswith("Iris-setosa"): return 0
    if s.startswith("Iris-versicolor"): return 1
    return 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputDir", type=str, default="benchmark/results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cpu") # Iris is a tiny dataset, CPU is optimal
    print(f"Device: {device}")

    # Load data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    os.makedirs("../tests", exist_ok=True)
    csv_file = "../tests/iris.csv"
    if not os.path.exists(csv_file):
        import urllib.request
        urllib.request.urlretrieve(url, csv_file)

    data = []
    labels = []
    with open(csv_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5:
                data.append([float(x) for x in parts[:4]])
                labels.append(species_to_index(parts[4]))
                
    X = np.array(data, dtype=np.float32)
    Y = np.array(labels, dtype=np.int64)

    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    train_n = int(len(X) * 0.8)
    X_train, Y_train = X[:train_n], Y[:train_n]
    X_test, Y_test = X[train_n:], Y[train_n:]

    model = IrisMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    run_id = f"iris_mlp_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}"
    os.makedirs(os.path.join(args.outputDir, "pytorch", "iris", run_id), exist_ok=True)

    X_train_t = torch.tensor(X_train).to(device)
    Y_train_t = torch.tensor(Y_train).to(device)
    X_test_t = torch.tensor(X_test).to(device)
    Y_test_t = torch.tensor(Y_test).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(targets).sum().item()
            
        if (epoch) % 100 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_t)
                _, predicted = outputs.max(1)
                correct_test = predicted.eq(Y_test_t).sum().item()
                
            train_acc = correct_train / len(X_train)
            test_acc = correct_test / len(X_test)
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch} loss={avg_loss:.6f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = outputs.max(1)
        final_acc = predicted.eq(Y_test_t).sum().item() / len(X_test)
    print(f"Final test accuracy={final_acc}")

if __name__ == "__main__":
    main()
