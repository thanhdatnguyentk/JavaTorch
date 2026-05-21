import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from datetime import datetime
import csv

class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.size = 0

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.size
            self.id2word[self.size] = word
            self.size += 1

    def get_id(self, word):
        return self.word2id.get(word, -1)

def tokenize(text):
    text = text.lower()
    for p in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
        text = text.replace(p, " " + p + " ")
    return [w for w in text.split() if w.strip()]

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Java LSTM: hidden_dim=64, 2 layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        # Take the last time step output from the last layer
        last_hidden = output[:, -1, :]
        drop = self.dropout(last_hidden)
        return self.fc(drop)

def load_data(data_dir):
    entries = []
    with open(os.path.join(data_dir, "pos.txt"), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip(): entries.append((line.strip(), 1))
    with open(os.path.join(data_dir, "neg.txt"), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip(): entries.append((line.strip(), 0))
    return entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--maxLen", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputDir", type=str, default="benchmark/results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = "../data/rt-polarity"
    all_entries = load_data(data_dir)
    random.shuffle(all_entries)
    
    split = int(len(all_entries) * 0.8)
    train_entries = all_entries[:split]
    test_entries = all_entries[split:]
    print(f"[Benchmark][Sentiment] dataset loaded train={len(train_entries)} test={len(test_entries)}")

    vocab = Vocabulary()
    # Add a padding token
    vocab.add_word("<PAD>")
    for text, _ in train_entries:
        for t in tokenize(text):
            vocab.add_word(t)
            
    print(f"[Benchmark][Sentiment] vocabulary size={vocab.size}")

    model = SentimentModel(vocab.size, 32, 64, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    run_id = f"sentiment_rtpolarity_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}"
    os.makedirs(os.path.join(args.outputDir, "pytorch", "sentiment_rtpolarity", run_id), exist_ok=True)
    
    best_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        num_batches = (len(train_entries) + args.batchSize - 1) // args.batchSize
        
        for b in range(num_batches):
            batch_entries = train_entries[b * args.batchSize : (b + 1) * args.batchSize]
            current_bs = len(batch_entries)
            
            x_data = np.zeros((current_bs, args.maxLen), dtype=np.int64)
            y_labels = np.zeros(current_bs, dtype=np.int64)
            
            for i, (text, label) in enumerate(batch_entries):
                tokens = tokenize(text)
                for j in range(args.maxLen):
                    if j < len(tokens):
                        idx = vocab.get_id(tokens[j])
                        x_data[i, j] = idx if idx != -1 else 0
                    else:
                        x_data[i, j] = 0
                y_labels[i] = label
                
            x_tensor = torch.tensor(x_data).to(device)
            y_tensor = torch.tensor(y_labels).to(device)
            
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y_tensor).sum().item()
            total_train += current_bs
            
            if (b + 1) == 1 or (b + 1) % 50 == 0 or (b + 1) == num_batches:
                print(f"[Benchmark][Sentiment][Train] epoch={epoch+1}/{args.epochs} batch={b+1}/{num_batches} loss={loss.item():.5f}")
                
        # Eval
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for b in range((len(test_entries) + args.batchSize - 1) // args.batchSize):
                batch_entries = test_entries[b * args.batchSize : (b + 1) * args.batchSize]
                current_bs = len(batch_entries)
                x_data = np.zeros((current_bs, args.maxLen), dtype=np.int64)
                y_labels = np.zeros(current_bs, dtype=np.int64)
                for i, (text, label) in enumerate(batch_entries):
                    tokens = tokenize(text)
                    for j in range(args.maxLen):
                        if j < len(tokens):
                            idx = vocab.get_id(tokens[j])
                            x_data[i, j] = idx if idx != -1 else 0
                        else:
                            x_data[i, j] = 0
                    y_labels[i] = label
                x_tensor = torch.tensor(x_data).to(device)
                y_tensor = torch.tensor(y_labels).to(device)
                logits = model(x_tensor)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y_tensor).sum().item()
                total_val += current_bs
                
        train_acc = correct_train / max(1, total_train)
        val_acc = correct_val / max(1, total_val)
        avg_loss = total_loss / max(1, num_batches)
        epoch_ms = int((time.time() - t0) * 1000)
        
        if val_acc > best_acc:
            best_acc = val_acc
            
        print(f"[Benchmark][Sentiment] epoch={epoch+1}/{args.epochs} loss={avg_loss:.5f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time_ms={epoch_ms}")

if __name__ == "__main__":
    main()
