import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

class MultiTaskLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, sentiment_classes, topic_classes):
        super(MultiTaskLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Using batch_first=True to match standard PyTorch patterns
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # Bidirectional means output is hidden_dim * 2. In JavaTorch it might be unidirectional.
        # Looking at JavaTorch `LSTM(embedDim, hiddenDim, true, true)`, the 4th param might be bidir?
        # Let's assume unidirectional for standard parity unless specified.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        
        # Two heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, sentiment_classes)
        )
        self.topic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, topic_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        # Take last time step
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        sent_logits = self.sentiment_head(last_hidden)
        topic_logits = self.topic_head(last_hidden)
        
        return sent_logits, topic_logits

def tokenize(text):
    text = text.lower()
    for p in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
        text = text.replace(p, " " + p + " ")
    return [w for w in text.split() if w.strip()]

def load_uit_vsfc_data(data_dir, phase):
    sents_path = os.path.join(data_dir, phase, "sents.txt")
    sent_labels_path = os.path.join(data_dir, phase, "sentiments.txt")
    topic_labels_path = os.path.join(data_dir, phase, "topics.txt")
    
    if not os.path.exists(sents_path):
        return []
        
    with open(sents_path, "r", encoding="utf-8") as f:
        sents = [line.strip() for line in f]
    with open(sent_labels_path, "r", encoding="utf-8") as f:
        sentiments = [int(line.strip()) for line in f]
    with open(topic_labels_path, "r", encoding="utf-8") as f:
        topics = [int(line.strip()) for line in f]
        
    return list(zip(sents, sentiments, topics))

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--maxLen", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataDir", type=str, default="../examples/data/uit-vsfc")
    parser.add_argument("--outputDir", type=str, default="benchmark/results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = args.dataDir
    # Fallback to alternative paths if not found
    if not os.path.exists(os.path.join(data_dir, "train", "sents.txt")):
        alternatives = ["../data/uit-vsfc", "examples/data/uit-vsfc", "data/uit-vsfc"]
        for alt in alternatives:
            if os.path.exists(os.path.join(alt, "train", "sents.txt")):
                data_dir = alt
                break

    train_data = load_uit_vsfc_data(data_dir, "train")
    dev_data = load_uit_vsfc_data(data_dir, "dev")
    test_data = load_uit_vsfc_data(data_dir, "test")
    
    if not train_data:
        print("UIT-VSFC data not found. Skipping...")
        return

    print(f"Train={len(train_data)} Dev={len(dev_data)} Test={len(test_data)}")
    
    # 3 sentiment classes, 4 topic classes
    sentiment_classes = 3
    topic_classes = 4

    vocab = Vocabulary()
    vocab.add_word("<PAD>")
    for text, _, _ in train_data:
        for t in tokenize(text):
            vocab.add_word(t)
            
    print(f"Vocabulary size: {vocab.size}")

    model = MultiTaskLSTMModel(vocab.size, 256, 512, sentiment_classes, topic_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    run_id = f"uit_vsfc_multitask_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}"
    os.makedirs(os.path.join(args.outputDir, "pytorch", "uit_vsfc_multitask", run_id), exist_ok=True)
    
    best_loss = float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        total_sent_loss = 0
        total_topic_loss = 0
        correct_sent = 0
        correct_topic = 0
        total_samples = 0
        
        num_batches = (len(train_data) + args.batchSize - 1) // args.batchSize
        
        for b in range(num_batches):
            batch = train_data[b * args.batchSize : (b + 1) * args.batchSize]
            current_bs = len(batch)
            
            x_data = np.zeros((current_bs, args.maxLen), dtype=np.int64)
            y_sent = np.zeros(current_bs, dtype=np.int64)
            y_topic = np.zeros(current_bs, dtype=np.int64)
            
            for i, (text, sent, top) in enumerate(batch):
                tokens = tokenize(text)
                for j in range(args.maxLen):
                    if j < len(tokens):
                        idx = vocab.get_id(tokens[j])
                        x_data[i, j] = idx if idx != -1 else 0
                    else:
                        x_data[i, j] = 0
                y_sent[i] = sent
                y_topic[i] = top
                
            x_tensor = torch.tensor(x_data).to(device)
            y_sent_tensor = torch.tensor(y_sent).to(device)
            y_topic_tensor = torch.tensor(y_topic).to(device)
            
            optimizer.zero_grad()
            sent_logits, topic_logits = model(x_tensor)
            
            loss_sent = criterion(sent_logits, y_sent_tensor)
            loss_topic = criterion(topic_logits, y_topic_tensor)
            loss = loss_sent + loss_topic
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_sent_loss += loss_sent.item()
            total_topic_loss += loss_topic.item()
            
            preds_sent = torch.argmax(sent_logits, dim=1)
            preds_topic = torch.argmax(topic_logits, dim=1)
            correct_sent += (preds_sent == y_sent_tensor).sum().item()
            correct_topic += (preds_topic == y_topic_tensor).sum().item()
            total_samples += current_bs
            
        train_acc_sent = correct_sent / max(1, total_samples)
        train_acc_topic = correct_topic / max(1, total_samples)
        
        # Eval on dev
        model.eval()
        dev_loss = 0
        correct_sent_dev = 0
        correct_topic_dev = 0
        total_samples_dev = 0
        
        with torch.no_grad():
            for b in range((len(dev_data) + args.batchSize - 1) // args.batchSize):
                batch = dev_data[b * args.batchSize : (b + 1) * args.batchSize]
                current_bs = len(batch)
                x_data = np.zeros((current_bs, args.maxLen), dtype=np.int64)
                y_sent = np.zeros(current_bs, dtype=np.int64)
                y_topic = np.zeros(current_bs, dtype=np.int64)
                for i, (text, sent, top) in enumerate(batch):
                    tokens = tokenize(text)
                    for j in range(args.maxLen):
                        if j < len(tokens):
                            idx = vocab.get_id(tokens[j])
                            x_data[i, j] = idx if idx != -1 else 0
                        else:
                            x_data[i, j] = 0
                    y_sent[i] = sent
                    y_topic[i] = top
                x_tensor = torch.tensor(x_data).to(device)
                y_sent_tensor = torch.tensor(y_sent).to(device)
                y_topic_tensor = torch.tensor(y_topic).to(device)
                sent_logits, topic_logits = model(x_tensor)
                l_s = criterion(sent_logits, y_sent_tensor)
                l_t = criterion(topic_logits, y_topic_tensor)
                dev_loss += (l_s + l_t).item()
                preds_sent = torch.argmax(sent_logits, dim=1)
                preds_topic = torch.argmax(topic_logits, dim=1)
                correct_sent_dev += (preds_sent == y_sent_tensor).sum().item()
                correct_topic_dev += (preds_topic == y_topic_tensor).sum().item()
                total_samples_dev += current_bs
                
        dev_acc_sent = correct_sent_dev / max(1, total_samples_dev)
        dev_acc_topic = correct_topic_dev / max(1, total_samples_dev)
        
        epoch_ms = int((time.time() - t0) * 1000)
        epoch_sec = epoch_ms / 1000.0
        
        print(f"Epoch {epoch+1}/{args.epochs} | lr=0.001000 | train_loss={total_loss/max(1, num_batches):.4f} "
              f"(sent={total_sent_loss/max(1, num_batches):.4f} topic={total_topic_loss/max(1, num_batches):.4f}) | "
              f"train_sent_acc={train_acc_sent*100:.2f}% | train_topic_acc={train_acc_topic*100:.2f}% | "
              f"dev_loss={dev_loss/max(1, num_batches):.4f} | dev_sent_acc={dev_acc_sent*100:.2f}% | "
              f"dev_topic_acc={dev_acc_topic*100:.2f}% | epoch_time={epoch_sec:.3f}s")

if __name__ == "__main__":
    main()
