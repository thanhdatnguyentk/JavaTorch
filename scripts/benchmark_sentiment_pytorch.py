#!/usr/bin/env python3
"""PyTorch RT-Polarity Sentiment LSTM benchmark - Phase 3"""
import os, sys, csv, time, argparse, logging, gc
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

class Tokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

class Vocabulary:
    def __init__(self):
        self.word_to_id, self.id_to_word, self.next_id = {}, {}, 1
    def add_word(self, word: str):
        if word not in self.word_to_id:
            self.word_to_id[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.next_id += 1
    def get_id(self, word: str) -> int:
        return self.word_to_id.get(word, 0)
    def size(self) -> int:
        return len(self.word_to_id) + 1

class ReviewDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, tokenizer: Tokenizer, max_len: int=20):
        self.texts, self.labels, self.vocab, self.tokenizer, self.max_len = texts, labels, vocab, tokenizer, max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.texts[idx])
        token_ids = [self.vocab.get_id(tokens[i]) if i < len(tokens) else 0 for i in range(self.max_len)]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, output_dim)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(0.3)
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)
        out = self.relu(self.fc1(self.dropout(hidden[-1])))
        return self.fc2(out)

def get_device(device_str: str) -> torch.device:
    if device_str.lower() == 'gpu' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def get_memory_usage() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except:
        return 0.0

def get_peak_vram() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0

def load_rtpolarity(tokenizer, vocab, max_len):
    logger.info("[PyTorch][Sentiment] Loading RT-Polarity data...")
    data_dir = Path(__file__).parent.parent / 'data' / 'rt-polarity'
    pos_texts = (data_dir / 'pos.txt').read_text(encoding='utf-8', errors='ignore').splitlines() if (data_dir / 'pos.txt').exists() else []
    neg_texts = (data_dir / 'neg.txt').read_text(encoding='utf-8', errors='ignore').splitlines() if (data_dir / 'neg.txt').exists() else []
    
    all_texts, all_labels = pos_texts + neg_texts, [1]*len(pos_texts) + [0]*len(neg_texts)
    split_idx = int(len(all_texts) * 0.8)
    
    for text in all_texts[:split_idx]:
        for token in tokenizer.tokenize(text):
            vocab.add_word(token)
    
    logger.info(f"[PyTorch][Sentiment] Dataset: total={len(all_texts)}, train={split_idx}, test={len(all_texts)-split_idx}")
    logger.info(f"[PyTorch][Sentiment] Vocabulary size={vocab.size()}")
    
    train_ds = ReviewDataset(all_texts[:split_idx], all_labels[:split_idx], vocab, tokenizer, max_len)
    test_ds = ReviewDataset(all_texts[split_idx:], all_labels[split_idx:], vocab, tokenizer, max_len)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=0)
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (text, label) in enumerate(train_loader):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            _, predicted = predictions.max(1)
            correct += predicted.eq(label).sum().item()
            total += label.size(0)
        if batch_idx == 0 or (batch_idx+1) % 50 == 0:
            logger.info(f"[PyTorch][Sentiment][Train] epoch={epoch+1}/{total_epochs} batch={batch_idx+1}/{len(train_loader)} loss={loss.item():.5f}")
    return total_loss / len(train_loader), 100.0*correct/total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for text, label in test_loader:
            text, label = text.to(device), label.to(device)
            predictions = model(text)
            _, predicted = predictions.max(1)
            correct += predicted.eq(label).sum().item()
            total += label.size(0)
    return 100.0*correct/total

def benchmark_inference(model, test_loader, device, warmup_steps, measure_steps):
    model.eval()
    latencies, total_samples, step = [], 0, 0
    with torch.no_grad():
        for text, label in test_loader:
            text = text.to(device)
            if step < warmup_steps:
                model(text)
                step += 1
                continue
            if len(latencies) >= measure_steps:
                break
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.time()
            model(text)
            if device.type == 'cuda': torch.cuda.synchronize()
            latencies.append((time.time()-start)*1000)
            total_samples += text.size(0)
            step += 1
    
    if not latencies:
        return {'p50_ms': 0.0, 'p95_ms': 0.0, 'throughput_sps': 0.0}
    latencies.sort()
    return {
        'p50_ms': latencies[len(latencies)//2],
        'p95_ms': latencies[int(len(latencies)*0.95)],
        'throughput_sps': total_samples/(sum(latencies)/1000) if sum(latencies)>0 else 0.0
    }

def append_csv_row(csv_path: Path, row: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def fmt(v: float) -> str:
    return f"{v:.4f}"

def base_row(run_id, device, seed, batch_size, epochs, mixed_precision):
    return {
        'run_id': run_id, 'timestamp': datetime.now().isoformat(), 'device': device,
        'seed': str(seed), 'train_batch_size': str(batch_size), 'epochs': str(epochs),
        'mixed_precision': str(mixed_precision).lower()
    }

def main():
    parser = argparse.ArgumentParser(description='PyTorch LSTM Sentiment Benchmark')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batchSize', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixedPrecision', action='store_true')
    parser.add_argument('--inferWarmup', type=int, default=10)
    parser.add_argument('--inferSteps', type=int, default=100)
    parser.add_argument('--maxLen', type=int, default=20)
    parser.add_argument('--outputDir', default='benchmark/results')
    parser.add_argument('--runId', default=None)
    args = parser.parse_args()
    
    if args.runId is None:
        args.runId = f"sentiment_rtpolarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}"
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    logger.info(f"[PyTorch][Sentiment] Using device: {device}")
    
    tokenizer, vocab = Tokenizer(), Vocabulary()
    train_loader, test_loader = load_rtpolarity(tokenizer, vocab, args.maxLen)
    
    model = SentimentModel(vocab_size=vocab.size()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    run_dir = Path(args.outputDir) / 'pytorch' / 'sentiment_rtpolarity' / args.runId
    epoch_csv = run_dir / 'epoch_metrics.csv'
    summary_csv = run_dir / 'run_summary.csv'
    
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    peak_heap_mb, peak_vram_mb = get_memory_usage(), get_peak_vram()
    best_acc, best_epoch = -1.0, -1
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_acc = evaluate(model, test_loader, criterion, device)
        epoch_time_ms = int((time.time()-epoch_start)*1000)
        
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
        
        peak_heap_mb = max(peak_heap_mb, get_memory_usage())
        peak_vram_mb = max(peak_vram_mb, get_peak_vram())
        
        row = base_row(args.runId, args.device, args.seed, args.batchSize, args.epochs, args.mixedPrecision)
        row.update({'framework': 'pytorch', 'task': 'sentiment_rtpolarity', 'epoch': str(epoch+1),
                   'train_loss': fmt(train_loss), 'train_acc': fmt(train_acc), 'val_acc': fmt(val_acc),
                   'epoch_time_ms': str(epoch_time_ms), 'peak_heap_mb': fmt(peak_heap_mb), 'peak_vram_mb': fmt(peak_vram_mb)})
        append_csv_row(epoch_csv, row)
        logger.info(f"[PyTorch][Sentiment] epoch={epoch+1}/{args.epochs} loss={train_loss:.5f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time_ms={epoch_time_ms}")
    
    logger.info("[PyTorch][Sentiment] Benchmarking inference...")
    infer = benchmark_inference(model, test_loader, device, args.inferWarmup, args.inferSteps)
    total_time_ms = int((time.time()-start_time)*1000)
    
    summary = base_row(args.runId, args.device, args.seed, args.batchSize, args.epochs, args.mixedPrecision)
    summary.update({'framework': 'pytorch', 'task': 'sentiment_rtpolarity', 'best_val_acc': fmt(best_acc),
                   'best_epoch': str(best_epoch+1), 'total_train_time_ms': str(total_time_ms),
                   'inference_p50_ms': fmt(infer['p50_ms']), 'inference_p95_ms': fmt(infer['p95_ms']),
                   'inference_throughput_sps': fmt(infer['throughput_sps']),
                   'peak_heap_mb': fmt(peak_heap_mb), 'peak_vram_mb': fmt(peak_vram_mb)})
    append_csv_row(summary_csv, summary)
    
    logger.info(f"[PyTorch][Sentiment] Finished. Artifacts in: {run_dir.absolute()}")
    del model
    gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
