#!/usr/bin/env python3
"""
Phase 3: PyTorch LSTM Sentiment Classification Benchmark (RT-Polarity)
Matches Java Phase 1 framework metrics and CSV schema
"""

import os
import sys
import csv
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s'
)
logger = logging.getLogger(__name__)


class Tokenizer:
    """Simple whitespace-based tokenizer matching Java BasicTokenizer"""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        # Convert to lowercase and split on whitespace
        tokens = text.lower().split()
        # Simple cleanup: remove empty strings
        return [t for t in tokens if t]


class Vocabulary:
    """Vocabulary builder matching Java Data.Vocabulary"""
    
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 1  # Reserve 0 for padding
    
    def add_word(self, word: str):
        """Add word to vocabulary"""
        if word not in self.word_to_id:
            self.word_to_id[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.next_id += 1
    
    def get_id(self, word: str) -> int:
        """Get word ID (0 if not in vocab)"""
        return self.word_to_id.get(word, 0)
    
    def size(self) -> int:
        """Get vocabulary size"""
        return len(self.word_to_id) + 1  # +1 for padding


class MovieCommentDataset(Dataset):
    """Movie review dataset matching Java MovieCommentLoader"""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, 
                 tokenizer: Tokenizer, max_len: int = 20):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Convert to IDs and pad/truncate
        token_ids = []
        for i in range(self.max_len):
            if i < len(tokens):
                token_ids.append(self.vocab.get_id(tokens[i]))
            else:
                token_ids.append(0)  # Padding
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class SentimentModel(nn.Module):
    """Sentiment classification model with Embedding + LSTM"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 output_dim: int, n_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embedding_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        out = self.relu(self.fc1(hidden))
        out = self.fc2(out)
        return out


def load_rt_polarity_data(data_dir: Path, tokenizer: Tokenizer, vocab: Vocabulary,
                          max_len: int, batch_size: int) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """Load RT-Polarity dataset"""
    
    logger.info("[PyTorch][Sentiment] Loading RT-Polarity data...")
    
    pos_file = data_dir / 'pos.txt'
    neg_file = data_dir / 'neg.txt'
    
    # Load positive reviews
    pos_texts = []
    if pos_file.exists():
        with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
            pos_texts = f.readlines()
    
    # Load negative reviews
    neg_texts = []
    if neg_file.exists():
        with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
            neg_texts = f.readlines()
    
    # Combine and create labels
    all_texts = pos_texts + neg_texts
    all_labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    
    # Build vocabulary from training set (we split 80/20)
    split_idx = int(len(all_texts) * 0.8)
    train_texts = all_texts[:split_idx]
    
    for text in train_texts:
        for token in tokenizer.tokenize(text):
            vocab.add_word(token)
    
    logger.info(f"[PyTorch][Sentiment] Dataset: total={len(all_texts)}, "
               f"train={split_idx}, test={len(all_texts)-split_idx}")
    logger.info(f"[PyTorch][Sentiment] Vocabulary size={vocab.size()}")
    
    # Create train and test sets
    train_dataset = MovieCommentDataset(
        train_texts, all_labels[:split_idx], vocab, tokenizer, max_len
    )
    
    test_dataset = MovieCommentDataset(
        all_texts[split_idx:], all_labels[split_idx:], vocab, tokenizer, max_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader, vocab


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    max_train_batches: int
) -> Tuple[float, float]:
    """Train one epoch"""
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    num_batches = len(train_loader)
    
    for batch_idx, (text, label) in enumerate(train_loader):
        if max_train_batches > 0 and batch_idx >= max_train_batches:
            break
        text = text.to(device)
        label = label.to(device)
        
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
        
        # Logging
        if batch_idx == 0 or (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
            logger.info(f"[PyTorch][Sentiment][Train] epoch={epoch+1}/{total_epochs} "
                       f"batch={batch_idx+1}/{num_batches} loss={loss.item():.5f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_eval_batches: int
) -> float:
    """Evaluate model on test set"""
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (text, label) in enumerate(test_loader):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
            text = text.to(device)
            label = label.to(device)
            
            predictions = model(text)
            _, predicted = predictions.max(1)
            correct += predicted.eq(label).sum().item()
            total += label.size(0)
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy


def benchmark_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    max_eval_batches: int,
    infer_csv: Path,
    run_id: str,
    seed: int,
    train_batch_size: int,
    epochs: int,
    mixed_precision: bool
) -> Dict[str, float]:
    """Benchmark inference latency and throughput"""
    
    model.eval()
    
    latencies = []
    total_samples = 0
    step = 0
    
    with torch.no_grad():
        for batch_idx, (text, label) in enumerate(test_loader):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
            text = text.to(device)
            
            # Warmup iterations
            if step < warmup_steps:
                model(text)
                step += 1
                continue
            
            # Measurement iterations
            if len(latencies) >= measure_steps:
                break
            
            # Synchronize if using GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            predictions = model(text)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            batch_size = text.size(0)
            
            latencies.append(latency_ms)
            total_samples += batch_size
            step += 1

            row = base_row(run_id, device.type, seed, train_batch_size, epochs, mixed_precision)
            row['framework'] = 'pytorch'
            row['task'] = 'sentiment_rtpolarity'
            row['step'] = str(len(latencies))
            row['batch_size'] = str(batch_size)
            row['latency_ms'] = format_float(latency_ms)
            append_csv_row(infer_csv, row)
    
    if not latencies:
        return {
            'p50_ms': 0.0,
            'p95_ms': 0.0,
            'throughput_sps': 0.0
        }
    
    latencies.sort()
    p50_idx = len(latencies) // 2
    p95_idx = int(len(latencies) * 0.95)
    
    p50_ms = latencies[p50_idx]
    p95_ms = latencies[min(p95_idx, len(latencies) - 1)]
    total_latency = sum(latencies)
    throughput_sps = total_samples / (total_latency / 1000) if total_latency > 0 else 0.0
    
    return {
        'p50_ms': p50_ms,
        'p95_ms': p95_ms,
        'throughput_sps': throughput_sps
    }


def get_device(device_str: str) -> torch.device:
    """Get PyTorch device"""
    if device_str.lower() == 'gpu':
        if not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA not available, falling back to CPU")
            return torch.device('cpu')
        return torch.device('cuda')
    return torch.device('cpu')


def get_peak_vram() -> float:
    """Get peak VRAM usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_peak_vram():
    """Reset peak VRAM counter"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_memory_usage() -> float:
    """Get current process memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0


def append_csv_row(csv_path: Path, row: Dict[str, str]):
    """Append row to CSV file"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)


def format_float(value: float) -> str:
    """Format float to 4 decimal places"""
    return f"{value:.4f}"


def base_row(run_id: str, device: str, seed: int, batch_size: int, epochs: int,
             mixed_precision: bool) -> Dict[str, str]:
    """Create base row with common fields"""
    return {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'seed': str(seed),
        'train_batch_size': str(batch_size),
        'epochs': str(epochs),
        'mixed_precision': str(mixed_precision).lower(),
    }


def main():
    parser = argparse.ArgumentParser(description='PyTorch LSTM Sentiment Benchmark')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=8,
                       help='Number of epochs')
    parser.add_argument('--batchSize', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--mixedPrecision', action='store_true',
                       help='Use mixed precision')
    parser.add_argument('--inferWarmup', type=int, default=10,
                       help='Inference warmup steps')
    parser.add_argument('--inferSteps', type=int, default=100,
                       help='Inference measurement steps')
    parser.add_argument('--maxTrainBatches', type=int, default=0,
                       help='Limit train batches per epoch (0 = all)')
    parser.add_argument('--maxEvalBatches', type=int, default=0,
                       help='Limit eval/inference batches (0 = all)')
    parser.add_argument('--maxLen', type=int, default=20,
                       help='Max sequence length')
    parser.add_argument('--outputDir', type=str, default='benchmark/results',
                       help='Output directory')
    parser.add_argument('--runId', type=str, default=None,
                       help='Run ID')
    
    args = parser.parse_args()
    
    if args.runId is None:
        args.runId = f"sentiment_rtpolarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}"
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = get_device(args.device)
    logger.info(f"[PyTorch][Sentiment] Using device: {device}")
    
    # Load data
    data_dir = Path(__file__).parent.parent / 'data' / 'rt-polarity'
    
    tokenizer = Tokenizer()
    vocab = Vocabulary()
    
    train_loader, test_loader, vocab = load_rt_polarity_data(
        data_dir, tokenizer, vocab, args.maxLen, args.batchSize
    )
    
    # Build model
    model = SentimentModel(
        vocab_size=vocab.size(),
        embedding_dim=32,
        hidden_dim=64,
        output_dim=2
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Output paths
    run_dir = Path(args.outputDir) / 'pytorch' / 'sentiment_rtpolarity' / args.runId
    epoch_csv = run_dir / 'epoch_metrics.csv'
    infer_csv = run_dir / 'inference_samples.csv'
    summary_csv = run_dir / 'run_summary.csv'
    
    reset_peak_vram()
    start_time = time.time()
    peak_heap_mb = get_memory_usage()
    peak_vram_mb = get_peak_vram()
    
    best_acc = -1.0
    best_epoch = -1
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs, args.maxTrainBatches
        )
        
        val_acc = evaluate(model, test_loader, criterion, device, args.maxEvalBatches)
        
        epoch_time_ms = int((time.time() - epoch_start) * 1000)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
        
        # Track peak memory
        peak_heap_mb = max(peak_heap_mb, get_memory_usage())
        peak_vram_mb = max(peak_vram_mb, get_peak_vram())
        
        # Log epoch
        row = base_row(args.runId, args.device, args.seed, args.batchSize, args.epochs,
                       args.mixedPrecision)
        row['framework'] = 'pytorch'
        row['task'] = 'sentiment_rtpolarity'
        row['epoch'] = str(epoch + 1)
        row['train_loss'] = format_float(train_loss)
        row['train_acc'] = format_float(train_acc)
        row['val_acc'] = format_float(val_acc)
        row['epoch_time_ms'] = str(epoch_time_ms)
        row['peak_heap_mb'] = format_float(peak_heap_mb)
        row['peak_vram_mb'] = format_float(peak_vram_mb)
        
        append_csv_row(epoch_csv, row)
        
        logger.info(f"[PyTorch][Sentiment] epoch={epoch+1}/{args.epochs} "
                   f"loss={train_loss:.5f} train_acc={train_acc:.4f} "
                   f"val_acc={val_acc:.4f} time_ms={epoch_time_ms}")
    
    # Benchmark inference
    logger.info("[PyTorch][Sentiment] Benchmarking inference...")
    infer_metrics = benchmark_inference(
        model,
        test_loader,
        device,
        args.inferWarmup,
        args.inferSteps,
        args.maxEvalBatches,
        infer_csv,
        args.runId,
        args.seed,
        args.batchSize,
        args.epochs,
        args.mixedPrecision,
    )
    
    total_time_ms = int((time.time() - start_time) * 1000)
    
    # Summary
    summary_row = base_row(args.runId, args.device, args.seed, args.batchSize, args.epochs,
                          args.mixedPrecision)
    summary_row['framework'] = 'pytorch'
    summary_row['task'] = 'sentiment_rtpolarity'
    summary_row['best_val_acc'] = format_float(best_acc)
    summary_row['best_epoch'] = str(best_epoch + 1)
    summary_row['total_train_time_ms'] = str(total_time_ms)
    summary_row['inference_p50_ms'] = format_float(infer_metrics['p50_ms'])
    summary_row['inference_p95_ms'] = format_float(infer_metrics['p95_ms'])
    summary_row['inference_throughput_sps'] = format_float(infer_metrics['throughput_sps'])
    summary_row['peak_heap_mb'] = format_float(peak_heap_mb)
    summary_row['peak_vram_mb'] = format_float(peak_vram_mb)
    
    append_csv_row(summary_csv, summary_row)
    
    logger.info(f"[PyTorch][Sentiment] Finished. Artifacts in: {run_dir.absolute()}")
    
    # Cleanup
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
