#!/usr/bin/env python3
"""PyTorch UIT-VSFC multitask benchmark with Java-parity artifacts."""

import argparse
import csv
import gc
import logging
import os
import random
import re
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

TASK_NAME = "uit_vsfc_multitask"
DEFAULT_DATA_DIR = "examples/data/uit-vsfc"


def fmt(v: float) -> str:
    return f"{v:.6f}"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_csv_row(path: Path, row: OrderedDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_confusion_csv(path: Path, matrix: np.ndarray, labels: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\pred", *labels])
        for i, row in enumerate(matrix.tolist()):
            row_label = labels[i] if i < len(labels) else str(i)
            writer.writerow([row_label, *row])


def get_memory_usage_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def get_device(device_str: str) -> torch.device:
    if device_str.lower() == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LabelEncoder:
    def __init__(self) -> None:
        self.label_to_id: Dict[str, int] = OrderedDict()
        self.id_to_label: List[str] = []

    def fit_or_get(self, label: str) -> int:
        key = self._normalize_label(label)
        if key in self.label_to_id:
            return self.label_to_id[key]
        idx = len(self.id_to_label)
        self.label_to_id[key] = idx
        self.id_to_label.append(key)
        return idx

    def get_id(self, label: str) -> int:
        key = self._normalize_label(label)
        if key not in self.label_to_id:
            raise ValueError(f"Unknown label: {label}")
        return self.label_to_id[key]

    @property
    def labels(self) -> List[str]:
        return self.id_to_label

    @staticmethod
    def _normalize_label(s: str) -> str:
        return "" if s is None else s.strip().lower()


class VietnameseTokenizer:
    REPLACEMENTS = {
        "ko": "khong",
        "k": "khong",
        "kh": "khong",
        "hok": "khong",
        "dc": "duoc",
        "đc": "duoc",
        "cx": "cung",
        "vs": "voi",
        "mik": "minh",
        "mk": "minh",
        "bt": "binhthuong",
        "ntn": "nhu_the_nao",
        "j": "gi",
    }

    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        s = text.strip().lower()
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"https?://\S+", " url ", s)
        s = re.sub(r"[@#][\w\d_]+", " mention ", s)
        s = re.sub(r"[^\w\d\s_]", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def tokenize(self, text: str) -> List[str]:
        normalized = self.normalize(text)
        if not normalized:
            return []
        out: List[str] = []
        for part in normalized.split():
            out.append(self.REPLACEMENTS.get(part, part))
        return out


class Vocabulary:
    def __init__(self) -> None:
        self.word_to_id: Dict[str, int] = {}
        self.next_id = 1

    def add_word(self, word: str) -> None:
        if word not in self.word_to_id:
            self.word_to_id[word] = self.next_id
            self.next_id += 1

    def get_id(self, word: str) -> int:
        return self.word_to_id.get(word, 0)

    def size(self) -> int:
        return len(self.word_to_id) + 1


@dataclass
class Entry:
    text: str
    sentiment_label: str
    topic_label: str
    sentiment_id: int
    topic_id: int


@dataclass
class DatasetSplits:
    train: List[Entry]
    dev: List[Entry]
    test: List[Entry]
    sentiment_labels: List[str]
    topic_labels: List[str]


@dataclass
class TaskMetrics:
    accuracy: float
    macro_f1: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    confusion: np.ndarray


@dataclass
class EvalResult:
    loss: float
    sentiment: TaskMetrics
    topic: TaskMetrics
    joint_exact_match: float


class MultiTaskLSTM(nn.Module):
    def __init__(self, vocab_size: int, sentiment_classes: int, topic_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.sentiment_head = nn.Linear(256, sentiment_classes)
        self.topic_head = nn.Linear(256, topic_classes)

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled = self.dropout(out[:, -1, :])
        return self.sentiment_head(pooled), self.topic_head(pooled)


class MultiTaskTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, sentiment_classes: int, topic_classes: int) -> None:
        super().__init__()
        d_model = 128
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.sentiment_head = nn.Linear(d_model, sentiment_classes)
        self.topic_head = nn.Linear(d_model, topic_classes)

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maxLen={self.max_len}")
        emb = self.embedding(x)
        h = emb + self.pos_embed[:, :seq_len, :]
        h = self.encoder(h)
        pooled = self.norm(h[:, 0, :])
        return self.sentiment_head(pooled), self.topic_head(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_root(data_dir: str) -> Path:
    candidates = [Path(data_dir), Path(DEFAULT_DATA_DIR), Path("data/uit-vsfc")]
    seen = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"UIT-VSFC directory not found. Checked: {[str(x) for x in candidates]}")


def read_split(root: Path, split: str) -> Tuple[List[str], List[str], List[str]]:
    split_dir = root / split
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    text_file = split_dir / "sents.txt"
    sent_file = split_dir / "sentiments.txt"
    topic_file = split_dir / "topics.txt"
    if not text_file.exists() or not sent_file.exists() or not topic_file.exists():
        raise FileNotFoundError(f"Split {split} must contain sents.txt, sentiments.txt, topics.txt")

    texts = text_file.read_text(encoding="utf-8").splitlines()
    sentiments = [x.strip() for x in sent_file.read_text(encoding="utf-8").splitlines()]
    topics = [x.strip() for x in topic_file.read_text(encoding="utf-8").splitlines()]

    if not (len(texts) == len(sentiments) == len(topics)):
        raise ValueError(
            f"Mismatched line counts in split {split} "
            f"(sents={len(texts)}, sentiments={len(sentiments)}, topics={len(topics)})"
        )
    if len(texts) == 0:
        raise ValueError(f"Split {split} is empty")

    return texts, sentiments, topics


def encode_split(
    texts: List[str],
    sentiments: List[str],
    topics: List[str],
    sentiment_encoder: LabelEncoder,
    topic_encoder: LabelEncoder,
    fit: bool,
) -> List[Entry]:
    out: List[Entry] = []
    for i, text in enumerate(texts):
        sent_label = sentiments[i]
        topic_label = topics[i]
        if not text.strip() or not sent_label or not topic_label:
            continue
        if fit:
            sent_id = sentiment_encoder.fit_or_get(sent_label)
            topic_id = topic_encoder.fit_or_get(topic_label)
        else:
            sent_id = sentiment_encoder.get_id(sent_label)
            topic_id = topic_encoder.get_id(topic_label)
        out.append(Entry(text=text, sentiment_label=sent_label, topic_label=topic_label, sentiment_id=sent_id, topic_id=topic_id))
    return out


def load_uit_vsfc(data_dir: str) -> DatasetSplits:
    root = resolve_data_root(data_dir)
    train_texts, train_sent, train_topic = read_split(root, "train")
    dev_texts, dev_sent, dev_topic = read_split(root, "dev")
    test_texts, test_sent, test_topic = read_split(root, "test")

    sent_enc = LabelEncoder()
    topic_enc = LabelEncoder()
    train = encode_split(train_texts, train_sent, train_topic, sent_enc, topic_enc, fit=True)
    dev = encode_split(dev_texts, dev_sent, dev_topic, sent_enc, topic_enc, fit=False)
    test = encode_split(test_texts, test_sent, test_topic, sent_enc, topic_enc, fit=False)

    return DatasetSplits(
        train=train,
        dev=dev,
        test=test,
        sentiment_labels=sent_enc.labels,
        topic_labels=topic_enc.labels,
    )


def build_vocab(entries: List[Entry], tokenizer: VietnameseTokenizer) -> Vocabulary:
    vocab = Vocabulary()
    for e in entries:
        for token in tokenizer.tokenize(e.text):
            vocab.add_word(token)
    return vocab


def create_batch(entries: List[Entry], start: int, end: int, tokenizer: VietnameseTokenizer, vocab: Vocabulary, max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = entries[start:end]
    x = np.zeros((len(batch), max_len), dtype=np.int64)
    y_sent = np.zeros((len(batch),), dtype=np.int64)
    y_topic = np.zeros((len(batch),), dtype=np.int64)

    for i, e in enumerate(batch):
        tokens = tokenizer.tokenize(e.text)
        limit = min(max_len, len(tokens))
        for j in range(limit):
            x[i, j] = vocab.get_id(tokens[j])
        y_sent[i] = e.sentiment_id
        y_topic[i] = e.topic_id

    return torch.from_numpy(x), torch.from_numpy(y_sent), torch.from_numpy(y_topic)


def compute_task_metrics(confusion: np.ndarray) -> TaskMetrics:
    classes = confusion.shape[0]
    precision = np.zeros(classes, dtype=np.float64)
    recall = np.zeros(classes, dtype=np.float64)
    f1 = np.zeros(classes, dtype=np.float64)

    total = confusion.sum()
    correct = np.trace(confusion)

    for i in range(classes):
        tp = confusion[i, i]
        row_sum = confusion[i, :].sum()
        col_sum = confusion[:, i].sum()
        p = tp / col_sum if col_sum > 0 else 0.0
        r = tp / row_sum if row_sum > 0 else 0.0
        precision[i] = p
        recall[i] = r
        f1[i] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    macro_f1 = float(f1.mean()) if classes > 0 else 0.0
    accuracy = float(correct / total) if total > 0 else 0.0
    return TaskMetrics(accuracy=accuracy, macro_f1=macro_f1, precision=precision, recall=recall, f1=f1, confusion=confusion)


def evaluate(
    model: nn.Module,
    entries: List[Entry],
    tokenizer: VietnameseTokenizer,
    vocab: Vocabulary,
    max_len: int,
    batch_size: int,
    device: torch.device,
    alpha: float,
    beta: float,
    sentiment_classes: int,
    topic_classes: int,
) -> EvalResult:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = (len(entries) + batch_size - 1) // batch_size

    sent_conf = np.zeros((sentiment_classes, sentiment_classes), dtype=np.int64)
    topic_conf = np.zeros((topic_classes, topic_classes), dtype=np.int64)
    joint_correct = 0
    total_samples = 0

    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, len(entries))
            x, y_sent, y_topic = create_batch(entries, start, end, tokenizer, vocab, max_len)
            x = x.to(device)
            y_sent = y_sent.to(device)
            y_topic = y_topic.to(device)

            sent_logits, topic_logits = model.forward_both(x)
            loss_sent = ce(sent_logits, y_sent)
            loss_topic = ce(topic_logits, y_topic)
            loss = alpha * loss_sent + beta * loss_topic
            total_loss += float(loss.item())

            sent_pred = torch.argmax(sent_logits, dim=1)
            topic_pred = torch.argmax(topic_logits, dim=1)

            y_sent_cpu = y_sent.cpu().numpy()
            y_topic_cpu = y_topic.cpu().numpy()
            sent_pred_cpu = sent_pred.cpu().numpy()
            topic_pred_cpu = topic_pred.cpu().numpy()

            for i in range(len(y_sent_cpu)):
                sent_conf[y_sent_cpu[i], sent_pred_cpu[i]] += 1
                topic_conf[y_topic_cpu[i], topic_pred_cpu[i]] += 1
                if sent_pred_cpu[i] == y_sent_cpu[i] and topic_pred_cpu[i] == y_topic_cpu[i]:
                    joint_correct += 1
                total_samples += 1

    sent_metrics = compute_task_metrics(sent_conf)
    topic_metrics = compute_task_metrics(topic_conf)
    avg_loss = total_loss / max(1, num_batches)
    joint = joint_correct / total_samples if total_samples > 0 else 0.0
    model.train()
    return EvalResult(loss=avg_loss, sentiment=sent_metrics, topic=topic_metrics, joint_exact_match=joint)


def select_objective(dev_eval: EvalResult, selection: str, alpha: float, beta: float) -> float:
    if selection == "sentiment":
        return dev_eval.sentiment.macro_f1
    if selection == "topic":
        return dev_eval.topic.macro_f1
    w = alpha + beta
    wa = alpha / w if w > 0 else 0.5
    wb = beta / w if w > 0 else 0.5
    return wa * dev_eval.sentiment.macro_f1 + wb * dev_eval.topic.macro_f1


def benchmark_inference(
    model: nn.Module,
    entries: List[Entry],
    tokenizer: VietnameseTokenizer,
    vocab: Vocabulary,
    max_len: int,
    batch_size: int,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    infer_csv: Path,
    base_meta: OrderedDict,
) -> Dict[str, float]:
    model.eval()
    latencies: List[float] = []
    total_samples = 0
    seen = 0
    cursor = 0

    with torch.no_grad():
        while seen < warmup_steps + measure_steps and len(entries) > 0:
            end = min(cursor + batch_size, len(entries))
            x, _, _ = create_batch(entries, cursor, end, tokenizer, vocab, max_len)
            x = x.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model.forward_both(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.time() - t0) * 1000.0

            if seen >= warmup_steps:
                latencies.append(dt_ms)
                total_samples += x.size(0)
                row = OrderedDict(base_meta)
                row["step"] = str(len(latencies))
                row["batch_size"] = str(x.size(0))
                row["latency_ms"] = fmt(dt_ms)
                append_csv_row(infer_csv, row)

            seen += 1
            cursor = end
            if cursor >= len(entries) and seen < warmup_steps + measure_steps:
                cursor = 0

    if not latencies:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "throughput_sps": float("nan")}

    latencies.sort()
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    throughput = total_samples / (sum(latencies) / 1000.0) if sum(latencies) > 0 else float("nan")
    return {"p50_ms": p50, "p95_ms": p95, "throughput_sps": throughput}


def build_base_row(
    run_id: str,
    model_name: str,
    device: str,
    seed: int,
    batch_size: int,
    epochs: int,
    max_len: int,
    alpha: float,
    beta: float,
) -> OrderedDict:
    row = OrderedDict()
    row["run_id"] = run_id
    row["timestamp"] = timestamp()
    row["framework"] = "pytorch"
    row["task"] = TASK_NAME
    row["model"] = model_name
    row["device"] = device
    row["seed"] = str(seed)
    row["train_batch_size"] = str(batch_size)
    row["mixed_precision"] = "false"
    row["batch_size"] = str(batch_size)
    row["epochs"] = str(epochs)
    row["max_len"] = str(max_len)
    row["alpha"] = fmt(alpha)
    row["beta"] = fmt(beta)
    return row


def write_per_class_rows(path: Path, base_meta: OrderedDict, split: str, head: str, labels: List[str], metrics: TaskMetrics) -> None:
    for i, cls in enumerate(labels):
        row = OrderedDict()
        row["run_id"] = base_meta["run_id"]
        row["task"] = TASK_NAME
        row["model"] = base_meta["model"]
        row["device"] = base_meta["device"]
        row["split"] = split
        row["head"] = head
        row["class_id"] = str(i)
        row["class_name"] = cls
        row["precision"] = fmt(float(metrics.precision[i]))
        row["recall"] = fmt(float(metrics.recall[i]))
        row["f1"] = fmt(float(metrics.f1[i]))
        append_csv_row(path, row)


def array_to_str(values: np.ndarray) -> str:
    return ";".join(fmt(float(v)) for v in values.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch UIT-VSFC Multitask Benchmark")
    parser.add_argument("--dataDir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--maxLen", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--selection", default="weighted", choices=["weighted", "sentiment", "topic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inferWarmup", type=int, default=10)
    parser.add_argument("--inferSteps", type=int, default=50)
    parser.add_argument("--outputDir", default="benchmark/results")
    parser.add_argument("--runId", default=None)
    parser.add_argument("--checkpointPath", default="")
    args = parser.parse_args()

    if args.alpha < 0 or args.beta < 0 or (args.alpha + args.beta) <= 0:
        raise ValueError("alpha/beta must be non-negative and not both zero")

    run_id = args.runId or f"uit_vsfc_{args.model}_{timestamp()}_{args.device}"
    set_seed(args.seed)
    device = get_device(args.device)

    logger.info("[PyTorch][UIT] Loading UIT-VSFC dataset from: %s", args.dataDir)
    splits = load_uit_vsfc(args.dataDir)
    logger.info("[PyTorch][UIT] Train=%d Dev=%d Test=%d", len(splits.train), len(splits.dev), len(splits.test))
    logger.info("[PyTorch][UIT] Sentiment classes=%d Topic classes=%d", len(splits.sentiment_labels), len(splits.topic_labels))

    tokenizer = VietnameseTokenizer()
    vocab = build_vocab(splits.train, tokenizer)
    logger.info("[PyTorch][UIT] Vocabulary size=%d", vocab.size())

    if args.model == "lstm":
        model: nn.Module = MultiTaskLSTM(vocab.size(), len(splits.sentiment_labels), len(splits.topic_labels))
    else:
        model = MultiTaskTransformer(vocab.size(), args.maxLen, len(splits.sentiment_labels), len(splits.topic_labels))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    run_dir = Path(args.outputDir) / "pytorch" / TASK_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    epoch_csv = run_dir / "epoch_metrics.csv"
    infer_csv = run_dir / "inference_samples.csv"
    summary_csv = run_dir / "run_summary.csv"
    per_class_csv = run_dir / "per_class_metrics.csv"

    best_checkpoint = Path(args.checkpointPath) if args.checkpointPath else (run_dir / "best_model.pt")
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_start = time.time()
    best_dev_objective = -1e9
    best_epoch = -1
    cumulative_epoch_ms = 0
    peak_heap_mb = get_memory_usage_mb()
    peak_vram_mb = get_peak_vram_mb()

    shuffled = list(splits.train)

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(shuffled)
        model.train()

        total_loss = 0.0
        total_sent_loss = 0.0
        total_topic_loss = 0.0
        train_sent_correct = 0
        train_topic_correct = 0
        train_total = 0

        num_batches = (len(shuffled) + args.batchSize - 1) // args.batchSize
        epoch_start = time.time()

        for b in range(num_batches):
            start = b * args.batchSize
            end = min((b + 1) * args.batchSize, len(shuffled))
            x, y_sent, y_topic = create_batch(shuffled, start, end, tokenizer, vocab, args.maxLen)
            x = x.to(device)
            y_sent = y_sent.to(device)
            y_topic = y_topic.to(device)

            optimizer.zero_grad()
            sent_logits, topic_logits = model.forward_both(x)

            loss_sent = ce(sent_logits, y_sent)
            loss_topic = ce(topic_logits, y_topic)
            loss = args.alpha * loss_sent + args.beta * loss_topic
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_sent_loss += float(loss_sent.item())
            total_topic_loss += float(loss_topic.item())

            sent_pred = torch.argmax(sent_logits, dim=1)
            topic_pred = torch.argmax(topic_logits, dim=1)
            train_sent_correct += int((sent_pred == y_sent).sum().item())
            train_topic_correct += int((topic_pred == y_topic).sum().item())
            train_total += int(y_sent.size(0))

        train_loss = total_loss / max(1, num_batches)
        train_sent_loss = total_sent_loss / max(1, num_batches)
        train_topic_loss = total_topic_loss / max(1, num_batches)
        train_sent_acc = train_sent_correct / max(1, train_total)
        train_topic_acc = train_topic_correct / max(1, train_total)

        dev_eval = evaluate(
            model,
            splits.dev,
            tokenizer,
            vocab,
            args.maxLen,
            args.batchSize,
            device,
            args.alpha,
            args.beta,
            len(splits.sentiment_labels),
            len(splits.topic_labels),
        )

        dev_objective = select_objective(dev_eval, args.selection, args.alpha, args.beta)
        if dev_objective > best_dev_objective:
            best_dev_objective = dev_objective
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_checkpoint)

        epoch_ms = int((time.time() - epoch_start) * 1000)
        cumulative_epoch_ms += epoch_ms
        epoch_sec = epoch_ms / 1000.0
        avg_batch_ms = epoch_ms / max(1, num_batches)
        throughput = len(shuffled) / epoch_sec if epoch_sec > 0 else 0.0

        peak_heap_mb = max(peak_heap_mb, get_memory_usage_mb())
        peak_vram_mb = max(peak_vram_mb, get_peak_vram_mb())

        row = build_base_row(run_id, args.model, args.device, args.seed, args.batchSize, args.epochs, args.maxLen, args.alpha, args.beta)
        row["epoch"] = str(epoch + 1)
        row["train_loss_total"] = fmt(train_loss)
        row["train_loss_sentiment"] = fmt(train_sent_loss)
        row["train_loss_topic"] = fmt(train_topic_loss)
        row["train_sent_acc"] = fmt(train_sent_acc)
        row["train_topic_acc"] = fmt(train_topic_acc)
        row["dev_loss"] = fmt(dev_eval.loss)
        row["dev_sent_acc"] = fmt(dev_eval.sentiment.accuracy)
        row["dev_topic_acc"] = fmt(dev_eval.topic.accuracy)
        row["dev_sent_macro_f1"] = fmt(dev_eval.sentiment.macro_f1)
        row["dev_topic_macro_f1"] = fmt(dev_eval.topic.macro_f1)
        row["dev_joint_exact_match"] = fmt(dev_eval.joint_exact_match)
        row["dev_objective"] = fmt(dev_objective)
        row["epoch_time_ms"] = str(epoch_ms)
        row["cumulative_time_ms"] = str(cumulative_epoch_ms)
        row["avg_batch_time_ms"] = fmt(avg_batch_ms)
        row["throughput_samples_per_sec"] = fmt(throughput)
        row["peak_heap_mb"] = fmt(peak_heap_mb)
        row["peak_vram_mb"] = fmt(peak_vram_mb)
        append_csv_row(epoch_csv, row)

        logger.info(
            "[PyTorch][UIT] epoch=%d/%d train_loss=%.4f sent_acc=%.2f%% topic_acc=%.2f%% dev_sent_f1=%.4f dev_topic_f1=%.4f dev_joint=%.4f objective=%.4f time=%.3fs",
            epoch + 1,
            args.epochs,
            train_loss,
            train_sent_acc * 100.0,
            train_topic_acc * 100.0,
            dev_eval.sentiment.macro_f1,
            dev_eval.topic.macro_f1,
            dev_eval.joint_exact_match,
            dev_objective,
            epoch_sec,
        )

    if best_epoch <= 0:
        raise RuntimeError("No best checkpoint selected during training")

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))

    best_dev_eval = evaluate(
        model,
        splits.dev,
        tokenizer,
        vocab,
        args.maxLen,
        args.batchSize,
        device,
        args.alpha,
        args.beta,
        len(splits.sentiment_labels),
        len(splits.topic_labels),
    )
    test_eval = evaluate(
        model,
        splits.test,
        tokenizer,
        vocab,
        args.maxLen,
        args.batchSize,
        device,
        args.alpha,
        args.beta,
        len(splits.sentiment_labels),
        len(splits.topic_labels),
    )

    infer_meta = build_base_row(run_id, args.model, args.device, args.seed, args.batchSize, args.epochs, args.maxLen, args.alpha, args.beta)
    infer = benchmark_inference(
        model,
        splits.test,
        tokenizer,
        vocab,
        args.maxLen,
        args.batchSize,
        device,
        args.inferWarmup,
        args.inferSteps,
        infer_csv,
        infer_meta,
    )

    final_model_path = run_dir / f"uit_vsfc_{args.model}_multitask.pt"
    torch.save(model.state_dict(), final_model_path)

    write_confusion_csv(run_dir / "dev_confusion_sentiment.csv", best_dev_eval.sentiment.confusion, splits.sentiment_labels)
    write_confusion_csv(run_dir / "dev_confusion_topic.csv", best_dev_eval.topic.confusion, splits.topic_labels)
    write_confusion_csv(run_dir / "test_confusion_sentiment.csv", test_eval.sentiment.confusion, splits.sentiment_labels)
    write_confusion_csv(run_dir / "test_confusion_topic.csv", test_eval.topic.confusion, splits.topic_labels)

    base_meta = build_base_row(run_id, args.model, args.device, args.seed, args.batchSize, args.epochs, args.maxLen, args.alpha, args.beta)
    write_per_class_rows(per_class_csv, base_meta, "dev", "sentiment", splits.sentiment_labels, best_dev_eval.sentiment)
    write_per_class_rows(per_class_csv, base_meta, "dev", "topic", splits.topic_labels, best_dev_eval.topic)
    write_per_class_rows(per_class_csv, base_meta, "test", "sentiment", splits.sentiment_labels, test_eval.sentiment)
    write_per_class_rows(per_class_csv, base_meta, "test", "topic", splits.topic_labels, test_eval.topic)

    total_train_ms = int((time.time() - total_start) * 1000)

    summary = build_base_row(run_id, args.model, args.device, args.seed, args.batchSize, args.epochs, args.maxLen, args.alpha, args.beta)
    summary["selection"] = args.selection
    summary["best_epoch"] = str(best_epoch)
    summary["best_dev_objective"] = fmt(best_dev_objective)
    summary["best_checkpoint"] = str(best_checkpoint.absolute())

    summary["dev_loss"] = fmt(best_dev_eval.loss)
    summary["dev_sent_acc"] = fmt(best_dev_eval.sentiment.accuracy)
    summary["dev_topic_acc"] = fmt(best_dev_eval.topic.accuracy)
    summary["dev_sent_macro_f1"] = fmt(best_dev_eval.sentiment.macro_f1)
    summary["dev_topic_macro_f1"] = fmt(best_dev_eval.topic.macro_f1)
    summary["dev_joint_exact_match"] = fmt(best_dev_eval.joint_exact_match)

    summary["test_loss"] = fmt(test_eval.loss)
    summary["test_sent_acc"] = fmt(test_eval.sentiment.accuracy)
    summary["test_topic_acc"] = fmt(test_eval.topic.accuracy)
    summary["test_sent_macro_f1"] = fmt(test_eval.sentiment.macro_f1)
    summary["test_topic_macro_f1"] = fmt(test_eval.topic.macro_f1)
    summary["test_joint_exact_match"] = fmt(test_eval.joint_exact_match)

    summary["dev_sent_precision"] = array_to_str(best_dev_eval.sentiment.precision)
    summary["dev_sent_recall"] = array_to_str(best_dev_eval.sentiment.recall)
    summary["dev_sent_f1"] = array_to_str(best_dev_eval.sentiment.f1)
    summary["dev_topic_precision"] = array_to_str(best_dev_eval.topic.precision)
    summary["dev_topic_recall"] = array_to_str(best_dev_eval.topic.recall)
    summary["dev_topic_f1"] = array_to_str(best_dev_eval.topic.f1)

    summary["test_sent_precision"] = array_to_str(test_eval.sentiment.precision)
    summary["test_sent_recall"] = array_to_str(test_eval.sentiment.recall)
    summary["test_sent_f1"] = array_to_str(test_eval.sentiment.f1)
    summary["test_topic_precision"] = array_to_str(test_eval.topic.precision)
    summary["test_topic_recall"] = array_to_str(test_eval.topic.recall)
    summary["test_topic_f1"] = array_to_str(test_eval.topic.f1)

    summary["inference_p50_ms"] = fmt(infer["p50_ms"])
    summary["inference_p95_ms"] = fmt(infer["p95_ms"])
    summary["inference_throughput_sps"] = fmt(infer["throughput_sps"])
    summary["total_train_time_ms"] = str(total_train_ms)
    summary["peak_heap_mb"] = fmt(peak_heap_mb)
    summary["peak_vram_mb"] = fmt(peak_vram_mb)
    summary["final_model_path"] = str(final_model_path.absolute())
    append_csv_row(summary_csv, summary)

    logger.info(
        "[PyTorch][UIT] Best epoch=%d | Dev sent_macro_f1=%.4f topic_macro_f1=%.4f joint=%.4f",
        best_epoch,
        best_dev_eval.sentiment.macro_f1,
        best_dev_eval.topic.macro_f1,
        best_dev_eval.joint_exact_match,
    )
    logger.info(
        "[PyTorch][UIT] Test sent_acc=%.2f%% topic_acc=%.2f%% sent_macro_f1=%.4f topic_macro_f1=%.4f joint=%.4f",
        test_eval.sentiment.accuracy * 100.0,
        test_eval.topic.accuracy * 100.0,
        test_eval.sentiment.macro_f1,
        test_eval.topic.macro_f1,
        test_eval.joint_exact_match,
    )
    logger.info(
        "[PyTorch][UIT] Inference p50=%.4f ms p95=%.4f ms throughput=%.2f samples/s",
        infer["p50_ms"],
        infer["p95_ms"],
        infer["throughput_sps"],
    )
    logger.info("[PyTorch][UIT] Artifacts in: %s", str(run_dir.absolute()))

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
