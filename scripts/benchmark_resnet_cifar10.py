#!/usr/bin/env python3
"""Phase 3: PyTorch ResNet benchmark on local CIFAR-10 binary files."""

import argparse
import csv
import gc
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)


def get_device(device_str: str) -> torch.device:
    if device_str.lower() == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_memory_usage_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def read_cifar_bin_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 3073 != 0:
        raise ValueError(f"Invalid CIFAR binary format: {path}")
    rows = raw.reshape(-1, 3073)
    labels = rows[:, 0].astype(np.int64)
    images = rows[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    images = (images - CIFAR_MEAN) / CIFAR_STD
    return images, labels


def load_local_cifar10(batch_size: int, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    logger.info("[PyTorch][ResNet] Preparing CIFAR-10 data from local binary files...")
    base = Path(__file__).resolve().parent.parent / "data" / "cifar-10" / "cifar-10-batches-bin"

    train_parts: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    for i in range(1, 6):
        imgs, lbs = read_cifar_bin_file(base / f"data_batch_{i}.bin")
        train_parts.append(imgs)
        train_labels.append(lbs)

    x_train = np.concatenate(train_parts, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    x_test, y_test = read_cifar_bin_file(base / "test_batch.bin")

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    logger.info("[PyTorch][ResNet] CIFAR-10 loaded: train=%d test=%d", len(train_ds), len(test_ds))
    return train_loader, test_loader


def build_model(device: torch.device) -> nn.Module:
    logger.info("[PyTorch][ResNet] Building ResNet18 architecture...")
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def append_csv_row(path: Path, row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def fmt(v: float) -> str:
    return f"{v:.4f}"


def base_row(run_id: str, device: str, seed: int, batch_size: int, epochs: int, mixed_precision: bool) -> Dict[str, str]:
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "seed": str(seed),
        "train_batch_size": str(batch_size),
        "epochs": str(epochs),
        "mixed_precision": str(mixed_precision).lower(),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    max_train_batches: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for idx, (x, y) in enumerate(loader):
        if max_train_batches > 0 and idx >= max_train_batches:
            break
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

        if idx == 0 or (idx + 1) % 100 == 0 or (idx + 1) == len(loader):
            logger.info(
                "[PyTorch][ResNet][Train] epoch=%d/%d batch=%d/%d loss=%.5f",
                epoch + 1,
                total_epochs,
                idx + 1,
                len(loader),
                loss.item(),
            )

    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_eval_batches: int) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if max_eval_batches > 0 and idx >= max_eval_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(1, total)


def benchmark_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    infer_csv: Path,
    run_id: str,
    seed: int,
    batch_size: int,
    epochs: int,
    mixed_precision: bool,
    max_eval_batches: int,
) -> Dict[str, float]:
    model.eval()
    lat_ms: List[float] = []
    total_samples = 0
    step = 0

    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            if max_eval_batches > 0 and idx >= max_eval_batches:
                break
            if step >= warmup_steps + measure_steps:
                break

            x = x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.time() - t0) * 1000.0

            if step >= warmup_steps:
                lat_ms.append(dt_ms)
                total_samples += x.size(0)
                row = base_row(run_id, device.type, seed, batch_size, epochs, mixed_precision)
                row.update(
                    {
                        "framework": "pytorch",
                        "task": "resnet_cifar10",
                        "step": str(step - warmup_steps + 1),
                        "batch_size": str(x.size(0)),
                        "latency_ms": fmt(dt_ms),
                    }
                )
                append_csv_row(infer_csv, row)

            step += 1

    if not lat_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "throughput_sps": 0.0}

    lat_ms.sort()
    p50 = lat_ms[len(lat_ms) // 2]
    p95 = lat_ms[min(int(len(lat_ms) * 0.95), len(lat_ms) - 1)]
    throughput = total_samples / (sum(lat_ms) / 1000.0)
    return {"p50_ms": p50, "p95_ms": p95, "throughput_sps": throughput}


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch ResNet18 CIFAR-10 Benchmark")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixedPrecision", action="store_true")
    parser.add_argument("--inferWarmup", type=int, default=10)
    parser.add_argument("--inferSteps", type=int, default=50)
    parser.add_argument("--maxTrainBatches", type=int, default=0)
    parser.add_argument("--maxEvalBatches", type=int, default=0)
    parser.add_argument("--outputDir", type=str, default="benchmark/results")
    parser.add_argument("--runId", type=str, default=None)
    args = parser.parse_args()

    run_id = args.runId or f"resnet_cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.device}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    logger.info("[PyTorch][ResNet] Using device: %s", device)

    train_loader, test_loader = load_local_cifar10(args.batchSize, device)
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    run_dir = Path(args.outputDir) / "pytorch" / "resnet_cifar10" / run_id
    epoch_csv = run_dir / "epoch_metrics.csv"
    infer_csv = run_dir / "inference_samples.csv"
    summary_csv = run_dir / "run_summary.csv"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_start = time.time()
    peak_heap_mb = get_memory_usage_mb()
    peak_vram_mb = get_peak_vram_mb()
    best_val = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.epochs,
            args.maxTrainBatches,
        )
        val_acc = evaluate(model, test_loader, device, args.maxEvalBatches)
        epoch_ms = int((time.time() - t0) * 1000)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch

        peak_heap_mb = max(peak_heap_mb, get_memory_usage_mb())
        peak_vram_mb = max(peak_vram_mb, get_peak_vram_mb())

        row = base_row(run_id, args.device, args.seed, args.batchSize, args.epochs, args.mixedPrecision)
        row.update(
            {
                "framework": "pytorch",
                "task": "resnet_cifar10",
                "epoch": str(epoch + 1),
                "train_loss": fmt(train_loss),
                "train_acc": fmt(train_acc),
                "val_acc": fmt(val_acc),
                "epoch_time_ms": str(epoch_ms),
                "peak_heap_mb": fmt(peak_heap_mb),
                "peak_vram_mb": fmt(peak_vram_mb),
            }
        )
        append_csv_row(epoch_csv, row)

        logger.info(
            "[PyTorch][ResNet] epoch=%d/%d loss=%.5f train_acc=%.4f val_acc=%.4f time_ms=%d",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_acc,
            epoch_ms,
        )

    logger.info("[PyTorch][ResNet] Benchmarking inference...")
    infer = benchmark_inference(
        model,
        test_loader,
        device,
        args.inferWarmup,
        args.inferSteps,
        infer_csv,
        run_id,
        args.seed,
        args.batchSize,
        args.epochs,
        args.mixedPrecision,
        args.maxEvalBatches,
    )

    total_ms = int((time.time() - total_start) * 1000)
    summary = base_row(run_id, args.device, args.seed, args.batchSize, args.epochs, args.mixedPrecision)
    summary.update(
        {
            "framework": "pytorch",
            "task": "resnet_cifar10",
            "best_val_acc": fmt(best_val),
            "best_epoch": str(best_epoch + 1),
            "total_train_time_ms": str(total_ms),
            "inference_p50_ms": fmt(infer["p50_ms"]),
            "inference_p95_ms": fmt(infer["p95_ms"]),
            "inference_throughput_sps": fmt(infer["throughput_sps"]),
            "peak_heap_mb": fmt(peak_heap_mb),
            "peak_vram_mb": fmt(peak_vram_mb),
        }
    )
    append_csv_row(summary_csv, summary)

    logger.info("[PyTorch][ResNet] Finished. Artifacts in: %s", run_dir.resolve())

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
