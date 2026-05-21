"""
Parse JavaTorch training logs and visualize Loss and Accuracy curves.
"""

from typing import Dict, List, Tuple
import os
import re
import matplotlib.pyplot as plt


def parse_vit_log(filepath: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Parses ViT CIFAR-10 training logs."""
    epochs, losses, train_accs, test_accs = [], [], [], []
    with open(filepath, "r", encoding="utf-16") as f:
        for line in f:
            if ">>> Epoch" in line:
                match = re.search(r"Epoch\s+(\d+)/\d+\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Train Acc:\s*([\d.]+)\s*\|\s*Test Acc:\s*([\d.]+)", line)
                if match:
                    epochs.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
                    train_accs.append(float(match.group(3)))
                    test_accs.append(float(match.group(4)))
    return epochs, losses, train_accs, test_accs


def parse_cifar10_fashion_log(filepath: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Parses CNN CIFAR-10 and LeNet FashionMNIST logs."""
    epochs, losses, train_accs, test_accs = [], [], [], []
    with open(filepath, "r", encoding="utf-16") as f:
        for line in f:
            if "avg_loss" in line and "batch" not in line and "Epoch" in line:
                match = re.search(r"Epoch\s+(\d+)/\d+\s+avg_loss=([\d.]+)\s+train_acc=([\d.]+)\s+test_acc=([\d.]+)", line)
                if match:
                    epochs.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
                    train_accs.append(float(match.group(3)))
                    test_accs.append(float(match.group(4)))
    return epochs, losses, train_accs, test_accs


def parse_sentiment_log(filepath: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Parses LSTM Sentiment movie review logs."""
    epochs, losses, train_accs, test_accs = [], [], [], []
    with open(filepath, "r", encoding="utf-16") as f:
        for line in f:
            if "Epoch" in line and "loss:" in line:
                match = re.search(r"Epoch\s+(\d+)/\d+\s+-\s+loss:\s*([\d.]+)\s+-\s+train_acc:\s*([\d.]+)%\s+-\s+test_acc:\s*([\d.]+)%", line)
                if match:
                    epochs.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
                    train_accs.append(float(match.group(3)) / 100.0)
                    test_accs.append(float(match.group(4)) / 100.0)
    return epochs, losses, train_accs, test_accs


def parse_resnet_log(filepath: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Parses ResNet-18 CIFAR-10 training logs."""
    epochs, losses, train_accs, test_accs = [], [], [], []
    with open(filepath, "r", encoding="utf-16") as f:
        for line in f:
            if "Epoch" in line and "lr=" in line:
                match = re.search(r"Epoch\s+(\d+)/\d+\s+lr=[\d.]+\s+avg_loss=([\d.]+)\s+train_acc=([\d.]+)\s+test_acc=([\d.]+)", line)
                if match:
                    epochs.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
                    train_accs.append(float(match.group(3)))
                    test_accs.append(float(match.group(4)))
    return epochs, losses, train_accs, test_accs


def parse_iris_log(filepath: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Parses MLP Iris dataset training logs."""
    epochs, losses, train_accs, test_accs = [], [], [], []
    with open(filepath, "r", encoding="utf-16") as f:
        for line in f:
            if "Epoch" in line and "loss=" in line:
                match = re.search(r"Epoch\s+(\d+)\s+loss=([\d.]+)\s+train_acc=([\d.]+)\s+test_acc=([\d.]+)", line)
                if match:
                    epochs.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
                    train_accs.append(float(match.group(3)))
                    test_accs.append(float(match.group(4)))
    return epochs, losses, train_accs, test_accs


def plot_training_curves(model_name: str, epochs: List[int], losses: List[float], train_accs: List[float], test_accs: List[float], output_filename: str) -> None:
    """Generates and saves a premium dark-themed training history plot."""
    if not epochs:
        print(f"No training data parsed for {model_name}. Skipping plot.")
        return

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#121212")
    
    fig.suptitle(f"{model_name} Training History (JavaTorch)", color="#FFFFFF", fontsize=18, fontweight="bold", y=0.98)

    # 1. Loss Curve
    ax1.set_facecolor("#1A1A1A")
    ax1.plot(epochs, losses, color="#FF5E5E", linewidth=2.5, marker="o", label="Training Loss")
    ax1.set_title("Loss Trajectory (Lower is Better)", color="#E0E0E0", fontsize=13, pad=12)
    ax1.set_xlabel("Epoch", color="#A0A0A0", fontsize=11, labelpad=8)
    ax1.set_ylabel("Loss", color="#A0A0A0", fontsize=11, labelpad=8)
    ax1.grid(True, linestyle="--", linewidth=0.5, color="#333333")
    ax1.legend(facecolor="#1E1E1E", edgecolor="#333333", loc="upper right")
    ax1.tick_params(colors="#C0C0C0", labelsize=10)

    # 2. Accuracy Curve
    ax2.set_facecolor("#1A1A1A")
    ax2.plot(epochs, train_accs, color="#FFC048", linewidth=2.5, marker="o", label="Train Acc")
    ax2.plot(epochs, test_accs, color="#2E8BFF", linewidth=2.5, marker="s", label="Test Acc")
    ax2.set_title("Accuracy Trajectory (Higher is Better)", color="#E0E0E0", fontsize=13, pad=12)
    ax2.set_xlabel("Epoch", color="#A0A0A0", fontsize=11, labelpad=8)
    ax2.set_ylabel("Accuracy", color="#A0A0A0", fontsize=11, labelpad=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", linewidth=0.5, color="#333333")
    ax2.legend(facecolor="#1E1E1E", edgecolor="#333333", loc="lower right")
    ax2.tick_params(colors="#C0C0C0", labelsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_filename, dpi=300, facecolor="#121212")
    plt.close()
    print(f"Saved training curve: {output_filename}")


def parse_multitask_log(filepath: str) -> Dict[str, List]:
    """Parses multitask LSTM UIT-VSFC training logs."""
    data = {
        "epochs": [],
        "total_losses": [],
        "sent_losses": [],
        "topic_losses": [],
        "train_sent_accs": [],
        "train_topic_accs": [],
        "dev_sent_f1s": [],
        "dev_topic_f1s": [],
        "dev_joints": [],
        "objectives": []
    }
    with open(filepath, "r", encoding="utf-16") as f:
        for line in f:
            if "Epoch" in line and "train_loss" in line:
                match = re.search(
                    r"Epoch\s+(\d+)/\d+\s*\|\s*lr=[\d.]+\s*\|\s*train_loss=([\d.]+)\s*\(sent=([\d.]+)\s+topic=([\d.]+)\)\s*\|\s*train_sent_acc=([\d.]+)%\s*\|\s*train_topic_acc=([\d.]+)%\s*\|\s*dev_sent_macro_f1=([\d.]+)\s*\|\s*dev_topic_macro_f1=([\d.]+)\s*\|\s*dev_joint=([\d.]+)\s*\|\s*objective=([\d.]+)",
                    line
                )
                if match:
                    data["epochs"].append(int(match.group(1)))
                    data["total_losses"].append(float(match.group(2)))
                    data["sent_losses"].append(float(match.group(3)))
                    data["topic_losses"].append(float(match.group(4)))
                    data["train_sent_accs"].append(float(match.group(5)) / 100.0)
                    data["train_topic_accs"].append(float(match.group(6)) / 100.0)
                    data["dev_sent_f1s"].append(float(match.group(7)))
                    data["dev_topic_f1s"].append(float(match.group(8)))
                    data["dev_joints"].append(float(match.group(9)))
                    data["objectives"].append(float(match.group(10)))
    return data


def plot_multitask_curves(
    model_name: str,
    data: Dict[str, List],
    output_filename: str
) -> None:
    """Generates and saves a premium dark-themed multitask training history plot."""
    epochs = data["epochs"]
    if not epochs:
        print(f"No training data parsed for {model_name}. Skipping plot.")
        return

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#121212")
    
    fig.suptitle(f"{model_name} Training History (JavaTorch)", color="#FFFFFF", fontsize=18, fontweight="bold", y=0.98)

    # 1. Loss Curves
    ax1.set_facecolor("#1A1A1A")
    ax1.plot(epochs, data["total_losses"], color="#FF5E5E", linewidth=2.5, marker="o", label="Total Loss")
    ax1.plot(epochs, data["sent_losses"], color="#FFC048", linewidth=2.0, linestyle="--", marker="s", label="Sentiment Loss")
    ax1.plot(epochs, data["topic_losses"], color="#2E8BFF", linewidth=2.0, linestyle="--", marker="^", label="Topic Loss")
    
    ax1.set_title("Loss Trajectories", color="#E0E0E0", fontsize=13, pad=12)
    ax1.set_xlabel("Epoch", color="#A0A0A0", fontsize=11, labelpad=8)
    ax1.set_ylabel("Loss", color="#A0A0A0", fontsize=11, labelpad=8)
    ax1.grid(True, linestyle="--", linewidth=0.5, color="#333333")
    ax1.legend(facecolor="#1E1E1E", edgecolor="#333333", loc="upper right")
    ax1.tick_params(colors="#C0C0C0", labelsize=10)

    # 2. Performance Metrics
    ax2.set_facecolor("#1A1A1A")
    ax2.plot(epochs, data["train_sent_accs"], color="#2ED573", linewidth=2.0, marker="o", label="Train Sentiment Acc")
    ax2.plot(epochs, data["train_topic_accs"], color="#10AC84", linewidth=2.0, marker="s", label="Train Topic Acc")
    ax2.plot(epochs, data["dev_sent_f1s"], color="#FDA7DF", linewidth=2.0, linestyle=":", marker="d", label="Dev Sentiment F1")
    ax2.plot(epochs, data["dev_topic_f1s"], color="#54A0FF", linewidth=2.0, linestyle=":", marker="v", label="Dev Topic F1")
    ax2.plot(epochs, data["dev_joints"], color="#FF9F43", linewidth=2.5, marker="*", label="Dev Joint Acc")
    
    ax2.set_title("Performance Metrics", color="#E0E0E0", fontsize=13, pad=12)
    ax2.set_xlabel("Epoch", color="#A0A0A0", fontsize=11, labelpad=8)
    ax2.set_ylabel("Score / Accuracy", color="#A0A0A0", fontsize=11, labelpad=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", linewidth=0.5, color="#333333")
    ax2.legend(facecolor="#1E1E1E", edgecolor="#333333", loc="lower right")
    ax2.tick_params(colors="#C0C0C0", labelsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_filename, dpi=300, facecolor="#121212")
    plt.close()
    print(f"Saved training curve: {output_filename}")


def parse_example_multitask_log(filepath: str) -> Dict[str, List]:
    """Parses multitask LSTM example training logs (exampleUitVsfc.log)."""
    data = {
        "epochs": [],
        "losses": [],
        "train_sent_accs": [],
        "train_topic_accs": [],
        "dev_sent_accs": [],
        "dev_topic_accs": [],
        "dev_sent_f1s": []
    }
    with open(filepath, "r", encoding="utf-16") as f:
        for line in f:
            if "Epoch" in line and "- loss:" in line:
                match = re.search(
                    r"Epoch\s+(\d+)/\d+\s+-\s+loss:\s*([\d.]+)\s*\|\s*Train Acc\s*\(Sent:\s*([\d.]+)%,\s*Topic:\s*([\d.]+)%\)\s*\|\s*Dev Acc\s*\(Sent:\s*([\d.]+)%,\s*Topic:\s*([\d.]+)%\)\s*\|\s*Dev MacroF1\s*\(Sent:\s*([\d.]+)\)",
                    line
                )
                if match:
                    data["epochs"].append(int(match.group(1)))
                    data["losses"].append(float(match.group(2)))
                    data["train_sent_accs"].append(float(match.group(3)) / 100.0)
                    data["train_topic_accs"].append(float(match.group(4)) / 100.0)
                    data["dev_sent_accs"].append(float(match.group(5)) / 100.0)
                    data["dev_topic_accs"].append(float(match.group(6)) / 100.0)
                    data["dev_sent_f1s"].append(float(match.group(7)))
    return data


def plot_example_multitask_curves(
    model_name: str,
    data: Dict[str, List],
    output_filename: str
) -> None:
    """Generates and saves a premium dark-themed multitask training history plot for example logs."""
    epochs = data["epochs"]
    if not epochs:
        print(f"No training data parsed for {model_name}. Skipping plot.")
        return

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#121212")
    
    fig.suptitle(f"{model_name} Training History (JavaTorch)", color="#FFFFFF", fontsize=18, fontweight="bold", y=0.98)

    # 1. Loss Curve
    ax1.set_facecolor("#1A1A1A")
    ax1.plot(epochs, data["losses"], color="#FF5E5E", linewidth=2.5, marker="o", label="Training Loss")
    
    ax1.set_title("Loss Trajectory (Lower is Better)", color="#E0E0E0", fontsize=13, pad=12)
    ax1.set_xlabel("Epoch", color="#A0A0A0", fontsize=11, labelpad=8)
    ax1.set_ylabel("Loss", color="#A0A0A0", fontsize=11, labelpad=8)
    ax1.grid(True, linestyle="--", linewidth=0.5, color="#333333")
    ax1.legend(facecolor="#1E1E1E", edgecolor="#333333", loc="upper right")
    ax1.tick_params(colors="#C0C0C0", labelsize=10)

    # 2. Performance Metrics
    ax2.set_facecolor("#1A1A1A")
    ax2.plot(epochs, data["train_sent_accs"], color="#2ED573", linewidth=2.0, marker="o", label="Train Sentiment Acc")
    ax2.plot(epochs, data["train_topic_accs"], color="#10AC84", linewidth=2.0, marker="s", label="Train Topic Acc")
    ax2.plot(epochs, data["dev_sent_accs"], color="#2E8BFF", linewidth=2.0, linestyle="--", marker="o", label="Dev Sentiment Acc")
    ax2.plot(epochs, data["dev_topic_accs"], color="#00FA9A", linewidth=2.0, linestyle="--", marker="s", label="Dev Topic Acc")
    ax2.plot(epochs, data["dev_sent_f1s"], color="#FDA7DF", linewidth=2.0, linestyle=":", marker="d", label="Dev Sentiment F1")
    
    ax2.set_title("Performance Metrics", color="#E0E0E0", fontsize=13, pad=12)
    ax2.set_xlabel("Epoch", color="#A0A0A0", fontsize=11, labelpad=8)
    ax2.set_ylabel("Score / Accuracy", color="#A0A0A0", fontsize=11, labelpad=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", linewidth=0.5, color="#333333")
    ax2.legend(facecolor="#1E1E1E", edgecolor="#333333", loc="lower right")
    ax2.tick_params(colors="#C0C0C0", labelsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_filename, dpi=300, facecolor="#121212")
    plt.close()
    print(f"Saved training curve: {output_filename}")


def main() -> None:
    logs_dir = "logs"
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define mapping of log files, their parser, model display name, and output plot filename
    log_configs = [
        ("runTrainViTCifar10.log", parse_vit_log, "Vision Transformer (ViT) on CIFAR-10", "logs_vit_cifar10.png"),
        ("runTrainCifar10.log", parse_cifar10_fashion_log, "CNN on CIFAR-10", "logs_cnn_cifar10.png"),
        ("runTrainFashionMNIST.log", parse_cifar10_fashion_log, "LeNet on FashionMNIST", "logs_fashionmnist_lenet.png"),
        ("runTrainSentiment.log", parse_sentiment_log, "LSTM Sentiment on Movie Reviews", "logs_sentiment_lstm.png"),
        ("runTrainResNet.log", parse_resnet_log, "ResNet-18 on CIFAR-10", "logs_resnet_cifar10.png"),
        ("runTrainIris.log", parse_iris_log, "MLP on Iris Dataset", "logs_iris_mlp.png")
    ]

    for log_file, parser, model_name, out_img in log_configs:
        path = os.path.join(logs_dir, log_file)
        if os.path.exists(path):
            print(f"Parsing {log_file}...")
            epochs, losses, train_accs, test_accs = parser(path)
            plot_training_curves(model_name, epochs, losses, train_accs, test_accs, os.path.join(output_dir, out_img))
        else:
            print(f"Log file not found: {path}")

    # Process multitask LSTM log
    multitask_log = "runTrainUitVsfcMultitask.log"
    path = os.path.join(logs_dir, multitask_log)
    if os.path.exists(path):
        print(f"Parsing multitask log: {multitask_log}...")
        data = parse_multitask_log(path)
        plot_multitask_curves("Multitask LSTM (UIT-VSFC)", data, os.path.join(output_dir, "logs_uit_vsfc_multitask.png"))
    else:
        print(f"Log file not found: {path}")

    # Process example multitask LSTM log (exampleUitVsfc.log)
    example_log = "exampleUitVsfc.log"
    path = os.path.join(logs_dir, example_log)
    if os.path.exists(path):
        print(f"Parsing example multitask log: {example_log}...")
        data = parse_example_multitask_log(path)
        plot_example_multitask_curves("Multitask LSTM (exampleUitVsfc)", data, os.path.join(output_dir, "logs_example_uit_vsfc.png"))
    else:
        print(f"Log file not found: {path}")


if __name__ == "__main__":
    main()
