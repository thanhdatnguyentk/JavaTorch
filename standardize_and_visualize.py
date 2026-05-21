"""
Standardize benchmark model names and generate visualization comparisons.
"""

from typing import Dict, List, Optional, Any
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def clean_benchmark_data(csv_path: str) -> pd.DataFrame:
    """
    Loads benchmark CSV, strips whitespaces, and standardizes model names.
    
    Args:
        csv_path: Path to the benchmark CSV file.
        
    Returns:
        pd.DataFrame: Cleaned and standardized DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark file not found at: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Strip whitespaces from string columns
    for col in ["Framework", "Model", "Device"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Standardize model names
    name_mapping: Dict[str, str] = {
        "Sentiment (RT-Polarity)": "Sentiment (RT-Polarity)",
        "runTrainSentiment": "Sentiment (RT-Polarity)",
        "ResNet (CIFAR-10)": "ResNet-18 (CIFAR-10)",
        "ResNet-18 (CIFAR-10)": "ResNet-18 (CIFAR-10)",
        "runTrainResNet": "ResNet-18 (CIFAR-10)",
        "runTrainFashionMNIST": "LeNet (FashionMNIST)",
        "LeNet (FashionMNIST)": "LeNet (FashionMNIST)",
        "runTrainCifar10": "CNN (CIFAR-10)",
        "runTrainViTCifar10": "ViT (CIFAR-10)",
        "ViT (CIFAR-10)": "ViT (CIFAR-10)",
        "exampleUitVsfc": "UIT-VSFC (Multitask LSTM)",
        "UIT-VSFC (Multitask LSTM)": "UIT-VSFC (Multitask LSTM)",
        "Iris (MLP)": "Iris (MLP)",
        "runTrainIris": "Iris (MLP)",
        "LeNet (MNIST)": "LeNet (MNIST)",
        "runTrainLeNet": "LeNet (MNIST)",
        "runTrainUitVsfcMultitask": "UIT-VSFC (Multitask LSTM)"
    }

    df["Model"] = df["Model"].map(name_mapping).fillna(df["Model"])
    return df


def generate_plots(df: pd.DataFrame, output_dashboard: str, output_dir: str = "visualizations") -> None:
    """
    Generates and saves the combined dashboard as well as 4 separate high-resolution images.
    
    Args:
        df: The cleaned DataFrame.
        output_dashboard: Path where the combined dashboard image will be saved.
        output_dir: Path where individual plots will be saved.
    """
    # Set premium dark style
    plt.style.use("dark_background")
    
    # Define color palette for the frameworks
    colors: Dict[str, str] = {
        "JavaTorch": "#FF5E5E",  # Premium coral red
        "PyTorch": "#2E8BFF",    # Premium vibrant blue
        "DL4J": "#00FA9A"        # Premium spring green
    }

    # Aggregate duplicates by taking the mean for clean plotting
    df_mean = df.groupby(["Model", "Framework"], as_index=False).mean(numeric_only=True)
    df_pivot = df.groupby(["Model", "Framework"])[["E2E_Time_Seconds", "Peak_RAM_MB"]].mean()

    # Get list of models run by JavaTorch for comparison
    jt_models = df[df["Framework"] == "JavaTorch"]["Model"].unique()
    jt_models = sorted(list(jt_models))

    # Calculate relative metrics of JavaTorch compared to baselines
    relative_data: List[Dict[str, Any]] = []
    for model in jt_models:
        jt_time = df_pivot.loc[(model, "JavaTorch"), "E2E_Time_Seconds"]
        jt_ram = df_pivot.loc[(model, "JavaTorch"), "Peak_RAM_MB"]
        
        # PyTorch comparison
        pt_speed = np.nan
        pt_ram = np.nan
        if (model, "PyTorch") in df_pivot.index:
            pt_time = df_pivot.loc[(model, "PyTorch"), "E2E_Time_Seconds"]
            pt_ram_val = df_pivot.loc[(model, "PyTorch"), "Peak_RAM_MB"]
            pt_speed = (pt_time / jt_time) * 100
            pt_ram = (jt_ram / pt_ram_val) * 100
            
        # DL4J comparison
        dl4j_speed = np.nan
        dl4j_ram = np.nan
        if (model, "DL4J") in df_pivot.index:
            dl4j_time = df_pivot.loc[(model, "DL4J"), "E2E_Time_Seconds"]
            dl4j_ram_val = df_pivot.loc[(model, "DL4J"), "Peak_RAM_MB"]
            dl4j_speed = (dl4j_time / jt_time) * 100
            dl4j_ram = (jt_ram / dl4j_ram_val) * 100
            
        relative_data.append({
            "Model": model,
            "vs. PyTorch Speed": pt_speed,
            "vs. DL4J Speed": dl4j_speed,
            "vs. PyTorch RAM": pt_ram,
            "vs. DL4J RAM": dl4j_ram
        })
        
    df_rel = pd.DataFrame(relative_data).set_index("Model")

    # =========================================================================
    # PART A: GENERATE COMBINED 2x2 DASHBOARD
    # =========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(22, 16), facecolor="#121212")
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]
    
    fig.suptitle("JavaTorch vs. PyTorch vs. DL4J Performance Dashboard & Relative Comparison", 
                 color="#FFFFFF", fontsize=24, fontweight="bold", y=0.98)

    # 1. Combined E2E Execution Time Plot
    ax1.set_facecolor("#1A1A1A")
    sns.barplot(data=df_mean, x="Model", y="E2E_Time_Seconds", hue="Framework", palette=colors, ax=ax1, edgecolor="#2D2D2D", linewidth=1.2)
    ax1.set_yscale("log")
    ax1.set_title("End-to-End Execution Time (Lower is Better)", color="#E0E0E0", fontsize=15, pad=15)
    ax1.set_xlabel("Model Architecture", color="#A0A0A0", fontsize=12, labelpad=10)
    ax1.set_ylabel("Execution Time (Seconds, Log Scale)", color="#A0A0A0", fontsize=12, labelpad=10)
    ax1.tick_params(colors="#C0C0C0", labelsize=10)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha="right")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, color="#333333")

    for container in ax1.containers:
        labels: List[str] = []
        for rect in container:
            height = rect.get_height()
            if np.isnan(height) or height <= 0:
                labels.append("")
            elif height >= 1:
                labels.append(f"{height:.2f}s")
            else:
                labels.append(f"{height*1000:.1f}ms")
        ax1.bar_label(container, labels=labels, color="#FFFFFF", fontsize=9, padding=4, rotation=25)

    # 2. Combined Peak RAM Usage Plot
    ax2.set_facecolor("#1A1A1A")
    sns.barplot(data=df_mean, x="Model", y="Peak_RAM_MB", hue="Framework", palette=colors, ax=ax2, edgecolor="#2D2D2D", linewidth=1.2)
    ax2.set_title("Peak RAM Utilization (Lower is Better)", color="#E0E0E0", fontsize=15, pad=15)
    ax2.set_xlabel("Model Architecture", color="#A0A0A0", fontsize=12, labelpad=10)
    ax2.set_ylabel("Peak RAM (MB)", color="#A0A0A0", fontsize=12, labelpad=10)
    ax2.tick_params(colors="#C0C0C0", labelsize=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha="right")
    ax2.grid(True, which="major", linestyle="--", linewidth=0.5, color="#333333")

    for container in ax2.containers:
        labels: List[str] = []
        for rect in container:
            height = rect.get_height()
            if np.isnan(height) or height <= 0:
                labels.append("")
            elif height < 1024:
                labels.append(f"{height:.0f} MB")
            else:
                labels.append(f"{height/1024:.2f} GB")
        ax2.bar_label(container, labels=labels, color="#FFFFFF", fontsize=9, padding=4, rotation=25)

    for ax in [ax1, ax2]:
        legend = ax.legend(title="Framework", facecolor="#1E1E1E", edgecolor="#333333", loc="upper right")
        plt.setp(legend.get_title(), color="#A0A0A0")
        for text in legend.get_texts():
            text.set_color("#E0E0E0")

    # 3. Combined Relative Speed (%) Heatmap
    ax3.set_facecolor("#121212")
    speed_df = df_rel[["vs. PyTorch Speed", "vs. DL4J Speed"]].T
    annot_speed = speed_df.copy()
    for col in annot_speed.columns:
        annot_speed[col] = annot_speed[col].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
    sns.heatmap(speed_df.astype(float), annot=annot_speed.values, fmt="", cmap="RdYlGn", vmin=0, cbar_kws={'label': 'Relative Speed (%)'}, ax=ax3, linewidths=1.5, linecolor="#2D2D2D", annot_kws={"fontsize": 11, "fontweight": "bold", "color": "#FFFFFF"})
    ax3.set_title("JavaTorch Speed as % of Baselines (Higher is Better)\nFormula: (Time_Baseline / Time_JavaTorch) * 100%", color="#E0E0E0", fontsize=14, pad=15)
    ax3.tick_params(axis='y', rotation=0, colors="#C0C0C0", labelsize=11)
    ax3.tick_params(axis='x', rotation=25, colors="#C0C0C0", labelsize=11)
    ax3.set_xlabel("Model Architecture", color="#A0A0A0", fontsize=12, labelpad=10)

    # 4. Combined Relative RAM (%) Heatmap
    ax4.set_facecolor("#121212")
    ram_df = df_rel[["vs. PyTorch RAM", "vs. DL4J RAM"]].T
    annot_ram = ram_df.copy()
    for col in annot_ram.columns:
        annot_ram[col] = annot_ram[col].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
    sns.heatmap(ram_df.astype(float), annot=annot_ram.values, fmt="", cmap="RdYlGn_r", vmin=0, cbar_kws={'label': 'Relative RAM Ratio (%)'}, ax=ax4, linewidths=1.5, linecolor="#2D2D2D", annot_kws={"fontsize": 11, "fontweight": "bold", "color": "#FFFFFF"})
    ax4.set_title("JavaTorch RAM as % of Baselines (Lower is Better)\nFormula: (RAM_JavaTorch / RAM_Baseline) * 100%", color="#E0E0E0", fontsize=14, pad=15)
    ax4.tick_params(axis='y', rotation=0, colors="#C0C0C0", labelsize=11)
    ax4.tick_params(axis='x', rotation=25, colors="#C0C0C0", labelsize=11)
    ax4.set_xlabel("Model Architecture", color="#A0A0A0", fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(output_dashboard, dpi=300, facecolor="#121212")
    plt.close()
    print(f"Combined dashboard saved to: {output_dashboard}")

    # =========================================================================
    # PART B: GENERATE 4 INDIVIDUAL HIGH-RES PLOTS
    # =========================================================================
    # 1. Individual End-to-End Time Plot
    plt.figure(figsize=(12, 7.5), facecolor="#121212")
    ax = plt.gca()
    ax.set_facecolor("#1A1A1A")
    sns.barplot(data=df_mean, x="Model", y="E2E_Time_Seconds", hue="Framework", palette=colors, ax=ax, edgecolor="#2D2D2D", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_title("End-to-End Execution Time (Lower is Better)", color="#FFFFFF", fontsize=16, pad=18, fontweight="bold")
    ax.set_xlabel("Model Architecture", color="#C0C0C0", fontsize=13, labelpad=10)
    ax.set_ylabel("Execution Time (Seconds, Log Scale)", color="#C0C0C0", fontsize=13, labelpad=10)
    ax.tick_params(colors="#C0C0C0", labelsize=10)
    plt.xticks(rotation=25, ha="right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#333333")
    
    for container in ax.containers:
        labels: List[str] = []
        for rect in container:
            height = rect.get_height()
            if np.isnan(height) or height <= 0:
                labels.append("")
            elif height >= 1:
                labels.append(f"{height:.2f}s")
            else:
                labels.append(f"{height*1000:.1f}ms")
        ax.bar_label(container, labels=labels, color="#FFFFFF", fontsize=9, padding=4, rotation=25)
        
    legend = ax.legend(title="Framework", facecolor="#1E1E1E", edgecolor="#333333", loc="upper right")
    plt.setp(legend.get_title(), color="#A0A0A0")
    for text in legend.get_texts():
        text.set_color("#E0E0E0")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_e2e_time.png"), dpi=300, facecolor="#121212")
    plt.close()
    print(f"Saved individual plot: {os.path.join(output_dir, 'benchmark_e2e_time.png')}")

    # 2. Individual Peak RAM Usage Plot
    plt.figure(figsize=(12, 7.5), facecolor="#121212")
    ax = plt.gca()
    ax.set_facecolor("#1A1A1A")
    sns.barplot(data=df_mean, x="Model", y="Peak_RAM_MB", hue="Framework", palette=colors, ax=ax, edgecolor="#2D2D2D", linewidth=1.2)
    ax.set_title("Peak RAM Utilization (Lower is Better)", color="#FFFFFF", fontsize=16, pad=18, fontweight="bold")
    ax.set_xlabel("Model Architecture", color="#C0C0C0", fontsize=13, labelpad=10)
    ax.set_ylabel("Peak RAM (MB)", color="#C0C0C0", fontsize=13, labelpad=10)
    ax.tick_params(colors="#C0C0C0", labelsize=10)
    plt.xticks(rotation=25, ha="right")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="#333333")
    
    for container in ax.containers:
        labels: List[str] = []
        for rect in container:
            height = rect.get_height()
            if np.isnan(height) or height <= 0:
                labels.append("")
            elif height < 1024:
                labels.append(f"{height:.0f} MB")
            else:
                labels.append(f"{height/1024:.2f} GB")
        ax.bar_label(container, labels=labels, color="#FFFFFF", fontsize=9, padding=4, rotation=25)
        
    legend = ax.legend(title="Framework", facecolor="#1E1E1E", edgecolor="#333333", loc="upper right")
    plt.setp(legend.get_title(), color="#A0A0A0")
    for text in legend.get_texts():
        text.set_color("#E0E0E0")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_peak_ram.png"), dpi=300, facecolor="#121212")
    plt.close()
    print(f"Saved individual plot: {os.path.join(output_dir, 'benchmark_peak_ram.png')}")

    # 3. Individual JavaTorch Relative Speed (%) Heatmap
    plt.figure(figsize=(12, 5.5), facecolor="#121212")
    ax = plt.gca()
    ax.set_facecolor("#121212")
    sns.heatmap(speed_df.astype(float), annot=annot_speed.values, fmt="", cmap="RdYlGn", vmin=0, cbar_kws={'label': 'Relative Speed (%)'}, ax=ax, linewidths=1.5, linecolor="#2D2D2D", annot_kws={"fontsize": 11, "fontweight": "bold", "color": "#FFFFFF"})
    ax.set_title("JavaTorch Speed as % of Baselines (Higher is Better)\nFormula: (Time_Baseline / Time_JavaTorch) * 100%", color="#FFFFFF", fontsize=15, pad=18, fontweight="bold")
    ax.tick_params(axis='y', rotation=0, colors="#C0C0C0", labelsize=11)
    ax.tick_params(axis='x', rotation=25, colors="#C0C0C0", labelsize=11)
    ax.set_xlabel("Model Architecture", color="#C0C0C0", fontsize=13, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_relative_speed.png"), dpi=300, facecolor="#121212")
    plt.close()
    print(f"Saved individual plot: {os.path.join(output_dir, 'benchmark_relative_speed.png')}")

    # 4. Individual JavaTorch Relative RAM (%) Heatmap
    plt.figure(figsize=(12, 5.5), facecolor="#121212")
    ax = plt.gca()
    ax.set_facecolor("#121212")
    sns.heatmap(ram_df.astype(float), annot=annot_ram.values, fmt="", cmap="RdYlGn_r", vmin=0, cbar_kws={'label': 'Relative RAM Ratio (%)'}, ax=ax, linewidths=1.5, linecolor="#2D2D2D", annot_kws={"fontsize": 11, "fontweight": "bold", "color": "#FFFFFF"})
    ax.set_title("JavaTorch RAM as % of Baselines (Lower is Better)\nFormula: (RAM_JavaTorch / RAM_Baseline) * 100%", color="#FFFFFF", fontsize=15, pad=18, fontweight="bold")
    ax.tick_params(axis='y', rotation=0, colors="#C0C0C0", labelsize=11)
    ax.tick_params(axis='x', rotation=25, colors="#C0C0C0", labelsize=11)
    ax.set_xlabel("Model Architecture", color="#C0C0C0", fontsize=13, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_relative_ram.png"), dpi=300, facecolor="#121212")
    plt.close()
    print(f"Saved individual plot: {os.path.join(output_dir, 'benchmark_relative_ram.png')}")


def main() -> None:
    csv_file = "benchmark_comparison_summary.csv"
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_image = os.path.join(output_dir, "benchmark_comparison_visual.png")
    
    try:
        # 1. Clean data
        print("Cleaning and standardizing benchmark data...")
        cleaned_df = clean_benchmark_data(csv_file)
        
        # Save cleaned CSV
        cleaned_df.to_csv(csv_file, index=False)
        print(f"Standardized CSV successfully saved to: {csv_file}")
        
        # 2. Visualize with relative percentage table heatmaps
        print("Generating comprehensive comparison dashboards...")
        generate_plots(cleaned_df, output_image, output_dir)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
