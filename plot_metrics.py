import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def to_float_or_none(value):
    if value is None:
        return None
    value_str = str(value).strip()
    if value_str == "":
        return None
    return float(value_str)


def read_metrics_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            phase_time_sec = to_float_or_none(row.get("phase_time_sec"))
            epoch_time_sec = to_float_or_none(row.get("epoch_time_sec"))
            cumulative_time_sec = to_float_or_none(row.get("cumulative_time_sec"))
            global_progress_percent = to_float_or_none(row.get("global_progress_percent"))
            vram_allocated_mb = to_float_or_none(row.get("vram_allocated_mb"))
            vram_reserved_mb = to_float_or_none(row.get("vram_reserved_mb"))
            vram_peak_allocated_mb = to_float_or_none(row.get("vram_peak_allocated_mb"))
            cpu_usage_percent = to_float_or_none(row.get("cpu_usage_percent"))
            ram_used_mb = to_float_or_none(row.get("ram_used_mb"))
            ram_percent = to_float_or_none(row.get("ram_percent"))

            rows.append({
                "run_id": row.get("run_id", ""),
                "model_name": row.get("model_name", ""),
                "epoch": int(row["epoch"]),
                "phase": row["phase"].strip().lower(),
                "loss": float(row["loss"]),
                "accuracy": float(row["accuracy"]),
                "precision_macro": float(row["precision_macro"]),
                "recall_macro": float(row["recall_macro"]),
                "f1_macro": float(row["f1_macro"]),
                "precision_weighted": float(row["precision_weighted"]),
                "recall_weighted": float(row["recall_weighted"]),
                "f1_weighted": float(row["f1_weighted"]),
                "phase_time_sec": phase_time_sec,
                "epoch_time_sec": epoch_time_sec,
                "cumulative_time_sec": cumulative_time_sec,
                "global_progress_percent": global_progress_percent,
                "vram_allocated_mb": vram_allocated_mb,
                "vram_reserved_mb": vram_reserved_mb,
                "vram_peak_allocated_mb": vram_peak_allocated_mb,
                "cpu_usage_percent": cpu_usage_percent,
                "ram_used_mb": ram_used_mb,
                "ram_percent": ram_percent,
            })
    if not rows:
        raise ValueError("CSV không có dữ liệu.")
    return rows


def split_by_phase(rows):
    train_rows = sorted((r for r in rows if r["phase"] == "train"), key=lambda r: r["epoch"])
    val_rows = sorted((r for r in rows if r["phase"] == "val"), key=lambda r: r["epoch"])
    test_rows = sorted((r for r in rows if r["phase"] == "test"), key=lambda r: r["epoch"])
    if not train_rows and not val_rows and not test_rows:
        raise ValueError("Không tìm thấy dòng phase='train', 'val' hoặc 'test' trong CSV.")
    return train_rows, val_rows, test_rows


def detect_latest_run_id(rows):
    run_ids = sorted({r.get("run_id", "") for r in rows if r.get("run_id", "")})
    if not run_ids:
        return None
    return run_ids[-1]


def filter_rows_by_run_id(rows, run_id):
    if run_id is None:
        return rows
    filtered_rows = [r for r in rows if r.get("run_id", "") == run_id]
    return filtered_rows if filtered_rows else rows


def count_csv_data_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        row_count = sum(1 for _ in file) - 1
    return max(0, row_count)


def read_batch_metrics_csv(batch_csv_path: Path, max_points=50000):
    if not batch_csv_path.exists():
        return []

    total_rows = count_csv_data_rows(batch_csv_path)
    if total_rows == 0:
        return []

    max_points = max(1, int(max_points))
    sample_step = max(1, math.ceil(total_rows / max_points))
    if sample_step > 1:
        print(
            f"[Batch CSV] Downsample từ {total_rows} xuống khoảng {math.ceil(total_rows / sample_step)} điểm "
            f"(step={sample_step}) để giảm RAM."
        )

    rows = []
    with batch_csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader):
            if sample_step > 1 and (row_index % sample_step != 0):
                continue
            rows.append({
                "run_id": row.get("run_id", ""),
                "model_name": row.get("model_name", ""),
                "global_step": int(float(row["global_step"])),
                "epoch": int(float(row["epoch"])),
                "batch_index": int(float(row["batch_index"])),
                "num_batches": int(float(row["num_batches"])),
                "progress_percent": to_float_or_none(row.get("progress_percent")),
                "batch_loss": to_float_or_none(row.get("batch_loss")),
                "batch_accuracy": to_float_or_none(row.get("batch_accuracy")),
                "elapsed_sec": to_float_or_none(row.get("elapsed_sec")),
                "vram_allocated_mb": to_float_or_none(row.get("vram_allocated_mb")),
                "vram_reserved_mb": to_float_or_none(row.get("vram_reserved_mb")),
                "vram_peak_allocated_mb": to_float_or_none(row.get("vram_peak_allocated_mb")),
                "cpu_usage_percent": to_float_or_none(row.get("cpu_usage_percent")),
                "ram_used_mb": to_float_or_none(row.get("ram_used_mb")),
                "ram_percent": to_float_or_none(row.get("ram_percent")),
            })
    return rows


def mean_ignore_none(values):
    valid_values = [value for value in values if value is not None and not np.isnan(value)]
    if not valid_values:
        return None
    return float(np.mean(valid_values))


def infer_default_batch_csv_path(metrics_csv_path: Path):
    metrics_name = metrics_csv_path.name
    if metrics_name.startswith("training_metrics"):
        return metrics_csv_path.with_name(metrics_name.replace("training_metrics", "training_batch_metrics", 1))
    return metrics_csv_path.with_name(f"training_batch_{metrics_name}")


def parse_named_paths(entries, argument_name):
    mapping = {}
    for raw_entry in entries:
        if "=" not in raw_entry:
            raise ValueError(
                f"{argument_name} phải theo định dạng model_name=duong_dan_csv, nhận được: {raw_entry}"
            )
        model_name, path_text = raw_entry.split("=", 1)
        model_name = model_name.strip()
        path_text = path_text.strip()
        if not model_name or not path_text:
            raise ValueError(
                f"{argument_name} không hợp lệ, cần đủ model_name và đường dẫn: {raw_entry}"
            )
        mapping[model_name] = Path(path_text)
    return mapping


def resolve_run_id(rows, run_id_arg):
    return detect_latest_run_id(rows) if run_id_arg == "latest" else run_id_arg


def summarize_model_for_comparison(model_name, train_rows, val_rows, test_rows):
    train_phase_time = mean_ignore_none([r.get("phase_time_sec") for r in train_rows])
    gpu_peak_mb = mean_ignore_none([r.get("vram_peak_allocated_mb") for r in train_rows])
    ram_used_mb = mean_ignore_none([r.get("ram_used_mb") for r in train_rows])
    cpu_usage_percent = mean_ignore_none([r.get("cpu_usage_percent") for r in train_rows])

    best_val_accuracy = None
    if val_rows:
        best_val_accuracy = max(r["accuracy"] for r in val_rows)

    latest_test_accuracy = None
    if test_rows:
        latest_test_accuracy = test_rows[-1].get("accuracy")

    return {
        "model_name": model_name,
        "avg_train_epoch_time_sec": train_phase_time,
        "avg_gpu_peak_mb": gpu_peak_mb,
        "avg_ram_used_mb": ram_used_mb,
        "avg_cpu_percent": cpu_usage_percent,
        "best_val_accuracy": best_val_accuracy,
        "latest_test_accuracy": latest_test_accuracy,
    }


def smooth_curve(values, window=20):
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def save_batch_plots(batch_rows, output_dir: Path):
    if not batch_rows:
        return

    sorted_rows = sorted(batch_rows, key=lambda r: r["global_step"])
    x_steps = [r["global_step"] for r in sorted_rows]

    loss_values = [r["batch_loss"] if r["batch_loss"] is not None else np.nan for r in sorted_rows]
    progress_values = [r["progress_percent"] if r["progress_percent"] is not None else np.nan for r in sorted_rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(x_steps, loss_values, alpha=0.4, linewidth=1.5, label="Batch Loss")
    axes[0].plot(x_steps, smooth_curve(np.array(loss_values, dtype=np.float64), window=25), linewidth=2.0, label="Smoothed")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Batch Training Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_steps, progress_values, color="#2ca02c", linewidth=2)
    axes[1].set_xlabel("Global Step")
    axes[1].set_ylabel("Progress (%)")
    axes[1].set_title("Batch-Level Training Progress")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "batch_loss_progress.png", dpi=150)
    plt.close(fig)

    vram_allocated = [r["vram_allocated_mb"] if r["vram_allocated_mb"] is not None else np.nan for r in sorted_rows]
    vram_reserved = [r["vram_reserved_mb"] if r["vram_reserved_mb"] is not None else np.nan for r in sorted_rows]
    vram_peak = [r["vram_peak_allocated_mb"] if r["vram_peak_allocated_mb"] is not None else np.nan for r in sorted_rows]

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(x_steps, vram_allocated, linewidth=1.8, label="Allocated MB")
    ax2.plot(x_steps, vram_reserved, linewidth=1.8, label="Reserved MB")
    ax2.plot(x_steps, vram_peak, linewidth=1.8, label="Peak Allocated MB")
    ax2.set_title("Batch-Level VRAM Usage")
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("VRAM (MB)")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "batch_vram.png", dpi=150)
    plt.close(fig2)


def plot_single_metric(ax, train_rows, val_rows, metric_key, title, ylabel):
    if train_rows:
        ax.plot(
            [r["epoch"] for r in train_rows],
            [r[metric_key] for r in train_rows],
            marker="o",
            linewidth=2,
            label="Train",
        )
    if val_rows:
        ax.plot(
            [r["epoch"] for r in val_rows],
            [r[metric_key] for r in val_rows],
            marker="s",
            linewidth=2,
            label="Val",
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


def save_main_plots(train_rows, val_rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_single_metric(axes[0, 0], train_rows, val_rows, "loss", "Loss", "Loss")
    plot_single_metric(axes[0, 1], train_rows, val_rows, "accuracy", "Accuracy", "Accuracy")
    plot_single_metric(axes[1, 0], train_rows, val_rows, "f1_macro", "F1 Macro", "F1")
    plot_single_metric(axes[1, 1], train_rows, val_rows, "f1_weighted", "F1 Weighted", "F1")
    fig.suptitle("Training Metrics Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_overview.png", dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    plot_single_metric(axes2[0, 0], train_rows, val_rows, "precision_macro", "Precision Macro", "Precision")
    plot_single_metric(axes2[0, 1], train_rows, val_rows, "recall_macro", "Recall Macro", "Recall")
    plot_single_metric(axes2[1, 0], train_rows, val_rows, "precision_weighted", "Precision Weighted", "Precision")
    plot_single_metric(axes2[1, 1], train_rows, val_rows, "recall_weighted", "Recall Weighted", "Recall")
    fig2.suptitle("Precision/Recall Metrics", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(output_dir / "metrics_precision_recall.png", dpi=150)
    plt.close(fig2)


def save_runtime_plots(train_rows, val_rows, output_dir: Path):
    all_rows = sorted(train_rows + val_rows, key=lambda r: (r["epoch"], r["phase"]))
    if not all_rows:
        return

    rows_with_progress = [r for r in all_rows if r.get("global_progress_percent") is not None]
    rows_with_time = [r for r in all_rows if r.get("cumulative_time_sec") is not None]
    rows_with_vram = [
        r for r in all_rows
        if r.get("vram_allocated_mb") is not None
        or r.get("vram_reserved_mb") is not None
        or r.get("vram_peak_allocated_mb") is not None
    ]

    if rows_with_progress:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            [r["epoch"] for r in rows_with_progress],
            [r["global_progress_percent"] for r in rows_with_progress],
            marker="o",
            linewidth=2,
            color="#1f77b4",
        )
        ax.set_title("Global Training Progress")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Progress (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "runtime_progress.png", dpi=150)
        plt.close(fig)

    if rows_with_time:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            [r["epoch"] for r in rows_with_time],
            [r["cumulative_time_sec"] / 60.0 for r in rows_with_time],
            marker="s",
            linewidth=2,
            color="#2ca02c",
        )
        ax.set_title("Cumulative Training Time")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time (minutes)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "runtime_cumulative_time.png", dpi=150)
        plt.close(fig)

    if rows_with_vram:
        fig, ax = plt.subplots(figsize=(12, 5))
        x_values = [r["epoch"] for r in rows_with_vram]
        allocated = [r["vram_allocated_mb"] if r["vram_allocated_mb"] is not None else np.nan for r in rows_with_vram]
        reserved = [r["vram_reserved_mb"] if r["vram_reserved_mb"] is not None else np.nan for r in rows_with_vram]
        peak = [r["vram_peak_allocated_mb"] if r["vram_peak_allocated_mb"] is not None else np.nan for r in rows_with_vram]

        ax.plot(x_values, allocated, marker="o", linewidth=2, label="Allocated MB")
        ax.plot(x_values, reserved, marker="s", linewidth=2, label="Reserved MB")
        ax.plot(x_values, peak, marker="^", linewidth=2, label="Peak Allocated MB")

        ax.set_title("GPU VRAM Usage Across Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("VRAM (MB)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "runtime_vram.png", dpi=150)
        plt.close(fig)

    rows_with_system_resource = [
        r for r in train_rows
        if r.get("cpu_usage_percent") is not None
        or r.get("ram_used_mb") is not None
        or r.get("ram_percent") is not None
    ]

    if rows_with_system_resource:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        x_values = [r["epoch"] for r in rows_with_system_resource]
        cpu_values = [r.get("cpu_usage_percent") if r.get("cpu_usage_percent") is not None else np.nan for r in rows_with_system_resource]
        ram_used_values = [r.get("ram_used_mb") if r.get("ram_used_mb") is not None else np.nan for r in rows_with_system_resource]
        ram_percent_values = [r.get("ram_percent") if r.get("ram_percent") is not None else np.nan for r in rows_with_system_resource]

        axes[0].plot(x_values, cpu_values, marker="o", linewidth=2, color="#ff7f0e")
        axes[0].set_title("CPU Usage During Training")
        axes[0].set_ylabel("CPU (%)")
        axes[0].grid(alpha=0.3)

        axes[1].plot(x_values, ram_used_values, marker="s", linewidth=2, color="#2ca02c")
        axes[1].set_title("RAM Used During Training")
        axes[1].set_ylabel("RAM (MB)")
        axes[1].grid(alpha=0.3)

        axes[2].plot(x_values, ram_percent_values, marker="^", linewidth=2, color="#1f77b4")
        axes[2].set_title("RAM Utilization During Training")
        axes[2].set_ylabel("RAM (%)")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "runtime_cpu_ram.png", dpi=150)
        plt.close(fig)


def save_comparison_plots(model_summaries, output_dir: Path):
    if not model_summaries:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [summary["model_name"] for summary in model_summaries]
    x_positions = np.arange(len(model_names))

    fig_resource, resource_axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    resource_specs = [
        ("avg_gpu_peak_mb", "GPU Peak VRAM (MB)", "#1f77b4"),
        ("avg_ram_used_mb", "RAM Used (MB)", "#2ca02c"),
        ("avg_cpu_percent", "CPU Usage (%)", "#ff7f0e"),
    ]
    for axis, (metric_key, axis_title, color) in zip(resource_axes, resource_specs):
        values = [summary.get(metric_key) if summary.get(metric_key) is not None else 0.0 for summary in model_summaries]
        bars = axis.bar(x_positions, values, color=color, alpha=0.85)
        axis.set_title(axis_title)
        axis.grid(axis="y", alpha=0.3)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    resource_axes[-1].set_xticks(x_positions)
    resource_axes[-1].set_xticklabels(model_names, rotation=20, ha="right")
    fig_resource.suptitle("So sánh tài nguyên train giữa các mô hình", fontsize=14)
    fig_resource.tight_layout()
    fig_resource.savefig(output_dir / "compare_resources_gpu_ram_cpu.png", dpi=150)
    plt.close(fig_resource)

    epoch_time_values = [
        summary.get("avg_train_epoch_time_sec") if summary.get("avg_train_epoch_time_sec") is not None else 0.0
        for summary in model_summaries
    ]
    fig_time, ax_time = plt.subplots(figsize=(11, 5))
    bars_time = ax_time.bar(model_names, epoch_time_values, color="#8c564b", alpha=0.9)
    ax_time.set_title("So sánh thời gian trung bình 1 epoch (phase=train)")
    ax_time.set_ylabel("Thời gian (giây)")
    ax_time.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars_time, epoch_time_values):
        ax_time.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig_time.tight_layout()
    fig_time.savefig(output_dir / "compare_epoch_time.png", dpi=150)
    plt.close(fig_time)

    val_accuracy_values = [summary.get("best_val_accuracy") for summary in model_summaries]
    test_accuracy_values = [summary.get("latest_test_accuracy") for summary in model_summaries]
    has_test_values = any(value is not None for value in test_accuracy_values)

    fig_acc, ax_acc = plt.subplots(figsize=(11, 5))
    bar_width = 0.35
    x_idx = np.arange(len(model_names))
    val_values = [value if value is not None else 0.0 for value in val_accuracy_values]
    bars_val = ax_acc.bar(x_idx - (bar_width / 2 if has_test_values else 0), val_values, bar_width, label="Best Val Accuracy")

    if has_test_values:
        test_values = [value if value is not None else 0.0 for value in test_accuracy_values]
        bars_test = ax_acc.bar(x_idx + bar_width / 2, test_values, bar_width, label="Latest Test Accuracy")
    else:
        bars_test = []

    ax_acc.set_title("So sánh độ chính xác giữa các mô hình")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_xticks(x_idx)
    ax_acc.set_xticklabels(model_names, rotation=20, ha="right")
    ax_acc.grid(axis="y", alpha=0.3)
    ax_acc.legend()

    for bar in list(bars_val) + list(bars_test):
        value = bar.get_height()
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig_acc.tight_layout()
    fig_acc.savefig(output_dir / "compare_accuracy.png", dpi=150)
    plt.close(fig_acc)


def save_test_plot(test_rows, output_dir: Path):
    if not test_rows:
        return

    latest_test = test_rows[-1]
    metric_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    metric_labels = [
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "F1 Macro",
        "Precision Weighted",
        "Recall Weighted",
        "F1 Weighted",
    ]
    metric_values = [latest_test[key] for key in metric_keys]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(metric_labels, metric_values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test Metrics Summary")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "test_metrics_summary.png", dpi=150)
    plt.close(fig)


def read_confusion_csv(confusion_csv_path: Path):
    with confusion_csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = list(csv.reader(file))

    if len(reader) < 2 or len(reader[0]) < 2:
        raise ValueError(f"File confusion matrix không hợp lệ: {confusion_csv_path}")

    class_names = reader[0][1:]
    matrix = []
    for row in reader[1:]:
        if len(row) < len(class_names) + 1:
            continue
        matrix.append([int(float(value)) for value in row[1:len(class_names) + 1]])

    confusion_matrix = np.array(matrix, dtype=np.int64)
    return class_names, confusion_matrix


def reduce_confusion_for_plot(class_names, confusion_matrix, max_classes):
    max_classes = max(1, int(max_classes))
    num_classes = len(class_names)
    if num_classes <= max_classes:
        return class_names, confusion_matrix, False

    support = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0)
    top_indices = np.argsort(support)[-max_classes:]
    top_indices = top_indices[np.argsort(support[top_indices])[::-1]]

    reduced_class_names = [class_names[index] for index in top_indices]
    reduced_confusion = confusion_matrix[np.ix_(top_indices, top_indices)]
    return reduced_class_names, reduced_confusion, True


def plot_confusion_heatmap(class_names, confusion_matrix, title: str, output_path: Path, max_heatmap_classes: int):
    plot_class_names, plot_confusion_matrix, reduced = reduce_confusion_for_plot(
        class_names,
        confusion_matrix,
        max_classes=max_heatmap_classes,
    )
    if reduced:
        print(
            f"[Heatmap] {title}: giảm từ {len(class_names)} class xuống "
            f"{len(plot_class_names)} class có support cao nhất."
        )

    num_classes = len(plot_class_names)
    fig_width = min(24, max(8, num_classes * 0.25))
    fig_height = min(20, max(6, num_classes * 0.22))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(plot_confusion_matrix, interpolation="nearest", cmap="Blues", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    title_text = title
    if reduced:
        title_text += f" (top {len(plot_class_names)} classes by support)"

    ax.set_title(title_text)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    tick_step = max(1, math.ceil(num_classes / 40))
    tick_indices = np.arange(0, num_classes, tick_step)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([plot_class_names[i] for i in tick_indices], rotation=45, ha="right")
    ax.set_yticklabels([plot_class_names[i] for i in tick_indices])

    if num_classes <= 30:
        threshold = plot_confusion_matrix.max() / 2 if plot_confusion_matrix.size > 0 else 0
        for row_index in range(plot_confusion_matrix.shape[0]):
            for col_index in range(plot_confusion_matrix.shape[1]):
                value = plot_confusion_matrix[row_index, col_index]
                text_color = "white" if value > threshold else "black"
                ax.text(col_index, row_index, f"{value}", ha="center", va="center", color=text_color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def save_confusion_heatmaps(confusion_dir: Path, output_dir: Path, max_heatmap_classes: int):
    if not confusion_dir.exists():
        print(f"Không tìm thấy thư mục confusion matrix: {confusion_dir}")
        return

    heatmap_dir = output_dir / "confusion_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    confusion_files = sorted(confusion_dir.glob("*.csv"))
    if not confusion_files:
        print(f"Không có file confusion matrix CSV trong: {confusion_dir}")
        return

    generated_count = 0
    for confusion_file in confusion_files:
        class_names, confusion_matrix = read_confusion_csv(confusion_file)
        if confusion_matrix.size == 0:
            continue
        output_path = heatmap_dir / f"{confusion_file.stem}.png"
        plot_confusion_heatmap(
            class_names,
            confusion_matrix,
            title=f"Confusion Matrix - {confusion_file.stem}",
            output_path=output_path,
            max_heatmap_classes=max_heatmap_classes,
        )
        generated_count += 1

    print(f"Đã lưu {generated_count} confusion heatmap vào: {heatmap_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ metric từ training_metrics.csv")
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="training_metrics.csv",
        help="Đường dẫn tới file training_metrics.csv",
    )
    parser.add_argument(
        "--batch-metrics-csv",
        type=str,
        default="training_batch_metrics.csv",
        help="Đường dẫn tới file CSV metrics theo batch",
    )
    parser.add_argument(
        "--test-metrics-csv",
        type=str,
        default="test_metrics.csv",
        help="File metrics test riêng (dùng khi metrics-csv không có phase=test)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="latest",
        help="Run ID cần vẽ. Mặc định 'latest' để lấy phiên train mới nhất",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Thư mục output chứa ảnh biểu đồ",
    )
    parser.add_argument(
        "--confusion-dir",
        type=str,
        default="confusion_matrices",
        help="Thư mục chứa các file confusion matrix CSV (val_confusion_epoch_*.csv)",
    )
    parser.add_argument(
        "--max-heatmap-classes",
        type=int,
        default=120,
        help="Giới hạn số class tối đa khi vẽ confusion heatmap để tránh OOM",
    )
    parser.add_argument(
        "--max-batch-points",
        type=int,
        default=50000,
        help="Giới hạn số điểm tối đa đọc từ batch metrics CSV để giảm RAM",
    )
    parser.add_argument(
        "--skip-confusion-heatmaps",
        action="store_true",
        help="Bỏ qua bước vẽ confusion heatmap nếu chỉ cần metric plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị biểu đồ trên màn hình ngoài việc lưu file",
    )
    parser.add_argument(
        "--compare-metrics",
        action="append",
        default=[],
        help="Thêm mô hình để so sánh theo định dạng model_name=duong_dan_training_metrics.csv (có thể dùng nhiều lần)",
    )
    parser.add_argument(
        "--compare-batch-metrics",
        action="append",
        default=[],
        help="Batch metrics cho từng model theo định dạng model_name=duong_dan_training_batch_metrics.csv",
    )
    parser.add_argument(
        "--compare-run-id",
        type=str,
        default="latest",
        help="Run ID dùng cho tất cả model khi ở chế độ compare (mặc định latest)",
    )
    return parser.parse_args()


def run_single_mode(args, output_dir: Path):
    metrics_csv_path = Path(args.metrics_csv)
    batch_metrics_csv_path = Path(args.batch_metrics_csv)
    test_metrics_csv_path = Path(args.test_metrics_csv)
    confusion_dir = Path(args.confusion_dir)

    if not metrics_csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file metrics CSV: {metrics_csv_path}")

    rows = read_metrics_csv(metrics_csv_path)
    selected_run_id = resolve_run_id(rows, args.run_id)
    rows = filter_rows_by_run_id(rows, selected_run_id)
    train_rows, val_rows, test_rows = split_by_phase(rows)

    if not test_rows and test_metrics_csv_path.exists():
        try:
            test_metrics_rows = read_metrics_csv(test_metrics_csv_path)
            _, _, fallback_test_rows = split_by_phase(test_metrics_rows)
            if fallback_test_rows:
                test_rows = fallback_test_rows
                print(
                    f"Không thấy phase=test trong {metrics_csv_path}. "
                    f"Đã dùng dữ liệu test từ {test_metrics_csv_path}."
                )
        except Exception as error:
            print(f"Không thể đọc test metrics từ {test_metrics_csv_path}: {error}")

    batch_rows = read_batch_metrics_csv(batch_metrics_csv_path, max_points=args.max_batch_points)
    batch_rows = filter_rows_by_run_id(batch_rows, selected_run_id)

    if train_rows or val_rows:
        save_main_plots(train_rows, val_rows, output_dir)
        save_runtime_plots(train_rows, val_rows, output_dir)
        save_batch_plots(batch_rows, output_dir)
    if test_rows:
        save_test_plot(test_rows, output_dir)

    if args.skip_confusion_heatmaps:
        print("Đã bỏ qua vẽ confusion heatmap theo tùy chọn --skip-confusion-heatmaps")
    else:
        save_confusion_heatmaps(confusion_dir, output_dir, max_heatmap_classes=args.max_heatmap_classes)
    if selected_run_id:
        print(f"Run ID đang vẽ: {selected_run_id}")


def run_compare_mode(args, output_dir: Path):
    compare_metrics_map = parse_named_paths(args.compare_metrics, "--compare-metrics")
    compare_batch_map = parse_named_paths(args.compare_batch_metrics, "--compare-batch-metrics")

    model_summaries = []
    for model_name, metrics_csv_path in compare_metrics_map.items():
        if not metrics_csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file metrics cho model '{model_name}': {metrics_csv_path}")

        rows = read_metrics_csv(metrics_csv_path)
        selected_run_id = resolve_run_id(rows, args.compare_run_id)
        rows = filter_rows_by_run_id(rows, selected_run_id)
        train_rows, val_rows, test_rows = split_by_phase(rows)

        batch_csv_path = compare_batch_map.get(model_name, infer_default_batch_csv_path(metrics_csv_path))
        batch_rows = read_batch_metrics_csv(batch_csv_path, max_points=args.max_batch_points)
        batch_rows = filter_rows_by_run_id(batch_rows, selected_run_id)

        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        if train_rows or val_rows:
            save_main_plots(train_rows, val_rows, model_output_dir)
            save_runtime_plots(train_rows, val_rows, model_output_dir)
            save_batch_plots(batch_rows, model_output_dir)
        if test_rows:
            save_test_plot(test_rows, model_output_dir)

        summary = summarize_model_for_comparison(model_name, train_rows, val_rows, test_rows)
        model_summaries.append(summary)
        print(
            f"[{model_name}] run_id={selected_run_id} | "
            f"avg_epoch={summary['avg_train_epoch_time_sec']}s | "
            f"best_val_acc={summary['best_val_accuracy']}"
        )

    save_comparison_plots(model_summaries, output_dir)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_metrics:
        run_compare_mode(args, output_dir)
    else:
        run_single_mode(args, output_dir)

    print(f"Đã lưu biểu đồ vào: {output_dir.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
