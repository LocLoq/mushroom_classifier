import argparse
import csv
import gc
import inspect
import os

import torch
import torch.nn as nn
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def is_cuda_oom_error(error: Exception) -> bool:
    message = str(error).lower()
    return "out of memory" in message and "cuda" in message


def compute_metrics(all_labels: torch.Tensor, all_preds: torch.Tensor, num_classes: int):
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true_label, pred_label in zip(all_labels, all_preds):
        confusion_matrix[true_label, pred_label] += 1

    true_positive = confusion_matrix.diag().float()
    predicted_positive = confusion_matrix.sum(dim=0).float()
    actual_positive = confusion_matrix.sum(dim=1).float()

    precision_per_class = torch.where(
        predicted_positive > 0,
        true_positive / predicted_positive,
        torch.zeros_like(true_positive),
    )
    recall_per_class = torch.where(
        actual_positive > 0,
        true_positive / actual_positive,
        torch.zeros_like(true_positive),
    )
    f1_per_class = torch.where(
        (precision_per_class + recall_per_class) > 0,
        (2 * precision_per_class * recall_per_class) / (precision_per_class + recall_per_class),
        torch.zeros_like(true_positive),
    )

    support = actual_positive
    support_sum = support.sum()
    support_weights = support / support_sum if support_sum > 0 else torch.zeros_like(support)

    return {
        "precision_macro": precision_per_class.mean().item(),
        "recall_macro": recall_per_class.mean().item(),
        "f1_macro": f1_per_class.mean().item(),
        "precision_weighted": (precision_per_class * support_weights).sum().item(),
        "recall_weighted": (recall_per_class * support_weights).sum().item(),
        "f1_weighted": (f1_per_class * support_weights).sum().item(),
        "confusion_matrix": confusion_matrix,
    }


def to_rgb(image):
    return image.convert("RGB")


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, max_skip_attempts=20):
        imagefolder_init_params = inspect.signature(datasets.ImageFolder.__init__).parameters
        init_kwargs = {
            "root": root,
            "transform": transform,
        }
        if "allow_empty" in imagefolder_init_params:
            init_kwargs["allow_empty"] = True

        super().__init__(**init_kwargs)
        self.max_skip_attempts = max(1, int(max_skip_attempts))
        self._bad_indices = set()

    def _mark_bad_index(self, sample_index: int, error: Exception):
        if sample_index in self._bad_indices:
            return
        self._bad_indices.add(sample_index)

        sample_path = ""
        if 0 <= sample_index < len(self.samples):
            sample_path = self.samples[sample_index][0]

        print(
            "[Dataset] Bỏ qua ảnh lỗi: "
            f"idx={sample_index}, path={sample_path}, error={type(error).__name__}: {error}"
        )

    def __getitem__(self, index):
        if len(self.samples) == 0:
            raise RuntimeError("Dataset test không có sample hợp lệ để evaluate.")

        dataset_size = len(self.samples)
        current_index = int(index) % dataset_size
        max_attempts = min(dataset_size, self.max_skip_attempts)
        last_error = None

        for _ in range(max_attempts):
            if current_index in self._bad_indices:
                current_index = (current_index + 1) % dataset_size
                continue

            try:
                return super().__getitem__(current_index)
            except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as error:
                last_error = error
                self._mark_bad_index(current_index, error)
                current_index = (current_index + 1) % dataset_size

        raise RuntimeError(
            "Không thể lấy sample test hợp lệ sau nhiều lần thử. "
            f"Lỗi cuối: {type(last_error).__name__}: {last_error}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập test và xuất CSV metric/confusion matrix")
    parser.add_argument(
        "--model",
        type=str,
        choices=["efficientnet_b0", "resnet18", "vgg16"],
        default="efficientnet_b0",
        help="Kiến trúc mô hình để evaluate",
    )
    parser.add_argument("--data-dir", type=str, default="local_dataset", help="Thư mục gốc dataset")
    parser.add_argument("--train-subdir", type=str, default="train", help="Tên thư mục train con trong data-dir")
    parser.add_argument("--test-subdir", type=str, default="test", help="Tên thư mục test con trong data-dir")
    parser.add_argument("--model-path", type=str, default="nammushroom_efficientnet_b0.pth", help="Đường dẫn model weights")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size khi evaluate")
    parser.add_argument("--num-workers", type=int, default=2, help="Số worker DataLoader")
    parser.add_argument("--metrics-csv", type=str, default="test_metrics.csv", help="File CSV output metrics test")
    parser.add_argument(
        "--confusion-dir",
        type=str,
        default="confusion_matrices_test",
        help="Thư mục chứa confusion matrix CSV của test",
    )
    return parser.parse_args()


def infer_num_classes_from_state_dict(state_dict, model_name: str) -> int:
    if model_name == "efficientnet_b0":
        weight_key = "classifier.1.weight"
        bias_key = "classifier.1.bias"
    elif model_name == "resnet18":
        weight_key = "fc.weight"
        bias_key = "fc.bias"
    elif model_name == "vgg16":
        weight_key = "classifier.6.weight"
        bias_key = "classifier.6.bias"
    else:
        raise ValueError(f"Model không được hỗ trợ: {model_name}")

    if weight_key in state_dict:
        return int(state_dict[weight_key].shape[0])
    if bias_key in state_dict:
        return int(state_dict[bias_key].shape[0])
    raise KeyError(
        f"Không tìm thấy key output head ({weight_key}/{bias_key}) trong state_dict để suy ra số class."
    )


def build_model_for_eval(model_name: str, num_classes: int):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        return model

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    if model_name == "vgg16":
        model = models.vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        return model

    raise ValueError(f"Model không được hỗ trợ: {model_name}")


def resolve_eval_class_names(data_dir: str, train_subdir: str, test_dataset_classes, model_num_classes: int):
    train_dir = os.path.join(data_dir, train_subdir)
    if os.path.isdir(train_dir):
        train_dataset = datasets.ImageFolder(train_dir)
        train_classes = train_dataset.classes
        if len(train_classes) == model_num_classes:
            return train_classes

    # Fallback: dùng class test + class giả để đủ chiều model output.
    eval_class_names = list(test_dataset_classes)
    for class_index in range(len(eval_class_names), model_num_classes):
        eval_class_names.append(f"class_{class_index}")
    return eval_class_names


def run_eval_pass(
    model,
    test_loader,
    device,
    criterion,
    valid_test_indices,
    remap_table,
):
    running_loss = 0.0
    running_corrects = 0
    valid_sample_count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            valid_mask = torch.isin(labels, valid_test_indices)
            if not torch.any(valid_mask):
                continue

            inputs = inputs[valid_mask]
            labels = labels[valid_mask]

            remapped_labels = remap_table[labels]

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, remapped_labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == remapped_labels.data)
            valid_sample_count += inputs.size(0)
            all_preds.append(preds.detach().cpu())
            all_labels.append(remapped_labels.detach().cpu())

    return running_loss, running_corrects, valid_sample_count, all_preds, all_labels


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(to_rgb),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dir = os.path.join(args.data_dir, args.test_subdir)
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

    test_dataset = SafeImageFolder(test_dir, test_transform)
    if len(test_dataset) == 0:
        raise RuntimeError("Tập test không có ảnh hợp lệ để evaluate.")
    test_class_names = test_dataset.classes
    print(f"Các class test: {test_class_names}")

    model_state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model_num_classes = infer_num_classes_from_state_dict(model_state_dict, args.model)

    eval_class_names = resolve_eval_class_names(
        data_dir=args.data_dir,
        train_subdir=args.train_subdir,
        test_dataset_classes=test_class_names,
        model_num_classes=model_num_classes,
    )
    num_classes = len(eval_class_names)
    print(f"Số class từ model: {model_num_classes}")
    print(f"Số class dùng để evaluate: {num_classes}")

    # Remap index class từ test set sang không gian class đầy đủ để khớp output model.
    class_name_to_eval_index = {class_name: idx for idx, class_name in enumerate(eval_class_names)}
    unknown_test_classes = [
        class_name for class_name in test_class_names if class_name not in class_name_to_eval_index
    ]
    test_to_eval_index = {
        test_index: class_name_to_eval_index[class_name]
        for test_index, class_name in enumerate(test_class_names)
        if class_name in class_name_to_eval_index
    }

    if unknown_test_classes:
        unknown_count = sum(
            1 for target_index in test_dataset.targets
            if test_dataset.classes[target_index] in set(unknown_test_classes)
        )
        print(
            "[Warning] Bỏ qua class test không có trong model/eval: "
            f"{unknown_test_classes} | số ảnh bị bỏ qua: {unknown_count}"
        )

    if not test_to_eval_index:
        raise RuntimeError(
            "Không có class nào trong tập test khớp với không gian class của model. "
            "Không thể evaluate."
        )

    valid_test_indices = torch.tensor(sorted(test_to_eval_index.keys()), dtype=torch.long, device=device)
    remap_table = torch.full((len(test_class_names),), -1, dtype=torch.long, device=device)
    for test_index, eval_index in test_to_eval_index.items():
        remap_table[test_index] = eval_index

    model = build_model_for_eval(args.model, num_classes)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    requested_batch_size = max(1, int(args.batch_size))
    effective_batch_size = requested_batch_size

    while True:
        test_loader = DataLoader(
            test_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        try:
            (
                running_loss,
                running_corrects,
                valid_sample_count,
                all_preds,
                all_labels,
            ) = run_eval_pass(
                model=model,
                test_loader=test_loader,
                device=device,
                criterion=criterion,
                valid_test_indices=valid_test_indices,
                remap_table=remap_table,
            )
            if effective_batch_size != requested_batch_size:
                print(
                    "[Info] Evaluate đã tự giảm batch size do OOM: "
                    f"{requested_batch_size} -> {effective_batch_size}"
                )
            break
        except (torch.OutOfMemoryError, RuntimeError) as error:
            if device.type != "cuda" or not is_cuda_oom_error(error):
                raise
            if effective_batch_size == 1:
                raise

            next_batch_size = max(1, effective_batch_size // 2)
            if next_batch_size == effective_batch_size:
                raise

            print(
                "[Warning] CUDA OOM khi evaluate với batch_size="
                f"{effective_batch_size}. Thử lại với batch_size={next_batch_size}."
            )
            effective_batch_size = next_batch_size
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()

    if valid_sample_count == 0:
        raise RuntimeError("Không có ảnh hợp lệ để evaluate sau khi lọc class không khớp model.")

    test_loss = running_loss / valid_sample_count
    test_acc = running_corrects.double().item() / valid_sample_count

    labels_tensor = torch.cat(all_labels)
    preds_tensor = torch.cat(all_preds)
    metrics = compute_metrics(labels_tensor, preds_tensor, num_classes)

    print(
        f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} "
        f"Precision(macro): {metrics['precision_macro']:.4f} "
        f"Recall(macro): {metrics['recall_macro']:.4f} "
        f"F1(macro): {metrics['f1_macro']:.4f}"
    )
    print(
        f"Test Precision(weighted): {metrics['precision_weighted']:.4f} "
        f"Recall(weighted): {metrics['recall_weighted']:.4f} "
        f"F1(weighted): {metrics['f1_weighted']:.4f}"
    )

    with open(args.metrics_csv, mode="w", newline="", encoding="utf-8") as metrics_file:
        fieldnames = [
            "epoch",
            "phase",
            "loss",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ]
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "epoch": 1,
            "phase": "test",
            "loss": test_loss,
            "accuracy": test_acc,
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
        })

    os.makedirs(args.confusion_dir, exist_ok=True)
    cm_path = os.path.join(args.confusion_dir, "test_confusion.csv")
    with open(cm_path, mode="w", newline="", encoding="utf-8") as cm_file:
        cm_writer = csv.writer(cm_file)
        cm_writer.writerow(["true\\pred", *eval_class_names])
        confusion_matrix_np = metrics["confusion_matrix"].cpu().numpy()
        for row_index, row_values in enumerate(confusion_matrix_np):
            cm_writer.writerow([eval_class_names[row_index], *row_values.tolist()])

    print(f"Đã lưu test metrics CSV: {args.metrics_csv}")
    print(f"Đã lưu test confusion matrix CSV: {cm_path}")


if __name__ == "__main__":
    main()
