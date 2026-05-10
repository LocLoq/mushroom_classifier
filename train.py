import argparse
import csv
import inspect
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    import psutil
except ImportError:
    psutil = None


METRICS_FIELDNAMES = [
    'run_id',
    'model_name',
    'epoch',
    'phase',
    'loss',
    'accuracy',
    'precision_macro',
    'recall_macro',
    'f1_macro',
    'precision_weighted',
    'recall_weighted',
    'f1_weighted',
    'phase_time_sec',
    'epoch_time_sec',
    'cumulative_time_sec',
    'global_progress_percent',
    'vram_allocated_mb',
    'vram_reserved_mb',
    'vram_peak_allocated_mb',
    'cpu_usage_percent',
    'ram_used_mb',
    'ram_percent',
    'learning_rate',
    'batch_size',
    'num_workers',
]

BATCH_METRICS_FIELDNAMES = [
    'run_id',
    'model_name',
    'global_step',
    'epoch',
    'batch_index',
    'num_batches',
    'progress_percent',
    'batch_loss',
    'batch_accuracy',
    'elapsed_sec',
    'vram_allocated_mb',
    'vram_reserved_mb',
    'vram_peak_allocated_mb',
    'cpu_usage_percent',
    'ram_used_mb',
    'ram_percent',
]


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
        'precision_macro': precision_per_class.mean().item(),
        'recall_macro': recall_per_class.mean().item(),
        'f1_macro': f1_per_class.mean().item(),
        'precision_weighted': (precision_per_class * support_weights).sum().item(),
        'recall_weighted': (recall_per_class * support_weights).sum().item(),
        'f1_weighted': (f1_per_class * support_weights).sum().item(),
        'confusion_matrix': confusion_matrix,
    }


def to_rgb(image):
    return image.convert('RGB')


class FixedClassImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, class_to_idx=None, allow_empty=False):
        self._fixed_class_to_idx = class_to_idx
        imagefolder_init_params = inspect.signature(datasets.ImageFolder.__init__).parameters
        init_kwargs = {
            'root': root,
            'transform': transform,
        }
        if 'allow_empty' in imagefolder_init_params:
            init_kwargs['allow_empty'] = allow_empty
        super().__init__(**init_kwargs)

    def find_classes(self, directory):
        if self._fixed_class_to_idx is None:
            return super().find_classes(directory)
        classes = sorted(self._fixed_class_to_idx.keys(), key=lambda class_name: self._fixed_class_to_idx[class_name])
        return classes, dict(self._fixed_class_to_idx)


class SafeImageFolder(FixedClassImageFolder):
    def __init__(self, *args, max_skip_attempts=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_skip_attempts = max(1, int(max_skip_attempts))
        self._bad_indices = set()

    def _mark_bad_index(self, sample_index: int, error: Exception):
        if sample_index in self._bad_indices:
            return
        self._bad_indices.add(sample_index)

        sample_path = ''
        if 0 <= sample_index < len(self.samples):
            sample_path = self.samples[sample_index][0]

        print(
            '[Dataset] Bỏ qua ảnh lỗi: '
            f'idx={sample_index}, path={sample_path}, error={type(error).__name__}: {error}'
        )

    def __getitem__(self, index):
        if len(self.samples) == 0:
            raise RuntimeError('Dataset không có sample hợp lệ để đọc.')

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
            'Không thể lấy sample hợp lệ sau nhiều lần thử. '
            f'Lỗi cuối: {type(last_error).__name__}: {last_error}'
        )


def log_empty_classes(dataset, split_name: str):
    if not hasattr(dataset, 'targets'):
        return

    class_counts = [0 for _ in dataset.classes]
    for target_index in dataset.targets:
        class_counts[int(target_index)] += 1

    empty_classes = [dataset.classes[i] for i, sample_count in enumerate(class_counts) if sample_count == 0]
    if empty_classes:
        print(
            f"[{split_name}] Bỏ qua {len(empty_classes)} class không có ảnh hợp lệ: "
            f"{', '.join(empty_classes)}"
        )


def build_safe_imagefolder(root_dir: str, transform, split_name: str, class_to_idx=None):
    try:
        dataset = SafeImageFolder(
            root=root_dir,
            transform=transform,
            class_to_idx=class_to_idx,
            allow_empty=True,
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}\n"
            "Không thể bỏ qua class rỗng với phiên bản torchvision hiện tại. "
            "Hãy nâng torchvision lên bản hỗ trợ allow_empty hoặc dọn dữ liệu lỗi."
        ) from error

    log_empty_classes(dataset, split_name)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train image classifier with checkpoint/resume and runtime reports')
    parser.add_argument(
        '--model',
        type=str,
        choices=['efficientnet_b0', 'resnet18', 'vgg16', 'mobilenet_v3'],
        default='efficientnet_b0',
        help='Backbone model architecture',
    )
    parser.add_argument('--data-dir', type=str, default='local_dataset', help='Dataset root folder')
    parser.add_argument('--train-subdir', type=str, default='train', help='Train subfolder inside data-dir')
    parser.add_argument('--val-subdir', type=str, default='val', help='Validation subfolder inside data-dir')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Max training epochs (early stopping may stop sooner; this stays as custom epoch cap)',
    )
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--backbone-lr', type=float, default=2e-4, help='Learning rate for backbone parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing for CrossEntropyLoss')
    parser.add_argument('--unfreeze-epoch', type=int, default=4, help='Epoch index (1-based) to unfreeze backbone blocks')
    parser.add_argument('--unfreeze-blocks', type=int, default=3, help='Number of last backbone blocks to unfreeze')
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['none', 'cosine'],
        default='cosine',
        help='Learning rate scheduler strategy',
    )
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum LR for cosine scheduler')
    parser.add_argument(
        '--class-weighted-loss',
        action='store_true',
        help='Use inverse-frequency class weights in CrossEntropyLoss',
    )
    parser.add_argument(
        '--no-class-weighted-loss',
        action='store_true',
        help='Disable class-weighted CrossEntropyLoss',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--metrics-csv', type=str, default='training_metrics.csv', help='Metrics CSV output path')
    parser.add_argument(
        '--batch-metrics-csv',
        type=str,
        default='training_batch_metrics.csv',
        help='Per-batch metrics CSV output path (for detailed progress charts)',
    )
    parser.add_argument('--confusion-dir', type=str, default='confusion_matrices', help='Confusion matrix folder')
    parser.add_argument('--model-out', type=str, default='nammushroom_efficientnet_b0.pth', help='Final model path')
    parser.add_argument(
        '--best-model-out',
        type=str,
        default='best_nammushroom_efficientnet_b0.pth',
        help='Best model path by validation accuracy',
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default='checkpoints/latest_checkpoint.pth',
        help='Checkpoint path for autosave/resume',
    )
    parser.add_argument(
        '--best-checkpoint-path',
        type=str,
        default='checkpoints/best_checkpoint.pth',
        help='Best checkpoint path by validation accuracy',
    )
    parser.add_argument(
        '--save-every-batches',
        type=int,
        default=50,
        help='Autosave checkpoint every N train batches (<=0 to disable mid-epoch checkpoint)',
    )
    parser.add_argument(
        '--min-epochs',
        type=int,
        default=10,
        help='Minimum epochs to run before early stopping can trigger',
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=8,
        help='Stop if validation accuracy does not improve for N epochs after min-epochs',
    )
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint path')
    parser.add_argument(
        '--detach',
        action='store_true',
        help='Run training in background (safe to close SSH after startup)',
    )
    parser.add_argument(
        '--detach-log',
        type=str,
        default='train_detach.log',
        help='Detached process log file path',
    )
    parser.add_argument(
        '--detach-pid-file',
        type=str,
        default='train_detach.pid',
        help='Detached process PID file path',
    )
    parser.add_argument('--_detached-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--report-path', type=str, default='training_report.json', help='Training report JSON output path')
    parser.add_argument(
        '--discord-webhook-url',
        type=str,
        default=os.getenv('DISCORD_WEBHOOK_URL', ''),
        help='Discord webhook URL for training progress notifications',
    )
    parser.add_argument(
        '--discord-notify-every-epochs',
        type=int,
        default=1,
        help='Send Discord progress every N epochs (<=0 to disable periodic updates)',
    )
    parser.add_argument(
        '--append-metrics',
        action='store_true',
        help='Append previous metric rows into current output file (default on)',
    )
    parser.add_argument(
        '--no-append-metrics',
        action='store_true',
        help='Do not keep old rows when writing training_metrics.csv',
    )
    parser.set_defaults(append_metrics=True, class_weighted_loss=True)
    return parser.parse_args()


def apply_model_specific_default_paths(args):
    model_suffix = args.model.lower()

    if args.metrics_csv == 'training_metrics.csv':
        args.metrics_csv = f'training_metrics_{model_suffix}.csv'
    if args.batch_metrics_csv == 'training_batch_metrics.csv':
        args.batch_metrics_csv = f'training_batch_metrics_{model_suffix}.csv'
    if args.confusion_dir == 'confusion_matrices':
        args.confusion_dir = f'confusion_matrices_{model_suffix}'
    if args.model_out == 'nammushroom_efficientnet_b0.pth':
        args.model_out = f'nammushroom_{model_suffix}.pth'
    if args.best_model_out == 'best_nammushroom_efficientnet_b0.pth':
        args.best_model_out = f'best_nammushroom_{model_suffix}.pth'
    if args.checkpoint_path == 'checkpoints/latest_checkpoint.pth':
        args.checkpoint_path = f'checkpoints/latest_checkpoint_{model_suffix}.pth'
    if args.best_checkpoint_path == 'checkpoints/best_checkpoint.pth':
        args.best_checkpoint_path = f'checkpoints/best_checkpoint_{model_suffix}.pth'
    if args.report_path == 'training_report.json':
        args.report_path = f'training_report_{model_suffix}.json'


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)


def ensure_parent_dir(path: str):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def get_vram_stats(device):
    if device.type != 'cuda':
        return 0.0, 0.0, 0.0
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return allocated_mb, reserved_mb, peak_allocated_mb


def get_system_resource_stats():
    if psutil is None:
        return 0.0, 0.0, 0.0

    cpu_usage_percent = psutil.cpu_percent(interval=None)
    virtual_memory = psutil.virtual_memory()
    ram_used_mb = (virtual_memory.total - virtual_memory.available) / (1024 ** 2)
    ram_percent = virtual_memory.percent
    return float(cpu_usage_percent), float(ram_used_mb), float(ram_percent)


def build_dataloader(dataset, phase: str, batch_size: int, num_workers: int, seed: int, epoch: int):
    shuffle = phase == 'train'
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def get_model_parameter_groups(model, model_name: str):
    if model_name in ('efficientnet_b0', 'vgg16', 'mobilenet_v3'):
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())
        return backbone_params, head_params

    if model_name == 'resnet18':
        backbone_modules = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
        backbone_params = []
        for module in backbone_modules:
            backbone_params.extend(list(module.parameters()))
        head_params = list(model.fc.parameters())
        return backbone_params, head_params
    raise ValueError(f'Model không được hỗ trợ: {model_name}')


def freeze_backbone_enable_head(model, model_name: str):
    backbone_params, head_params = get_model_parameter_groups(model, model_name)
    for param in backbone_params:
        param.requires_grad = False
    for param in head_params:
        param.requires_grad = True


def build_model(model_name: str, num_classes: int, device):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f'Model không được hỗ trợ: {model_name}')

    freeze_backbone_enable_head(model, model_name)
    return model.to(device)


def unfreeze_last_feature_blocks(model, model_name: str, unfreeze_blocks: int):
    if model_name in ('efficientnet_b0', 'vgg16'):
        feature_blocks = list(model.features.children())
    elif model_name == 'resnet18':
        feature_blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    elif model_name == 'mobilenet_v3':
        feature_blocks = list(model.features.children())
    else:
        raise ValueError(f'Model không được hỗ trợ: {model_name}')

    total_blocks = len(feature_blocks)

    freeze_backbone_enable_head(model, model_name)

    if unfreeze_blocks <= 0:
        return 0, total_blocks

    num_to_unfreeze = min(unfreeze_blocks, total_blocks)
    for block in feature_blocks[-num_to_unfreeze:]:
        for param in block.parameters():
            param.requires_grad = True

    return num_to_unfreeze, total_blocks


def build_optimizer(model, model_name: str, head_lr: float, backbone_lr: float, weight_decay: float):
    backbone_params, head_params = get_model_parameter_groups(model, model_name)
    if not backbone_params or not head_params:
        return optim.AdamW(model.parameters(), lr=head_lr, weight_decay=weight_decay)

    return optim.AdamW(
        [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, scheduler_name: str, max_epochs: int, min_lr: float):
    if scheduler_name == 'none':
        return None
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_epochs),
            eta_min=min_lr,
        )
    raise ValueError(f'Scheduler không được hỗ trợ: {scheduler_name}')


def compute_inverse_frequency_class_weights(targets, num_classes: int, device):
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for target_index in targets:
        class_counts[int(target_index)] += 1.0

    safe_counts = torch.clamp(class_counts, min=1.0)
    inverse_frequency = class_counts.sum() / safe_counts
    normalized_weights = inverse_frequency / inverse_frequency.mean()
    return normalized_weights.to(device), class_counts


def load_existing_metrics_rows(metrics_csv_path: str):
    if not os.path.isfile(metrics_csv_path):
        return []

    rows = []
    with open(metrics_csv_path, mode='r', newline='', encoding='utf-8') as metrics_file:
        reader = csv.DictReader(metrics_file)
        for row in reader:
            normalized = {key: row.get(key, '') for key in METRICS_FIELDNAMES}
            rows.append(normalized)
    return rows


def write_metrics_csv(metrics_csv_path: str, metrics_log_rows):
    ensure_parent_dir(metrics_csv_path)
    with open(metrics_csv_path, mode='w', newline='', encoding='utf-8') as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(metrics_log_rows)


def init_batch_metrics_csv(batch_metrics_csv_path: str):
    ensure_parent_dir(batch_metrics_csv_path)
    if os.path.isfile(batch_metrics_csv_path):
        return

    with open(batch_metrics_csv_path, mode='w', newline='', encoding='utf-8') as batch_metrics_file:
        writer = csv.DictWriter(batch_metrics_file, fieldnames=BATCH_METRICS_FIELDNAMES)
        writer.writeheader()


def append_batch_metric(batch_metrics_csv_path: str, row):
    with open(batch_metrics_csv_path, mode='a', newline='', encoding='utf-8') as batch_metrics_file:
        writer = csv.DictWriter(batch_metrics_file, fieldnames=BATCH_METRICS_FIELDNAMES)
        writer.writerow(row)


def write_json_report(report_path: str, report_data):
    ensure_parent_dir(report_path)
    with open(report_path, 'w', encoding='utf-8') as report_file:
        json.dump(report_data, report_file, ensure_ascii=False, indent=2)


def send_discord_webhook(webhook_url: str, message: str):
    if not webhook_url:
        return

    message_text = (message or '').strip()
    if not message_text:
        return

    max_message_length = 1900
    if len(message_text) > max_message_length:
        message_text = message_text[: max_message_length - 3] + '...'

    payload = json.dumps({'content': message_text}, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status_code = getattr(response, 'status', None)
            if status_code and status_code >= 400:
                print(f'[Discord] Webhook trả về status không thành công: {status_code}')
    except urllib.error.URLError as error:
        print(f'[Discord] Không gửi được webhook: {error}')


def save_checkpoint(checkpoint_path: str, checkpoint_data):
    ensure_parent_dir(checkpoint_path)
    torch.save(checkpoint_data, checkpoint_path)


def save_model(model_path: str, model):
    ensure_parent_dir(model_path)
    torch.save(model.state_dict(), model_path)


def launch_detached_training(args):
    if args._detached_child:
        return False

    filtered_argv = []
    for token in sys.argv[1:]:
        if token in ('--detach', '--_detached-child'):
            continue
        filtered_argv.append(token)
    filtered_argv.append('--_detached-child')

    train_script_path = os.path.abspath(__file__)
    command = [sys.executable, train_script_path, *filtered_argv]

    ensure_parent_dir(args.detach_log)
    ensure_parent_dir(args.detach_pid_file)

    with open(args.detach_log, 'a', encoding='utf-8') as log_file:
        if os.name == 'nt':
            creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                close_fds=True,
                creationflags=creationflags,
            )
        else:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                close_fds=True,
                start_new_session=True,
            )

    with open(args.detach_pid_file, 'w', encoding='utf-8') as pid_file:
        pid_file.write(f'{child.pid}\n')

    print(f'[Detach] PID: {child.pid}')
    print(f'[Detach] Log file: {os.path.abspath(args.detach_log)}')
    print(f'[Detach] PID file: {os.path.abspath(args.detach_pid_file)}')
    print('[Detach] Parent process will exit now. Training continues in background.')
    return True


def build_report(
    status,
    started_at_iso,
    args,
    device,
    num_classes,
    class_names,
    completed_epochs,
    max_epochs,
    best_val_acc,
    best_epoch,
    final_elapsed_sec,
    epoch_summaries,
    stop_reason,
    run_id,
):
    now_iso = datetime.now().isoformat(timespec='seconds')
    report = {
        'status': status,
        'run_id': run_id,
        'model_name': args.model,
        'started_at': started_at_iso,
        'last_updated_at': now_iso,
        'device': str(device),
        'num_classes': num_classes,
        'class_names': class_names,
        'completed_epochs': completed_epochs,
        'target_epochs': max_epochs,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'total_train_time_sec': float(final_elapsed_sec),
        'train_config': vars(args),
        'epoch_summaries': epoch_summaries,
        'checkpoint_path': args.checkpoint_path,
        'best_checkpoint_path': args.best_checkpoint_path,
        'metrics_csv_path': args.metrics_csv,
        'batch_metrics_csv_path': args.batch_metrics_csv,
        'model_output_path': args.model_out,
        'best_model_output_path': args.best_model_out,
        'stop_reason': stop_reason,
    }
    if status == 'completed':
        report['ended_at'] = now_iso
    return report


if __name__ == '__main__':
    args = parse_args()
    if args.no_append_metrics:
        args.append_metrics = False
    if args.no_class_weighted_loss:
        args.class_weighted_loss = False

    apply_model_specific_default_paths(args)

    if args.detach and launch_detached_training(args):
        raise SystemExit(0)

    if args.min_epochs < 1:
        raise ValueError('--min-epochs phải >= 1')
    if args.epochs < 1:
        raise ValueError('--epochs phải >= 1')
    if args.early_stop_patience < 1:
        raise ValueError('--early-stop-patience phải >= 1')
    if args.unfreeze_epoch < 1:
        raise ValueError('--unfreeze-epoch phải >= 1')
    if args.unfreeze_blocks < 0:
        raise ValueError('--unfreeze-blocks phải >= 0')
    if args.backbone_lr <= 0:
        raise ValueError('--backbone-lr phải > 0')
    if args.weight_decay < 0:
        raise ValueError('--weight-decay phải >= 0')
    if args.min_lr < 0:
        raise ValueError('--min-lr phải >= 0')
    if not (0.0 <= args.label_smoothing < 1.0):
        raise ValueError('--label-smoothing phải nằm trong [0.0, 1.0)')
    if args.discord_notify_every_epochs < 0:
        raise ValueError('--discord-notify-every-epochs phải >= 0')

    seed_everything(args.seed)

    session_start_time = time.time()
    started_at_iso = datetime.now().isoformat(timespec='seconds')
    run_id = started_at_iso.replace(':', '-').replace('T', '_')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Đang sử dụng thiết bị: {device}')
    print(f'Model: {args.model}')
    if psutil is None:
        print('[Resource] Chưa cài psutil, CPU/RAM metric sẽ là 0.0. Có thể cài: pip install psutil')
    else:
        psutil.cpu_percent(interval=None)

    send_discord_webhook(
        args.discord_webhook_url,
        (
            f"[Train Start] model={args.model} | run_id={run_id} | device={device} | "
            f"epochs={args.epochs} | batch_size={args.batch_size}"
        ),
    )

    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(to_rgb),
            transforms.RandomResizedCrop(
                224,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.02),
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                shear=4,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.03),
            transforms.RandomGrayscale(p=0.02),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.12,
                scale=(0.02, 0.08),
                ratio=(0.3, 3.3),
                value='random',
            ),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(to_rgb),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dir = os.path.join(args.data_dir, args.train_subdir)
    val_dir = os.path.join(args.data_dir, args.val_subdir)
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f'Không tìm thấy thư mục train: {train_dir}')
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f'Không tìm thấy thư mục val: {val_dir}')

    train_dataset = build_safe_imagefolder(
        root_dir=train_dir,
        transform=data_transforms['train'],
        split_name='train',
    )
    if len(train_dataset) == 0:
        raise RuntimeError('Tập train không có ảnh hợp lệ sau khi bỏ qua class lỗi.')

    val_dataset = build_safe_imagefolder(
        root_dir=val_dir,
        transform=data_transforms['val'],
        split_name='val',
        class_to_idx=train_dataset.class_to_idx,
    )

    image_datasets = {
        'train': train_dataset,
        'val': val_dataset,
    }

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f'Các loại nấm: {class_names}')

    os.makedirs(args.confusion_dir, exist_ok=True)

    metrics_log_rows = []
    epoch_summaries = []

    if args.append_metrics and not args.resume:
        metrics_log_rows.extend(load_existing_metrics_rows(args.metrics_csv))

    model = build_model(args.model, num_classes, device)

    class_weights = None
    if args.class_weighted_loss:
        class_weights, class_counts = compute_inverse_frequency_class_weights(
            targets=train_dataset.targets,
            num_classes=num_classes,
            device=device,
        )
        print(f'Áp dụng class-weighted loss | class_counts={class_counts.tolist()}')

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(
        model=model,
        model_name=args.model,
        head_lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        max_epochs=args.epochs,
        min_lr=args.min_lr,
    )

    start_epoch = 0
    start_phase = 'train'
    start_batch_idx = 0
    best_val_acc = 0.0
    best_epoch = 0
    accumulated_elapsed_sec = 0.0
    completed_epochs = 0
    epochs_without_improvement = 0
    global_step = 0
    backbone_unfrozen = False

    if args.resume:
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f'Không tìm thấy checkpoint để resume: {args.checkpoint_path}')

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        if optimizer_state_dict is not None:
            try:
                optimizer.load_state_dict(optimizer_state_dict)
            except ValueError as error:
                print(
                    '[Resume] Không thể nạp optimizer state từ checkpoint cũ; '
                    f'optimizer sẽ được khởi tạo mới. Chi tiết: {error}'
                )

        scheduler_state_dict = checkpoint.get('scheduler_state_dict')
        if scheduler is not None and scheduler_state_dict is not None:
            try:
                scheduler.load_state_dict(scheduler_state_dict)
            except ValueError as error:
                print(
                    '[Resume] Không thể nạp scheduler state từ checkpoint cũ; '
                    f'scheduler sẽ chạy mới. Chi tiết: {error}'
                )

        start_epoch = int(checkpoint.get('next_epoch', 0))
        start_phase = checkpoint.get('next_phase', 'train')
        start_batch_idx = int(checkpoint.get('next_batch_idx', 0))
        best_val_acc = float(checkpoint.get('best_val_acc', 0.0))
        best_epoch = int(checkpoint.get('best_epoch', 0))
        metrics_log_rows = checkpoint.get('metrics_log_rows', metrics_log_rows)
        epoch_summaries = checkpoint.get('epoch_summaries', [])
        accumulated_elapsed_sec = float(checkpoint.get('accumulated_elapsed_sec', 0.0))
        completed_epochs = int(checkpoint.get('completed_epochs', 0))
        epochs_without_improvement = int(checkpoint.get('epochs_without_improvement', 0))
        global_step = int(checkpoint.get('global_step', 0))
        backbone_unfrozen = bool(checkpoint.get('backbone_unfrozen', False))
        run_id = checkpoint.get('run_id', run_id)

        print(
            f'Resume từ checkpoint: epoch={start_epoch + 1}, phase={start_phase}, '
            f'batch bắt đầu={start_batch_idx + 1}'
        )
        send_discord_webhook(
            args.discord_webhook_url,
            (
                f"[Train Resume] model={args.model} | run_id={run_id} | "
                f"start_epoch={start_epoch + 1} | phase={start_phase}"
            ),
        )

    init_batch_metrics_csv(args.batch_metrics_csv)

    max_epochs = args.epochs
    phases = ['train']
    if len(image_datasets['val']) > 0:
        phases.append('val')
    else:
        print('[val] Không có ảnh hợp lệ, sẽ bỏ qua phase val trong quá trình train.')
    phase_to_index = {phase_name: index for index, phase_name in enumerate(phases)}

    if start_phase not in phase_to_index:
        print(
            f"Phase '{start_phase}' trong checkpoint không còn hợp lệ với dữ liệu hiện tại. "
            "Chuyển về phase train từ đầu epoch."
        )
        start_phase = 'train'
        start_batch_idx = 0

    if start_epoch >= max_epochs:
        print('Checkpoint cho thấy training đã hoàn tất theo số epoch hiện tại. Không train thêm.')

    stop_reason = 'max_epochs_reached'

    if not backbone_unfrozen and start_epoch + 1 >= args.unfreeze_epoch:
        backbone_unfrozen = True
    if backbone_unfrozen:
        unfrozen_blocks, total_blocks = unfreeze_last_feature_blocks(model, args.model, args.unfreeze_blocks)
        print(
            f'[FineTune] Resume với backbone mở {unfrozen_blocks}/{total_blocks} blocks cuối '
            f'(unfreeze_blocks={args.unfreeze_blocks}).'
        )

    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{max_epochs}')
        print('-' * 10)

        if not backbone_unfrozen and (epoch + 1) >= args.unfreeze_epoch:
            unfrozen_blocks, total_blocks = unfreeze_last_feature_blocks(model, args.model, args.unfreeze_blocks)
            backbone_unfrozen = True
            print(
                f'[FineTune] Epoch {epoch + 1}: mở {unfrozen_blocks}/{total_blocks} blocks cuối '
                f'của backbone để fine-tune.'
            )

        backbone_lr_now = optimizer.param_groups[0]['lr']
        head_lr_now = optimizer.param_groups[1]['lr']
        print(
            f'LR head/backbone: {head_lr_now:.6f}/{backbone_lr_now:.6f} | '
            f'backbone_unfrozen={backbone_unfrozen}'
        )

        epoch_train_metrics = None
        epoch_val_metrics = None
        epoch_train_acc = None
        epoch_train_loss = None
        epoch_val_acc = None
        epoch_val_loss = None

        for phase_index, phase in enumerate(phases):
            if epoch == start_epoch and phase_index < phase_to_index[start_phase]:
                continue

            dataloader = build_dataloader(
                image_datasets[phase],
                phase=phase,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                epoch=epoch,
            )

            skip_batches = 0
            if epoch == start_epoch and phase == start_phase:
                skip_batches = start_batch_idx

            if phase == 'train':
                model.train()
            else:
                model.eval()

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)

            phase_start_time = time.time()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            processed_samples = 0

            num_batches = len(dataloader)
            if skip_batches > 0:
                print(f"Bỏ qua {skip_batches}/{num_batches} batch đã xử lý trước đó ở phase '{phase}'")

            for batch_idx, (inputs, labels) in enumerate(dataloader):
                if batch_idx < skip_batches:
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                batch_correct = torch.sum(preds == labels.data).item()
                batch_size_now = inputs.size(0)
                batch_accuracy = batch_correct / batch_size_now if batch_size_now > 0 else 0.0

                running_loss += loss.item() * batch_size_now
                running_corrects += torch.sum(preds == labels.data)
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())
                processed_samples += batch_size_now

                if phase == 'train':
                    global_step += 1
                    cumulative_time_sec = accumulated_elapsed_sec + (time.time() - session_start_time)
                    train_progress = (epoch + (batch_idx + 1) / max(1, num_batches)) / max_epochs
                    progress_percent = min(100.0, max(0.0, train_progress * 100.0))
                    vram_allocated_mb, vram_reserved_mb, vram_peak_allocated_mb = get_vram_stats(device)
                    cpu_usage_percent, ram_used_mb, ram_percent = get_system_resource_stats()

                    append_batch_metric(args.batch_metrics_csv, {
                        'run_id': run_id,
                        'model_name': args.model,
                        'global_step': global_step,
                        'epoch': epoch + 1,
                        'batch_index': batch_idx + 1,
                        'num_batches': num_batches,
                        'progress_percent': float(progress_percent),
                        'batch_loss': float(loss.item()),
                        'batch_accuracy': float(batch_accuracy),
                        'elapsed_sec': float(cumulative_time_sec),
                        'vram_allocated_mb': float(vram_allocated_mb),
                        'vram_reserved_mb': float(vram_reserved_mb),
                        'vram_peak_allocated_mb': float(vram_peak_allocated_mb),
                        'cpu_usage_percent': float(cpu_usage_percent),
                        'ram_used_mb': float(ram_used_mb),
                        'ram_percent': float(ram_percent),
                    })

                    if args.save_every_batches > 0 and (batch_idx + 1) % args.save_every_batches == 0:
                        checkpoint_data = {
                            'run_id': run_id,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                            'next_epoch': epoch,
                            'next_phase': phase,
                            'next_batch_idx': batch_idx + 1,
                            'best_val_acc': best_val_acc,
                            'best_epoch': best_epoch,
                            'metrics_log_rows': metrics_log_rows,
                            'epoch_summaries': epoch_summaries,
                            'accumulated_elapsed_sec': cumulative_time_sec,
                            'completed_epochs': completed_epochs,
                            'epochs_without_improvement': epochs_without_improvement,
                            'global_step': global_step,
                            'backbone_unfrozen': backbone_unfrozen,
                            'train_config': vars(args),
                            'class_names': class_names,
                        }
                        save_checkpoint(args.checkpoint_path, checkpoint_data)
                        print(
                            f'[Checkpoint] Đã autosave tại epoch {epoch + 1}, phase {phase}, '
                            f'batch {batch_idx + 1}/{num_batches}'
                        )

            if not all_labels:
                raise RuntimeError(
                    f'Không có batch nào được xử lý ở epoch={epoch + 1}, phase={phase}. '
                    'Hãy kiểm tra dữ liệu hoặc tham số resume/checkpoint.'
                )

            phase_time_sec = time.time() - phase_start_time
            epoch_loss = running_loss / processed_samples
            epoch_acc = running_corrects.double() / processed_samples
            labels_tensor = torch.cat(all_labels)
            preds_tensor = torch.cat(all_preds)
            metrics = compute_metrics(labels_tensor, preds_tensor, num_classes)
            vram_allocated_mb, vram_reserved_mb, vram_peak_allocated_mb = get_vram_stats(device)
            cpu_usage_percent, ram_used_mb, ram_percent = get_system_resource_stats()

            global_progress_percent = ((epoch + 1) / max_epochs) * 100.0
            cumulative_time_sec = accumulated_elapsed_sec + (time.time() - session_start_time)
            learning_rate = max(param_group['lr'] for param_group in optimizer.param_groups)

            print(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} "
                f"Acc: {epoch_acc:.4f} "
                f"Precision(macro): {metrics['precision_macro']:.4f} "
                f"Recall(macro): {metrics['recall_macro']:.4f} "
                f"F1(macro): {metrics['f1_macro']:.4f}"
            )
            print(
                f"{phase.capitalize()} Precision(weighted): {metrics['precision_weighted']:.4f} "
                f"Recall(weighted): {metrics['recall_weighted']:.4f} "
                f"F1(weighted): {metrics['f1_weighted']:.4f}"
            )
            print(
                f"{phase.capitalize()} Time: {phase_time_sec:.2f}s | "
                f"Progress: {global_progress_percent:.2f}% | "
                f"VRAM allocated/reserved/peak (MB): "
                f"{vram_allocated_mb:.1f}/{vram_reserved_mb:.1f}/{vram_peak_allocated_mb:.1f} | "
                f"CPU: {cpu_usage_percent:.1f}% | RAM: {ram_used_mb:.1f}MB ({ram_percent:.1f}%)"
            )

            metrics_log_rows.append({
                'run_id': run_id,
                'model_name': args.model,
                'epoch': epoch + 1,
                'phase': phase,
                'loss': float(epoch_loss),
                'accuracy': float(epoch_acc),
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'f1_macro': metrics['f1_macro'],
                'precision_weighted': metrics['precision_weighted'],
                'recall_weighted': metrics['recall_weighted'],
                'f1_weighted': metrics['f1_weighted'],
                'phase_time_sec': float(phase_time_sec),
                'epoch_time_sec': 0.0,
                'cumulative_time_sec': float(cumulative_time_sec),
                'global_progress_percent': float(global_progress_percent),
                'vram_allocated_mb': float(vram_allocated_mb),
                'vram_reserved_mb': float(vram_reserved_mb),
                'vram_peak_allocated_mb': float(vram_peak_allocated_mb),
                'cpu_usage_percent': float(cpu_usage_percent),
                'ram_used_mb': float(ram_used_mb),
                'ram_percent': float(ram_percent),
                'learning_rate': float(learning_rate),
                'batch_size': int(args.batch_size),
                'num_workers': int(args.num_workers),
            })

            if phase == 'train':
                epoch_train_metrics = metrics
                epoch_train_acc = float(epoch_acc)
                epoch_train_loss = float(epoch_loss)
            else:
                epoch_val_metrics = metrics
                epoch_val_acc = float(epoch_acc)
                epoch_val_loss = float(epoch_loss)
                current_val_acc = float(epoch_acc)
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0

                    save_model(args.best_model_out, model)
                    best_checkpoint_data = {
                        'run_id': run_id,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'epoch': epoch + 1,
                        'best_val_acc': best_val_acc,
                        'best_epoch': best_epoch,
                        'global_step': global_step,
                        'backbone_unfrozen': backbone_unfrozen,
                        'train_config': vars(args),
                        'class_names': class_names,
                    }
                    save_checkpoint(args.best_checkpoint_path, best_checkpoint_data)
                    print(
                        f'[Best] Epoch {best_epoch} | Val Acc={best_val_acc:.4f} | '
                        f'Lưu best model/checkpoint thành công.'
                    )
                    send_discord_webhook(
                        args.discord_webhook_url,
                        (
                            f"[New Best] model={args.model} | run_id={run_id} | "
                            f"epoch={best_epoch}/{max_epochs} | val_acc={best_val_acc:.4f}"
                        ),
                    )
                else:
                    epochs_without_improvement += 1

            if phase == 'val':
                print('Val Confusion Matrix (rows=true, cols=pred):')
                print(metrics['confusion_matrix'])
                print('Class index mapping:')
                for index, class_name in enumerate(class_names):
                    print(f'  {index}: {class_name}')

                cm_path = os.path.join(args.confusion_dir, f'val_confusion_epoch_{epoch + 1}.csv')
                with open(cm_path, mode='w', newline='', encoding='utf-8') as cm_file:
                    cm_writer = csv.writer(cm_file)
                    cm_writer.writerow(['true\\pred', *class_names])
                    confusion_matrix_np = metrics['confusion_matrix'].cpu().numpy()
                    for row_index, row_values in enumerate(confusion_matrix_np):
                        cm_writer.writerow([class_names[row_index], *row_values.tolist()])

            write_metrics_csv(args.metrics_csv, metrics_log_rows)

            next_epoch = epoch
            next_phase = phase
            next_batch_idx = num_batches
            if phase == 'train':
                next_phase = 'val'
                next_batch_idx = 0
            else:
                next_epoch = epoch + 1
                next_phase = 'train'
                next_batch_idx = 0

            checkpoint_data = {
                'run_id': run_id,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'next_epoch': next_epoch,
                'next_phase': next_phase,
                'next_batch_idx': next_batch_idx,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'metrics_log_rows': metrics_log_rows,
                'epoch_summaries': epoch_summaries,
                'accumulated_elapsed_sec': accumulated_elapsed_sec + (time.time() - session_start_time),
                'completed_epochs': completed_epochs,
                'epochs_without_improvement': epochs_without_improvement,
                'global_step': global_step,
                'backbone_unfrozen': backbone_unfrozen,
                'train_config': vars(args),
                'class_names': class_names,
            }
            save_checkpoint(args.checkpoint_path, checkpoint_data)

        epoch_time_sec = time.time() - epoch_start_time
        completed_epochs = max(completed_epochs, epoch + 1)

        for row in reversed(metrics_log_rows):
            if row.get('run_id') == run_id and int(row.get('epoch', 0)) == epoch + 1:
                row['epoch_time_sec'] = float(epoch_time_sec)
            if int(row.get('epoch', 0)) < epoch + 1:
                break

        epoch_summary = {
            'epoch': epoch + 1,
            'epoch_time_sec': float(epoch_time_sec),
            'train_f1_macro': epoch_train_metrics['f1_macro'] if epoch_train_metrics else None,
            'val_f1_macro': epoch_val_metrics['f1_macro'] if epoch_val_metrics else None,
            'val_accuracy': float(best_val_acc) if epoch_val_metrics else None,
            'epochs_without_improvement': int(epochs_without_improvement),
        }
        epoch_summaries.append(epoch_summary)

        if scheduler is not None:
            scheduler.step()

        write_metrics_csv(args.metrics_csv, metrics_log_rows)

        running_report = build_report(
            status='running',
            started_at_iso=started_at_iso,
            args=args,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            completed_epochs=completed_epochs,
            max_epochs=max_epochs,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            final_elapsed_sec=accumulated_elapsed_sec + (time.time() - session_start_time),
            epoch_summaries=epoch_summaries,
            stop_reason='in_progress',
            run_id=run_id,
        )
        write_json_report(args.report_path, running_report)

        if args.discord_notify_every_epochs > 0 and ((epoch + 1) % args.discord_notify_every_epochs == 0):
            train_acc_text = f'{epoch_train_acc:.4f}' if epoch_train_acc is not None else 'n/a'
            train_loss_text = f'{epoch_train_loss:.4f}' if epoch_train_loss is not None else 'n/a'
            val_acc_text = f'{epoch_val_acc:.4f}' if epoch_val_acc is not None else 'n/a'
            val_loss_text = f'{epoch_val_loss:.4f}' if epoch_val_loss is not None else 'n/a'
            elapsed_min = (accumulated_elapsed_sec + (time.time() - session_start_time)) / 60.0
            send_discord_webhook(
                args.discord_webhook_url,
                (
                    f"[Train Progress] model={args.model} | run_id={run_id} | epoch={epoch + 1}/{max_epochs} | "
                    f"train_loss={train_loss_text} train_acc={train_acc_text} | "
                    f"val_loss={val_loss_text} val_acc={val_acc_text} | "
                    f"best_val_acc={best_val_acc:.4f} | elapsed={elapsed_min:.1f}m"
                ),
            )

        if completed_epochs >= args.min_epochs and epochs_without_improvement >= args.early_stop_patience:
            stop_reason = (
                f'early_stopping_triggered: no val acc improvement for '
                f'{epochs_without_improvement} epochs (patience={args.early_stop_patience})'
            )
            print(f'Dừng sớm: {stop_reason}')
            send_discord_webhook(
                args.discord_webhook_url,
                (
                    f"[Train Early Stop] model={args.model} | run_id={run_id} | "
                    f"epoch={completed_epochs}/{max_epochs} | best_val_acc={best_val_acc:.4f}"
                ),
            )
            break

    final_elapsed_sec = accumulated_elapsed_sec + (time.time() - session_start_time)

    write_metrics_csv(args.metrics_csv, metrics_log_rows)

    print(f'Đã lưu metrics CSV: {args.metrics_csv}')
    print(f'Đã lưu batch metrics CSV: {args.batch_metrics_csv}')
    print(f'Đã lưu confusion matrix CSV trong thư mục: {args.confusion_dir}')

    print('Huấn luyện hoàn tất!')

    save_model(args.model_out, model)
    print(f'Đã lưu mô hình cuối cùng: {args.model_out}')

    final_report = build_report(
        status='completed',
        started_at_iso=started_at_iso,
        args=args,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        completed_epochs=completed_epochs,
        max_epochs=max_epochs,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        final_elapsed_sec=final_elapsed_sec,
        epoch_summaries=epoch_summaries,
        stop_reason=stop_reason,
        run_id=run_id,
    )
    write_json_report(args.report_path, final_report)
    print(f'Đã lưu báo cáo train: {args.report_path}')

    send_discord_webhook(
        args.discord_webhook_url,
        (
            f"[Train Done] model={args.model} | run_id={run_id} | completed_epochs={completed_epochs} | "
            f"best_epoch={best_epoch} | best_val_acc={best_val_acc:.4f} | total_time={final_elapsed_sec / 60.0:.1f}m"
        ),
    )

    final_checkpoint = {
        'run_id': run_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'next_epoch': completed_epochs,
        'next_phase': 'train',
        'next_batch_idx': 0,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'metrics_log_rows': metrics_log_rows,
        'epoch_summaries': epoch_summaries,
        'accumulated_elapsed_sec': float(final_elapsed_sec),
        'completed_epochs': completed_epochs,
        'epochs_without_improvement': epochs_without_improvement,
        'global_step': global_step,
        'backbone_unfrozen': backbone_unfrozen,
        'train_config': vars(args),
        'class_names': class_names,
    }
    save_checkpoint(args.checkpoint_path, final_checkpoint)
