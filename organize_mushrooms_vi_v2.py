import random
import shutil
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

RAW_ROOT = Path(r"c:\Users\loql\Desktop\train-predict\raw\Classes")
OUT_ROOT = Path(r"c:\Users\loql\Desktop\train-predict\mushroom_dataset_vi")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RANDOM_SEED = 42


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def slugify_ascii(text: str) -> str:
    text = strip_accents(text)
    text = text.lower().strip()
    text = text.replace("&", " va ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def to_vietnamese_class_name(raw_class_name: str) -> str:
    # Ten lop theo form ten_nam (ASCII): nam_<ten_goc_duoc_chuan_hoa>
    return f"nam_{slugify_ascii(raw_class_name)}"


def list_source_class_dirs(root: Path):
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            yield category_dir.name, class_dir


def list_images(class_dir: Path):
    return [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def ensure_clean_output(out_root: Path):
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val", "test"):
        (out_root / split).mkdir(parents=True, exist_ok=True)


def split_files(files):
    total = len(files)
    train_n = int(total * TRAIN_RATIO)
    val_n = int(total * VAL_RATIO)
    test_n = total - train_n - val_n

    train_files = files[:train_n]
    val_files = files[train_n:train_n + val_n]
    test_files = files[train_n + val_n:]

    # Sanity check
    assert len(train_files) + len(val_files) + len(test_files) == total
    return train_files, val_files, test_files


def copy_split(files, split_name, class_name, counters, stats):
    class_out = OUT_ROOT / split_name / class_name
    class_out.mkdir(parents=True, exist_ok=True)

    for src in files:
        counters[(split_name, class_name)] += 1
        idx = counters[(split_name, class_name)]
        ext = src.suffix.lower()
        dst = class_out / f"{class_name}_{idx:06d}{ext}"
        shutil.copy2(src, dst)
        stats[split_name] += 1


def main():
    random.seed(RANDOM_SEED)
    ensure_clean_output(OUT_ROOT)

    per_split_image_count = defaultdict(int)
    per_split_class_count = defaultdict(set)
    class_image_counters = defaultdict(int)

    total_source_images = 0
    class_dir_count = 0

    for category_name, class_dir in list_source_class_dirs(RAW_ROOT):
        class_dir_count += 1
        raw_name = class_dir.name
        vi_name = to_vietnamese_class_name(raw_name)

        images = list_images(class_dir)
        if not images:
            continue

        random.shuffle(images)
        total_source_images += len(images)

        train_files, val_files, test_files = split_files(images)

        copy_split(train_files, "train", vi_name, class_image_counters, per_split_image_count)
        copy_split(val_files, "val", vi_name, class_image_counters, per_split_image_count)
        copy_split(test_files, "test", vi_name, class_image_counters, per_split_image_count)

        if train_files:
            per_split_class_count["train"].add(vi_name)
        if val_files:
            per_split_class_count["val"].add(vi_name)
        if test_files:
            per_split_class_count["test"].add(vi_name)

    total_out = sum(per_split_image_count.values())

    print("=== DONE ===")
    print(f"Source class dirs scanned: {class_dir_count}")
    print(f"Source images found: {total_source_images}")
    print(f"Output images written: {total_out}")

    for split in ("train", "val", "test"):
        img_n = per_split_image_count[split]
        cls_n = len(per_split_class_count[split])
        pct = (img_n / total_out * 100.0) if total_out else 0.0
        print(f"{split:5}: images={img_n}, classes={cls_n}, ratio={pct:.2f}%")

    if total_out != total_source_images:
        print("WARNING: output image count does not match source image count!")


if __name__ == "__main__":
    main()
