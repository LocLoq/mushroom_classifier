import os
from pathlib import Path
from collections import defaultdict

def verify_dataset():
    """Verify the reorganized Vietnamese mushroom dataset"""
    
    dataset_path = Path(r"c:\Users\loql\Desktop\train-predict\mushroom_dataset_vi")
    
    splits = ["train", "val", "test"]
    
    print("\n" + "="*70)
    print("VIETNAMESE MUSHROOM DATASET VERIFICATION")
    print("="*70 + "\n")
    
    # Count mushroom types per split
    mushroom_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    total_images = {"train": 0, "val": 0, "test": 0}
    
    for split in splits:
        split_path = dataset_path / split
        mushroom_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        
        for mushroom_dir in mushroom_dirs:
            mushroom_name = mushroom_dir.name
            image_files = list(mushroom_dir.glob("*.png")) + list(mushroom_dir.glob("*.jpg"))
            count = len(image_files)
            
            mushroom_counts[mushroom_name][split] = count
            total_images[split] += count
    
    # Print summary by split
    for split in splits:
        mushroom_types = len([k for k, v in mushroom_counts.items() if v[split] > 0])
        print(f"{split.upper():5} folder:")
        print(f"  - Mushroom types: {mushroom_types}")
        print(f"  - Total images: {total_images[split]}")
    
    # Calculate and display split percentages
    grand_total = sum(total_images.values())
    print(f"\nTotal images: {grand_total}")
    print(f"\nSplit distribution:")
    for split in splits:
        pct = (total_images[split] / grand_total) * 100
        print(f"  {split:5}: {total_images[split]:6} images ({pct:5.1f}%)")
    
    # Find a mushroom to show detailed split
    sample_mushrooms = [k for k, v in mushroom_counts.items() if v["train"] > 50 and "nấm" in k]
    
    if sample_mushrooms:
        sample = sample_mushrooms[0]
        counts = mushroom_counts[sample]
        total = sum(counts.values())
        
        print(f"\nSample mushroom: {sample}")
        print(f"  Train: {counts['train']} ({(counts['train']/total)*100:.1f}%)")
        print(f"  Val:   {counts['val']} ({(counts['val']/total)*100:.1f}%)")
        print(f"  Test:  {counts['test']} ({(counts['test']/total)*100:.1f}%)")
        print(f"  Total: {total}")
    
    # List sample Vietnamese mushroom names
    vi_mushrooms = sorted([k for k in mushroom_counts.keys() if k.startswith("nấm_")])[:10]
    
    print(f"\nSample Vietnamese mushroom names:")
    for mushroom in vi_mushrooms:
        print(f"  - {mushroom}")
    
    print("\n✓ Dataset organization complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    verify_dataset()
