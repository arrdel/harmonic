#!/usr/bin/env python3
"""
Prepare COCO dataset for HARMONIC training.
Converts COCO annotations to captions.json format.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def prepare_coco_dataset(dataset_path):
    """
    Prepare COCO dataset for HARMONIC.
    
    Args:
        dataset_path: Path to the dataset directory
    """
    dataset_path = Path(dataset_path)
    
    print("="*60)
    print("COCO Dataset Preparation for HARMONIC")
    print("="*60)
    print(f"Dataset path: {dataset_path}")
    
    # 1. Extract annotations
    annotations_zip = dataset_path / "annotations_trainval2017.zip"
    if annotations_zip.exists():
        print(f"\n[1/4] Extracting annotations...")
        import zipfile
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print("✓ Annotations extracted")
    else:
        print(f"\n[1/4] Annotations already extracted or missing")
    
    # 2. Extract images (if not already done)
    train_zip = dataset_path / "train2017.zip"
    train_dir = dataset_path / "train2017"
    
    if not train_dir.exists() and train_zip.exists():
        print(f"\n[2/4] Extracting training images (this takes a while)...")
        import zipfile
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print("✓ Images extracted")
    else:
        print(f"\n[2/4] Images already extracted or still downloading")
    
    # 3. Load COCO annotations
    print(f"\n[3/4] Processing COCO annotations...")
    captions_file = dataset_path / "annotations" / "captions_train2017.json"
    
    if not captions_file.exists():
        print(f"✗ Error: {captions_file} not found")
        return False
    
    with open(captions_file) as f:
        coco = json.load(f)
    
    print(f"  Images: {len(coco['images'])}")
    print(f"  Captions: {len(coco['annotations'])}")
    
    # 4. Create captions mapping
    print(f"\n[4/4] Creating captions.json...")
    
    # Map image_id to filename
    images = {img['id']: img['file_name'] for img in coco['images']}
    
    # Group captions by image (each image has 5 captions, we'll use the first)
    captions = {}
    caption_counts = defaultdict(int)
    
    for ann in coco['annotations']:
        img_name = images[ann['image_id']]
        if img_name not in captions:
            captions[img_name] = ann['caption']
            caption_counts[img_name] += 1
    
    # Save captions.json in train2017 directory
    output_file = train_dir / "captions.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    print(f"✓ Created captions for {len(captions)} images")
    print(f"✓ Saved to: {output_file}")
    
    # 5. Summary
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print(f"Dataset location: {train_dir}")
    print(f"Total images: {len(captions)}")
    print(f"Captions file: {output_file}")
    print("\nYou can now train with:")
    print(f"  python train_harmonic.py --train_data {train_dir}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "/media/scratch/adele/harmonic/dataset"
    
    prepare_coco_dataset(dataset_path)
