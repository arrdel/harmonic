#!/usr/bin/env python3
"""
Create COCO training subset for HARMONIC.
This creates captions.json from annotations while images are still downloading.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def create_captions_json():
    """Create captions.json from COCO annotations."""
    
    # Paths
    annotations_file = Path("/media/scratch/adele/harmonic/dataset/annotations/captions_train2017.json")
    output_dir = Path("/media/scratch/adele/harmonic/dataset")
    output_file = output_dir / "captions.json"
    
    print("="*70)
    print("Creating captions.json for HARMONIC Training")
    print("="*70)
    
    # Load annotations
    print("\n[1/2] Loading COCO annotations...")
    if not annotations_file.exists():
        print(f"✗ Error: {annotations_file} not found")
        print("  Make sure annotations are extracted")
        return False
    
    with open(annotations_file) as f:
        coco = json.load(f)
    
    print(f"  ✓ Loaded {len(coco['images'])} images")
    print(f"  ✓ Loaded {len(coco['annotations'])} captions")
    
    # Create mapping
    print("\n[2/2] Creating captions mapping...")
    
    # Map image_id to filename
    images = {img['id']: img['file_name'] for img in coco['images']}
    
    # Each image has ~5 captions, we'll take the first one per image
    captions = {}
    for ann in coco['annotations']:
        img_name = images[ann['image_id']]
        if img_name not in captions:
            captions[img_name] = ann['caption']
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    print(f"  ✓ Created captions for {len(captions)} images")
    print(f"  ✓ Saved to: {output_file}")
    
    print("\n" + "="*70)
    print("✅ Setup Complete!")
    print("="*70)
    print(f"\nTotal images with captions: {len(captions):,}")
    print(f"\nOnce train2017.zip finishes downloading and extracting,")
    print(f"you can start training with:")
    print(f"\n  cd /home/adelechinda/home/semester_projects/spring_26/computer_vision/HARMONIC")
    print(f"  bash train_multi_gpu.sh")
    print("\n" + "="*70)
    
    return True

if __name__ == "__main__":
    create_captions_json()
