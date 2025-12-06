"""
Helper script to download required model checkpoints.
Run this before using the pipeline for the first time.
"""

import os
import urllib.request
from pathlib import Path

# Model URLs and paths
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_URLS = {
    "grounding_dino_swinb": {
        "config": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB.cfg.py",
        "checkpoint": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swinb_cogcoor.pth"
    },
    "sam_vit_h": {
        "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "sam_vit_l": {
        "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "sam_vit_b": {
        "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
}

def download_file(url: str, filepath: Path, description: str):
    """Download a file with progress indication."""
    if filepath.exists():
        print(f"✓ {description} already exists: {filepath}")
        return
    
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Saving to: {filepath}")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Successfully downloaded {description}")
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        print(f"  Please download manually from: {url}")

def main():
    print("=" * 60)
    print("Model Downloader for Product Vision Pipeline")
    print("=" * 60)
    print()
    
    # Download GroundingDINO
    print("\n1. GroundingDINO Model:")
    gd_config_path = MODELS_DIR / "GroundingDINO_SwinB.cfg.py"
    gd_checkpoint_path = MODELS_DIR / "groundingdino_swinb_cogcoor.pth"
    
    download_file(
        MODEL_URLS["grounding_dino_swinb"]["config"],
        gd_config_path,
        "GroundingDINO config"
    )
    download_file(
        MODEL_URLS["grounding_dino_swinb"]["checkpoint"],
        gd_checkpoint_path,
        "GroundingDINO checkpoint"
    )
    
    # Download SAM (recommend vit_h for best quality, vit_b for speed)
    print("\n2. SAM Model (Segment Anything):")
    print("   Choose model size:")
    print("   - vit_h: Best quality, largest (~2.4GB)")
    print("   - vit_l: Good balance (~1.2GB)")
    print("   - vit_b: Fastest, smallest (~375MB)")
    
    sam_choice = input("   Enter choice (vit_h/vit_l/vit_b) [default: vit_b]: ").strip().lower()
    if not sam_choice:
        sam_choice = "vit_b"
    
    if sam_choice not in ["vit_h", "vit_l", "vit_b"]:
        print(f"   Invalid choice, using vit_b")
        sam_choice = "vit_b"
    
    sam_checkpoint_path = MODELS_DIR / f"sam_{sam_choice}.pth"
    download_file(
        MODEL_URLS[f"sam_{sam_choice}"]["checkpoint"],
        sam_checkpoint_path,
        f"SAM {sam_choice} checkpoint"
    )
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nModel paths:")
    print(f"  GroundingDINO config: {gd_config_path}")
    print(f"  GroundingDINO checkpoint: {gd_checkpoint_path}")
    print(f"  SAM checkpoint: {sam_checkpoint_path}")
    print("\nNote: CLIP and EasyOCR models are downloaded automatically on first use.")

if __name__ == "__main__":
    main()

