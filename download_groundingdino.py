"""
Download only GroundingDINO models (you already have SAM).
"""

import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Downloading GroundingDINO Models")
print("=" * 60)
print()

# Copy config from cloned repo (or download if not available)
config_source = Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py")
config_path = MODELS_DIR / "GroundingDINO_SwinB_cfg.py"

if config_path.exists():
    print(f"✓ Config already exists: {config_path}")
elif config_source.exists():
    print(f"Copying config from cloned repository...")
    import shutil
    shutil.copy(config_source, config_path)
    print(f"✓ Config copied: {config_path}")
else:
    # Try downloading
    config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    print(f"Downloading GroundingDINO config...")
    try:
        urllib.request.urlretrieve(config_url, config_path)
        print(f"✓ Config downloaded: {config_path}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        print(f"  Please copy manually from: {config_source}")

# Download checkpoint
checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swinb_cogcoor.pth"
checkpoint_path = MODELS_DIR / "groundingdino_swinb_cogcoor.pth"

if checkpoint_path.exists():
    print(f"✓ Checkpoint already exists: {checkpoint_path}")
else:
    print(f"\nDownloading GroundingDINO checkpoint (~1.2GB, this may take a few minutes)...")
    try:
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        print(f"✓ Checkpoint downloaded: {checkpoint_path}")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\n" + "=" * 60)
print("Done! You now have all required models:")
print(f"  ✓ SAM: models/sam_vit_b.pth")
if config_path.exists():
    print(f"  ✓ GroundingDINO config: {config_path}")
else:
    print(f"  ✗ GroundingDINO config: MISSING")
if checkpoint_path.exists():
    print(f"  ✓ GroundingDINO checkpoint: {checkpoint_path}")
else:
    print(f"  ✗ GroundingDINO checkpoint: MISSING")
    print(f"\n  Download manually from:")
    print(f"  https://github.com/IDEA-Research/GroundingDINO/releases")
print("=" * 60)

