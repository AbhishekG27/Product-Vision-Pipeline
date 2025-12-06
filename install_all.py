"""
Complete installation script for all dependencies.
This handles the tricky installations that may fail with direct pip install.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, capture_output=False)
    if check and result.returncode != 0:
        print(f"\n✗ Command failed with exit code {result.returncode}")
        return False
    return True

def main():
    print("=" * 60)
    print("Product Vision Pipeline - Complete Installation")
    print("=" * 60)
    print()
    
    # Step 1: Base requirements
    print("\n[1/5] Installing base requirements...")
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]):
        print("⚠ Some packages may have failed, but continuing...")
    
    # Step 2: Install CLIP (usually works fine)
    print("\n[2/5] Installing CLIP...")
    run_command([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"], check=False)
    
    # Step 3: Install SAM (usually works fine)
    print("\n[3/5] Installing SAM (Segment Anything)...")
    run_command([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"], check=False)
    
    # Step 4: Install GroundingDINO (tricky one)
    print("\n[4/5] Installing GroundingDINO...")
    print("This may take a few minutes and might show warnings...")
    
    # Try direct install first
    if not run_command([sys.executable, "-m", "pip", "install", "git+https://github.com/IDEA-Research/GroundingDINO.git"], check=False):
        print("\n⚠ Direct install failed. Trying manual installation...")
        print("Run this separately: python install_groundingdino.py")
    
    # Step 5: Download models
    print("\n[5/5] Downloading model checkpoints...")
    print("You can run this separately: python download_models.py")
    
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    print("\nNext steps:")
    print("1. If GroundingDINO failed, run: python install_groundingdino.py")
    print("2. Download models: python download_models.py")
    print("3. Test installation: python test_pipeline.py")
    print()

if __name__ == "__main__":
    main()

