"""
Helper script to install GroundingDINO manually.
This works around the setup.py issue with torch installation.
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil

def run_command(cmd, cwd=None):
    """Run a command and print output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("GroundingDINO Installation Helper")
    print("=" * 60)
    print()
    
    # Check if torch is installed
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} is already installed")
    except ImportError:
        print("✗ PyTorch not found. Please install it first:")
        print("  pip install torch torchvision")
        sys.exit(1)
    
    # Clone GroundingDINO if not exists
    gd_dir = Path("GroundingDINO")
    
    if gd_dir.exists():
        print(f"✓ GroundingDINO directory already exists: {gd_dir}")
        response = input("Remove and re-clone? (y/n): ").strip().lower()
        if response == 'y':
            print(f"Removing {gd_dir}...")
            shutil.rmtree(gd_dir)
        else:
            print("Using existing directory...")
    
    if not gd_dir.exists():
        print("\nCloning GroundingDINO repository...")
        if not run_command(["git", "clone", "https://github.com/IDEA-Research/GroundingDINO.git"]):
            print("✗ Failed to clone repository")
            sys.exit(1)
        print("✓ Repository cloned successfully")
    
    # Install dependencies first
    print("\nInstalling GroundingDINO dependencies...")
    requirements_file = gd_dir / "requirements.txt"
    
    if requirements_file.exists():
        if not run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]):
            print("⚠ Some dependencies may have failed, but continuing...")
    else:
        print("⚠ requirements.txt not found, installing common dependencies...")
        deps = [
            "transformers",
            "opencv-python",
            "pillow",
            "numpy",
            "supervision"
        ]
        for dep in deps:
            run_command([sys.executable, "-m", "pip", "install", dep])
    
    # Install GroundingDINO in editable mode
    print("\nInstalling GroundingDINO package...")
    if not run_command([sys.executable, "-m", "pip", "install", "-e", "."], cwd=gd_dir):
        print("\n⚠ Installation had issues. Trying alternative method...")
        
        # Alternative: manually add to path
        print("\nTrying to add GroundingDINO to Python path...")
        import site
        site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
        
        if site_packages:
            # Create a .pth file
            pth_file = Path(site_packages) / "groundingdino.pth"
            try:
                with open(pth_file, 'w') as f:
                    f.write(str(gd_dir.absolute()))
                print(f"✓ Created .pth file: {pth_file}")
            except Exception as e:
                print(f"✗ Could not create .pth file: {e}")
    
    # Test import
    print("\nTesting import...")
    sys.path.insert(0, str(gd_dir.absolute()))
    try:
        from groundingdino.util.inference import load_model
        print("✓ GroundingDINO imported successfully!")
        print("\nInstallation complete!")
    except ImportError as e:
        print(f"⚠ Import test failed: {e}")
        print("\nYou may need to add GroundingDINO to your Python path manually:")
        print(f"  Add this to your script: sys.path.insert(0, r'{gd_dir.absolute()}')")
        print("\nOr set PYTHONPATH environment variable:")
        print(f"  set PYTHONPATH={gd_dir.absolute()};%PYTHONPATH%")

if __name__ == "__main__":
    main()

