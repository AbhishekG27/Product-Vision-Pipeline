"""
Simple test script to verify pipeline installation and basic functionality.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found")
        return False
    
    try:
        import easyocr
        print("✓ EasyOCR")
    except ImportError:
        print("✗ EasyOCR not found")
        return False
    
    try:
        import clip
        print("✓ CLIP")
    except ImportError:
        print("✗ CLIP not found")
        return False
    
    try:
        from groundingdino.util.inference import load_model
        print("✓ GroundingDINO")
    except ImportError:
        print("⚠ GroundingDINO not found (install: pip install git+https://github.com/IDEA-Research/GroundingDINO.git)")
    
    try:
        from segment_anything import sam_model_registry
        print("✓ SAM (Segment Anything)")
    except ImportError:
        print("⚠ SAM not found (install: pip install git+https://github.com/facebookresearch/segment-anything.git)")
    
    return True

def test_pipeline_init():
    """Test if pipeline can be initialized."""
    print("\nTesting pipeline initialization...")
    
    try:
        from product_vision_pipeline import ProductVisionPipeline
        
        # Try to initialize without models (should work but models won't be loaded)
        pipeline = ProductVisionPipeline()
        print("✓ Pipeline class can be instantiated")
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return False

def test_model_paths():
    """Check if model files exist."""
    print("\nChecking model files...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("⚠ models/ directory does not exist")
        print("  Run: python download_models.py")
        return False
    
    required_files = [
        "GroundingDINO_SwinB_cfg.py",  # Note: underscore, not dot
        "groundingdino_swinb_cogcoor.pth"
    ]
    
    sam_files = [
        "sam_vit_b.pth",
        "sam_vit_l.pth",
        "sam_vit_h.pth"
    ]
    
    all_found = True
    
    for file in required_files:
        path = models_dir / file
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {file} not found")
            all_found = False
    
    sam_found = False
    for file in sam_files:
        path = models_dir / file
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.1f} MB)")
            sam_found = True
    
    if not sam_found:
        print("⚠ No SAM checkpoint found")
        print("  Run: python download_models.py")
        all_found = False
    
    return all_found

def main():
    print("=" * 60)
    print("Product Vision Pipeline - Installation Test")
    print("=" * 60)
    print()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test pipeline
    pipeline_ok = test_pipeline_init()
    
    # Test model paths
    models_ok = test_model_paths()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if imports_ok and pipeline_ok:
        print("✓ Basic installation is working")
    else:
        print("✗ Installation issues detected")
        print("\nNext steps:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Install GroundingDINO: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        print("3. Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")
        sys.exit(1)
    
    if models_ok:
        print("✓ Model files are present")
        print("\n✓ Pipeline is ready to use!")
    else:
        print("⚠ Model files missing")
        print("\nNext steps:")
        print("1. Run: python download_models.py")
        print("2. Or download models manually (see README.md)")
    
    print()

if __name__ == "__main__":
    main()

