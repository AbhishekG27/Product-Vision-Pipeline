# Setup Instructions

## Step-by-Step Installation

### 1. Install Base Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install GroundingDINO

GroundingDINO needs to be installed from source:

```bash
# Option A: Direct pip install (recommended)
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Option B: Clone and install manually
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

### 3. Install SAM (Segment Anything)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 4. Install CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 5. Download Model Checkpoints

Run the download script:

```bash
python download_models.py
```

This will download:
- GroundingDINO config and checkpoint
- SAM checkpoint (you'll choose the size)

### 6. Verify Installation

Test the installation:

```bash
python -c "from product_vision_pipeline import ProductVisionPipeline; print('Installation successful!')"
```

## Alternative: Using Conda

If you prefer conda:

```bash
conda create -n product_vision python=3.10
conda activate product_vision
pip install -r requirements.txt
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git
```

## Troubleshooting

### Issue: "No module named 'groundingdino'"

**Solution 1**: Try the helper script (recommended):
```bash
python install_groundingdino.py
```

**Solution 2**: Manual installation:
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

**Solution 3**: Direct pip install (may fail on Windows):
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### Issue: GroundingDINO installation fails with "No module named pip" or torch errors

**Solution**: This is a known issue with GroundingDINO's setup.py. Use the helper script:
```bash
python install_groundingdino.py
```

This script clones the repo and installs it properly, working around the setup.py issues.

### Issue: "No module named 'segment_anything'"

**Solution**: Install SAM:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Issue: CUDA out of memory

**Solutions**:
1. Use smaller SAM model (vit_b instead of vit_h)
2. Process images at lower resolution
3. Use CPU mode (set device='cpu' in pipeline initialization)

### Issue: Model download fails

**Solution**: Download manually:
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO/releases
- SAM: https://github.com/facebookresearch/segment-anything#model-checkpoints

### Issue: Import errors with GroundingDINO

**Solution**: Make sure you're using the correct import path. The pipeline uses:
```python
from groundingdino.util.inference import load_model, load_image, predict, annotate
```

If this doesn't work, you may need to check the GroundingDINO repository structure and adjust the imports in `product_vision_pipeline.py`.

## System Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended, but CPU works)
- **RAM**: At least 8GB (16GB recommended)
- **Disk Space**: ~5-10GB for models and dependencies

## Quick Test

After installation, test with a simple script:

```python
from product_vision_pipeline import ProductVisionPipeline
import cv2
import numpy as np

# Create a dummy image
test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.imwrite("test_image.jpg", test_image)

# Initialize pipeline (models may not be loaded, but should not crash)
try:
    pipeline = ProductVisionPipeline()
    print("✓ Pipeline initialized successfully")
except Exception as e:
    print(f"✗ Error: {e}")
```

