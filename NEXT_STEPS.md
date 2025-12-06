# Next Steps - Quick Guide

## ✅ What You Have
- ✓ SAM model: `models/sam_vit_b.pth` (you only need ONE SAM model, this is perfect!)
- ✓ GroundingDINO config: `models/GroundingDINO_SwinB_cfg.py`

## 📥 What You Need to Download

You still need **1 file**:

**Checkpoint file** (~1.2GB): `groundingdino_swinb_cogcoor.pth`

## 🚀 Download the Checkpoint

### Option 1: From GitHub Releases (Recommended)
1. Go to: https://github.com/IDEA-Research/GroundingDINO/releases
2. Find the latest release
3. Download: `groundingdino_swinb_cogcoor.pth` (or similar name)
4. Save to: `models/groundingdino_swinb_cogcoor.pth`

### Option 2: Direct Download (if URL works)
The checkpoint URL may vary. Check the releases page for the correct download link.

## ✅ After Download

Your `models/` folder should have:
```
models/
  ├── sam_vit_b.pth                          ✓ (you have this)
  ├── GroundingDINO_SwinB_cfg.py            ✓ (you have this)
  └── groundingdino_swinb_cogcoor.pth        (need to download ~1.2GB)
```

## 🎯 Then You're Ready!

Once all files are downloaded, you can use the pipeline:

```python
from product_vision_pipeline import ProductVisionPipeline
import cv2

pipeline = ProductVisionPipeline(
    grounding_dino_config_path="models/GroundingDINO_SwinB_cfg.py",
    grounding_dino_checkpoint_path="models/groundingdino_swinb_cogcoor.pth",
    sam_checkpoint_path="models/sam_vit_b.pth",
    sam_model_type="vit_b"
)

results = pipeline.process_shelf(
    image_path="your_image.jpg",
    product_query="toothpaste tube",
    expected_brands=["Colgate", "Crest"]
)
```

---

**Note**: You do NOT need to download all three SAM models (vit_h, vit_l, vit_b). You only need ONE, and `sam_vit_b.pth` is perfect for most use cases!

