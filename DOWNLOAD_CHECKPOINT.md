# Download GroundingDINO Checkpoint

## ✅ What You Have
- ✓ SAM model: `models/sam_vit_b.pth`
- ✓ GroundingDINO config: `models/GroundingDINO_SwinB_cfg.py`

## 📥 What You Need

You need to download the **GroundingDINO checkpoint** (~1.2GB):

**File**: `groundingdino_swinb_cogcoor.pth`  
**Size**: ~1.2GB  
**Save to**: `models/groundingdino_swinb_cogcoor.pth`

## 🔗 Download Options

### Option 1: Direct Download (Recommended)
Go to the GroundingDINO releases page and download:
- **URL**: https://github.com/IDEA-Research/GroundingDINO/releases
- Look for: `groundingdino_swinb_cogcoor.pth` or similar
- Save to: `models/groundingdino_swinb_cogcoor.pth`

### Option 2: Using Hugging Face (if available)
Some models are also available on Hugging Face:
- Search for "GroundingDINO" on https://huggingface.co/models

### Option 3: Using wget/curl (if URL is available)
```bash
# If you find a direct download URL, use:
wget <URL> -O models/groundingdino_swinb_cogcoor.pth
# or
curl -L <URL> -o models/groundingdino_swinb_cogcoor.pth
```

## ✅ After Download

Your `models/` folder should have:
```
models/
  ├── sam_vit_b.pth                          ✓
  ├── GroundingDINO_SwinB_cfg.py             ✓
  └── groundingdino_swinb_cogcoor.pth         (download this)
```

## 🎯 Then Test

Once downloaded, test with:
```bash
python test_pipeline.py
```

You should see all checkmarks! ✓

