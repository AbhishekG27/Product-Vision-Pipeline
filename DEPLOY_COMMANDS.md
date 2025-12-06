# 🚀 Render Deployment - Commands Only

## Render Dashboard Setup

### 1. Create Web Service
- Dashboard → New → **Web Service** (NOT Static Site)
- Connect: `AbhishekG27/Product-Vision-Pipeline`
- Branch: `main`

### 2. Build Command
```bash
pip install -r requirements.txt && pip install git+https://github.com/IDEA-Research/GroundingDINO.git && pip install git+https://github.com/facebookresearch/segment-anything.git && pip install git+https://github.com/openai/CLIP.git
```

### 3. Start Command
```bash
python app.py
```

### 4. Environment Variables
```
PYTHON_VERSION=3.10
PORT=7860
```

---

## After First Deploy - Download Models

### Open Render Shell
- Dashboard → Your Service → **Shell** tab

### Run These Commands
```bash
cd /opt/render/project/src
python download_models.py
```

### Restart Service
- Dashboard → Your Service → **Manual Deploy** → **Clear build cache & deploy**

---

## Quick Reference

**Service Type:** Web Service  
**Build Command:** See above  
**Start Command:** `python app.py`  
**Environment:** Python 3.10  
**Port:** 7860 (auto-set via env var)

---

## Your App URL
```
https://your-service-name.onrender.com
```

