# 🚀 Deploy to Render

## Quick Deployment Steps

### Step 1: Create Web Service (NOT Static Site)

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository: `AbhishekG27/Product-Vision-Pipeline`
4. Select branch: `main`

### Step 2: Configure Settings

**Name:** `product-vision-pipeline`

**Environment:** `Python 3`

**Build Command:**
```bash
pip install -r requirements.txt && pip install git+https://github.com/IDEA-Research/GroundingDINO.git && pip install git+https://github.com/facebookresearch/segment-anything.git && pip install git+https://github.com/openai/CLIP.git
```

**Start Command:**
```bash
python app.py
```

**Plan:** Choose **Starter** (free) or **Standard** (for better performance)

### Step 3: Set Environment Variables

In Render Dashboard → Environment Variables, add:

```
PYTHON_VERSION=3.10
PORT=7860
```

### Step 4: Deploy

Click **"Create Web Service"** and wait for build to complete.

### Step 5: Download Models (After First Deploy)

1. Go to Render Dashboard → Your Service → **Shell**
2. Run these commands:

```bash
cd /opt/render/project/src
python download_models.py
```

**Note:** Models are large (~2-3GB total). This may take 10-15 minutes.

### Step 6: Restart Service

After models download, restart the service from Render Dashboard.

---

## Alternative: Using render.yaml

If you have `render.yaml` in your repo:

1. Push `render.yaml` to GitHub
2. In Render Dashboard → **"New"** → **"Blueprint"**
3. Connect repository
4. Render will auto-detect `render.yaml` and configure everything

---

## Important Notes

- **First deploy will fail** until models are downloaded (this is normal)
- Models need to be downloaded via Shell after first deploy
- Free tier has **512MB RAM** - may need to use smaller SAM model (`vit_b`)
- Service will auto-sleep after 15 min inactivity (free tier)
- First request after sleep takes ~30 seconds to wake up

---

## Troubleshooting

### Build Fails

**Error:** "No module named 'groundingdino'"
- **Fix:** Ensure build command includes all pip installs

### Service Crashes

**Error:** "Out of memory"
- **Fix:** Use smaller SAM model (`vit_b` instead of `vit_h`)

### Models Not Found

**Error:** "File not found: models/..."
- **Fix:** Run `python download_models.py` in Shell

### Port Already in Use

**Error:** "Address already in use"
- **Fix:** Use `PORT` environment variable (already set in render.yaml)

---

## Your App URL

After deployment, your app will be available at:
```
https://product-vision-pipeline.onrender.com
```

(Replace `product-vision-pipeline` with your service name)

---

## Cost

- **Free Tier:** 750 hours/month, auto-sleep after 15 min
- **Starter:** $7/month - always on, 512MB RAM
- **Standard:** $25/month - always on, 2GB RAM (recommended for models)

---

## Next Steps

1. Deploy service
2. Download models via Shell
3. Test your app at the provided URL
4. Share the link with others!

