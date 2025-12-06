# Quick Start Guide

Get up and running with the Product Vision Pipeline in 5 minutes!

## 1. Install Dependencies (2 minutes)

```bash
# Install base packages
pip install -r requirements.txt

# Install GroundingDINO
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

## 2. Download Models (2 minutes)

```bash
python download_models.py
```

Choose the SAM model size:
- `vit_b` - Fastest, smallest (recommended for testing)
- `vit_l` - Balanced
- `vit_h` - Best quality, largest

## 3. Test Installation (30 seconds)

```bash
python test_pipeline.py
```

You should see:
```
✓ PyTorch X.X.X
✓ OpenCV X.X.X
✓ EasyOCR
✓ CLIP
✓ GroundingDINO
✓ SAM (Segment Anything)
✓ Pipeline is ready to use!
```

## 4. Run Your First Analysis (1 minute)

Create a simple script `my_first_analysis.py`:

```python
from product_vision_pipeline import ProductVisionPipeline
import cv2

# Initialize
pipeline = ProductVisionPipeline(
    grounding_dino_config_path="models/GroundingDINO_SwinB.cfg.py",
    grounding_dino_checkpoint_path="models/groundingdino_swinb_cogcoor.pth",
    sam_checkpoint_path="models/sam_vit_b.pth",
    sam_model_type="vit_b"
)

# Analyze shelf image
results = pipeline.process_shelf(
    image_path="your_shelf_image.jpg",  # Replace with your image
    product_query="toothpaste tube",
    expected_brands=["Colgate", "Crest"],
    return_visualization=True
)

# Print results
print(f"Found {results['num_products']} products")
print(f"Total units: {results['total_units']}")

# Save visualization
cv2.imwrite("results.jpg", results['visualization'])
print("Results saved to results.jpg")
```

Run it:
```bash
python my_first_analysis.py
```

## That's It! 🎉

You now have a working product vision pipeline. Try different queries:

- `"shampoo bottle"` - Detect shampoo products
- `"soda can"` - Detect drink cans
- `"cereal box"` - Detect cereal boxes
- `"detergent bottle"` - Detect cleaning products

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example_usage.py](example_usage.py) for more examples
- See [setup_instructions.md](setup_instructions.md) for troubleshooting

## Common Issues

**"No module named 'groundingdino'"**
→ Run: `pip install git+https://github.com/IDEA-Research/GroundingDINO.git`

**"CUDA out of memory"**
→ Use smaller SAM model (`vit_b`) or process smaller images

**"Model files not found"**
→ Run: `python download_models.py`

## Need Help?

Check the troubleshooting section in [setup_instructions.md](setup_instructions.md) or open an issue on GitHub.

