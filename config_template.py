"""
Configuration template for Product Vision Pipeline.
Copy this file to config.py and update with your paths.
"""

# GroundingDINO Configuration
GROUNDING_DINO_CONFIG = "models/GroundingDINO_SwinB.cfg.py"
GROUNDING_DINO_CHECKPOINT = "models/groundingdino_swinb_cogcoor.pth"

# SAM Configuration
SAM_CHECKPOINT = "models/sam_vit_b.pth"  # Options: sam_vit_b.pth, sam_vit_l.pth, sam_vit_h.pth
SAM_MODEL_TYPE = "vit_b"  # Options: "vit_b", "vit_l", "vit_h"

# Device Configuration
DEVICE = "cuda"  # Options: "cuda" or "cpu" (auto-detected if None)

# OCR Configuration
OCR_LANGUAGES = ['en']  # Add more: ['en', 'es', 'fr', 'de', etc.]

# Detection Thresholds
BOX_THRESHOLD = 0.3  # Confidence threshold for bounding boxes (0.0-1.0)
TEXT_THRESHOLD = 0.25  # Confidence threshold for text matching (0.0-1.0)

# Placement Analysis Thresholds
ALIGNMENT_TOLERANCE = 0.1  # Fraction of image dimension for alignment check
SPACING_TOLERANCE = 0.15  # Fraction of average box size for spacing check

# Stack Detection
STACK_OVERLAP_THRESHOLD = 0.3  # Minimum overlap ratio to consider as stacked

