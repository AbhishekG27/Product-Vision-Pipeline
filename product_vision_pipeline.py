"""
Product Vision Pipeline: Detects products on shelves, reads brand names,
analyzes placement, and counts products/units using pretrained models.

Components:
- GroundingDINO: Text-driven object detection
- SAM: Instance segmentation
- EasyOCR: Text recognition
- CLIP: Brand/logo verification
"""

import cv2
import numpy as np
import torch
from PIL import Image
import easyocr
import clip
from typing import List, Dict, Tuple, Optional, Any
import warnings
import sys
import os
import tempfile
from pathlib import Path
warnings.filterwarnings('ignore')

# Try to import GroundingDINO, with fallback to local path
GROUNDING_DINO_AVAILABLE = False
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from groundingdino.util import box_ops
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    # Try adding local GroundingDINO directory to path
    local_gd_path = Path(__file__).parent / "GroundingDINO"
    if local_gd_path.exists():
        sys.path.insert(0, str(local_gd_path))
        try:
            from groundingdino.util.inference import load_model, load_image, predict, annotate
            from groundingdino.util import box_ops
            GROUNDING_DINO_AVAILABLE = True
            print("✓ GroundingDINO loaded from local directory")
        except ImportError:
            pass
    
    if not GROUNDING_DINO_AVAILABLE:
        print("Warning: GroundingDINO not available. Install with: python install_groundingdino.py")

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: SAM not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")


class ProductVisionPipeline:
    """
    Complete pipeline for product detection, brand reading, and placement analysis.
    """
    
    def __init__(
        self,
        grounding_dino_config_path: str = None,
        grounding_dino_checkpoint_path: str = None,
        sam_checkpoint_path: str = None,
        sam_model_type: str = "vit_h",
        device: str = None,
        ocr_languages: List[str] = ['en']
    ):
        """
        Initialize the pipeline with pretrained models.
        
        Args:
            grounding_dino_config_path: Path to GroundingDINO config file (e.g., "models/GroundingDINO_SwinB_cfg.py")
            grounding_dino_checkpoint_path: Path to GroundingDINO checkpoint
            sam_checkpoint_path: Path to SAM checkpoint
            sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            device: Device to use ('cuda' or 'cpu')
            ocr_languages: Languages for OCR (default: English)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize GroundingDINO
        self.grounding_dino_model = None
        if GROUNDING_DINO_AVAILABLE and grounding_dino_config_path and grounding_dino_checkpoint_path:
            try:
                self.grounding_dino_model = load_model(
                    grounding_dino_config_path,
                    grounding_dino_checkpoint_path
                )
                print("GroundingDINO loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load GroundingDINO: {e}")
        
        # Initialize SAM
        self.sam_predictor = None
        if SAM_AVAILABLE and sam_checkpoint_path:
            try:
                sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
                sam.to(device=self.device)
                self.sam_predictor = SamPredictor(sam)
                print("SAM loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load SAM: {e}")
        
        # Initialize EasyOCR
        try:
            self.ocr_reader = easyocr.Reader(ocr_languages, gpu=torch.cuda.is_available())
            print("EasyOCR loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load EasyOCR: {e}")
            self.ocr_reader = None
        
        # Initialize CLIP
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("CLIP loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CLIP: {e}")
            self.clip_model = None
    
    def detect_products(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Detect products using GroundingDINO based on text prompt.
        
        Args:
            image: Input image (numpy array, BGR format)
            text_prompt: Text description of products to detect (e.g., "Colgate toothpaste", "Pepsi can")
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
        
        Returns:
            Dictionary with detections containing boxes, scores, and labels
        """
        if self.grounding_dino_model is None:
            raise ValueError("GroundingDINO model not loaded")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # GroundingDINO's load_image expects a file path, so we need to save temporarily
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            image_pil.save(tmp_path, 'JPEG')
        
        try:
            # Load and preprocess image
            image_source, image = load_image(tmp_path)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        # Run detection
        boxes, logits, phrases = predict(
            model=self.grounding_dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Convert boxes to image coordinates
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        
        return {
            'boxes': boxes_xyxy.cpu().numpy(),
            'scores': logits.cpu().numpy(),
            'phrases': phrases,
            'image_shape': (H, W)
        }
    
    def segment_products(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate segmentation masks for detected products using SAM.
        
        Args:
            image: Input image (numpy array, BGR format)
            boxes: Bounding boxes in xyxy format
        
        Returns:
            List of binary masks (one per box)
        """
        if self.sam_predictor is None:
            raise ValueError("SAM model not loaded")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM
        self.sam_predictor.set_image(image_rgb)
        
        masks = []
        for box in boxes:
            # Ensure box is a numpy array
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            elif not isinstance(box, np.ndarray):
                box = np.array(box)
            
            # Ensure box is float type numpy array (SAM's transform expects numpy, not tensor)
            box = box.astype(np.float32)
            
            # SAM's predict expects numpy array, not tensor (it will convert internally)
            # The box should be in format [x1, y1, x2, y2]
            mask, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],  # Add batch dimension but keep as numpy
                multimask_output=False
            )
            masks.append(mask[0])
        
        return masks
    
    def read_text(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        masks: List[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Read text from detected product regions using EasyOCR.
        
        Args:
            image: Input image (numpy array, BGR format)
            boxes: Bounding boxes in xyxy format
            masks: Optional segmentation masks to focus OCR
        
        Returns:
            List of text detection results per product
        """
        if self.ocr_reader is None:
            raise ValueError("EasyOCR not loaded")
        
        results = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            # Extract region
            if masks and i < len(masks):
                # Use mask to focus OCR
                mask_region = masks[i][y1:y2, x1:x2]
                region = image[y1:y2, x1:x2].copy()
                region[mask_region == False] = [255, 255, 255]  # White background
            else:
                region = image[y1:y2, x1:x2]
            
            # Run OCR
            ocr_results = self.ocr_reader.readtext(region)
            
            # Extract text and confidence
            texts = []
            confidences = []
            for (bbox, text, conf) in ocr_results:
                texts.append(text)
                confidences.append(conf)
            
            results.append({
                'texts': texts,
                'combined_text': ' '.join(texts),
                'confidences': confidences,
                'box': box
            })
        
        return results
    
    def verify_brand(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        expected_brands: List[str],
        masks: List[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Verify brand identity using CLIP by comparing product regions with brand names.
        
        Args:
            image: Input image (numpy array, BGR format)
            boxes: Bounding boxes in xyxy format
            expected_brands: List of expected brand names to match against
            masks: Optional segmentation masks
        
        Returns:
            List of brand verification results with similarity scores
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not loaded")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            # Extract product region
            if masks and i < len(masks):
                mask_region = masks[i][y1:y2, x1:x2]
                region = image_rgb[y1:y2, x1:x2].copy()
                region[mask_region == False] = [255, 255, 255]
            else:
                region = image_rgb[y1:y2, x1:x2]
            
            # Preprocess for CLIP
            region_pil = Image.fromarray(region).resize((224, 224))
            image_tensor = self.clip_preprocess(region_pil).unsqueeze(0).to(self.device)
            
            # Encode image
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Encode text prompts
            text_inputs = clip.tokenize([f"a photo of {brand}" for brand in expected_brands]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = (image_features @ text_features.T).cpu().numpy()[0]
            best_match_idx = np.argmax(similarities)
            
            results.append({
                'detected_brand': expected_brands[best_match_idx],
                'confidence': float(similarities[best_match_idx]),
                'all_similarities': {brand: float(sim) for brand, sim in zip(expected_brands, similarities)},
                'box': box
            })
        
        return results
    
    def analyze_placement(
        self,
        boxes: np.ndarray,
        image_shape: Tuple[int, int],
        alignment_tolerance: float = 0.1,
        spacing_tolerance: float = 0.15
    ) -> Dict[str, Any]:
        """
        Analyze product placement: alignment, spacing, and arrangement.
        
        Args:
            boxes: Bounding boxes in xyxy format
            image_shape: (height, width) of the image
            alignment_tolerance: Tolerance for alignment check (fraction of image dimension)
            spacing_tolerance: Tolerance for spacing check (fraction of average box size)
        
        Returns:
            Dictionary with placement analysis results
        """
        if len(boxes) == 0:
            return {
                'is_aligned': False,
                'alignment_score': 0.0,
                'spacing_consistent': False,
                'spacing_score': 0.0,
                'arrangement_quality': 'poor'
            }
        
        boxes = np.array(boxes)
        H, W = image_shape
        
        # Calculate box properties
        box_centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        box_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        
        # Check horizontal alignment (same row)
        # Group boxes by vertical position (y-center)
        y_tolerance = H * alignment_tolerance
        rows = []
        used = set()
        
        for i in range(len(boxes)):
            if i in used:
                continue
            row = [i]
            used.add(i)
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                if abs(box_centers_y[i] - box_centers_y[j]) < y_tolerance:
                    row.append(j)
                    used.add(j)
            rows.append(row)
        
        # Check alignment within rows
        alignment_scores = []
        for row in rows:
            if len(row) < 2:
                continue
            row_centers = box_centers_y[row]
            std_dev = np.std(row_centers)
            # Normalize by tolerance - lower std = better alignment
            normalized_std = std_dev / (H * alignment_tolerance + 1e-6)
            score = max(0.0, 1.0 - normalized_std)
            alignment_scores.append(score)
        
        # If no multi-product rows, check overall vertical alignment
        if not alignment_scores and len(rows) > 1:
            # Check if rows are evenly spaced vertically
            row_y_centers = [np.mean(box_centers_y[row]) for row in rows]
            row_y_centers.sort()
            if len(row_y_centers) >= 2:
                row_spacings = [row_y_centers[i+1] - row_y_centers[i] for i in range(len(row_y_centers)-1)]
                if len(row_spacings) > 0:
                    spacing_std = np.std(row_spacings)
                    avg_spacing = np.mean(row_spacings)
                    if avg_spacing > 0:
                        normalized_std = spacing_std / avg_spacing
                        avg_alignment = max(0.0, 1.0 - min(1.0, normalized_std))
                    else:
                        avg_alignment = 0.0
                else:
                    avg_alignment = 0.0
            else:
                avg_alignment = 0.0
        else:
            avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
        
        is_aligned = avg_alignment > 0.7
        
        # Check spacing consistency
        spacing_scores = []
        for row in rows:
            if len(row) < 2:
                continue
            row_indices = sorted(row, key=lambda i: box_centers_x[i])
            spacings = []
            for k in range(len(row_indices) - 1):
                i1, i2 = row_indices[k], row_indices[k + 1]
                spacing = box_centers_x[i2] - box_centers_x[i1] - (box_widths[i1] + box_widths[i2]) / 2
                spacings.append(spacing)
            
            if len(spacings) > 0:
                avg_spacing = np.mean(spacings)
                spacing_std = np.std(spacings)
                # Normalize spacing score: lower std relative to mean = better consistency
                if avg_spacing > 0:
                    normalized_std = spacing_std / (avg_spacing + 1e-6)
                    spacing_score = max(0.0, min(1.0, 1.0 - normalized_std))
                else:
                    spacing_score = 0.0
                spacing_scores.append(spacing_score)
        
        # If no multi-product rows, check horizontal distribution
        if not spacing_scores and len(boxes) > 1:
            # Check overall horizontal spacing consistency
            sorted_indices = sorted(range(len(boxes)), key=lambda i: box_centers_x[i])
            spacings = []
            for k in range(len(sorted_indices) - 1):
                i1, i2 = sorted_indices[k], sorted_indices[k + 1]
                spacing = box_centers_x[i2] - box_centers_x[i1] - (box_widths[i1] + box_widths[i2]) / 2
                if spacing > 0:  # Only positive spacings
                    spacings.append(spacing)
            
            if len(spacings) > 0:
                avg_spacing = np.mean(spacings)
                spacing_std = np.std(spacings)
                if avg_spacing > 0:
                    normalized_std = spacing_std / avg_spacing
                    avg_spacing_score = max(0.0, 1.0 - min(1.0, normalized_std))
                else:
                    avg_spacing_score = 0.0
            else:
                avg_spacing_score = 0.0
        else:
            avg_spacing_score = np.mean(spacing_scores) if spacing_scores else 0.0
        
        spacing_consistent = avg_spacing_score > 0.7
        
        # Overall arrangement quality
        # Consider both alignment and spacing, but also number of products detected
        # A well-stocked shelf with many products should score higher
        num_products = len(boxes)
        products_per_row_avg = np.mean([len(row) for row in rows]) if rows else 0
        multi_product_rows = sum(1 for row in rows if len(row) >= 2)
        
        # Base quality on alignment and spacing
        # High alignment with good spacing = excellent
        if avg_alignment > 0.8 and avg_spacing_score > 0.8:
            quality = 'excellent'
        # Good alignment with decent spacing = good
        elif avg_alignment > 0.6 and avg_spacing_score > 0.6:
            quality = 'good'
        # Good alignment alone (spacing might be hard to measure with few multi-product rows)
        elif avg_alignment > 0.7:
            quality = 'good'
        # Fair alignment or spacing
        elif avg_alignment > 0.4 or avg_spacing_score > 0.4:
            quality = 'fair'
        # Many products in multiple rows suggests organization (even if scores are low)
        elif num_products >= 5 and len(rows) >= 2 and avg_alignment > 0.3:
            quality = 'fair'
        else:
            quality = 'poor'
        
        # Add diagnostic information
        diagnostics = {
            'num_products': num_products,
            'num_rows': len(rows),
            'products_per_row': [len(row) for row in rows],
            'multi_product_rows': sum(1 for row in rows if len(row) >= 2),
            'single_product_rows': sum(1 for row in rows if len(row) == 1)
        }
        
        return {
            'is_aligned': is_aligned,
            'alignment_score': float(avg_alignment),
            'spacing_consistent': spacing_consistent,
            'spacing_score': float(avg_spacing_score),
            'arrangement_quality': quality,
            'num_rows': len(rows),
            'products_per_row': [len(row) for row in rows],
            'diagnostics': diagnostics
        }
    
    def count_stacked_units(
        self,
        boxes: np.ndarray,
        masks: List[np.ndarray] = None,
        overlap_threshold: float = 0.3
    ) -> List[int]:
        """
        Count stacked units for each detected product.
        Uses vertical overlap analysis to detect stacking.
        
        Args:
            boxes: Bounding boxes in xyxy format
            masks: Optional segmentation masks
            overlap_threshold: Minimum overlap ratio to consider as stacked
        
        Returns:
            List of unit counts per product (typically 1, but can be higher for stacked items)
        """
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        unit_counts = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Find overlapping boxes in vertical direction
            stacked_count = 1  # At least one unit
            
            for j, other_box in enumerate(boxes):
                if i == j:
                    continue
                
                ox1, oy1, ox2, oy2 = other_box
                
                # Check if boxes are horizontally aligned (same column)
                horizontal_overlap = max(0, min(x2, ox2) - max(x1, ox1))
                if horizontal_overlap / box_width < 0.5:  # Not in same column
                    continue
                
                # Check vertical stacking
                vertical_distance = min(abs(y2 - oy1), abs(oy2 - y1))
                if vertical_distance < box_height * overlap_threshold:
                    stacked_count += 1
            
            unit_counts.append(stacked_count)
        
        return unit_counts
    
    def process_shelf(
        self,
        image_path: str,
        product_query: str,
        expected_brands: List[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        return_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline: detect products, read brands, analyze placement, and count units.
        
        Args:
            image_path: Path to shelf image
            product_query: Text query for product detection (e.g., "toothpaste", "shampoo bottles")
            expected_brands: Optional list of expected brand names for verification
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            return_visualization: Whether to return annotated visualization
        
        Returns:
            Complete analysis results dictionary
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Step 1: Detect products
        print(f"Detecting products: {product_query}")
        detections = self.detect_products(image, product_query, box_threshold, text_threshold)
        boxes = detections['boxes']
        scores = detections['scores']
        
        print(f"Found {len(boxes)} products")
        
        if len(boxes) == 0:
            return {
                'num_products': 0,
                'products': [],
                'placement_analysis': None,
                'total_units': 0
            }
        
        # Step 2: Segment products
        masks = []
        if self.sam_predictor is not None:
            print("Segmenting products...")
            try:
                # Ensure boxes are numpy arrays (convert from tensor if needed)
                if isinstance(boxes, torch.Tensor):
                    boxes_np = boxes.cpu().numpy()
                elif isinstance(boxes, np.ndarray):
                    boxes_np = boxes.copy()
                else:
                    boxes_np = np.array(boxes)
                masks = self.segment_products(image, boxes_np)
            except Exception as e:
                print(f"Warning: Segmentation failed: {e}")
                import traceback
                traceback.print_exc()
                masks = []
        
        # Step 3: Read text/brand names
        ocr_results = []
        if self.ocr_reader is not None:
            print("Reading text from products...")
            try:
                ocr_results = self.read_text(image, boxes, masks if masks else None)
            except Exception as e:
                print(f"Warning: OCR failed: {e}")
        
        # Step 4: Verify brands (if expected brands provided)
        brand_results = []
        if expected_brands and self.clip_model is not None:
            print("Verifying brands...")
            try:
                brand_results = self.verify_brand(image, boxes, expected_brands, masks if masks else None)
            except Exception as e:
                print(f"Warning: Brand verification failed: {e}")
        
        # Step 5: Analyze placement
        print("Analyzing placement...")
        placement_analysis = self.analyze_placement(boxes, detections['image_shape'])
        
        # Step 6: Count stacked units
        print("Counting stacked units...")
        unit_counts = self.count_stacked_units(boxes, masks if masks else None)
        total_units = sum(unit_counts) if unit_counts else len(boxes)
        
        # Compile results
        products = []
        for i in range(len(boxes)):
            product_info = {
                'id': i + 1,
                'box': boxes[i].tolist(),
                'confidence': float(scores[i]),
                'units_stacked': unit_counts[i] if unit_counts else 1
            }
            
            if ocr_results and i < len(ocr_results):
                product_info['detected_text'] = ocr_results[i]['combined_text']
                product_info['text_confidence'] = float(np.mean(ocr_results[i]['confidences'])) if ocr_results[i]['confidences'] else 0.0
            
            if brand_results and i < len(brand_results):
                product_info['brand'] = brand_results[i]['detected_brand']
                product_info['brand_confidence'] = brand_results[i]['confidence']
            
            products.append(product_info)
        
        # Analyze brand distribution and print summary
        if brand_results:
            brand_distribution = {}
            for product in products:
                brand = product.get('brand', 'Unknown')
                if brand not in brand_distribution:
                    brand_distribution[brand] = {'count': 0, 'units': 0}
                brand_distribution[brand]['count'] += 1
                brand_distribution[brand]['units'] += product.get('units_stacked', 1)
            
            unique_brands = len(brand_distribution)
            if unique_brands == 1:
                brand_name = list(brand_distribution.keys())[0]
                brand_info = brand_distribution[brand_name]
                print(f"✓ Brand: {brand_name} (All {brand_info['count']} products are the same brand)")
                print(f"  Total units: {brand_info['units']}")
            else:
                print(f"✓ Found {unique_brands} different brands:")
                for brand, info in brand_distribution.items():
                    print(f"  - {brand}: {info['count']} product(s), {info['units']} unit(s)")
        
        results = {
            'num_products': len(boxes),
            'total_units': total_units,
            'products': products,
            'placement_analysis': placement_analysis,
            'product_query': product_query
        }
        
        # Create visualization if requested
        if return_visualization:
            vis_image = self.visualize_results(image, boxes, scores, masks, ocr_results, brand_results, placement_analysis)
            results['visualization'] = vis_image
        
        return results
    
    def visualize_results(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        masks: List[np.ndarray] = None,
        ocr_results: List[Dict] = None,
        brand_results: List[Dict] = None,
        placement_analysis: Dict = None
    ) -> np.ndarray:
        """
        Create visualization of detection and analysis results.
        """
        vis_image = image.copy()
        
        # Draw boxes and labels
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label_parts = [f"#{i+1}: {score:.2f}"]
            
            if brand_results and i < len(brand_results):
                label_parts.append(f"Brand: {brand_results[i]['detected_brand']}")
            
            if ocr_results and i < len(ocr_results) and ocr_results[i]['combined_text']:
                text = ocr_results[i]['combined_text'][:20]  # Truncate long text
                label_parts.append(f"Text: {text}")
            
            label = " | ".join(label_parts)
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        # Draw masks if available
        if masks:
            for mask in masks:
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask] = [0, 255, 0]
                vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
        
        # Add placement analysis text
        if placement_analysis:
            analysis_text = [
                f"Arrangement: {placement_analysis['arrangement_quality']}",
                f"Alignment: {placement_analysis['alignment_score']:.2f}",
                f"Spacing: {placement_analysis['spacing_score']:.2f}",
                f"Rows: {placement_analysis['num_rows']}"
            ]
            
            y_offset = 30
            for text in analysis_text:
                cv2.putText(
                    vis_image,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                y_offset += 25
        
        return vis_image


