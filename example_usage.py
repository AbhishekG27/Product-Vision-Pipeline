"""
Example usage of the Product Vision Pipeline.

This script demonstrates how to use the pipeline to:
- Detect products on shelves
- Read brand names and text
- Verify brand identity
- Analyze product placement
- Count products and stacked units
"""

import cv2
import json
from pathlib import Path
from product_vision_pipeline import ProductVisionPipeline

def main():
    # Configuration
    # Update these paths after running download_models.py
    GROUNDING_DINO_CONFIG = "models/GroundingDINO_SwinB_cfg.py"
    GROUNDING_DINO_CHECKPOINT = "models/groundingdino_swinb_cogcoor.pth"
    SAM_CHECKPOINT = "models/sam_vit_b.pth"  # or sam_vit_l.pth, sam_vit_h.pth
    
    # Initialize pipeline
    print("Initializing Product Vision Pipeline...")
    pipeline = ProductVisionPipeline(
        grounding_dino_config_path=GROUNDING_DINO_CONFIG,
        grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        sam_checkpoint_path=SAM_CHECKPOINT,
        sam_model_type="vit_b",  # or "vit_l", "vit_h"
        ocr_languages=['en']  # Add more languages if needed: ['en', 'es', 'fr']
    )
    
    # Example 1: Detect toothpaste products
    print("\n" + "="*60)
    print("Example 1: Detecting toothpaste products")
    print("="*60)
    
    image_path = "shelf_image.jpg"  # Replace with your image path
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path.")
        return
    
    results = pipeline.process_shelf(
        image_path=image_path,
        product_query="toothpaste tube",
        expected_brands=["Colgate", "Crest", "Sensodyne", "Oral-B"],
        box_threshold=0.3,
        text_threshold=0.25,
        return_visualization=True
    )
    
    # Print results
    print(f"\nDetection Results:")
    print(f"  Number of products detected: {results['num_products']}")
    print(f"  Total units (including stacked): {results['total_units']}")
    
    print(f"\nProduct Details:")
    for product in results['products']:
        print(f"  Product #{product['id']}:")
        print(f"    Confidence: {product['confidence']:.2f}")
        print(f"    Units stacked: {product['units_stacked']}")
        if 'detected_text' in product:
            print(f"    Detected text: {product['detected_text']}")
        if 'brand' in product:
            print(f"    Brand: {product['brand']} (confidence: {product['brand_confidence']:.2f})")
    
    print(f"\nPlacement Analysis:")
    if results['placement_analysis']:
        pa = results['placement_analysis']
        print(f"  Arrangement quality: {pa['arrangement_quality']}")
        print(f"  Alignment score: {pa['alignment_score']:.2f}")
        print(f"  Spacing score: {pa['spacing_score']:.2f}")
        print(f"  Number of rows: {pa['num_rows']}")
        print(f"  Products per row: {pa['products_per_row']}")
    
    # Save visualization
    if 'visualization' in results:
        vis_path = "results_visualization.jpg"
        cv2.imwrite(vis_path, results['visualization'])
        print(f"\nVisualization saved to: {vis_path}")
    
    # Save detailed results as JSON
    results_json = {
        'num_products': results['num_products'],
        'total_units': results['total_units'],
        'products': results['products'],
        'placement_analysis': results['placement_analysis']
    }
    
    with open("results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print("Detailed results saved to: results.json")
    
    # Example 2: Detect different product types
    print("\n" + "="*60)
    print("Example 2: Detecting shampoo bottles")
    print("="*60)
    
    results2 = pipeline.process_shelf(
        image_path=image_path,
        product_query="shampoo bottle",
        expected_brands=["Dove", "Pantene", "Head & Shoulders", "Herbal Essences"],
        box_threshold=0.3,
        return_visualization=True
    )
    
    print(f"Detected {results2['num_products']} shampoo products")
    print(f"Total units: {results2['total_units']}")
    
    # Example 3: Detect drinks/cans
    print("\n" + "="*60)
    print("Example 3: Detecting drink cans")
    print("="*60)
    
    results3 = pipeline.process_shelf(
        image_path=image_path,
        product_query="soda can",
        expected_brands=["Coca-Cola", "Pepsi", "Sprite", "Fanta"],
        box_threshold=0.3,
        return_visualization=True
    )
    
    print(f"Detected {results3['num_products']} drink cans")
    print(f"Total units: {results3['total_units']}")

if __name__ == "__main__":
    main()

