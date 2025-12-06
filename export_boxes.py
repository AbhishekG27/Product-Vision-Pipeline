"""
Utility script to export bounding boxes from analysis results.
"""

import json
import cv2
from product_vision_pipeline import ProductVisionPipeline
from pathlib import Path

def export_bounding_boxes(image_path, product_query, expected_brands=None, output_format="json"):
    """
    Analyze image and export bounding boxes.
    
    Args:
        image_path: Path to shelf image
        product_query: Product description to detect
        expected_brands: List of expected brands (optional)
        output_format: "json", "csv", or "txt"
    
    Returns:
        Dictionary with bounding boxes and metadata
    """
    # Initialize pipeline
    pipeline = ProductVisionPipeline(
        grounding_dino_config_path="models/GroundingDINO_SwinB_cfg.py",
        grounding_dino_checkpoint_path="models/groundingdino_swinb_cogcoor.pth",
        sam_checkpoint_path="models/sam_vit_b.pth",
        sam_model_type="vit_b"
    )
    
    # Run analysis
    results = pipeline.process_shelf(
        image_path=image_path,
        product_query=product_query,
        expected_brands=expected_brands,
        return_visualization=True
    )
    
    # Extract bounding boxes
    boxes_data = []
    for product in results['products']:
        box = product['box']
        boxes_data.append({
            'product_id': product['id'],
            'bounding_box': {
                'x1': int(box[0]),
                'y1': int(box[1]),
                'x2': int(box[2]),
                'y2': int(box[3]),
                'width': int(box[2] - box[0]),
                'height': int(box[3] - box[1])
            },
            'confidence': product['confidence'],
            'brand': product.get('brand', 'N/A'),
            'detected_text': product.get('detected_text', 'N/A'),
            'units_stacked': product['units_stacked']
        })
    
    # Export based on format
    base_name = Path(image_path).stem
    
    if output_format == "json":
        output_path = f"{base_name}_boxes.json"
        with open(output_path, 'w') as f:
            json.dump({
                'image_path': image_path,
                'product_query': product_query,
                'num_products': results['num_products'],
                'total_units': results['total_units'],
                'bounding_boxes': boxes_data,
                'placement_analysis': results['placement_analysis']
            }, f, indent=2)
        print(f"✓ Bounding boxes exported to: {output_path}")
    
    elif output_format == "csv":
        import pandas as pd
        df_data = []
        for item in boxes_data:
            df_data.append({
                'Product ID': item['product_id'],
                'X1': item['bounding_box']['x1'],
                'Y1': item['bounding_box']['y1'],
                'X2': item['bounding_box']['x2'],
                'Y2': item['bounding_box']['y2'],
                'Width': item['bounding_box']['width'],
                'Height': item['bounding_box']['height'],
                'Confidence': item['confidence'],
                'Brand': item['brand'],
                'Text': item['detected_text'],
                'Units': item['units_stacked']
            })
        df = pd.DataFrame(df_data)
        output_path = f"{base_name}_boxes.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Bounding boxes exported to: {output_path}")
    
    elif output_format == "txt":
        output_path = f"{base_name}_boxes.txt"
        with open(output_path, 'w') as f:
            f.write(f"Bounding Boxes for: {image_path}\n")
            f.write(f"Product Query: {product_query}\n")
            f.write(f"Total Products: {results['num_products']}\n")
            f.write("=" * 60 + "\n\n")
            for item in boxes_data:
                f.write(f"Product #{item['product_id']}:\n")
                f.write(f"  Bounding Box: [{item['bounding_box']['x1']}, {item['bounding_box']['y1']}, {item['bounding_box']['x2']}, {item['bounding_box']['y2']}]\n")
                f.write(f"  Format: [x1, y1, x2, y2] (top-left to bottom-right)\n")
                f.write(f"  Width: {item['bounding_box']['width']}px, Height: {item['bounding_box']['height']}px\n")
                f.write(f"  Confidence: {item['confidence']:.2f}\n")
                f.write(f"  Brand: {item['brand']}\n")
                f.write(f"  Units: {item['units_stacked']}\n\n")
        print(f"✓ Bounding boxes exported to: {output_path}")
    
    return boxes_data

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python export_boxes.py <image_path> <product_query> [expected_brands] [format]")
        print("Example: python export_boxes.py shelf.jpg 'toothpaste tube' 'Colgate,Crest' json")
        sys.exit(1)
    
    image_path = sys.argv[1]
    product_query = sys.argv[2]
    expected_brands = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else None
    output_format = sys.argv[4] if len(sys.argv) > 4 else "json"
    
    export_bounding_boxes(image_path, product_query, expected_brands, output_format)

