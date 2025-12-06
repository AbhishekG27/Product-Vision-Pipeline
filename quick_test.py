"""
Quick test script to verify the pipeline works with your models.
Replace 'your_image.jpg' with an actual shelf image to test.
"""

from product_vision_pipeline import ProductVisionPipeline
import cv2
from pathlib import Path

print("=" * 60)
print("Product Vision Pipeline - Quick Test")
print("=" * 60)
print()

# Initialize pipeline
print("Initializing pipeline...")
pipeline = ProductVisionPipeline(
    grounding_dino_config_path="models/GroundingDINO_SwinB_cfg.py",
    grounding_dino_checkpoint_path="models/groundingdino_swinb_cogcoor.pth",
    sam_checkpoint_path="models/sam_vit_b.pth",
    sam_model_type="vit_b"
)

print("✓ Pipeline initialized successfully!")
print()

# Test with an image (replace with your image path)
image_path = r"D:\kogo\Vision\images\images.jpg"  # <-- Using raw string for Windows path
# Alternative: image_path = "images/2.jpg"  # Relative path also works

if not Path(image_path).exists():
    print(f"⚠ Image not found: {image_path}")
    print("\nTo test the pipeline:")
    print("1. Place a shelf image in this directory")
    print("2. Update 'image_path' in this script")
    print("3. Run again")
    print("\nExample usage:")
    print("""
results = pipeline.process_shelf(
    image_path="shelf_image.jpg",
    product_query="toothpaste tube",
    expected_brands=["Colgate", "Crest", "Sensodyne"],
    return_visualization=True
)

print(f"Found {results['num_products']} products")
cv2.imwrite("results.jpg", results['visualization'])
""")
else:
    print(f"Processing image: {image_path}")
    print()
    
    # Example: Detect toothpaste
    # Lower threshold to detect more products on well-stocked shelves
    results = pipeline.process_shelf(
        image_path=image_path,
        product_query="toothpaste tube",
        expected_brands=["Colgate", "Crest", "Sensodyne", "Oral-B"],
        box_threshold=0.25,  # Lower threshold to detect more products
        text_threshold=0.20,  # Lower threshold for better matching
        return_visualization=True
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Products detected: {results['num_products']}")
    print(f"Total units: {results['total_units']}")
    print()
    
    # Print detailed product information including brands
    print("Product Details:")
    for product in results['products']:
        print(f"\n  Product #{product['id']}:")
        print(f"    Confidence: {product['confidence']:.2f}")
        print(f"    Units stacked: {product['units_stacked']}")
        
        # Show detected text/OCR results
        if 'detected_text' in product and product['detected_text']:
            print(f"    Detected text: {product['detected_text']}")
            if 'text_confidence' in product:
                print(f"    Text confidence: {product['text_confidence']:.2f}")
        else:
            print(f"    Detected text: (none)")
        
        # Show brand information
        if 'brand' in product:
            print(f"    Brand: {product['brand']}")
            if 'brand_confidence' in product:
                print(f"    Brand confidence: {product['brand_confidence']:.2f}")
        else:
            print(f"    Brand: (not identified)")
    
    print()
    if results['placement_analysis']:
        pa = results['placement_analysis']
        print("Placement Analysis:")
        print(f"  Arrangement quality: {pa['arrangement_quality']}")
        print(f"  Alignment score: {pa['alignment_score']:.2f}")
        print(f"  Spacing score: {pa['spacing_score']:.2f}")
        print(f"  Number of rows: {pa['num_rows']}")
        if 'diagnostics' in pa:
            diag = pa['diagnostics']
            print(f"  Products per row: {pa['products_per_row']}")
            print(f"  Multi-product rows: {diag.get('multi_product_rows', 0)}")
            print(f"  Single-product rows: {diag.get('single_product_rows', 0)}")
    
    # Save visualization
    if 'visualization' in results:
        output_path = "test_results.jpg"
        cv2.imwrite(output_path, results['visualization'])
        print(f"\n✓ Visualization saved to: {output_path}")
    
    print("\n✓ Test complete!")

