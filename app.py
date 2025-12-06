"""
Streamlined UI for Product Vision Pipeline using Gradio.
Provides an easy-to-use interface for product detection, brand identification, and placement analysis.
"""

import gradio as gr
import cv2
import json
import numpy as np
from pathlib import Path
from product_vision_pipeline import ProductVisionPipeline
import pandas as pd

# Initialize pipeline (will be loaded when needed)
pipeline = None

def initialize_pipeline():
    """Initialize the pipeline with models."""
    global pipeline
    if pipeline is None:
        print("Initializing pipeline...")
        pipeline = ProductVisionPipeline(
            grounding_dino_config_path="models/GroundingDINO_SwinB_cfg.py",
            grounding_dino_checkpoint_path="models/groundingdino_swinb_cogcoor.pth",
            sam_checkpoint_path="models/sam_vit_b.pth",
            sam_model_type="vit_b"
        )
        print("Pipeline initialized!")
    return pipeline

def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def analyze_shelf(
    image,
    product_query,
    expected_brands,
    box_threshold,
    text_threshold,
    show_boxes,
    export_json
):
    """Main analysis function."""
    if image is None:
        return None, "Please upload an image first.", None, None
    
    try:
        # Initialize pipeline
        pipe = initialize_pipeline()
        
        # Parse expected brands (comma-separated)
        brands_list = [b.strip() for b in expected_brands.split(",") if b.strip()] if expected_brands else None
        
        # Save uploaded image temporarily
        temp_path = "temp_input.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Run analysis
        results = pipe.process_shelf(
            image_path=temp_path,
            product_query=product_query,
            expected_brands=brands_list,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            return_visualization=True
        )
        
        # Prepare visualization
        vis_image = results['visualization']
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        # Analyze brands - group by brand
        brand_counts = {}
        for product in results['products']:
            brand = product.get('brand', 'Unknown')
            if brand not in brand_counts:
                brand_counts[brand] = {
                    'count': 0,
                    'total_units': 0,
                    'products': []
                }
            brand_counts[brand]['count'] += 1
            brand_counts[brand]['total_units'] += product.get('units_stacked', 1)
            brand_counts[brand]['products'].append(product['id'])
        
        unique_brands = len(brand_counts)
        all_same_brand = unique_brands == 1
        
        # Create summary text
        summary = f"""
# Analysis Results

## Detection Summary
- **Products Detected:** {results['num_products']}
- **Total Units:** {results['total_units']}
- **Product Query:** {results['product_query']}

## Brand Analysis
"""
        if all_same_brand:
            brand_name = list(brand_counts.keys())[0]
            brand_info = brand_counts[brand_name]
            summary += f"""
- **Brand:** {brand_name} (All products are the same brand)
- **Products:** {brand_info['count']} items
- **Total Units:** {brand_info['total_units']} units
"""
        else:
            summary += f"""
- **Unique Brands:** {unique_brands} different brands detected
"""
            for brand, info in brand_counts.items():
                summary += f"""
- **{brand}:** {info['count']} product(s), {info['total_units']} unit(s)
"""

        summary += "\n## Placement Analysis\n"
        if results['placement_analysis']:
            pa = results['placement_analysis']
            summary += f"""
- **Arrangement Quality:** {pa['arrangement_quality'].upper()}
- **Alignment Score:** {pa['alignment_score']:.2f} / 1.00
- **Spacing Score:** {pa['spacing_score']:.2f} / 1.00
- **Number of Rows:** {pa['num_rows']}
- **Products per Row:** {pa['products_per_row']}
"""
        
        # Create detailed product table with bounding boxes prominently displayed
        product_data = []
        for product in results['products']:
            box = product['box']
            row = {
                'ID': product['id'],
                'X1': int(box[0]),
                'Y1': int(box[1]),
                'X2': int(box[2]),
                'Y2': int(box[3]),
                'Width': int(box[2] - box[0]),
                'Height': int(box[3] - box[1]),
                'Bounding Box [x1,y1,x2,y2]': f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]",
                'Confidence': f"{product['confidence']:.2f}",
                'Brand': product.get('brand', 'N/A'),
                'Brand Conf.': f"{product.get('brand_confidence', 0):.2f}" if 'brand_confidence' in product else 'N/A',
                'Detected Text': product.get('detected_text', 'N/A')[:40] + '...' if product.get('detected_text') and len(product.get('detected_text', '')) > 40 else product.get('detected_text', 'N/A'),
                'Units Stacked': product['units_stacked']
            }
            product_data.append(row)
        
        df = pd.DataFrame(product_data)
        
        # Reorder columns to put bounding box info first
        if not df.empty:
            priority_cols = ['ID', 'X1', 'Y1', 'X2', 'Y2', 'Width', 'Height', 'Bounding Box [x1,y1,x2,y2]']
            other_cols = [c for c in df.columns if c not in priority_cols]
            df = df[priority_cols + other_cols]
        
        # Prepare JSON export - convert numpy types to native Python types
        json_data = {
            'summary': {
                'num_products': int(results['num_products']),
                'total_units': int(results['total_units']),
                'product_query': str(results['product_query']),
                'unique_brands': unique_brands,
                'all_same_brand': all_same_brand,
                'brand_summary': f"All products are {list(brand_counts.keys())[0]} brand" if all_same_brand 
                               else f"{unique_brands} different brands detected"
            },
            'brand_analysis': convert_to_serializable(brand_counts),
            'placement_analysis': convert_to_serializable(results['placement_analysis']),
            'products': convert_to_serializable(results['products'])
        }
        
        json_str = json.dumps(json_data, indent=2)
        
        # Save JSON to file for download
        json_file_path = f"results_{Path(temp_path).stem}.json"
        with open(json_file_path, 'w') as f:
            f.write(json_str)
        
        # Always show bounding boxes info in summary
        boxes_info = "\n## Bounding Boxes\n\n"
        boxes_info += f"**Total Products with Bounding Boxes:** {len(results['products'])}\n\n"
        boxes_info += "**Format:** [x1, y1, x2, y2] where:\n"
        boxes_info += "- (x1, y1) = Top-left corner\n"
        boxes_info += "- (x2, y2) = Bottom-right corner\n"
        boxes_info += "- All coordinates in pixels\n\n"
        
        if show_boxes and len(results['products']) <= 10:  # Show details for up to 10 products
            for product in results['products']:
                box = product['box']
                boxes_info += f"**Product #{product['id']}:**\n"
                boxes_info += f"- Box: [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]\n"
                boxes_info += f"- Size: {int(box[2] - box[0])}×{int(box[3] - box[1])}px\n"
                if 'brand' in product:
                    boxes_info += f"- Brand: {product['brand']}\n"
                boxes_info += "\n"
        elif len(results['products']) > 10:
            boxes_info += f"*Showing details for first 10 products. See table below for all {len(results['products'])} products.*\n\n"
        
        summary += boxes_info
        
        return vis_image_rgb, summary, df, json_str
        
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}\n\nPlease check:\n1. Image is valid\n2. Models are loaded\n3. Product query is appropriate"
        import traceback
        traceback.print_exc()
        return None, error_msg, None, None

# Create Gradio interface
with gr.Blocks(title="Product Vision Pipeline") as demo:
    gr.Markdown("""
    # 🛒 Product Vision Pipeline
    
    Detect products on shelves, identify brands, read text, and analyze placement using AI.
    
    **Features:**
    - 🔍 Product detection using text queries
    - 🏷️ Brand identification (OCR + CLIP)
    - 📊 Placement analysis (alignment, spacing, arrangement)
    - 📦 Unit counting (including stacked products)
    - 📍 Bounding box coordinates
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Shelf Image", type="numpy")
            
            product_query = gr.Textbox(
                label="Product Query",
                value="toothpaste tube",
                placeholder="e.g., 'toothpaste tube', 'shampoo bottle', 'soda can'",
                info="Describe what products to detect"
            )
            
            expected_brands = gr.Textbox(
                label="Expected Brands (comma-separated)",
                value="Colgate, Crest, Sensodyne, Oral-B",
                placeholder="Colgate, Crest, Sensodyne",
                info="List brands to verify (optional)"
            )
            
            with gr.Row():
                box_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="Detection Threshold",
                    info="Lower = more detections (may include false positives)"
                )
                
                text_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.20,
                    step=0.05,
                    label="Text Matching Threshold",
                    info="Lower = more text matches"
                )
            
            show_boxes = gr.Checkbox(
                label="Show Bounding Box Details",
                value=True,
                info="Display bounding box coordinates in results"
            )
            
            export_json = gr.Checkbox(
                label="Export JSON",
                value=True,
                info="Include JSON export of results"
            )
            
            analyze_btn = gr.Button("🔍 Analyze Shelf", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            image_output = gr.Image(label="Analysis Results (with bounding boxes)")
            
            summary_output = gr.Markdown(label="Summary")
            
            with gr.Tabs():
                with gr.Tab("📦 Bounding Boxes & Product Details"):
                    gr.Markdown("### Bounding Box Coordinates")
                    gr.Markdown("""
                    **Format:** [x1, y1, x2, y2] where:
                    - **(x1, y1)** = Top-left corner (in pixels)
                    - **(x2, y2)** = Bottom-right corner (in pixels)
                    - **Width** = x2 - x1
                    - **Height** = y2 - y1
                    """)
                    table_output = gr.Dataframe(
                        label="All Products with Bounding Box Coordinates",
                        wrap=True,
                        interactive=False
                    )
                
                with gr.Tab("📄 Complete JSON Export"):
                    json_output = gr.Code(
                        label="Full Results (JSON Format)",
                        language="json",
                        lines=25
                    )
                    gr.Markdown("""
                    **JSON Structure:**
                    - `summary`: Detection summary
                    - `placement_analysis`: Arrangement metrics
                    - `products`: Array of all products with bounding boxes
                    """)
    
    # Example images section
    gr.Markdown("### 📸 Example Queries")
    gr.Examples(
        examples=[
            ["toothpaste tube", "Colgate, Crest, Sensodyne"],
            ["shampoo bottle", "Dove, Pantene, Head & Shoulders"],
            ["soda can", "Coca-Cola, Pepsi, Sprite"],
            ["cereal box", "Kellogg's, General Mills"],
        ],
        inputs=[product_query, expected_brands],
        label="Try these queries:"
    )
    
    # Connect the analyze button
    analyze_btn.click(
        fn=analyze_shelf,
        inputs=[image_input, product_query, expected_brands, box_threshold, text_threshold, show_boxes, export_json],
        outputs=[image_output, summary_output, table_output, json_output]
    )
    
    # Auto-analyze when image is uploaded (optional - can be enabled)
    # image_input.upload(
    #     fn=analyze_shelf,
    #     inputs=[image_input, product_query, expected_brands, box_threshold, text_threshold, show_boxes, export_json],
    #     outputs=[image_output, summary_output, table_output, json_output]
    # )
    
    gr.Markdown("""
    ### 💡 Tips
    - **Product Query**: Be specific (e.g., "toothpaste tube" works better than just "toothpaste")
    - **Detection Threshold**: Start with 0.25, lower if you need more detections
    - **Brands**: Leave empty if you don't want brand verification
    - **Bounding Boxes**: Coordinates are in [x1, y1, x2, y2] format (top-left to bottom-right)
    """)

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Product Vision Pipeline UI...")
    print("=" * 60)
    print()
    print("The UI will open in your browser automatically")
    print("If it doesn't open, manually go to: http://localhost:7860")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        demo.launch(
            share=False,
            server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
            server_port=7860,
            inbrowser=True,  # Auto-open browser
            show_error=True
        )
    except OSError as e:
        if "Address already in use" in str(e):
            print("\n⚠ Port 7860 is already in use!")
            print("Try:")
            print("1. Close any other Gradio apps")
            print("2. Or change the port in app.py (line 284)")
        else:
            raise

