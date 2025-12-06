# Product Vision Pipeline - UI Guide

## 🚀 Quick Start with UI

### 1. Install UI Dependencies

```bash
pip install gradio pandas
```

### 2. Launch the UI

```bash
python app.py
```

The UI will open in your browser at `http://localhost:7860`

## 📋 Features

### Main Interface

1. **Upload Image**: Drag and drop or click to upload a shelf image
2. **Product Query**: Enter what to detect (e.g., "toothpaste tube")
3. **Expected Brands**: Comma-separated list of brands (optional)
4. **Thresholds**: Adjust detection sensitivity
5. **Analyze**: Click to run the analysis

### Results Display

- **Visualization**: Image with bounding boxes and labels
- **Summary**: Key metrics and placement analysis
- **Product Table**: Detailed table with all products including:
  - Bounding box coordinates (X1, Y1, X2, Y2)
  - Brand information
  - Confidence scores
  - Detected text
  - Unit counts
- **JSON Export**: Complete results in JSON format

## 📍 Bounding Box Format

Bounding boxes are provided in **XYXY format**:
- `[x1, y1, x2, y2]`
- `x1, y1`: Top-left corner coordinates
- `x2, y2`: Bottom-right corner coordinates
- All coordinates are in pixels

Example:
```json
{
  "bounding_box": {
    "x1": 100,
    "y1": 50,
    "x2": 200,
    "y2": 150,
    "width": 100,
    "height": 100
  }
}
```

## 💻 Command Line Export

You can also export bounding boxes using the command line:

```bash
# Export as JSON
python export_boxes.py image.jpg "toothpaste tube" "Colgate,Crest" json

# Export as CSV
python export_boxes.py image.jpg "toothpaste tube" "Colgate,Crest" csv

# Export as TXT
python export_boxes.py image.jpg "toothpaste tube" "Colgate,Crest" txt
```

## 🎯 Example Queries

- `"toothpaste tube"` - Detect toothpaste
- `"shampoo bottle"` - Detect shampoo
- `"soda can"` - Detect drink cans
- `"cereal box"` - Detect cereal boxes
- `"detergent bottle"` - Detect cleaning products

## 📊 Understanding Results

### Placement Analysis
- **Excellent**: Alignment > 0.8 and Spacing > 0.8
- **Good**: Alignment > 0.7 or both > 0.6
- **Fair**: Alignment > 0.4 or Spacing > 0.4
- **Poor**: Below thresholds

### Confidence Scores
- **Detection Confidence**: How certain the model is about the product
- **Brand Confidence**: How well the product matches the expected brand
- **Text Confidence**: OCR reading confidence

## 🔧 Tips

1. **Lower thresholds** (0.2-0.25) for more detections
2. **Higher thresholds** (0.3-0.4) for fewer but more accurate detections
3. **Specific queries** work better (e.g., "toothpaste tube" vs "toothpaste")
4. **Multiple brands** help with verification accuracy

