Walkie-Talkie Screen Text Tracker

📋 Project Overview

A computer vision system that detects walkie-talkies, tracks them in real-time, and extracts text/numeric information from their screens using OCR. The system is designed to monitor multiple walkie-talkie devices simultaneously and read their display information.

🏗️ Project Structure

```
walkie-tracker/
├── data/
│   ├── raw_images/          # Raw photos of walkie-talkies
│   ├── annotated/           # Labeled images for training
│   ├── train/images/labels  # Training dataset
│   ├── val/images/labels    # Validation dataset
│   └── dataset.yaml         # Dataset configuration
├── src/s
│   ├── main.py              # Main application
│   ├── detector.py          # YOLO-based walkie-talkie detector
│   ├── train_detector.py    # Model training script
│   ├── annotation_tool.py   # Image annotation tool
│   ├── tracker.py           # Device tracking and re-identification
│   ├── ocr_processor.py     # OCR engine for screen text extraction
│   ├── ui.py               # User interface components
├── configs/
│   └── settings.yaml       # Configuration file
├── scripts/
│   ├── split_data.py       # Dataset splitting utility
│   └── capture_variations.py # Data collection script
├── models/                 # Trained model storage
├── runs/                  # Training runs and logs
├── requirements.txt       # Python dependencies
└── .vscode/              # VS Code configuration
```

🚀 Quick Start

1. Environment Setup

```bash
# Clone or create project directory
cd walkie-tracker

# Create Python 3.12 virtual environment
python -3.12 -m venv devenv

# Activate virtual environment
# Windows:
devenv\Scripts\activate
# Linux/Mac:
source devenv/bin/activate

# Clear pip cache and install dependencies
pip cache purge
pip install -r requirements.txt
```

2. Data Collection & Model Training Pipeline

```bash
# Step 1: Collect images of walkie-talkies (interactive)
python scripts/capture_variations.py

# Step 2: Annotate collected images (draw bounding boxes)
python src/annotation_tool.py

# Step 3: Split data into training/validation sets (80/20 ratio)
python scripts/split_data.py

# Step 4:  augmented copies(brightness up/down,contrast,clahe,blur,noise)
python scripts/augment_train.py

# Step 5: Train the detection model
python src/train_detector.py
```

3. Run the Application

```bash
# Start the walkie-talkie tracker
python src/main_msi_genai.py
```

⚙️ Configuration

Screen Region Coordinates

For accurate OCR, adjust the screen region in configs/settings.yaml:

```yaml
screen:
  roi_offsets:
    walkie_talkie:
      x1: 0.10  # Left offset (10% from left)
      y1: 0.55  # Top offset (55% from top)
      x2: 0.90  # Right offset (90% from left)
      y2: 0.70  # Bottom offset (70% from top)
```

Training Parameters

Adjust in configs/settings.yaml:

```yaml
training:
  epochs: 50          # Number of training epochs
  batch_size: 16      # Batch size
  learning_rate: 0.01 # Learning rate
  device: "cpu"       # "cuda" for GPU training
```

🎮 Application Controls

When running main.py:

· SPACE: Pause/Resume tracking
· M: Toggle device mapping mode
· D: Toggle debug mode
· T: Test detection on current frame
· S: Save screenshot
· Q: Quit application

🔧 Troubleshooting

Common Issues

1. Camera not detected
   · Check camera ID in configs/settings.yaml (default: 1)
   · Try camera IDs 0, 1, or 2
2. Poor OCR results
   · Ensure good lighting on walkie-talkie screen
   · Adjust screen region coordinates
   · Try different OCR engines in configuration
3. Model not detecting walkie-talkies
   · Retrain with more diverse images
   · Adjust confidence threshold in detector.py

Complete Reset

To start fresh with new data:

```bash
# Delete all collected data and trained models
# 1. Remove data folders
rm -rf data/annotated data/raw_images data/train data/val

# 2. Clear cache
rm -rf data/cache data/dataset_summary.txt

# 3. Remove training runs
rm -rf runs/

# 4. Optional: Remove downloaded model
rm -f yolov8n.pt
```

📊 Dataset Information

Data Split Ratio

The split_data.py script uses an 80/20 split:

· 80% for training
· 20% for validation

Supported Image Formats

· JPG (.jpg, .jpeg)
· PNG (.png)

Annotation Format

Uses YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Classes:

· 0: walkie_talkie
· 1: screen

🔍 OCR Engines Supported

The system supports multiple OCR engines (configure in settings.yaml):

1. Tesseract (Recommended for digital displays)
2. EasyOCR
3. PaddleOCR

🖥️ Development

VS Code Configuration

The project includes VS Code settings for:

· Python interpreter path
· Auto-formatting with Black
· Linting with Pylint
· Debug configurations

Testing

```bash
# Quick training test (30 epochs)
python src/train_detector.py --quick

# Test with specific configuration
python src/main_msi_genai.py --config configs/settings.yaml --camera 0
```

📁 File Descriptions

Core Application Files

· src/main_msi_genai.py - Main application entry point
· src/fast_detector.py - Walkie-talkie detection using YOLOv8
· src/simple_ocr.py - Text extraction from screens

Training Pipeline

· src/train_detector.py - Model training with YOLOv8
· src/annotation_tool.py - Interactive annotation tool
· scripts/split_data.py - Dataset splitting utility

Configuration

· configs/settings.yaml - All system parameters
· data/dataset.yaml - Dataset configuration for YOLO

📈 Performance Tips

1. For better detection:
   · Collect diverse images (different angles, lighting)
   · Include multiple walkie-talkies in training data
   · Annotate both walkie-talkie and screen separately
2. For better OCR:
   · Ensure screen is clearly visible
   · Use good lighting conditions
   · Adjust screen region coordinates for your device model
3. For real-time performance:
   · Use GPU if available (set device: "cuda")
   · Reduce frame resolution if needed
   · Adjust frame_skip in settings

🤝 Contributing

1. Follow the data collection pipeline for new walkie-talkie models
2. Test OCR accuracy with different screen types
3. Report issues with specific walkie-talkie models

🆘 Support

For issues:

1. Check the troubleshooting section
2. Ensure all dependencies are installed
3. Verify camera and lighting setup
4. Review console output for error messages

---

Note: This system requires a trained model. If no trained model is found, it will use a default YOLOv8n model which may not be optimized for walkie-talkie detection. Always train with your specific walkie-talkie models for best results.
