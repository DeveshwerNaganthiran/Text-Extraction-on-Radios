# Walkie-Talkie Screen Text Tracker (MSI GenAI Edition)

📋 **Project Overview**

A computer vision system that detects walkie-talkies, tracks them in real-time, and extracts text/numeric information from their screens. This updated version integrates **MSI GenAI** for advanced text extraction, multilingual support, and automated UI defect detection, while falling back to local OCR engines (like Tesseract or EasyOCR) if offline. The system is designed to monitor multiple walkie-talkie devices simultaneously and read their display information.

✨ **Key Features**
* **GenAI-Powered OCR:** Extracts complex, multilingual text and provides English translations.
* **Automated Defect Detection:** Automatically flags UI issues such as misalignments, text overlaps (bridge errors), and upside-down text.
* **GUI Launcher:** Includes a Tkinter-based user interface to easily configure device profiles, camera inputs, and output paths.
* **Live Camera Tuning:** Adjust Brightness, Sharpness, and Focus dynamically via an on-screen overlay.
* **Dual Capture Methods:** Seamlessly switch between OpenCV and FFmpeg camera backends.
* **Unicode Rendering:** Renders complex scripts (Hangul, CJK, Arabic, Cyrillic) flawlessly on the preview using Pillow.

🏗️ **Project Structure**

```text
walkie-tracker/
├── data/
│   ├── raw_images/          # Raw photos of walkie-talkies
│   ├── annotated/           # Labeled images for training
│   ├── train/images/labels  # Training dataset
│   ├── val/images/labels    # Validation dataset
│   └── dataset.yaml         # Dataset configuration
├── src/
│   ├── main_msi_genai.py    # Primary GenAI application & GUI Launcher
│   ├── fast_detector.py     # YOLOv8 walkie-talkie detection
│   ├── msi_genai_ocr.py     # MSI GenAI integration module
│   ├── simple_ocr.py        # Fallback local OCR (Tesseract/EasyOCR)
│   ├── train_detector.py    # Model training script
│   ├── annotation_tool.py   # Image annotation tool
│   └── tracker.py           # Device tracking and re-identification
├── configs/
│   ├── settings.yaml        # Configuration file
│   └── device_profiles.json # Saved device naming profiles
├── scripts/
│   ├── split_data.py        # Dataset splitting utility
│   ├── capture_variations.py# Data collection script
│   └── augment_train.py     # Data augmentation script
├── models/                  # Trained model storage
├── output/                  # Saved session captures and JSON logs
├── runs/                    # Training runs and logs
├── requirements.txt         # Python dependencies
├── .env                     # MSI GenAI API Credentials (Required)
└── .vscode/                 # VS Code configuration            # MSI GenAI API Credentials (Required)

🚀 Quick Start

1. Environment Setup

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

2. MSI GenAI Configuration (.env)

To use the GenAI text extraction, you must create a .env file in the root directory with your API credentials:

MSI_HOST=your_msi_host_url
MSI_API_KEY=your_api_key
MSI_USER_ID=your_user_id
MSI_DATASTORE_ID=your_datastore_id

3. Data Collection & Model Training Pipeline

# Step 1: Collect images of walkie-talkies (interactive)
python scripts/capture_variations.py

# Step 2: Annotate collected images (draw bounding boxes)
python src/annotation_tool.py

# Step 3: Split data into training/validation sets (80/20 ratio)
python scripts/split_data.py

# Step 4: Create augmented copies (brightness up/down, contrast, clahe, blur, noise)
python scripts/augment_train.py

# Step 5: Train the detection model
python src/train_detector.py

4. Run the Application

# 1. GUI Launcher Mode (Recommended for easy setup)
python main_msi_genai.py --gui

# 2. Standard CLI Mode
python main_msi_genai.py

# 3. Automated Single Capture (Useful for automated testing)
python main_msi_genai.py --once --warmup-sec 2.0

⚙️ Configuration

Screen Region Coordinates

For accurate OCR, adjust the screen region in configs/settings.yaml:

screen:
  roi_offsets:
    walkie_talkie:
      x1: 0.10  # Left offset (10% from left)
      y1: 0.55  # Top offset (55% from top)
      x2: 0.90  # Right offset (90% from left)
      y2: 0.70  # Bottom offset (70% from top)

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

When the camera preview window is active, use the following hotkeys:

 “SPACE”: Capture the current frame and extract text using MSI GenAI (or fallback OCR).
 “C”: Switch camera capture method (OpenCV ↔ FFmpeg).
 “T”: Toggle the Camera Settings Overlay (Adjust Brightness, Sharpness, Focus using your mouse).
“+” : Zoom In.
 “-” : Zoom Out.
 “Z”: Reset Zoom.
“ X”: Exit the application.

📁 Output Format

When you save a result (by pressing S after a capture), the system creates a structured folder in the output/ directory:

output/session_YYYYMMDD_HHMMSS/
├── raw_capture.jpg           # The original unedited frame
├── annotated_result.jpg      # Image with bounding boxes, text, and error warnings
├── session_summary.json      # JSON log of all devices detected and text extracted
└── device_1/                 # Subfolder for each detected device
    ├── device_info.json      # Specific device OCR text and error flags
    ├── screen_roi.jpg        # Cropped image of the screen
    └── sent_to_genai.jpg     # The exact crop sent to the GenAI API


🔧 Troubleshooting

Common Issues

Camera not detected
· Check camera ID in configs/settings.yaml (default: 1)
· Try camera IDs 0, 1, or 2
· Press C in the app to switch from OpenCV to the FFmpeg backend.

Poor OCR results
· Ensure good lighting on walkie-talkie screen
· Adjust screen region coordinates
· Try different OCR engines in configuration

Model not detecting walkie-talkies
· Retrain with more diverse images
· Adjust confidence threshold in settings.yaml

API_ERROR / CONNECTION ERROR
· Ensure your .env variables are correctly configured and that your network allows connections to the MSI host.

Complete Reset

To start fresh with new data:

Delete all collected data and trained models
1. Remove data folders
rm -rf data/annotated data/raw_images data/train data/val

 2. Clear cache
rm -rf data/cache data/dataset_summary.txt

3. Remove training runs
rm -rf runs/

4. Optional: Remove downloaded model
rm -f yolov8n.pt



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

MSI GenAI (Primary engine for complex, multilingual text and translation)
Tesseract (Recommended offline fallback for digital displays)
EasyOCR
PaddleOCR

🖥️ Development

VS Code Configuration

The project includes VS Code settings for:

· Python interpreter path
· Auto-formatting with Black
· Linting with Pylint
· Debug configurations

Testing

# Quick training test (30 epochs)
python src/train_detector.py --quick

# Test with specific configuration
python src/main_msi_genai.py --config configs/settings.yaml --camera 0

📈 Performance Tips

For better detection:
· Collect diverse images (different angles, lighting)
· Include multiple walkie-talkies in training data
· Annotate both walkie-talkie and screen separately

For better OCR:
· Ensure screen is clearly visible
· Use good lighting conditions
· Adjust screen region coordinates for your device model

For real-time performance:
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

Note: This system requires a trained model. If no trained model is found, it will use a default YOLOv8n model which may not be optimized for walkie-talkie detection. Always train with your specific walkie-talkie models for best results...