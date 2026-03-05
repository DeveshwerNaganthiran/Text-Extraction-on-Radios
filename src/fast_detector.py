from pathlib import Path
from ultralytics import YOLO

class FastDetector:
    def __init__(self, model_path="walkie_detector.pt"):
        print(f"🚀 Loading YOLO model from: {model_path}")
        
        try:
            # Check if model exists
            if not Path(model_path).exists():
                print(f"❌ Model not found at {model_path}")
                print("Using default YOLOv8n model...")
                model_path = "yolov8n.pt"
            
            # Load model
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully")
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"Classes: {self.class_names}")
            else:
                # Default classes
                self.class_names = {0: 'walkie_talkie', 1: 'screen'}
                print("⚠ Using default classes: walkie_talkie, screen")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Using simple fallback detection")
            self.model = None
    
    def detect_boxes(self, image, confidence=0.5):
        """
        Detect walkie-talkies using YOLO model
        Returns: List of bounding boxes [x1, y1, x2, y2]
        """
        if self.model is None:
            return self.detect_simple_boxes(image)
        
        try:
            # Run detection
            results = self.model(image, conf=confidence, verbose=False, device='cpu')
            
            boxes = []
            
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Check if it's a walkie_talkie
                    class_name = self.class_names.get(cls, '')
                    if class_name == 'walkie_talkie' and conf >= confidence:
                        boxes.append([x1, y1, x2, y2])
            
            return boxes
            
        except Exception as e:
            print(f"⚠ Detection error: {e}")
            return self.detect_simple_boxes(image)
    
    def detect_simple_boxes(self, image):
        """Fallback simple detection when model fails"""
        height, width = image.shape[:2]
        
        # Center region as fallback
        box_size = 300
        center_x, center_y = width // 2, height // 2
        
        return [[
            center_x - box_size // 2,
            center_y - box_size // 2,
            center_x + box_size // 2,
            center_y + box_size // 2
        ]]
    
    def detect_with_screens(self, image, confidence=0.5):
        """
        Detect both walkie-talkies and screens
        Returns: (device_boxes, screen_boxes)
        """
        if self.model is None:
            return self.detect_simple_boxes(image), []
        
        try:
            results = self.model(image, conf=confidence, verbose=False, device='cpu')
            
            device_boxes = []
            screen_boxes = []
            
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    class_name = self.class_names.get(cls, '')
                    
                    if conf >= confidence:
                        if class_name == 'walkie_talkie':
                            device_boxes.append([x1, y1, x2, y2])
                        elif class_name == 'screen':
                            screen_boxes.append([x1, y1, x2, y2])
            
            return device_boxes, screen_boxes
            
        except Exception as e:
            print(f"⚠ Detection error: {e}")
            return self.detect_simple_boxes(image), []
