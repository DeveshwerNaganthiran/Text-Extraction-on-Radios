import cv2
import os
from pathlib import Path

class FixedAnnotationTool:
    def __init__(self, image_dir, label_dir="data/annotated"):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        all_images = list(self.image_dir.glob("*.jpg"))
        all_images.extend(list(self.image_dir.glob("*.png")))
        all_images.extend(list(self.image_dir.glob("*.jpeg")))

        self.images = []
        for img_path in all_images:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                self.images.append(img_path)
        
        if not self.images:
            if not all_images:
                print(f"❌ No images found in {image_dir}!")
                print(f" Make sure you have images in: {self.image_dir.absolute()}")
            else:
                print(f"✅ Found {len(all_images)} images")
                print("✅ All images already annotated. Nothing to do.")
            return
        
        self.current_index = 0
        self.annotations = []
        self.current_boxes = []
        self.current_label = "walkie_talkie"
        self.labels = ["walkie_talkie", "screen"]
        
        # Colors for different labels
        self.colors = {
            "walkie_talkie": (0, 255, 0), # Green
            "screen": (255, 0, 0) # Red
        }
        
        print(f"✅ Found {len(all_images)} images")
        print(f"✅ Remaining to annotate: {len(self.images)} images")
    
    def load_annotation(self):
        """Load existing annotation for current image"""
        img_path = self.images[self.current_index]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        self.current_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        label = self.labels[int(class_id)]
                        
                        # Convert to pixel coordinates
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            self.current_boxes.append((x1, y1, x2, y2, label))
    
    def save_annotation(self):
        """Save annotation to YOLO format"""
        img_path = self.images[self.current_index]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        if not self.current_boxes:
            # If no boxes, delete the label file
            if label_path.exists():
                label_path.unlink()
            return
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Cannot read image: {img_path}")
            return
        
        h, w = img.shape[:2]
        
        with open(label_path, 'w') as f:
            for box in self.current_boxes:
                x1, y1, x2, y2, label = box
                
                # Convert to YOLO format
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                label_id = self.labels.index(label)
                f.write(f"{label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            x1, y1 = self.start_point
            x2, y2 = end_point
            
            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Minimum size check
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.current_boxes.append((x1, y1, x2, y2, self.current_label))
                # Auto-save after drawing
                self.save_annotation()
                print(f"✓ Box added and saved for {self.images[self.current_index].name}")
    
    def run(self):
        if not self.images:
            return
        
        print("\n" + "="*60)
        print("ANNOTATION TOOL - FIXED VERSION")
        print("="*60)
        print("CONTROLS:")
        print(" MOUSE: Click & drag to draw box")
        print(" SPACE: Save and go to next image")
        print(" S: Switch label (Green=Walkie, Blue=Screen)")
        print(" D: Delete last box")
        print(" C: Clear all boxes")
        print(" N: Next image (with save)")
        print(" P: Previous image (with save)")
        print(" Q: Quit")
        print("="*60)
        
        cv2.namedWindow("Annotation Tool", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotation Tool", self.mouse_callback)
        
        self.load_annotation()
        
        while True:
            img_path = self.images[self.current_index]
            image = cv2.imread(str(img_path))
            
            if image is None:
                print(f"❌ Cannot read: {img_path}")
                break
            
            image_copy = image.copy()
            
            # Draw existing boxes
            for box in self.current_boxes:
                x1, y1, x2, y2, label = box
                color = self.colors.get(label, (0, 255, 0))
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_copy, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display info
            status = f"Image {self.current_index+1}/{len(self.images)}: {img_path.name}"
            label_status = f"Label: {self.current_label} (Press 'S' to change)"
            box_status = f"Boxes: {len(self.current_boxes)}"
            
            cv2.putText(image_copy, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_copy, label_status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[self.current_label], 2)
            cv2.putText(image_copy, box_status, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Progress bar
            progress = (self.current_index + 1) / len(self.images)
            bar_width = 400
            bar_filled = int(progress * bar_width)
            cv2.rectangle(image_copy, (10, 120), (10 + bar_width, 130), (100, 100, 100), -1)
            cv2.rectangle(image_copy, (10, 120), (10 + bar_filled, 130), (0, 255, 0), -1)
            cv2.putText(image_copy, f"{progress*100:.1f}%", (10 + bar_width + 10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Annotation Tool", image_copy)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): # Quit
                self.save_annotation()
                break
            elif key == ord('n') or key == ord(' '): # Next image (SPACE or N)
                self.save_annotation()
                self.current_index = min(self.current_index + 1, len(self.images)-1)
                self.load_annotation()
            elif key == ord('p'): # Previous image
                self.save_annotation()
                self.current_index = max(self.current_index - 1, 0)
                self.load_annotation()
            elif key == ord('s'): # Switch label
                idx = self.labels.index(self.current_label)
                self.current_label = self.labels[(idx + 1) % len(self.labels)]
                print(f"✓ Switched to: {self.current_label}")
            elif key == ord('d'): # Delete last box
                if self.current_boxes:
                    self.current_boxes.pop()
                    self.save_annotation()
                    print(f"✓ Box deleted")
            elif key == ord('c'): # Clear all boxes
                self.current_boxes.clear()
                self.save_annotation()
                print(f"✓ All boxes cleared")
            elif key == 27: # ESC
                self.save_annotation()
                break
        
        cv2.destroyAllWindows()
        
        # Count saved annotations
        saved_count = len(list(self.label_dir.glob("*.txt")))
        print(f"\n✅ Annotation complete!")
        print(f" Saved {saved_count} annotations")
        print(f" Images: {len(self.images)}")
        print(f" Annotated: {saved_count}/{len(self.images)}")
        
        if saved_count < len(self.images):
            print(f"⚠️ Warning: {len(self.images) - saved_count} images not annotated!")

if __name__ == "__main__":
    # Check if images exist
    image_dir = "data/raw_images"
    
    if not os.path.exists(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        print(f" Run: python src/data_collection.py first!")
        exit(1)
    
    tool = FixedAnnotationTool(image_dir)
    tool.run()