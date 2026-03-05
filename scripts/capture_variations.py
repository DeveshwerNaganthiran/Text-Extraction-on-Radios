import cv2
from datetime import datetime
from pathlib import Path

class ManualDataCollector:
    def __init__(self, output_dir="data/raw_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        i = 1
        while True:
            candidate = path.with_name(f"{stem}({i}){suffix}")
            if not candidate.exists():
                return candidate
            i += 1
    
    def capture_scenarios(self, camera_id=1):
        """Capture different scenarios MANUALLY - press SPACE for each shot"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        scenarios = [
            {"name": "Front View", "count": 30, "instruction": "Show front of walkie - press SPACE when ready"},
            {"name": "45 Degree Angle", "count": 20, "instruction": "Tilt 45 degrees - press SPACE when ready"},
            {"name": "Side View", "count": 20, "instruction": "Show side - press SPACE when ready"},
            {"name": "Close Up", "count": 20, "instruction": "Bring very close - press SPACE when ready"},
            {"name": "Far Away", "count": 15, "instruction": "Move far back - press SPACE when ready"},
            {"name": "Different Lighting", "count": 20, "instruction": "Change lights - press SPACE when ready"},
            {"name": "Screen Text Changes", "count": 30, "instruction": "Change screen text - press SPACE when ready"},
            {"name": "Multiple Walkies", "count": 25, "instruction": "Show 2-3 devices - press SPACE when ready"},
        ]
        
        total_count = 0
        
        for scenario in scenarios:
            print(f"\n{'='*50}")
            print(f"SCENARIO: {scenario['name']}")
            print(f"INSTRUCTION: {scenario['instruction']}")
            print(f"Take {scenario['count']} photos")
            print(f"{'='*50}")
            
            input("Press Enter when ready to start this scenario...")
            
            count_in_scenario = 0
            
            while count_in_scenario < scenario['count']:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display live feed with instructions
                display_frame = frame.copy()
                
                # Add scenario info
                cv2.putText(display_frame, f"Scenario: {scenario['name']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, scenario['instruction'], 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(display_frame, f"Photo: {count_in_scenario+1}/{scenario['count']}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "SPACE: Capture | ESC: Skip scenario | Q: Quit all", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show a rectangle to help with framing
                h, w = frame.shape[:2]
                cv2.rectangle(display_frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 255), 2)
                cv2.putText(display_frame, "Center walkie here", (w//4, h//4 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Show focus/quality indicator
                # You can add focus check here if you want
                cv2.putText(display_frame, "[Focus and press SPACE]", 
                           (w//2 - 150, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.imshow("Manual Capture - Press SPACE when ready", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27: # ESC to skip this scenario
                    print(f"Skipping {scenario['name']}")
                    break
                elif key == ord('q'): # Q to quit all
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"\nStopped early. Total captured: {total_count} images") 
                    return
                elif key == ord(' '): # SPACE to capture
                    # Save the image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    base_filename = self.output_dir / f"{scenario['name'].replace(' ', '_').lower()}_{count_in_scenario:03d}.jpg"
                    filename = self._unique_path(base_filename)
                    cv2.imwrite(str(filename), frame)
                    total_count += 1
                    count_in_scenario += 1
                    
                    # Show confirmation
                    print(f" ✓ Captured: {filename.name}")
                    
                    # Show preview for 0.5 seconds
                    cv2.putText(frame, "CAPTURED!", (w//2 - 100, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("Manual Capture - Press SPACE when ready", frame)
                    cv2.waitKey(500) # Show for 0.5 seconds
            
            print(f"Completed {scenario['name']}: {count_in_scenario}/{scenario['count']} photos")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*50}")
        print(f"✅ DATA COLLECTION COMPLETE!")
        print(f" Total captured: {total_count} images")
        print(f" Saved in: {self.output_dir}")
        print(f"{'='*50}")
    
    def check_camera_preview(self, camera_id=0):
        """Check camera view before starting"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ ERROR: Cannot open camera!")
            return False
        
        print("\n📸 Camera Preview Mode")
        print(" Adjust camera, lighting, and walkie talkie position")
        print(" Press SPACE to take a test shot")
        print(" Press Q when ready to start main capture")
        
        test_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display instructions
            h, w = frame.shape[:2]
            cv2.putText(frame, "CAMERA PREVIEW MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Adjust walkie talkie position and lighting", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, "SPACE: Test shot | Q: Start main capture | ESC: Exit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center guidelines
            cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
            cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
            
            # Draw framing rectangle
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 255), 2)
            cv2.putText(frame, "Center walkie here", (w//4, h//4 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show test count
            cv2.putText(frame, f"Test shots: {test_count}", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Camera Preview - Adjust and Press Q when ready", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '): # SPACE - test shot
                base_test_path = self.output_dir / f"test_{test_count}.jpg"
                test_path = self._unique_path(base_test_path)
                cv2.imwrite(str(test_path), frame)
                test_count += 1
                print(f" Test shot #{test_count} saved")
                
                # Show confirmation
                cv2.putText(frame, "TEST SHOT SAVED", (w//2 - 100, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Camera Preview - Adjust and Press Q when ready", frame)
                cv2.waitKey(500)
                
            elif key == ord('q'): # Q - start main capture
                cap.release()
                cv2.destroyAllWindows()
                return True
            elif key == 27: # ESC - exit
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        return False

if __name__ == "__main__":
    print("🚀 MANUAL WALKIE TALKIE DATA COLLECTION")
    print("="*60)
    print("This version lets YOU control when to take each photo.")
    print("You can focus, adjust lighting, and press SPACE when ready.")
    print("="*60)
    
    collector = ManualDataCollector()
    
    # Step 1: Camera preview and test shots
    print("\n📸 STEP 1: Camera Preview")
    print(" Adjust camera position and lighting")
    ready = collector.check_camera_preview(camera_id=1)
    
    if not ready:
        print("\n❌ Camera setup cancelled.")
        exit()
    
    # Step 2: Main capture
    print("\n📸 STEP 2: Main Capture Scenarios")
    print(" We'll go through 8 scenarios, 180+ photos total")
    print(" Press SPACE for each photo when walkie is in position")
    print(" Press ESC to skip a scenario")
    print(" Press Q to quit completely")
    
    input("\nPress Enter to begin main capture...")
    
    collector.capture_scenarios(camera_id=1)
    
    print("\n🎉 Done! Next steps:")
    print("1. Label the images: python src/annotation_tool.py")
    print("2. Split data: python scripts/split_data.py")
    print("3. Train model: python src/train_detector.py")
