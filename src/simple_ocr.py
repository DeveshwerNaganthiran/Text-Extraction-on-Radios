import cv2
import numpy as np
import re
from PIL import Image
import pytesseract

class SimpleOCR:
    def __init__(self):
        """Initialize Pytesseract OCR with ROI extraction"""
        try:
            print("[Pytesseract] Loading OCR engine...")
            # Test if pytesseract works
            test_img = np.zeros((10, 10), dtype=np.uint8)
            pytesseract.image_to_string(Image.fromarray(test_img))
            print("[Pytesseract] Ready")
        except Exception as e:
            print(f"[ERROR] Pytesseract error: {e}")
            print("[ERROR] Make sure Tesseract is installed")
    
    
    def clean_text(self, text):
        """Clean and normalize OCR output"""
        if not text:
            return ""
        
        text = ' '.join(text.split())
        text = ''.join(char for char in text if char.isprintable())
        
        replacements = {
            '#': '1', 'O': '0', 'o': '0',
            'I': '1', 'l': '1', '|': '1',
            'B': '8', 'S': '5', 'G': '6', 'Z': '2',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        text = ' '.join(text.split())
        return text.strip()
    
    def validate_and_correct(self, text):
        """Light correction - preserve actual detection"""
        if not text:
            return text
        
        text = text.strip()
        text = re.sub(r'^[#]', '1', text)
        
        match = re.search(r'([0-9#{1])\s*(.+)', text)
        if match:
            prefix = match.group(1).replace('#', '1')
            rest = match.group(2)
            text = prefix + ' ' + rest
        
        text = re.sub(r'Exec[ct]?([0-9])', r'Exect\1', text, flags=re.IGNORECASE)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()

    def preprocess_screen(self, image):
        """Advanced preprocessing like Google Vision API"""
        if image is None or image.size == 0:
            return None
        
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Step 1: Denoise (Google uses bilateral filtering)
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Step 2: Enhance contrast (Google uses CLAHE)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Step 3: Adaptive brightness adjustment
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=20)
            
            # Step 4: Morphological operations (opening to remove noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
            
            # Step 5: Adaptive thresholding (like Google)
            binary = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            return binary
        except Exception as e:
            print(f"  [ERROR] Preprocessing error: {e}")
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image

    def extract_text_from_roi(self, roi_image):
        """Extract text from ROI using Pytesseract with confidence scoring"""
        if roi_image is None or roi_image.size == 0:
            return "", 0.0
        
        try:
            h, w = roi_image.shape[:2]
            
            # Only upscale if image is too small
            if min(h, w) < 80:
                scale = max(3, 300 // max(h, w))
                upscaled = cv2.resize(roi_image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                print(f"      Upscaled: {roi_image.shape} -> {upscaled.shape}")
            else:
                upscaled = roi_image
            
            # Convert to PIL Image
            if len(upscaled.shape) == 2:
                pil_img = Image.fromarray(upscaled, mode='L')
            else:
                pil_img = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
            
            # Use detailed OCR data (like Google) to get confidence scores
            try:
                # Get detailed data with confidence
                data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
                
                # Filter by confidence (>70% is good, >80% is excellent)
                confidences = []
                texts = []
                
                for i in range(len(data['text'])):
                    conf = int(data['conf'][i])
                    text = data['text'][i].strip()
                    
                    if text and conf > 0:  # 0 confidence means no detection
                        confidences.append(conf / 100.0)  # Convert to 0-1
                        texts.append(text)
                
                if texts:
                    # Average confidence from all detected words
                    text = ' '.join(texts)
                    confidence = np.mean(confidences)
                    print(f"      Detected: {len(texts)} words, avg conf={confidence:.2f}")
                else:
                    text = ""
                    confidence = 0.0
                    
            except:
                # Fallback to basic OCR if detailed data fails
                config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_. '
                text = pytesseract.image_to_string(pil_img, config=config)
                text = text.strip()
                confidence = min(0.95, 0.5 + len(text) * 0.05) if text else 0.0
            
            print(f"      OCR Result: '{text}' (len={len(text)}, conf={confidence:.2f})")
            
            return text, confidence
            
        except Exception as e:
            print(f"    [ERROR] OCR error: {e}")
            return "", 0.0

    def extract_text_accurate(self, image):
        """
        Extract text with HIGH ACCURACY - takes time but accurate
        Multiple preprocessing strategies to maximize accuracy
        """
        if image is None or image.size == 0:
            return "", 0.0
        
        best_result = None
        best_confidence = 0.0
        all_attempts = []
        
        try:
            print("\n    [Pytesseract] HIGH-ACCURACY TEXT EXTRACTION")
            print("    Multiple attempts for maximum accuracy (this takes time)...\n")
            
            # Attempt 1: Original image as-is
            print("    [Attempt 1] Original image:")
            text, conf = self.extract_text_from_roi(image)
            if text:
                cleaned = self.clean_text(text)
                corrected = self.validate_and_correct(cleaned)
                all_attempts.append((corrected, conf))
                if conf > best_confidence:
                    best_result = corrected
                    best_confidence = conf
            
            # Attempt 2: Preprocessed (enhanced contrast)
            print("    [Attempt 2] Enhanced contrast:")
            processed = self.preprocess_screen(image)
            if processed is not None:
                text, conf = self.extract_text_from_roi(processed)
                if text:
                    cleaned = self.clean_text(text)
                    corrected = self.validate_and_correct(cleaned)
                    all_attempts.append((corrected, conf))
                    if conf > best_confidence:
                        best_result = corrected
                        best_confidence = conf
            
            # Attempt 3: High contrast (2x alpha)
            print("    [Attempt 3] High contrast (2x):")
            if processed is not None:
                enhanced = cv2.convertScaleAbs(processed, alpha=2.0, beta=30)
                text, conf = self.extract_text_from_roi(enhanced)
                if text:
                    cleaned = self.clean_text(text)
                    corrected = self.validate_and_correct(cleaned)
                    all_attempts.append((corrected, conf))
                    if conf > best_confidence:
                        best_result = corrected
                        best_confidence = conf
            
            # Attempt 4: Very high contrast (3x alpha)
            print("    [Attempt 4] Very high contrast (3x):")
            if processed is not None:
                enhanced = cv2.convertScaleAbs(processed, alpha=3.0, beta=50)
                text, conf = self.extract_text_from_roi(enhanced)
                if text:
                    cleaned = self.clean_text(text)
                    corrected = self.validate_and_correct(cleaned)
                    all_attempts.append((corrected, conf))
                    if conf > best_confidence:
                        best_result = corrected
                        best_confidence = conf
            
            # Attempt 5: Inverted image (black text on white)
            print("    [Attempt 5] Inverted image:")
            if processed is not None:
                inverted = cv2.bitwise_not(processed)
                text, conf = self.extract_text_from_roi(inverted)
                if text:
                    cleaned = self.clean_text(text)
                    corrected = self.validate_and_correct(cleaned)
                    all_attempts.append((corrected, conf))
                    if conf > best_confidence:
                        best_result = corrected
                        best_confidence = conf
            
            # Attempt 6: Adaptive thresholding
            print("    [Attempt 6] Adaptive threshold:")
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
            text, conf = self.extract_text_from_roi(adaptive)
            if text:
                cleaned = self.clean_text(text)
                corrected = self.validate_and_correct(cleaned)
                all_attempts.append((corrected, conf))
                if conf > best_confidence:
                    best_result = corrected
                    best_confidence = conf
            
            # Attempt 7: Morphological opening
            print("    [Attempt 7] Morphological opening:")
            if processed is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                morph = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
                text, conf = self.extract_text_from_roi(morph)
                if text:
                    cleaned = self.clean_text(text)
                    corrected = self.validate_and_correct(cleaned)
                    all_attempts.append((corrected, conf))
                    if conf > best_confidence:
                        best_result = corrected
                        best_confidence = conf
            
            # Summary
            print("\n    [SUMMARY] All attempts:")
            for i, (attempt_text, attempt_conf) in enumerate(all_attempts, 1):
                marker = " <-- BEST" if attempt_text == best_result else ""
                print(f"      {i}. '{attempt_text}' (conf: {attempt_conf:.2f}){marker}")
            
            if best_result:
                print(f"\n    [FINAL RESULT] '{best_result}' (confidence: {best_confidence:.2f})")
                return best_result, best_confidence
            
            print("\n    [RESULT] No text detected in any attempt")
            return "", 0.0
            
        except Exception as e:
            print(f"  [ERROR] Extraction error: {e}")
            import traceback
            traceback.print_exc()
            return "", 0.0

    def extract_text_simple(self, image, confidence_threshold=0.3):
        """Simple extraction"""
        if image is None or image.size == 0:
            return "", 0.0
        
        try:
            return self.extract_text_from_roi(image)
        except Exception as e:
            print(f"  OCR error: {e}")
            return "", 0.0
