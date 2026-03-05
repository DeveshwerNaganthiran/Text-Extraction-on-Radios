import base64
import requests
import json
import os
import uuid
import time
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class MSIGenAIOCR:
    """OCR using MSI Corporate GenAI Service"""
    
    def __init__(self):
        # Get configuration from environment or use credentials from genai_client.py
        self.host = os.getenv('MSI_HOST', "https://genai-service.stage.commandcentral.com/app-gateway/api/v2")
        self.api_key = os.getenv('MSI_API_KEY', "GTy:YsSiQSt,cxCGOLsj(ZkjCZDFTh!OkML9WrEn")
        self.user_id = os.getenv('MSI_USER_ID', 'bgvk38@motorolasolutions.com')
        self.datastore_id = os.getenv('MSI_DATASTORE_ID', "1579319e-2b48-4bad-9825-4a7dd10ac0ef")
        self.model = os.getenv('MSI_MODEL', "Claude-Sonnet-4")
        
        if not self.api_key:
            raise ValueError("MSI_API_KEY not found in environment variables")
        
        if not self.datastore_id:
            raise ValueError("MSI_DATASTORE_ID not found in environment variables")
        
        # Session-based workflow endpoints
        self.chat_url = self.host + "/chat"
        self.upload_url = self.host + "/upload"
 
        # Performance tuning
        self.http = requests.Session()
        self.max_image_dim = int(os.getenv("MSI_MAX_IMAGE_DIM", "1600"))
        self.jpeg_quality = int(os.getenv("MSI_JPEG_QUALITY", "98"))
        self.init_timeouts = [int(t) for t in os.getenv("MSI_INIT_TIMEOUTS", "60,30").split(",") if t.strip()]
        self.init_attempts = int(os.getenv("MSI_INIT_ATTEMPTS", "3"))
        self.upload_timeout = int(os.getenv("MSI_UPLOAD_TIMEOUT", "50"))
        self.prompt_timeout = int(os.getenv("MSI_PROMPT_TIMEOUT", "80"))
        self.session_ttl_sec = int(os.getenv("MSI_SESSION_TTL_SEC", "360"))
        self.prompt_mode = os.getenv("MSI_PROMPT_MODE", "short").strip().lower()  # full|short

        self._cached_session_id: Optional[str] = None
        self._cached_session_ts: float = 0.0

        try:
            sid = os.getenv("MSI_SESSION_ID", "").strip()
            sid_file = os.getenv("MSI_SESSION_ID_FILE", "").strip()
            if not sid_file:
                sid_file = str((Path(__file__).resolve().parents[1] / ".msi_genai_session"))

            if not sid and sid_file:
                try:
                    p = Path(sid_file)
                    if p.exists():
                        sid = p.read_text(encoding="utf-8").strip()
                except Exception:
                    sid = ""

            if sid:
                self._cached_session_id = sid
                self._cached_session_ts = time.time()
        except Exception:
            pass
        
        print(f"[MSI GenAI] Using: {self.model}")
        print(f"[MSI GenAI] User: {self.user_id}")
        print(f"[MSI GenAI] Datastore: {self.datastore_id[:10]}...")
        print(f"[MSI GenAI] Chat URL: {self.chat_url}")
    
    def encode_image_to_base64(self, image):
        """Convert image to base64 string"""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            img = image
            if self.max_image_dim and self.max_image_dim > 0:
                h, w = img.shape[:2]
                max_dim = max(h, w)
                if max_dim > self.max_image_dim:
                    scale = self.max_image_dim / float(max_dim)
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            success, buffer = cv2.imencode(
                '.jpg',
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
            )
            if success:
                return base64.b64encode(buffer).decode("utf-8")
            else:
                raise ValueError("Failed to encode image")
        else:
            raise TypeError("Image must be file path or numpy array")
    
    def init_session(self) -> str:
        """Initialize a chat session with retry logic and progressive timeouts"""
        headers = {
            "Content-Type": "application/json",
            "x-msi-genai-api-key": self.api_key
        }
        payload = {
            "userId": self.user_id,
            "model": self.model,
            "datastoreId": self.datastore_id,
            "prompt": "init"
        }
        
        timeouts = self.init_timeouts if self.init_timeouts else [8, 15]
        attempts = max(1, int(self.init_attempts))

        for attempt in range(1, attempts + 1):
            timeout = timeouts[min(attempt - 1, len(timeouts) - 1)]
            try:
                if attempt > 1:
                    print(f"[MSI GenAI] Retry {attempt-1}: Initializing session (timeout: {timeout}s)...")
                    time.sleep(min(attempt * 2, 10))
                
                response = self.http.post(
                    self.chat_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code >= 400:
                    if response.status_code in [502, 503, 504] and attempt < attempts:
                        print(f"[MSI GenAI] Gateway error {response.status_code} on attempt {attempt}, retrying...")
                        time.sleep(min(attempt * 3, 15))
                        continue
                    else:
                        raise RuntimeError(f"Session init failed {response.status_code}: {response.text}")
                
                response_data = response.json()
                
                if response_data.get("status") and "sessionId" in response_data:
                    session_id = response_data["sessionId"]
                    print(f"[MSI GenAI] Session initialized: {session_id}")
                    return session_id
                else:
                    raise RuntimeError(f"Invalid session response: {response_data}")
                    
            except requests.exceptions.Timeout as e:
                print(f"[MSI GenAI] Timeout on attempt {attempt} ({timeout}s)")
                if attempt == attempts:
                    raise RuntimeError(f"Timeout after {attempts} attempts: {e}")
            except requests.exceptions.ConnectionError as e:
                print(f"[MSI GenAI] Connection error on attempt {attempt}: {e}")
                if attempt == attempts:
                    raise RuntimeError(f"Connection failed after {attempts} attempts: {e}")
            except Exception as e:
                print(f"[MSI GenAI] Error on attempt {attempt}: {e}")
                if attempt == attempts:
                    raise
    
    def upload_image(self, session_id: str, image_base64: str) -> bool:
        """Upload image to session"""
        headers = {
            "x-msi-genai-api-key": self.api_key
        }
        image_bytes = base64.b64decode(image_base64)
        url = f"{self.upload_url}/{session_id}?userId={self.user_id}"
        
        try:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            response = self.http.post(
                url,
                headers=headers,
                files=files,
                timeout=self.upload_timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Upload failed {response.status_code}: {response.text}")
            
            print(f"[MSI GenAI] Image uploaded to session {session_id}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to upload image: {e}")
            raise
    
    def send_prompt(self, session_id: str, prompt: str) -> dict:
        """Send prompt to existing session"""
        headers = {
            "Content-Type": "application/json",
            "x-msi-genai-api-key": self.api_key
        }
        payload = {
            "userId": self.user_id,
            "model": self.model,
            "datastoreId": self.datastore_id,
            "sessionId": session_id,
            "prompt": prompt
        }
        
        try:
            response = self.http.post(
                self.chat_url,
                headers=headers,
                json=payload,
                timeout=self.prompt_timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Prompt send failed {response.status_code}: {response.text}")
            
            return response.json()
        except Exception as e:
            print(f"[ERROR] Failed to send prompt: {e}")
            raise

    def get_or_init_session(self) -> str:
        now = time.time()
        if self._cached_session_id and (now - self._cached_session_ts) < self.session_ttl_sec:
            return self._cached_session_id

        session_id = self.init_session()
        self._cached_session_id = session_id
        self._cached_session_ts = now
        return session_id
    
    def extract_text(
        self,
        image,
        region: Optional[tuple] = None,
        expected_language: Optional[str] = None,
    ) -> Tuple[str, float]:

        def _print_detected_line(msg: str):
            print(f"Detected: '{msg}'", flush=True)

        def _parse_structured(text: str) -> dict:
            out = {
                "upside_down_error": False,
                "upside_down_evidence": "",
                "overlap_error": False,
                "overlap_evidence": "",
                "misalignment_error": False,  
                "misalignment_evidence": "",  
                "vertical_overlap_error": False,
                "vertical_overlap_evidence": "",
                "ui_render_overlap_error": False,
                "ui_render_overlap_evidence": "",
                "language": "",
                "original": "",
                "english": "",
            }
            if not text:
                return out

            mode = None
            buf_original = []
            buf_english = []

            def _set_mode(new_mode: Optional[str]):
                nonlocal mode
                mode = new_mode

            lines = [ln.rstrip("\r") for ln in str(text).splitlines()]
            for ln in lines:
                s = ln.strip("\n")
                low = s.strip().lower()

                if low.startswith("detected language:"):
                    out["language"] = s.split(":", 1)[-1].strip()
                    _set_mode(None)
                    continue

                if low.startswith("detected languages:"):
                    out["language"] = s.split(":", 1)[-1].strip()
                    _set_mode(None)
                    continue

                if "upside down error" in low and ":" in low:
                    val = low.split(":", 1)[-1].strip()
                    out["upside_down_error"] = any(x in val for x in ["yes", "true", "1"])
                    _set_mode(None)
                    continue

                if "upside down evidence" in low and ":" in low:
                    out["upside_down_evidence"] = s.split(":", 1)[-1].strip()
                    _set_mode(None)
                    continue

                if "overlap error" in low and ":" in low:
                    val = low.split(":", 1)[-1].strip()
                    out["overlap_error"] = any(x in val for x in ["yes", "true", "1"])
                    _set_mode(None)
                    continue

                if "overlap evidence" in low and ":" in low:
                    out["overlap_evidence"] = s.split(":", 1)[-1].strip()
                    _set_mode(None)
                    continue

                if "misalignment error" in low and ":" in low:
                    val = low.split(":", 1)[-1].strip()
                    out["misalignment_error"] = any(x in val for x in ["yes", "true", "1"])
                    _set_mode(None)
                    continue

                if "misalignment evidence" in low and ":" in low:
                    out["misalignment_evidence"] = s.split(":", 1)[-1].strip()
                    _set_mode(None)
                    continue

                if "vertical overlap error" in low and ":" in low:
                    val = low.split(":", 1)[-1].strip()
                    out["vertical_overlap_error"] = any(x in val for x in ["yes", "true", "1"])
                    _set_mode(None)
                    continue

                if "vertical overlap evidence" in low and ":" in low:
                    out["vertical_overlap_evidence"] = s.split(":", 1)[-1].strip()
                    _set_mode(None)
                    continue

                if "ui render overlap error" in low and ":" in low:
                    val = low.split(":", 1)[-1].strip()
                    out["ui_render_overlap_error"] = any(x in val for x in ["yes", "true", "1"])
                    _set_mode(None)
                    continue

                if "ui render overlap evidence" in low and ":" in low:
                    out["ui_render_overlap_evidence"] = s.split(":", 1)[-1].strip()
                    _set_mode(None)
                    continue

                if low.startswith("detected text(original):"):
                    _set_mode("original")
                    try:
                        rest = s.split(":", 1)[-1].strip()
                        if rest and rest.lower() != "detected text(original)":
                            buf_original.append(rest)
                    except Exception:
                        pass
                    continue

                if low.startswith("detected text(english translation):"):
                    _set_mode("english")
                    try:
                        rest = s.split(":", 1)[-1].strip()
                        if rest and rest.lower() != "detected text(english translation)":
                            buf_english.append(rest)
                    except Exception:
                        pass
                    continue

                if low in ["<<<", ">>>"]:
                    continue

                if mode == "original":
                    if s.strip():
                        buf_original.append(s.rstrip())
                elif mode == "english":
                    if s.strip():
                        buf_english.append(s.rstrip())

            out["original"] = "\n".join(buf_original).rstrip()
            out["english"] = "\n".join(buf_english).rstrip()

            def _looks_merged_token(s: str) -> bool:
                v = (s or "").strip()
                if not v: return False
                if " " in v: return False
                import re
                if re.search(r"([A-Za-z]{2,8})\1", v): return True
                if re.search(r"[a-z][A-Z]", v): return True
                if re.search(r"[A-Za-z]{3,}\d+[A-Za-z]{2,}", v): return True
                return False

            try:
                if out["overlap_error"]:
                    ev = (out.get("overlap_evidence") or "").strip()
                    if not ev:
                        out["overlap_error"] = False
                    elif not _looks_merged_token(ev):
                        out["overlap_error"] = False
                        out["overlap_evidence"] = ""
                    else:
                        import re
                        m = re.search(r"^([A-Za-z]{2,8})\1$", ev)
                        if m:
                            tok = m.group(1)
                            orig = out.get("original") or ""
                            if re.search(rf"\b{re.escape(tok)}\s+{re.escape(tok)}\b", orig):
                                out["overlap_error"] = False
                                out["overlap_evidence"] = ""
            except Exception:
                pass

            try:
                ev = (out.get("overlap_evidence") or "").strip()
                if ev:
                    joined = (out.get("original") or "") + "\n" + (out.get("english") or "")
                    jlow = joined.lower()
                    elow = ev.lower()
                    if ("wi-fi" in elow or "wifi" in elow) and ("wi-fi" not in jlow and "wifi" not in jlow):
                        out["overlap_error"] = False
                        out["overlap_evidence"] = ""
            except Exception: pass
            return out

        def _format_structured(parsed: dict) -> str:
            lang = (parsed.get("language") or "").strip() or "Unknown"
            original = (parsed.get("original") or "").strip() or "NO_TEXT"
            english = (parsed.get("english") or "").strip() or "NO_TEXT"
            
            upside_down_error = bool(parsed.get("upside_down_error"))
            upside_down_evidence = (parsed.get("upside_down_evidence") or "").strip()
            misalignment_error = bool(parsed.get("misalignment_error"))  
            misalignment_evidence = (parsed.get("misalignment_evidence") or "").strip() 
            overlap_error = bool(parsed.get("overlap_error"))
            overlap_evidence = (parsed.get("overlap_evidence") or "").strip()

            def _is_english_only(language_field: str) -> bool:
                v = (language_field or "").strip().lower()
                if not v: return False
                parts = [p.strip() for p in v.split(",") if p.strip()]
                if not parts: return False
                def _is_eng(p: str) -> bool:
                    return p in ["english", "en", "en-us", "en-gb"] or p.startswith("english")
                return all(_is_eng(p) for p in parts)

            is_english = _is_english_only(lang)

            lines = []
            
            if upside_down_error:
                lines.append("Error Detected: Upside Down")
                if upside_down_evidence:
                    lines.append(f"Likely 1: {upside_down_evidence}")
            elif misalignment_error:
                lines.append("Error Detected: Misalignment")
                if misalignment_evidence:
                    lines.append(f"Likely 1: {misalignment_evidence}")
            elif overlap_error:
                lines.append("Error Detected: Overlap")
                if overlap_evidence:
                    lines.append(f"Likely 1: {overlap_evidence}")
                    
            lines.append(f"Detected Language: {lang}")
            lines.append("Detected Text(Original):")
            lines.append(original)
            if (not is_english) and (english not in ["", "NO_TEXT"]):
                lines.append("Detected Text(English Translation):")
                lines.append(english)
            return "\n".join(lines)

        try:
            if region:
                x1, y1, x2, y2 = region
                roi = image[y1:y2, x1:x2]
                image_to_use = roi
            else:
                image_to_use = image
            
            image_base64 = self.encode_image_to_base64(image_to_use)
            
            if self.prompt_mode == "short":
                lang_hint = ""
                try:
                    v = (expected_language or "").strip()
                    if v:
                        lang_hint = (
                            f"The expected UI language is '{v}'. "
                            "If multiple languages are present, include the expected UI language in Detected Languages. "
                            "Do NOT label the language as English unless the Original text is entirely English. "
                        )
                except Exception: pass

                softkey_hint = ""
                try:
                    exp_softkeys = str(os.getenv("WALKIE_EXPECT_SOFTKEYS", "") or "").strip()
                    exp_softkeys = int(exp_softkeys) if exp_softkeys else 0
                    if exp_softkeys > 0:
                        exp_seps = max(0, int(exp_softkeys) - 1)
                        softkey_hint = (
                            f"This device has {int(exp_softkeys)} softkey buttons. "
                            "If you see fewer columns than expected, treat it as an overlap error. "
                        )
                except Exception: pass

                prompt = (
                    "Extract all visible text from this walkie-talkie screen image. "
                    "Ignore icons/pictograms. extract TEXT ONLY. "
                    "CRITICAL: You MUST preserve exact leading spaces and indentation for every line! "
                    "CRITICAL: Preserve column separators and layout exactly. "
                    "EXTREMELY CRITICAL: Carefully inspect the text for ANY upside-down or inverted characters. "
                    "Pay special attention to Arabic text. If the letter 'ع' (Ain) or any other letter is visually flipped upside down, you MUST set Upside Down Error to YES and provide the character in the evidence. "
                    + lang_hint
                    + softkey_hint
                    +
                    "Be STRICT on layout bugs, BUT handle Bi-Directional text correctly: "
                    "If the screen contains Right-to-Left text (like Arabic) mixed with LTR (English/Numbers), the entire block is usually RIGHT-ALIGNED. "
                    "Because lines have different lengths, their LEFT starting positions will be staggered/uneven. "
                    "DO NOT flag a Misalignment Error for uneven left margins in a Right-Aligned text block! "
                    "Return EXACTLY in this format, no extra text:\n"
                    "Detected Languages: <languages>\n"
                    "Detected Text(Original):\n<<<\n<original text exactly as it appears>\n>>>\n"
                    "Detected Text(English Translation):\n<<<\n<english translation>\n>>>\n"
                    "Upside Down Error: <YES or NO>\n"
                    "Upside Down Evidence: <if YES, output ONLY the upside down character or word>\n"
                    "Overlap Error: <YES or NO>\n"
                    "Overlap Evidence: <if YES, quote the merged word>\n"
                    "Misalignment Error: <YES or NO. Answer YES if ANY line is indented or shifted away from the left or right margin compared to the other lines.>\n"
                    "Misalignment Evidence: <If YES, extract ONLY the specific line of text that is visibly indented/shifted away from the margin. Do NOT output the first or second lines if the third line is the indented one.>\n"
                    "Vertical Overlap Error: <YES or NO>\n"
                    "Vertical Overlap Evidence: <if YES, quote the stacked snippet>\n"
                    "UI Render Overlap Error: <YES or NO>\n"
                    "UI Render Overlap Evidence: <if YES, quote the overlapped snippet>\n"
                )
            else:
                prompt = """Read ALL text visible in this walkie-talkie screen image..."""
            
            try:
                now = time.time()
                if self._cached_session_id and (now - self._cached_session_ts) < self.session_ttl_sec:
                    session_id = self._cached_session_id
                else:
                    session_id = self.get_or_init_session()
                
                self.upload_image(session_id, image_base64)
                result = self.send_prompt(session_id, prompt)

            except Exception as e:
                return f"REQUEST_ERROR: {str(e)[:50]}", 0.0
            
            raw_text = self.extract_text_from_response(result)
            parsed = _parse_structured(raw_text)
            # --- FORCE CV2 PIXEL-PERFECT FALLBACK TO OVERWRITE BAD EVIDENCE ---
            # Flawless physical line segmentation using Morphological Gradients.
            # Bypasses all glare and perfectly maps indented lines to OCR text using Hard Anchors.
            try:
                if image_to_use is not None:
                    gray = cv2.cvtColor(image_to_use, cv2.COLOR_BGR2GRAY) if len(image_to_use.shape) == 3 else image_to_use
                    h_img, w_img = gray.shape
                    
                    # RELAXED CROP: cy1=0.20, cy2=0.90
                    # Expands the vertical view to catch lower menu items like "convergnti".
                    cy1, cy2 = int(h_img * 0.20), int(h_img * 0.90)
                    cx1, cx2 = int(w_img * 0.05), int(w_img * 0.95)
                    
                    if cy2 > cy1 and cx2 > cx1:
                        roi = gray[cy1:cy2, cx1:cx2]
                        
                        # 1. Morphological Gradient: Extracts text outlines perfectly regardless of lighting/glare
                        kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        grad = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, kernel_grad)
                        
                        # 2. Otsu Thresholding
                        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        
                        # 3. STRICT HORIZONTAL DILATION
                        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 2))
                        dilated = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_h)
                        
                        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        lines_bounds = []
                        for c in contours:
                            x, y, w, h = cv2.boundingRect(c)
                            # RELAXED FILTERS: Catches smaller font sizes and shorter words
                            if w > 12 and h > 6 and h < int((cy2 - cy1) * 0.5):
                                lines_bounds.append([x, y, x+w, y+h])
                                
                        # Sort by Y axis
                        lines_bounds.sort(key=lambda b: b[1])
                        
                        # 4. Group intersecting bounds into solid physical lines
                        merged_lines = []
                        for b in lines_bounds:
                            if not merged_lines:
                                merged_lines.append(b)
                            else:
                                last = merged_lines[-1]
                                y_overlap = max(0, min(last[3], b[3]) - max(last[1], b[1]))
                                min_h = min(last[3]-last[1], b[3]-b[1])
                                # Require substantial vertical overlap to merge (prevents separate lines from fusing)
                                if y_overlap > 0.4 * min_h:
                                    last[0] = min(last[0], b[0])
                                    last[1] = min(last[1], b[1])
                                    last[2] = max(last[2], b[2])
                                    last[3] = max(last[3], b[3])
                                else:
                                    merged_lines.append(b)

                        # Clean up: Remove residual WAVE header if it sneaks in at the very top
                        if len(merged_lines) >= 2 and merged_lines[0][1] < int((cy2-cy1)*0.08):
                            merged_lines.pop(0)

                        if len(merged_lines) >= 2:
                            roi_w = cx2 - cx1
                            
                            # How close a letter needs to be to "touch" the line (2% variance)
                            touch_tol = max(3, int(roi_w * 0.02))   
                            # How far away it must be to be considered "misaligned" (5% shift)
                            shift_thresh = max(5, int(roi_w * 0.05)) 
                            
                            is_rtl = any(rtl_lang in str(parsed.get("language") or "").lower() for rtl_lang in ["arabic", "hebrew", "farsi", "urdu", "persian"])
                            
                            # 1. Get the starting edge for each row (Left edge for LTR, Right edge for RTL)
                            edges = [b[2] if is_rtl else b[0] for b in merged_lines]
                            
                            # 2. Draw the "Majority Line"
                            best_baseline = edges[0]
                            max_votes = 0
                            
                            for e in edges:
                                # Count how many rows "touch" this specific line
                                votes = sum(1 for other_e in edges if abs(e - other_e) <= touch_tol)
                                if votes > max_votes:
                                    max_votes = votes
                                    best_baseline = e
                            
                            # Edge case: If there are exactly 2 lines, we assume the top line is the correct baseline
                            if len(merged_lines) == 2 and abs(edges[0] - edges[1]) > shift_thresh:
                                best_baseline = edges[0]
                            
                            # 3. Check if any minority row didn't touch the line
                            shifted_idx = -1
                            max_deviation = 0
                            
                            for i, e in enumerate(edges):
                                dev = abs(e - best_baseline)
                                if dev > shift_thresh:
                                    # It's misaligned! Grab the one that strayed the furthest.
                                    if dev > max_deviation:
                                        max_deviation = dev
                                        shifted_idx = i
                                        
                            if shifted_idx != -1:
                                parsed["misalignment_error"] = True
                                
                                # 5. Map the physically indented box back to the OCR text
                                orig_text = parsed.get("original") or ""
                                ocr_lines = [ln.rstrip() for ln in orig_text.splitlines() if ln.strip() and not ln.strip().lower().startswith("wave") and len(ln.strip()) > 1]
                                
                                if ocr_lines:
                                    # Fallback 1: Trust explicit spaces if AI captured them organically
                                    indented_idx = -1
                                    for ocr_idx, ln in enumerate(ocr_lines):
                                        if ln.startswith("  ") or ln.startswith(" \t") or ln.startswith("\t"):
                                            indented_idx = ocr_idx
                                            break
                                            
                                    if indented_idx != -1:
                                        parsed["misalignment_evidence"] = ocr_lines[indented_idx].strip()
                                    else:
                                        # Fallback 2: Intelligent anchored mapping.
                                        # Misaligned text on these UI devices is almost universally the last wrapped line.
                                        if shifted_idx == len(merged_lines) - 1:
                                            # If CV2 says the last physical line is shifted, grab the last OCR line. Perfect anchor.
                                            target_idx = len(ocr_lines) - 1
                                        else:
                                            # Otherwise, use proportional Y-center mapping
                                            y_centers = [b[1] + (b[3] - b[1]) / 2.0 for b in merged_lines]
                                            shift_y = y_centers[shifted_idx]
                                            min_y, max_y = min(y_centers), max(y_centers)
                                            
                                            if max_y > min_y:
                                                rel_y = (shift_y - min_y) / (max_y - min_y)
                                                target_idx = int(round(rel_y * (len(ocr_lines) - 1)))
                                            else:
                                                target_idx = 0
                                                
                                        target_idx = max(0, min(len(ocr_lines) - 1, target_idx))
                                        parsed["misalignment_evidence"] = ocr_lines[target_idx].strip()
            except Exception as e:
                pass
            # -------------------------------------------------------------

            try:
                exp = (expected_language or "").strip()
                if exp:
                    exp_low = exp.lower()
                    model_low = (parsed.get("language") or "").strip().lower()
                    if not (exp_low in ["english", "en"] or exp_low.startswith("english")) and (model_low in ["english", "en"] or model_low.startswith("english")):
                        parsed["language"] = exp
            except Exception:
                pass
            
            if parsed.get("language") or parsed.get("original") or parsed.get("english") or parsed.get("misalignment_error") or parsed.get("upside_down_error"):
                text = _format_structured(parsed)
            else:
                text = raw_text
            
            text = self.clean_text(text)
            text = self.fix_ocr_errors(text)

            parsed2 = _parse_structured(text)
            detected_summary = (parsed2.get("english") or "").strip() or (parsed2.get("original") or "").strip() or text
            detected_summary = " ".join([ln.strip() for ln in str(detected_summary).splitlines() if ln.strip()])
            
            _print_detected_line(detected_summary)
            return text, 0.0
            
        except Exception as e:
            return f"EXTRACTION_ERROR: {str(e)[:50]}", 0.0
    
    def extract_text_from_response(self, response: dict) -> str:
        try:
            if "data" in response and isinstance(response["data"], dict):
                data = response["data"]
                for k in ["text", "message", "response", "output"]:
                    if k in data and isinstance(data[k], str) and data[k].strip():
                        return data[k].strip()

            for k in ["message", "response", "text"]:
                if k in response and isinstance(response[k], str) and response[k].strip():
                    return response[k].strip()

            if "msg" in response and isinstance(response["msg"], str):
                return response["msg"].strip()
                
            for key, value in response.items():
                if isinstance(value, str) and len(value) > 0:
                    return value.strip()
            
            return json.dumps(response)[:200]
                
        except Exception as e:
            return ""
    
    def clean_text(self, text: str) -> str:
        if not text:
            return "NO_TEXT"
        
        prefixes = ["The text says:", "Text:", "Display shows:", "I can see:", "Here's the text:", "Extracted text:", "The screen displays:", "Walkie-talkie screen shows:"]
        
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        text = text.strip('"').strip("'").strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        
        lines = [line.rstrip() for line in text.split('\n') if line.strip()]
        filtered_lines = []
        
        for line in lines:
            if len(line.strip()) < 2:
                continue
            alnum_count = sum(1 for c in line if c.isalnum())
            num_count = sum(1 for c in line if c.isdigit())
            
            if alnum_count >= 2 or num_count >= 1:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines) if filtered_lines else "NO_TEXT"
    
    def fix_ocr_errors(self, text: str) -> str:
        if not text or text == "NO_TEXT":
            return text
        
        fixes = [
            ("ce11u10r", "cellular"), ("ce11u1ar", "cellular"), ("ce11ular", "cellular"),
            ("he11o", "hello"), ("ca11", "call"), ("te11", "tell"), ("se11", "sell"),
            ("we11", "well"), ("fi11", "fill"), ("mi11", "mill"), ("pi11", "pill"),
            ("bi11", "bill"), ("si11y", "silly"), ("c0nnect", "connect"), ("t0tal", "total"),
            ("m0de", "mode"), ("r0ad", "road"), ("5can", "Scan"), ("5ignal", "Signal"),
            ("5tatus", "Status"), ("8att", "Batt"), ("8attery", "Battery"),
        ]
        
        for wrong, correct in fixes:
            text = text.replace(wrong, correct)
            text = text.replace(wrong.upper(), correct.upper())
            text = text.replace(wrong.capitalize(), correct.capitalize())
        
        return text

    def calculate_confidence(self, text: str) -> float:
        if not text or text == "NO_TEXT" or text.startswith(("API_ERROR", "CONNECTION", "TIMEOUT", "REQUEST", "EXTRACTION")):
            return 0.0
        if not text or len(text.strip()) == 0:
            return 0.0
        
        confidence = 0.5
        text_len = len(text.strip())
        if 5 <= text_len <= 30: confidence += 0.2
        elif text_len > 30: confidence += 0.3
        else: confidence += 0.1
        
        import re
        walkie_patterns = [
            (r'(CH|CHAN|CHANNEL)[\s\-_]?\d+', 0.15), (r'\d{3}\.\d{4}', 0.25),
            (r'\d{2,3}\.\d{3,4}', 0.20), (r'(TX|RX|SCAN|MON|LTR|MDC|PWR|VOL)', 0.10),
            (r'(BATT|BATTERY|LOW|FULL)', 0.10), (r'[\d\w]+\s*[\d\w]+', 0.05),
        ]
        
        for pattern, boost in walkie_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                confidence += boost
                break 
        
        if bool(re.search(r'[A-Za-z]', text)) and bool(re.search(r'\d', text)):
            confidence += 0.15
        
        total_chars = len(text)
        if total_chars > 0:
            confidence += (sum(1 for c in text if c.isalnum() or c in '.-_ ') / total_chars) * 0.2
        
        return round(max(0.1, min(confidence, 0.95)) * 20) / 20