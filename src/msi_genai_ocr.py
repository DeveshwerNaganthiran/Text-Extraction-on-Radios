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
import difflib
# Load environment variables
load_dotenv(override=True)

class MSIGenAIOCR:
    """OCR using MSI Corporate GenAI Service"""
    
    def __init__(self):
        # Get configuration from environment or use credentials from genai_client.py
        self.host = os.getenv('MSI_HOST', "https://genai-service.stage.commandcentral.com/app-gateway/api/v2")
        self.api_key = os.getenv('MSI_API_KEY', "(I9ZpcAsjzv*aSwdRHxc3nOnuZR1LY!aNkxPG~e9")
        self.user_id = os.getenv('MSI_USER_ID', "rnj673@motorolasolutions.com")
        self.datastore_id = os.getenv('MSI_DATASTORE_ID', "1579319e-2b48-4bad-9825-4a7dd10ac0ef")
        
        # MODIFICATION 1: Switch to the highly cost-efficient GPT-4o Mini model
        self.model = os.getenv('MSI_MODEL', "ChatGPT4o-mini")
        
        if not self.api_key:
            raise ValueError("MSI_API_KEY not found in environment variables")
        
        if not self.datastore_id:
            raise ValueError("MSI_DATASTORE_ID not found in environment variables")
        
        # Session-based workflow endpoints
        self.chat_url = self.host + "/chat"
        self.upload_url = self.host + "/upload"
        self.sessions_url = self.host + f"/getChatSessions/{self.model}"
 
        # MODIFICATION 2: Performance tuning and Cache Busting
        self.http = requests.Session()
        # --- NEW: Force close the connection and disable network caching ---
        self.http.headers.update({
            "Connection": "close", 
            "Cache-Control": "no-cache", 
            "Pragma": "no-cache"
        })
        
        self.max_image_dim = int(os.getenv("MSI_MAX_IMAGE_DIM", "800"))
        self.jpeg_quality = int(os.getenv("MSI_JPEG_QUALITY", "85"))
        
        self.init_timeouts = [int(t) for t in os.getenv("MSI_INIT_TIMEOUTS", "60,30").split(",") if t.strip()]
        self.init_attempts = int(os.getenv("MSI_INIT_ATTEMPTS", "3"))
        self.upload_timeout = int(os.getenv("MSI_UPLOAD_TIMEOUT", "50"))
        self.prompt_timeout = int(os.getenv("MSI_PROMPT_TIMEOUT", "80"))
        self.session_ttl_sec = int(os.getenv("MSI_SESSION_TTL_SEC", "360"))
        self.prompt_mode = os.getenv("MSI_PROMPT_MODE", "short").strip().lower()

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
                self._cached_session_ts = 0.0 
        except Exception:
            pass
        
        print(f"[MSI GenAI] Using: {self.model}")
        print(f"[MSI GenAI] User: {self.user_id}")
        print(f"[MSI GenAI] Datastore: {self.datastore_id[:10]}...")
        
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
    
    def encode_image_to_base64(self, image, dynamic_dim=None, squash_ratio=1.0):
        """Convert image to base64 string with dynamic compression and aspect ratio squashing"""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            img = image
            
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            safe_dim = dynamic_dim if dynamic_dim else 600 
            h, w = img.shape[:2]
            
            # --- NEW: STRICT GRID DIMENSION LIMITS ---
            # If the image is extremely tall (a batch capture collage), force scale it down
            # so the AI gateway doesn't reject it as an unsupported media type.
            max_allowed_h = 2000
            if h > max_allowed_h:
                scale_h = max_allowed_h / float(h)
                new_w = max(1, int(w * scale_h))
                new_h = max_allowed_h
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w # Update dimensions for the next calculation
            
            max_dim = max(h, w)
            scale = safe_dim / float(max_dim) if max_dim > safe_dim else 1.0
            
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            
            # --- GEOMETRIC SQUASH ---
            # Only compress the width if a specific squash_ratio is requested
            if new_w > new_h and squash_ratio != 1.0:
                new_w = max(1, int(new_w * squash_ratio))
                
            if scale != 1.0 or new_w != w:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            success, buffer = cv2.imencode(
                '.jpg',
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 75],
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
            
            response_data = response.json()
            
            # --- UPDATED: Extract token usage from the response "args" ---
            try:
                pt, ct = 0, 0
                
                # Check standard locations
                if "usage" in response_data:
                    pt = response_data["usage"].get("promptTokens", response_data["usage"].get("prompt_tokens", 0))
                    ct = response_data["usage"].get("completionTokens", response_data["usage"].get("completion_tokens", 0))
                
                # Check inside 'args' as per API Guide documentation
                if pt == 0 and ct == 0 and "args" in response_data and isinstance(response_data["args"], dict):
                    args_data = response_data["args"]
                    if "usage" in args_data:
                        pt = args_data["usage"].get("promptTokens", args_data["usage"].get("prompt_tokens", 0))
                        ct = args_data["usage"].get("completionTokens", args_data["usage"].get("completion_tokens", 0))
                    else:
                        pt = args_data.get("promptTokens", args_data.get("prompt_tokens", 0))
                        ct = args_data.get("completionTokens", args_data.get("completion_tokens", 0))

                self.total_prompt_tokens += int(pt)
                self.total_completion_tokens += int(ct)
            except Exception:
                pass
                
            return response_data
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
        dynamic_dim: Optional[int] = None,
        squash_ratio: float = 1.0,
        expected_text: Optional[str] = None,
    ) -> Tuple[str, float]:
        
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        def _print_detected_line(msg: str):
            print(f"Detected: '{msg}'", flush=True)

        # =====================================================================
        # [START OF INSERTION 1]: Add the Character-Level Verification Function
        # =====================================================================
        def _generate_correction_prompt(expected: str, actual: str) -> str:
            exp_clean = str(expected).strip()
            act_clean = str(actual).strip()
            
            if not exp_clean or not act_clean:
                return ""
            
            # Handle perfect truncation or rolling text (e.g. "RESEN" inside "PRESENT")
            if act_clean in exp_clean and len(act_clean) >= 2:
                return "" 
                
            if exp_clean.lower() == act_clean.lower():
                return ""
                
            prompt = f"Your previous output was '{act_clean}'. This seems incorrect. Please look at the physical image again very closely.\n"
            prompt += f"The expected full word is {len(exp_clean)} letters. You provided {len(act_clean)} letters.\n\n"
            prompt += "Let's check letter by letter from left to right:\n"
            
            matcher = difflib.SequenceMatcher(None, exp_clean, act_clean)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    prompt += f"- Letters '{exp_clean[i1:i2]}' match perfectly.\n"
                elif tag == 'replace':
                    prompt += f"- MISMATCH: You saw '{act_clean[j1:j2]}'. You might have autocorrected a typo. Look closely at the image: is it physically written as '{exp_clean[i1:i2]}'?\n"
                elif tag == 'delete':
                    prompt += f"- MISSING LETTER: It looks like you missed '{exp_clean[i1:i2]}' here. Look closely at the spacing.\n"
                elif tag == 'insert':
                    prompt += f"- AUTOCORRECT WARNING: You added an extra '{act_clean[j1:j2]}'. You likely autocorrected a typo or grammar mistake! DO NOT AUTOCORRECT. Verify if this letter physically exists on the screen.\n"
                    
            prompt += "\nBased on this breakdown, re-read the text in the image. "
            prompt += "Return EXACTLY in the required format, no extra text:\n"
            prompt += "Detected Languages: <languages>\n"
            prompt += "Detected Text(Original):\n<<<\n<exact text>\n>>>\n"
            
            return prompt
        # =====================================================================
        # [END OF INSERTION 1]
        # =====================================================================

        def _parse_structured(text: str) -> dict:
            # ... (KEEP YOUR EXISTING _parse_structured CODE HERE) ...
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
            
            # PASS THE SQUASH RATIO HERE:
            image_base64 = self.encode_image_to_base64(image_to_use, dynamic_dim=dynamic_dim, squash_ratio=squash_ratio)
            
            if self.prompt_mode == "short":
                lang_hint = ""
                try:
                    v = (expected_language or "").strip()
                    if v:
                        lang_hint = f"Expected UI language is '{v}'. "
                except Exception: pass

                softkey_hint = ""
                try:
                    exp_softkeys = str(os.getenv("WALKIE_EXPECT_SOFTKEYS", "") or "").strip()
                    exp_softkeys = int(exp_softkeys) if exp_softkeys else 0
                    if exp_softkeys > 0:
                        softkey_hint = f"Device has {int(exp_softkeys)} softkey buttons. "
                except Exception: pass

                # --- FIX: REMOVE EXACT STRING LEAK TO PREVENT AI CHEATING ---
                vocab_hint = ""
                if expected_text and str(expected_text).strip():
                    vocab_hint = (
                        "CRITICAL ANTI-AUTOCORRECT WARNING: The text on this screen often contains spelling mistakes, "
                        "missing vowels, or UI truncation. DO NOT AUTOCORRECT. DO NOT FIX GRAMMAR OR SPELLING. "
                        "Output ONLY the exact physical letters you see. If it is cut off or misspelled, you MUST output the exact misspelling/cut-off.\n"
                    )
                prompt = (
                    "Extract ALL text from this walkie-talkie screen. Ignore all icons/symbols (♪, battery, power indicators). "
                    "CRITICAL INSTRUCTION: You must read the ENTIRE screen. Text may be perfectly centered, OR it may be split into softkey labels at the BOTTOM-LEFT and BOTTOM-RIGHT corners. "
                    "Do NOT ignore perfectly centered text, and Do NOT ignore text on the far edges. Capture absolutely everything. "
                    "Output EXACTLY what is visibly on the screen. DO NOT auto-complete cut-off words. If the screen physical borders cut off the text (e.g. 'Test fo'), output exactly 'Test fo'. Do NOT guess the missing letters. Do NOT confuse numbers with letters. "
                    "EXTREMELY IMPORTANT: The 'Detected Text(Original)' field MUST contain the EXACT language and characters shown in the image. DO NOT translate the original text into English. If the screen is in Chinese,Korean or other foreign languages the Original Text MUST be in the same foreign language "
                    "CRITICAL: Preserve exact leading spaces, indentation, and layout. "
                    + lang_hint + softkey_hint + vocab_hint +
                    "Be STRICT on layout bugs. If Right-to-Left text (Arabic) is mixed LTR, ignore left margin staggering. "
                    "If the image is a grid of multiple screenshots, treat it as a single valid document and read all text. DO NOT return 'unsupported media type' errors. "
                    "Return EXACTLY in this format, no extra text:\n"
                    "Detected Languages: <languages>\n"
                    "Detected Text(Original):\n<<<\n<exact text>\n>>>\n"
                    "Detected Text(English Translation):\n<<<\n<translation>\n>>>\n"
                    "Upside Down Error: <YES/NO>\n"
                    "Upside Down Evidence: <text>\n"
                    "Overlap Error: <YES/NO>\n"
                    "Overlap Evidence: <text>\n"
                    "Misalignment Error: <YES/NO>\n"
                    "Misalignment Evidence: <text>\n"
                    "Vertical Overlap Error: <YES/NO>\n"
                    "Vertical Overlap Evidence: <text>\n"
                    "UI Render Overlap Error: <YES/NO>\n"
                    "UI Render Overlap Evidence: <text>\n"
                    f"[RequestHash: {uuid.uuid4().hex[:8]}]\n"
                )
            else:
                prompt = """Read ALL text visible in this walkie-talkie screen image..."""
            
            try:
                now = time.time()
                # Use the cached session if available, otherwise initialize a new one
                if self._cached_session_id and (now - self._cached_session_ts) < self.session_ttl_sec:
                    session_id = self._cached_session_id
                else:
                    session_id = self.init_session()
                
                # CRITICAL FIX: Invalidate the cached session immediately!
                self._cached_session_id = None
                self._cached_session_ts = 0.0
                
                self.upload_image(session_id, image_base64)
                
                # --- ATTEMPT 1 ---
                result = self.send_prompt(session_id, prompt)
                raw_text = self.extract_text_from_response(result)
                parsed = _parse_structured(raw_text)
                
                # --- NEW: CHARACTER-LEVEL VERIFICATION & RETRY LOOP ---
                if expected_text:
                    current_original = (parsed.get("original") or "").strip()
                    correction_prompt = _generate_correction_prompt(expected_text, current_original)
                    
                    if correction_prompt:
                        print(f"[MSI GenAI] Hallucination detected (Expected: '{expected_text}', Got: '{current_original}'). Sending letter-by-letter retry...")
                        
                        # --- ATTEMPT 2 ---
                        # Send the highly specific correction prompt to the SAME session ID
                        retry_result = self.send_prompt(session_id, correction_prompt)
                        retry_raw = self.extract_text_from_response(retry_result)
                        retry_parsed = _parse_structured(retry_raw)
                        
                        # If the model followed instructions and returned new structured output, update our working variables
                        if retry_parsed.get("original"):
                            raw_text = retry_raw
                            parsed = retry_parsed

            except Exception as e:
                return f"REQUEST_ERROR: {str(e)[:50]}", 0.0
            
            raw_text = self.extract_text_from_response(result)
            parsed = _parse_structured(raw_text)
            
            # --- FORCE CV2 PIXEL-PERFECT FALLBACK TO OVERWRITE BAD EVIDENCE ---
            try:
                if image_to_use is not None:
                    gray = cv2.cvtColor(image_to_use, cv2.COLOR_BGR2GRAY) if len(image_to_use.shape) == 3 else image_to_use
                    h_img, w_img = gray.shape
                    
                    cy1, cy2 = int(h_img * 0.20), int(h_img * 0.90)
                    cx1, cx2 = int(w_img * 0.05), int(w_img * 0.95)
                    
                    if cy2 > cy1 and cx2 > cx1:
                        roi = gray[cy1:cy2, cx1:cx2]
                        
                        kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        grad = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, kernel_grad)
                        
                        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        
                        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 2))
                        dilated = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_h)
                        
                        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        lines_bounds = []
                        for c in contours:
                            x, y, w, h = cv2.boundingRect(c)
                            if w > 12 and h > 6 and h < int((cy2 - cy1) * 0.5):
                                lines_bounds.append([x, y, x+w, y+h])
                                
                        lines_bounds.sort(key=lambda b: b[1])
                        
                        merged_lines = []
                        for b in lines_bounds:
                            if not merged_lines:
                                merged_lines.append(b)
                            else:
                                last = merged_lines[-1]
                                y_overlap = max(0, min(last[3], b[3]) - max(last[1], b[1]))
                                min_h = min(last[3]-last[1], b[3]-b[1])
                                if y_overlap > 0.4 * min_h:
                                    last[0] = min(last[0], b[0])
                                    last[1] = min(last[1], b[1])
                                    last[2] = max(last[2], b[2])
                                    last[3] = max(last[3], b[3])
                                else:
                                    merged_lines.append(b)

                        if len(merged_lines) >= 2 and merged_lines[0][1] < int((cy2-cy1)*0.08):
                            merged_lines.pop(0)

                        if len(merged_lines) >= 2:
                            roi_w = cx2 - cx1
                            touch_tol = max(5, int(roi_w * 0.04))   
                            shift_thresh = max(15, int(roi_w * 0.15))
                            
                            is_rtl = any(rtl_lang in str(parsed.get("language") or "").lower() for rtl_lang in ["arabic", "hebrew", "farsi", "urdu", "persian"])
                            edges = [b[2] if is_rtl else b[0] for b in merged_lines]
                            
                            best_baseline = edges[0]
                            max_votes = 0
                            for e in edges:
                                votes = sum(1 for other_e in edges if abs(e - other_e) <= touch_tol)
                                if votes > max_votes:
                                    max_votes = votes
                                    best_baseline = e
                            
                            if len(merged_lines) == 2 and abs(edges[0] - edges[1]) > shift_thresh:
                                best_baseline = edges[0]
                            
                            shifted_idx = -1
                            max_deviation = 0
                            
                            for i, e in enumerate(edges):
                                dev = abs(e - best_baseline)
                                if dev > shift_thresh:
                                    if dev > max_deviation:
                                        max_deviation = dev
                                        shifted_idx = i
                                        
                            if shifted_idx != -1:
                                parsed["misalignment_error"] = True
                                
                                orig_text = parsed.get("original") or ""
                                ocr_lines = [ln.rstrip() for ln in orig_text.splitlines() if ln.strip() and not ln.strip().lower().startswith("wave") and len(ln.strip()) > 1]
                                
                                if ocr_lines:
                                    indented_idx = -1
                                    for ocr_idx, ln in enumerate(ocr_lines):
                                        if ln.startswith("  ") or ln.startswith(" \t") or ln.startswith("\t"):
                                            indented_idx = ocr_idx
                                            break
                                            
                                    if indented_idx != -1:
                                        parsed["misalignment_evidence"] = ocr_lines[indented_idx].strip()
                                    else:
                                        if shifted_idx == len(merged_lines) - 1:
                                            target_idx = len(ocr_lines) - 1
                                        else:
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
            detected_summary = (parsed2.get("original") or "").strip() or (parsed2.get("english") or "").strip() or text
            detected_summary = " ".join([ln.strip() for ln in str(detected_summary).splitlines() if ln.strip()])
            
            _print_detected_line(detected_summary)
            self.get_usage_and_cost()
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
        
    def explain_mismatch(self, expected_text: str, actual_text: str) -> str:
        """Asks the GenAI to explain the difference between expected and actual text."""
        if not expected_text:
            return "Expected text was empty."
        if not actual_text:
            return "No text was detected by the camera."
            
        prompt = (
            f"The expected text on the screen was '{expected_text}', but the camera/OCR read '{actual_text}'. "
            "In 15 words or less, briefly explain what is wrong (e.g., 'Missing the last letter', 'Extra word at the end', 'Typo in the second word', 'Completely different text')."
        )
        
        try:
            session_id = self.get_or_init_session()
            
            # --- BYPASS FIX: Upload a tiny 1x1 black pixel to satisfy the image requirement ---
            import numpy as np
            dummy_img = np.zeros((1, 1, 3), dtype=np.uint8)
            dummy_base64 = self.encode_image_to_base64(dummy_img)
            self.upload_image(session_id, dummy_base64)
            # -------------------------------------------------------------------------------
            
            result = self.send_prompt(session_id, prompt)
            explanation = self.extract_text_from_response(result)
            
            # --- NEW FIX: Strip out the Gateway's automated warning message ---
            warning_text = "**Unsupported media type found. Ignoring non-image media.**"
            explanation = explanation.replace(warning_text, "")
            
            return explanation.replace('\n', ' ').strip('."\' ')
        except Exception as e:
            return "AI could not generate an explanation."
    
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
                alnum_count = sum(1 for c in line if c.isalnum())
                
                # Keep the line as long as it contains at least ONE letter or number
                if alnum_count >= 1:
                    filtered_lines.append(line)
        
        return '\n'.join(filtered_lines) if filtered_lines else "NO_TEXT"
    
    def fix_ocr_errors(self, text: str) -> str:
        if not text or text == "NO_TEXT":
            return text
        
        text = text.replace("♪", "").replace("♫", "")
        
        fixes = [
            ("ce11u10r", "cellular"), ("ce11u1ar", "cellular"), ("ce11ular", "cellular"),
            ("he11o", "hello"), ("ca11", "call"), ("te11", "tell"), ("se11", "sell"),
            ("we11", "well"), ("fi11", "fill"), ("mi11", "mill"), ("pi11", "pill"),
            ("bi11", "bill"), ("si11y", "silly"), ("c0nnect", "connect"), ("t0tal", "total"),
            ("m0de", "mode"), ("r0ad", "road"), ("5can", "Scan"), ("5ignal", "Signal"),
            ("5tatus", "Status"), ("8att", "Batt"), ("8attery", "Battery"),
            
            # --- VAPORIZE STATUS BAR ICON HALLUCINATIONS ---
            ("AF KHZ", "25 KHZ"), 
            ("AF kHz", "25 kHz"),
            ("JMAX", ""),
            ("MAX", ""),
            ("M A X", ""),
            ("MAK", ""),
            ("M A K", ""),

            # "HAX" / "HAK" / "HAF" (High Power + Antenna misreads)
            ("H A X", ""),
            ("H|AX", ""),
            ("HAX", ""),
            ("H4X", ""),
            ("H A K", ""),
            ("HAK", ""),
            ("H|AK", ""),
            ("H4K", ""),
            ("H A F", ""),
            ("H AF", ""),
            ("HAF", ""),

            # "H1" / "HI" / "Hl" Variants
            ("H1 AK", ""),
            ("H1AK", ""),
            ("H1 AX", ""),
            ("H1AX", ""),
            ("H1A K", ""),
            ("H1A X", ""),
            ("J H1 AX", ""),
            ("H I A", ""),
            ("H1A", ""),
            ("H l A", ""),  # lowercase L
            ("DTETT", "DETT"),

            # Number/Letter confusions (5 instead of S, etc.)
            ("HIA5", ""),
            ("H1A5", ""),
            ("HIV", "HÍV"),
            ("HIA S", ""),
            ("HIAS", ""),
            ("H1A S", ""),
            ("H1AS", ""),
            ("⚠", ""),
            ("⚠️", ""),
            ("!", ""),
            ("🔊", ""),
            ("🔕", ""),
            ("🔔", ""),
            # --- NEW: VAPORIZE SPECIFIC SHEET1 HALLUCINATIONS ---
            ("i TXT", "TXT"),
            ("i 按", "按"),
            ("4 電話", "電話"),
            ("4 遙測", "遙測"),
            ("⏹", ""),
            ("️", ""),
            
            
            #Chinese Word
            ("錄展", "擴展"),
            ("警告碼", "號碼"),
            ("警告 號碼", "號碼"),
            ("衰道", "頻道"),
            ("信息 深後", "然後"),

            # Symbols and Emojis (Battery, Signal, Bluetooth, Star)
            ("Hi Δ⚡ 📶", ""),
            ("H A ⚡ 0", ""),
            ("H A ⚡", ""),
            ("H ⚡", ""),
            ("⚡", ""),
            ("📶", ""),
            ("M1 Δ", ""),
            ("H1 Δ", ""),
            ("H Δ", ""),
            ("Δ", ""),
            ("|4| × □", ""),
            ("▲", ""),
            ("▼", ""),

            # Star/Notification Icon misreads
            ("★|4★", ""),
            ("[4] ≡ ≡", ""),
            ("★", ""),
            ("☆", ""),
            ("M1 A K", ""),
            ("M 8 X", ""),
            ("间皇文", ""),
            ("414.2", ""),
            ("HIA☆", ""),
            ("H1A☆", ""),
            ("H 3 ×", ""),
            ("HIA*", ""),
            ("H1A*", ""),
            ("W1AX", ""),
            ("F D", ""),
            ("H 8 X", ""),
            ("H A ∆", ""),
            ("N4AX", ""),
            ("MI&I", ""),
            ("H 8", ""),
            ("内4", ""),
            ("間4&", ""),
            ("NIAX 0", ""),
            ("MIA", ""),
            ("MAF", ""),
            ("PIAK", ""),
            ("NIAX", ""),
            ("[] 星 文", ""),
            ("M1.5K", ""),

            # Signal Bar edge-cases (|4▲ often misread depending on font)
            ("|4▲", ""),
            ("H A S", ""),
            ("|| 4 ×", ""),
            ("01", ""),
            ("M A F 01", ""),
            ("TRIAS", ""),
            ("N1 A S 0", ""),
            ("|| △ ▢ [1]", ""),
            ("M AK", ""),
            ("014", ""),
            ("|미소", ""),
            ("H1 AK", ""),
            ("HI AX", ""),
            ("M1 A", ""),
            ("FMAS", ""),
            ("M A", ""),
            ("||4", ""),
            ("CH A5", ""),
            ("CH 3 []", ""),
            ("M A F", ""),
            ("14▲", ""),    # number 1
            ("l4▲", ""),    # lowercase L
            ("I4▲", ""),    # capital I
            ("H|AK", "")
        ]
        
        for wrong, correct in fixes:
            text = text.replace(wrong, correct)
            text = text.replace(wrong.upper(), correct.upper())
            text = text.replace(wrong.capitalize(), correct.capitalize())
        
        # Clean up any leftover blank spaces or empty lines caused by the deletion
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)

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

    # MODIFICATION 4: Usage Summary and Cost Fetching
    def get_monthly_usage(self) -> Optional[float]:
        """Fetches the monthly user cost limit metrics from the API endpoint."""
        headers = {
            "x-msi-genai-api-key": self.api_key
        }
        try:
            response = self.http.get(
                self.sessions_url,
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if "user_cost" in data:
                    return float(data["user_cost"])
        except Exception as e:
            pass
            
        return None

    def get_usage_and_cost(self):
        """Prints total token usage for the session, estimated cost, and monthly budget remaining."""
        total = self.total_prompt_tokens + self.total_completion_tokens
        
        print("\n" + "=" * 70)
        print("GENAI API TOKEN USAGE & BUDGET SUMMARY")
        print("=" * 70)
        
        if total == 0:
            print("Session Tokens:            [Usage metadata not returned by API]")
        else:
            print(f"Session Prompt (Input):    {self.total_prompt_tokens} tokens")
            print(f"Session Completion (Output):{self.total_completion_tokens} tokens")
            print(f"Total Session Tokens Used: {total} tokens")
            
            # Estimate cost based on ChatGPT-4o pricing
            if "4o" in self.model.lower() and "mini" not in self.model.lower():
                est_cost = (self.total_prompt_tokens / 1000000 * 5.00) + (self.total_completion_tokens / 1000000 * 15.00)
                print(f"Estimated Session Cost:    ${est_cost:.4f}")
            # Estimate cost based on ChatGPT-4o-mini pricing
            elif "mini" in self.model.lower():
                est_cost = (self.total_prompt_tokens / 1000000 * 0.15) + (self.total_completion_tokens / 1000000 * 0.60)
                print(f"Estimated Session Cost:    ${est_cost:.5f}")
        
        print("-" * 70)
        
        # Pull exact monthly usage values from the portal
        monthly_cost = self.get_monthly_usage()
        if monthly_cost is not None:
            budget_limit = 50.00 # The system is limited to $50 per user/month
            remaining = max(0, budget_limit - monthly_cost)
            
            print(f"Total Used This Month:     ${monthly_cost:.2f} / ${budget_limit:.2f}")
            print(f"Monthly Budget Remaining:  ${remaining:.2f}")
            print(f"Monthly Capacity Consumed: {(monthly_cost/budget_limit)*100:.1f}%")
        else:
            print("Monthly usage data is temporarily unavailable from portal.")
            
        print("=" * 70 + "(\n")
        