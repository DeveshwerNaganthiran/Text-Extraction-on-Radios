from __future__ import annotations

import argparse
import os
import sys
import time
import unicodedata
import re
from pathlib import Path
import threading

import cv2
import numpy as np
import yaml
import json
import pandas as pd

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fast_detector import FastDetector
from src.msi_genai_ocr import MSIGenAIOCR

# ---------------------------------------------------------------------------
# Strict Error Detection Methods Ported from main_msi_genai.py
# ---------------------------------------------------------------------------

def _strict_count_vertical_separators(screen_roi: np.ndarray) -> int:
    try:
        if screen_roi is None or getattr(screen_roi, "size", 0) == 0:
            return 0
        h, w = screen_roi.shape[:2]
        if h < 20 or w < 30:
            return 0

        y0 = int(max(0, h * 0.68))
        y1 = int(min(h, h * 0.98))
        roi = screen_roi[y0:y1, :]
        if roi is None or getattr(roi, "size", 0) == 0:
            return 0
        hh, ww_img = roi.shape[:2]
        if hh < 10 or ww_img < 30:
            return 0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass
        try:
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
        except Exception:
            blur = gray

        try:
            thr_hi = int(max(160, min(245, np.percentile(blur, 85))))
        except Exception:
            thr_hi = 200
        _t, bw_bright = cv2.threshold(blur, int(thr_hi), 255, cv2.THRESH_BINARY)

        try:
            thr_lo = int(min(120, max(20, np.percentile(blur, 10))))
        except Exception:
            thr_lo = 60
        _t, bw_dark = cv2.threshold(blur, int(thr_lo), 255, cv2.THRESH_BINARY_INV)

        try:
            bw_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -10)
        except Exception:
            bw_adapt = bw_bright

        try:
            bw = cv2.max(cv2.max(bw_bright, bw_dark), bw_adapt)
        except Exception:
            bw = bw_bright

        k_h = max(8, int(hh * 0.55))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(k_h)))
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

        try:
            contours, _hier = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            _res = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = _res[0] if _res else []

        xs_w = []
        for c in contours or []:
            x, y, ww, hh0 = cv2.boundingRect(c)
            if hh0 < int(hh * 0.55): continue
            if ww > max(10, int(ww_img * 0.05)): continue
            if int(hh0) < int(ww) * 6: continue
            xc = int(x + (ww // 2))
            if xc < int(ww_img * 0.08) or xc > int(ww_img * 0.92): continue
            xs_w.append((int(xc), int(max(1, ww))))

        if not xs_w:
            try:
                col = np.mean((bw_bright.astype(np.uint8) > 0).astype(np.float32), axis=0)
                win = max(3, int(ww_img * 0.01) | 1)
                kernel_1d = np.ones((int(win),), dtype=np.float32) / float(max(1, int(win)))
                col_s = np.convolve(col, kernel_1d, mode="same")
                thr = float(max(0.55, min(0.90, np.percentile(col_s, 98) * 0.85)))
                mask = col_s >= thr
                groups, start = [], None
                for xi, on in enumerate(mask.tolist()):
                    if on and start is None: start = int(xi)
                    elif (not on) and start is not None:
                        groups.append((int(start), int(xi - 1)))
                        start = None
                if start is not None: groups.append((int(start), int(len(mask) - 1)))

                max_w = max(2, int(ww_img * 0.020))
                for a, b in groups:
                    gw = int(b - a + 1)
                    if gw > int(max_w): continue
                    xc = int((int(a) + int(b)) // 2)
                    if xc < int(ww_img * 0.08) or xc > int(ww_img * 0.92): continue
                    xs_w.append((int(xc), int(gw)))
            except Exception:
                pass

        if not xs_w:
            return 0

        xs_w.sort(key=lambda t: t[0])
        merged = []
        min_gap = max(6, int(ww_img * 0.035))
        for x, wline in xs_w:
            if not merged:
                merged.append([int(x), int(wline)])
                continue
            prev_x, prev_w = merged[-1]
            gap = max(int(min_gap), int(prev_w) + int(wline))
            if abs(int(x) - int(prev_x)) <= int(gap):
                merged[-1][0] = int((int(prev_x) + int(x)) // 2)
                merged[-1][1] = int(max(int(prev_w), int(wline)))
            else:
                merged.append([int(x), int(wline)])

        return int(len(merged))
    except Exception:
        return 0

def _strict_separator_bridge_error(screen_roi: np.ndarray) -> bool:
    try:
        if screen_roi is None or getattr(screen_roi, "size", 0) == 0: return False
        h, w = screen_roi.shape[:2]
        if h < 20 or w < 20: return False

        y0 = int(max(0, h * 0.68))
        y1 = int(min(h, h * 0.98))
        roi = screen_roi[y0:y1, :]
        if roi is None or getattr(roi, "size", 0) == 0: return False
        hh, ww_img = roi.shape[:2]
        if hh < 10 or ww_img < 20: return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        try:
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
        except Exception:
            blur = gray

        try:
            thr_hi = int(max(160, min(245, np.percentile(blur, 85))))
        except Exception:
            thr_hi = 200
        _t, bw_bright = cv2.threshold(blur, int(thr_hi), 255, cv2.THRESH_BINARY)
        
        k_h = max(8, int(hh * 0.55))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(k_h)))
        vert = cv2.morphologyEx(bw_bright, cv2.MORPH_OPEN, kernel, iterations=1)
        try:
            contours, _hier = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            _res = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = _res[0] if _res else []

        xs_w = []
        for c in contours or []:
            x0, y0_, ww, hh0 = cv2.boundingRect(c)
            if hh0 < int(hh * 0.75): continue
            if int(y0_) > int(hh * 0.20): continue
            if ww > max(6, int(ww_img * 0.03)): continue
            xc = int(x0 + (ww // 2))
            if xc < int(ww_img * 0.12) or xc > int(ww_img * 0.88): continue
            xs_w.append((int(xc), int(max(1, ww))))

        if not xs_w: return False
        xs_w.sort(key=lambda t: t[0])
        merged = []
        min_gap = max(6, int(ww_img * 0.035))
        for x, wline in xs_w:
            if not merged:
                merged.append([int(x), int(wline)])
                continue
            prev_x, prev_w = merged[-1]
            gap = max(int(min_gap), int(prev_w) + int(wline))
            if abs(int(x) - int(prev_x)) <= int(gap):
                merged[-1][0] = int((int(prev_x) + int(x)) // 2)
                merged[-1][1] = int(max(int(prev_w), int(wline)))
            else:
                merged.append([int(x), int(wline)])

        y_top = int(max(0, hh * 0.10))
        y_bot = int(min(hh, hh * 0.95))
        band_half = max(2, int(ww_img * 0.006))
        for x, _wline in merged:
            x1 = max(0, int(x) - band_half)
            x2 = min(ww_img, int(x) + band_half + 1)
            if x2 <= x1: continue
            band = gray[y_top:y_bot, x1:x2]
            if band.size == 0: continue
            dark = float(np.mean((band.astype(np.uint8) < 90).astype(np.float32)))
            if dark > 0.08: return True
        return False
    except Exception:
        return False

def _strict_mixed_script_merge_tokens(text_block: str, top_k: int = 3) -> list[str]:
    try:
        s = str(text_block or "")
        if not s: return []
        toks = re.split(r"[\s\|]+", s)
        out, seen = [], set()
        k = max(1, int(top_k or 1))
        for tok in toks:
            t = (tok or "").strip().strip("'\"`.,;:!?()[]{}<>")
            if not t: continue
            if not (re.search(r"[A-Za-z]", t) and re.search(r"[\uAC00-\uD7A3]", t)): continue
            key = t.lower()
            if key in seen: continue
            seen.add(key)
            out.append(t)
            if len(out) >= k: break
        return out
    except Exception: return []

def _strict_guess_overlap_from_text_no_sep(text_block: str, top_k: int = 2) -> list[str]:
    try:
        s = str(text_block or "").strip()
        if not s: return []
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if not lines: return []

        def _is_kr(ch: str) -> bool:
            return 0xAC00 <= ord(ch) <= 0xD7A3 if len(ch) else False

        def _boundary_snippet(t: str) -> str:
            txt, best = str(t or ""), ""
            for i in range(1, len(txt)):
                a, b = txt[i - 1], txt[i]
                if _is_kr(a) != _is_kr(b):
                    lo, hi = max(0, i - 6), min(len(txt), i + 6)
                    cand = txt[lo:hi].strip().strip("'\"`.,;:!?()[]{}<>")
                    if len(cand) > len(best): best = cand
            try:
                if " " in best: best = max([p for p in str(best).split() if p], key=len)
            except Exception: pass
            return str(best)

        out, seen, k = [], set(), max(1, int(top_k or 1))
        for m in _strict_mixed_script_merge_tokens(s, top_k=k):
            key = str(m).lower()
            if key in seen: continue
            seen.add(key)
            out.append(str(m))
            if len(out) >= k: return out

        lines.sort(key=len, reverse=True)
        for ln in lines[:4]:
            sn = _boundary_snippet(ln)
            if not sn: continue
            key = sn.lower()
            if key in seen: continue
            seen.add(key)
            out.append(sn)
            if len(out) >= k: break
        return out
    except Exception: return []

def _strict_pick_overlap_tokens_from_line(line: str, top_k: int = 3) -> list[str]:
    try:
        ln = str(line or "").strip()
        if not ln: return []
        out, seen = [], set()

        def _is_mixed_token(tok: str) -> bool:
            t = str(tok or "")
            if not t: return False
            return bool(re.search(r"[A-Za-z]", t) and re.search(r"[\uAC00-\uD7A3]", t))

        toks_all = [t for t in re.split(r"[\s\|]+", ln) if t and str(t).strip()]
        cleaned = []
        for t in toks_all:
            tt = str(t).strip().strip("'\"`.,;:!?()[]{}<>")
            if len(tt) < 2: continue
            if _is_mixed_token(tt): cleaned.append(tt)

        cleaned.sort(key=lambda x: (len(x), x), reverse=True)
        k = max(1, int(top_k or 1))
        for tt in cleaned:
            key = tt.lower()
            if key in seen: continue
            seen.add(key)
            out.append(tt)
            if len(out) >= k: break
        return out
    except Exception: return []

def _strict_guess_overlap_from_missing_sep(line: str, expected_cols: int, top_k: int = 1) -> list[str]:
    try:
        ln = str(line or "").strip()
        if not ln or "|" not in ln: return []
        parts = [p for p in (x.strip() for x in ln.split("|")) if p]
        if expected_cols and len(parts) >= int(expected_cols): return []

        def _boundary_snippet(s: str) -> str:
            t, best = str(s or ""), ""
            for i in range(1, len(t)):
                a, b = t[i - 1], t[i]
                if (0xAC00 <= ord(a) <= 0xD7A3) != (0xAC00 <= ord(b) <= 0xD7A3):
                    lo, hi = max(0, i - 6), min(len(t), i + 6)
                    cand = t[lo:hi].strip()
                    if len(cand) > len(best): best = cand
            best = best.strip().strip("'\"`.,;:!?()[]{}<>")
            try:
                if " " in best: best = max([p for p in best.split() if p], key=len)
            except Exception: pass
            return best

        scored = []
        for p in parts:
            pp = str(p)
            snip = _boundary_snippet(pp)
            is_mixed = bool(re.search(r"[A-Za-z]", pp) and re.search(r"[\uAC00-\uD7A3]", pp))
            if not snip and not is_mixed: continue
            scored.append((len(pp) + (10 if snip else 0), snip, pp))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        out, seen, k = [], set(), max(1, int(top_k or 1))
        for _s, snip, pp in scored:
            cand = str(snip if snip else pp).strip().strip("'\"`.,;:!?()[]{}<>")
            if not cand: continue
            key = cand.lower()
            if key in seen: continue
            seen.add(key)
            out.append(cand)
            if len(out) >= k: break
        return out
    except Exception: return []

def _strict_pick_column_line(text_block: str) -> str:
    try:
        if not text_block: return ""
        best_ln, best_cols = "", 0
        for raw in str(text_block).splitlines():
            ln = str(raw).strip()
            if "|" not in ln: continue
            cols = len([p for p in (x.strip() for x in ln.split("|")) if p])
            if cols > best_cols:
                best_cols = cols
                best_ln = ln
        return best_ln.strip()
    except Exception: return ""

def _parse_structured_fields(block: str):
    info = {
        "error_red": False,
        "error_evidence": "",
        "error_type": "",
        "language": "",
        "original": "",
        "english": "",
    }
    if not block: return info
    lines = [ln.rstrip("\r") for ln in str(block).splitlines()]
    compact = [ln.strip() for ln in lines if ln.strip()]
    if compact and compact[0].strip().lower().startswith("error detected"):
        info["error_red"] = True
        try:
            first_raw = compact[0].strip()
            first_lower = first_raw.lower()
            if "upside" in first_lower: info["error_type"] = "upside down"
            elif "overlap" in first_lower or "bridge" in first_lower: info["error_type"] = "overlap"
            elif "misalign" in first_lower: info["error_type"] = "misalignment"
            else: info["error_type"] = "generic"
                
            ev_lines = []
            if ":" in first_raw: ev_lines.append(first_raw.split(":", 1)[-1].strip().strip('"'))
            else: ev_lines.append(first_raw)
            
            for ev_ln in compact[1:]:
                ev_low = ev_ln.lower()
                if ev_low.startswith("detected language") or ev_low.startswith("detected text"): break
                if ev_low.startswith("likely") or ev_low.startswith("token"):
                    ev_lines.append(ev_ln.strip())
            info["error_evidence"] = "\n".join(ev_lines).strip()
        except Exception: pass
    
    mode = None
    buf_orig, buf_eng = [], []
    for raw in lines:
        s = raw.strip()
        low = s.lower()
        if low.startswith("error detected") or low.startswith("likely") or low.startswith("token"): continue
        if low.startswith("detected language:"):
            info["language"] = s.split(":", 1)[-1].strip()
            mode = None; continue
        if low.startswith("detected text(original):"):
            mode = "orig"
            rest = s.split(":", 1)[-1].strip()
            if rest and rest.lower() != "detected text(original)": buf_orig.append(rest)
            continue
        if low.startswith("detected text(english translation):"):
            mode = "eng"
            rest = s.split(":", 1)[-1].strip()
            if rest and rest.lower() != "detected text(english translation)": buf_eng.append(rest)
            continue

        if mode == "orig" and s: buf_orig.append(s)
        elif mode == "eng" and s: buf_eng.append(s)

    info["original"] = "\n".join([x for x in buf_orig if x]).strip()
    info["english"] = "\n".join([x for x in buf_eng if x]).strip()
    return info

# ---------------------------------------------------------------------------
# Existing `verify_string.py` Display/Load Logic
# ---------------------------------------------------------------------------

def _ts_log(t0: float, msg: str):
    try:
        dt = time.time() - float(t0)
        print(f"[VERIFY][{dt:7.3f}s] {msg}", flush=True)
    except Exception:
        try:
            print(f"[VERIFY] {msg}", flush=True)
        except Exception:
            pass

def _font_candidates_for_text(text: str, preferred: str = "") -> list:
    s = str(text or "")
    has_hangul = any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in s)
    has_cjk = any(
        (0x3040 <= ord(ch) <= 0x30FF)
        or (0x4E00 <= ord(ch) <= 0x9FFF)
        or (0x3400 <= ord(ch) <= 0x4DBF)
        or (0xF900 <= ord(ch) <= 0xFAFF)
        for ch in s
    )

    out = []
    if preferred:
        out.append(preferred)

    if has_hangul:
        out.extend([r"C:\\Windows\\Fonts\\malgun.ttf", r"C:\\Windows\\Fonts\\gulim.ttc", r"C:\\Windows\\Fonts\\batang.ttc"])
    elif has_cjk:
        out.extend([r"C:\\Windows\\Fonts\\msyh.ttc", r"C:\\Windows\\Fonts\\simsun.ttc", r"C:\\Windows\\Fonts\\meiryo.ttc", r"C:\\Windows\\Fonts\\msgothic.ttc"])

    out.extend([r"C:\\Windows\\Fonts\\tahoma.ttf", r"C:\\Windows\\Fonts\\segoeui.ttf", r"C:\\Windows\\Fonts\\arial.ttf", r"C:\\Windows\\Fonts\\arialbd.ttf", r"C:\\Windows\\Fonts\\arialuni.ttf"])
    seen, dedup = set(), []
    for p in out:
        if not p or p in seen: continue
        seen.add(p)
        dedup.append(p)
    return dedup

def _draw_text_unicode(img_bgr: np.ndarray, text: str, org: tuple, font_scale: float, color_bgr: tuple, thickness: int = 1, font_path: str = ""):
    try:
        s = str(text or "")
        if not s: return img_bgr
        needs_unicode = any(ord(ch) > 127 for ch in s)
        if not needs_unicode or Image is None or ImageDraw is None or ImageFont is None:
            cv2.putText(img_bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), tuple(int(x) for x in color_bgr), int(thickness), cv2.LINE_AA)
            return img_bgr

        size = max(12, int(round(28 * float(font_scale))))
        candidates = _font_candidates_for_text(s, font_path)
        font = None
        for c in candidates:
            if not c: continue
            try:
                if os.path.exists(c):
                    font = ImageFont.truetype(c, size)
                    break
            except Exception:
                font = None

        if font is None:
            cv2.putText(img_bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), tuple(int(x) for x in color_bgr), int(thickness), cv2.LINE_AA)
            return img_bgr

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        x, y = int(org[0]), int(org[1])
        y = max(0, y - int(size * 0.85))
        b, g, r = [int(v) for v in color_bgr]
        draw.text((x, y), s, font=font, fill=(r, g, b))
        out_rgb = np.asarray(pil_img)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return img_bgr

def _truncate_text_to_px(text: str, max_w: int, font_scale: float) -> str:
    try:
        s = str(text or "")
        if not s: return ""
        ell = "..."
        needs_unicode = any(ord(ch) > 127 for ch in s)

        if not needs_unicode or ImageFont is None:
            (tw, _th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), 1)
            if tw <= int(max_w): return s
            lo, hi, best = 0, len(s), ""
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = s[:mid].rstrip() + ell
                (cw, _ch), _ = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), 1)
                if cw <= int(max_w): best, lo = cand, mid + 1
                else: hi = mid - 1
            return best if best else ell

        size = max(12, int(round(28 * float(font_scale))))
        font = None
        for c in _font_candidates_for_text(s, ""):
            try:
                if os.path.exists(c):
                    font = ImageFont.truetype(c, size)
                    break
            except Exception: pass

        if font is None: return s if len(s) <= 60 else (s[:57] + ell)

        def _w(t: str) -> int:
            try:
                if hasattr(font, "getlength"): return int(round(float(font.getlength(t))))
            except Exception: pass
            try:
                bbox = font.getbbox(t)
                return int(bbox[2] - bbox[0])
            except Exception: return len(t) * size

        if _w(s) <= int(max_w): return s
        lo, hi, best = 0, len(s), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = s[:mid].rstrip() + ell
            if _w(cand) <= int(max_w): best, lo = cand, mid + 1
            else: hi = mid - 1
        return best if best else ell
    except Exception:
        return str(text or "")

def _wrap_text_to_px(text: str, max_w: int, font_scale: float) -> list:
    try:
        s = str(text or "").strip()
        if not s: return [""]
        out, rest = [], s
        while rest:
            if len(rest) <= 1 or _truncate_text_to_px(rest, max_w, font_scale) == rest:
                out.append(rest); break
            lo, hi, best = 1, len(rest), 1
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = rest[:mid].rstrip()
                if not cand: hi = mid - 1; continue
                if _truncate_text_to_px(cand, max_w, font_scale) == cand: best, lo = mid, mid + 1
                else: hi = mid - 1
            seg = rest[:best].rstrip()
            if not seg: seg, best = rest[:1], 1
            out.append(seg)
            rest = rest[best:].lstrip()
        return out if out else [""]
    except Exception: return [str(text or "")] 

def _create_camera_overlay_state(cap: cv2.VideoCapture) -> dict:
    state = {"enabled": False, "drag": None, "values": {"Brightness": 0, "Sharpness": 0, "Focus": 0}, "last_applied": {}, "layout": {}}
    max_vals = {"Brightness": 255, "Sharpness": 255, "Focus": 50}
    def _safe_get(prop):
        try: return cap.get(prop)
        except Exception: return None
    for label, prop in [("Brightness", getattr(cv2, "CAP_PROP_BRIGHTNESS", None)), ("Sharpness", getattr(cv2, "CAP_PROP_SHARPNESS", None)), ("Focus", getattr(cv2, "CAP_PROP_FOCUS", None))]:
        if prop is None: continue
        v = _safe_get(prop)
        try:
            vv = 0 if v is None or (isinstance(v, float) and (v != v)) else int(round(float(v)))
            vmax = int(max_vals.get(label, 255))
            state["values"][label] = max(0, min(vmax, vv))
        except Exception:
            state["values"][label] = 0
    try: state["last_applied"] = dict(state["values"])
    except Exception: state["last_applied"] = {}
    return state

def _apply_camera_overlay_settings(cap: cv2.VideoCapture, overlay: dict) -> None:
    if cap is None or not overlay: return
    vals, last = overlay.get("values") or {}, overlay.get("last_applied") or {}
    prop_autofocus = getattr(cv2, "CAP_PROP_AUTOFOCUS", None)
    props = {"Brightness": getattr(cv2, "CAP_PROP_BRIGHTNESS", None), "Sharpness": getattr(cv2, "CAP_PROP_SHARPNESS", None), "Focus": getattr(cv2, "CAP_PROP_FOCUS", None)}
    for k, prop in props.items():
        if prop is None or k not in vals: continue
        v = vals.get(k)
        if last.get(k) == v: continue
        try:
            if k == "Focus" and prop_autofocus is not None:
                try: cap.set(prop_autofocus, 0)
                except Exception: pass
            cap.set(prop, float(v))
            overlay.setdefault("last_applied", {})[k] = v
        except Exception: pass

def _draw_camera_overlay(img_bgr: np.ndarray, overlay: dict) -> np.ndarray:
    if img_bgr is None or not overlay or not overlay.get("enabled"): return img_bgr
    h, w = img_bgr.shape[:2]
    panel_w = min(360, max(260, int(w * 0.38)))
    panel_h = 150
    x0, y0 = 10, max(10, h - panel_h - 10)
    x1, y1 = x0 + panel_w, y0 + panel_h
    out = img_bgr
    try:
        overlay_img = out.copy()
        cv2.rectangle(overlay_img, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay_img, 0.45, out, 0.55, 0, out)
        cv2.rectangle(out, (x0, y0), (x1, y1), (40, 40, 40), 1)
    except Exception: return img_bgr

    vals = overlay.get("values") or {}
    labels, max_vals = ["Brightness", "Sharpness", "Focus"], {"Brightness": 255, "Sharpness": 255, "Focus": 50}
    slider_left, slider_right, row_y = x0 + 120, x1 - 15, [y0 + 35, y0 + 75, y0 + 115]
    overlay["layout"] = {}

    try: cv2.putText(out, "Camera Settings", (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    except Exception: pass

    for i, lab in enumerate(labels):
        y, v, vmax = row_y[i], int(vals.get(lab, 0) or 0), int(max_vals.get(lab, 255))
        v = max(0, min(vmax, v))
        try: cv2.putText(out, f"{lab}: {v}", (x0 + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        except Exception: pass
        try:
            cv2.line(out, (slider_left, y), (slider_right, y), (190, 190, 190), 2)
            denom = float(vmax) if float(vmax) > 0 else 1.0
            knob_x = int(slider_left + (slider_right - slider_left) * (float(v) / denom))
            cv2.circle(out, (knob_x, y), 7, (0, 200, 255), -1)
            cv2.circle(out, (knob_x, y), 7, (0, 0, 0), 1)
        except Exception: pass
        overlay["layout"][lab] = {"x1": int(slider_left), "x2": int(slider_right), "y": int(y)}
    return out

def _attach_camera_overlay_mouse(window_name: str, overlay: dict) -> None:
    if not window_name or not overlay: return
    def _set_value(label: str, x: int) -> None:
        try:
            lay = (overlay.get("layout") or {}).get(label) or {}
            x1, x2 = int(lay.get("x1") or 0), int(lay.get("x2") or 0)
            if x2 <= x1: return
            xx, vmax = max(x1, min(x2, int(x))), float({"Brightness": 255, "Sharpness": 255, "Focus": 50}.get(label, 255))
            v = int(round((float(xx - x1) / float(x2 - x1)) * vmax))
            overlay.setdefault("values", {})[label] = max(0, min(int(vmax), v))
        except Exception: return
    def _hit_label(x: int, y: int) -> str:
        try:
            for lab, lay in (overlay.get("layout") or {}).items():
                yy, x1, x2 = int(lay.get("y") or 0), int(lay.get("x1") or 0), int(lay.get("x2") or 0)
                if x1 <= x <= x2 and (yy - 12) <= y <= (yy + 12): return str(lab)
        except Exception: return ""
        return ""
    def _on_mouse(event, x, y, flags, _userdata):
        try:
            if not overlay.get("enabled"): return
            if event == cv2.EVENT_LBUTTONDOWN:
                lab = _hit_label(int(x), int(y))
                if lab: overlay["drag"] = lab; _set_value(lab, int(x))
            elif event == cv2.EVENT_MOUSEMOVE:
                if (flags & cv2.EVENT_FLAG_LBUTTON) and overlay.get("drag"): _set_value(str(overlay.get("drag")), int(x))
            elif event == cv2.EVENT_LBUTTONUP: overlay["drag"] = None
        except Exception: return
    try: cv2.setMouseCallback(window_name, _on_mouse)
    except Exception: pass

def _apply_camera_env_tuning(cap: cv2.VideoCapture) -> None:
    try:
        if cap is None: return
        focus = (os.getenv("WALKIE_CAMERA_FOCUS", "") or "").strip()
        if focus != "" and hasattr(cv2, "CAP_PROP_FOCUS"):
            cap.set(cv2.CAP_PROP_FOCUS, float(focus))
    except Exception: pass

def _norm_col(s: str) -> str:
    v = str(s or "").strip().lower().replace("_", " ")
    v = re.sub(r"[\(\)\[\]\{\}:,;/\\\-]+", " ", v)
    v = " ".join(v.split())
    if v.startswith("string "): v = v[len("string "):].strip()
    if v.startswith("str "): v = v[len("str "):].strip()
    return v

def _norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    try: s = unicodedata.normalize("NFKC", s)
    except Exception: pass
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in (" ".join(ln.strip().split()) for ln in s.split("\n")) if ln != ""]
    return "\n".join(lines).strip()

def _jp_strip_diacritics(s: str) -> str:
    s = "" if s is None else str(s)
    try: s = unicodedata.normalize("NFKC", s)
    except Exception: pass
    try:
        decomp = unicodedata.normalize("NFD", s).replace("\u3099", "").replace("\u309A", "")
        return unicodedata.normalize("NFC", decomp)
    except Exception: return s

def _parse_structured_original(block: str) -> str:
    return _parse_structured_fields(block).get("original", "")

def _parse_structured_language(block: str) -> str:
    return _parse_structured_fields(block).get("language", "")

def _parse_structured_english(block: str) -> str:
    return _parse_structured_fields(block).get("english", "")

def load_device_profiles() -> dict:
    profiles = {}
    try:
        raw = str(os.getenv("WALKIE_DEVICE_PROFILES_JSON", "") or "").strip()
        if raw:
            obj = json.loads(raw)
            devices = obj.get("devices") if isinstance(obj, dict) else None
            if isinstance(devices, list):
                for d in devices:
                    if isinstance(d, dict) and d.get("id"): profiles[int(d["id"])] = d
        else:
            cfgp = Path(__file__).resolve().parents[1] / "configs" / "device_profiles.json"
            if cfgp.exists():
                with open(cfgp, "r", encoding="utf-8") as f:
                    obj = json.load(f) or {}
                devices = obj.get("devices") if isinstance(obj, dict) else None
                if isinstance(devices, list):
                    for d in devices:
                        if isinstance(d, dict) and d.get("id"): profiles[int(d["id"])] = d
    except Exception: pass
    return profiles

def get_device_name(profiles: dict, device_id: int) -> str:
    d = profiles.get(device_id) or {}
    name = str(d.get("name") or "").strip()
    return name if name else f"Device {device_id}"

def _show_ocr_result_window(
    roi: np.ndarray,
    original: str,
    english: str,
    language: str,
    verdict: str = "",
    expected_lines: list | None = None,
    device_name: str = "",
    error_msg: str = "",
) -> None:
    try:
        if roi is None or getattr(roi, "size", 0) == 0: return
        lang_label = (language or "").strip()
        try: parts = [p.strip().lower() for p in lang_label.split(",") if p.strip()]
        except Exception: parts = []
        english_only = bool(parts) and all(p in ["english", "en"] for p in parts)
        show_english = bool(english and english.strip()) and (not english_only)

        o_lines = [ln.strip() for ln in str(original or "").splitlines() if ln.strip()]
        e_lines = [ln.strip() for ln in str(english or "").splitlines() if ln.strip()]

        pad, gap, line_h = 22, 44, 40
        roi_h0, roi_w0 = roi.shape[:2]
        out_w = int(max(roi_w0, 1250 if show_english else 1050))
        roi_scale = 2.0 if roi_w0 < 520 else (3.0 if roi_w0 < 360 else 1.0)
        
        roi_disp = roi
        if float(roi_scale) > 1.01:
            try: roi_disp = cv2.resize(roi, (int(round(roi_w0 * roi_scale)), int(round(roi_h0 * roi_scale))), interpolation=cv2.INTER_CUBIC)
            except Exception: roi_disp = roi
        roi_h, roi_w = roi_disp.shape[:2]

        scale, head_scale, color = (0.92 if out_w >= 1050 else 0.82), (1.05 if out_w >= 1050 else 0.95), (255, 255, 255)

        if show_english:
            col_w = max(220, (out_w - (2 * pad) - gap) // 2)
            left_x, right_x = pad, pad + col_w + gap
            left_head = f"Original ({lang_label})" if lang_label else "Original"
            right_head, max_w = "English", max(60, col_w - 10)
            left_head_lines, right_head_lines = _wrap_text_to_px(left_head, max_w, head_scale), _wrap_text_to_px(right_head, max_w, head_scale)
            header_rows = max(len(left_head_lines), len(right_head_lines), 1)
            left_wrapped, right_wrapped, max_rows = [], [], 0
            for i in range(max(len(o_lines), len(e_lines), 1)):
                lw = _wrap_text_to_px(o_lines[i] if i < len(o_lines) else "", max_w, scale)
                rw = _wrap_text_to_px(e_lines[i] if i < len(e_lines) else "", max_w, scale)
                rows = max(len(lw), len(rw), 1)
                while len(lw) < rows: lw.append("")
                while len(rw) < rows: rw.append("")
                left_wrapped.extend(lw); right_wrapped.extend(rw); max_rows += rows

            text_h = pad + (header_rows * line_h) + (max_rows * line_h) + pad
            out = np.zeros((roi_h + text_h, out_w, 3), dtype=np.uint8)
            x_img = int(max(0, (out_w - roi_w) // 2))
            out[:roi_h, x_img : x_img + roi_w] = roi_disp

            y = roi_h + pad + 24
            for r in range(header_rows):
                out = _draw_text_unicode(out, left_head_lines[r] if r < len(left_head_lines) else "", (left_x, y), head_scale, color, 1)
                out = _draw_text_unicode(out, right_head_lines[r] if r < len(right_head_lines) else "", (right_x, y), head_scale, color, 1)
                y += line_h

            for idx in range(max_rows):
                l, rr = left_wrapped[idx] if idx < len(left_wrapped) else "", right_wrapped[idx] if idx < len(right_wrapped) else ""
                if l: out = _draw_text_unicode(out, l, (left_x, y), scale, color, 1)
                if rr: out = _draw_text_unicode(out, rr, (right_x, y), scale, color, 1)
                y += line_h
        else:
            max_w, head = max(80, out_w - (2 * pad) - 10), f"Original ({lang_label})" if lang_label else "Original"
            head_lines, body_wrapped = _wrap_text_to_px(head, max_w, head_scale), []
            for ln in (o_lines or [""]): body_wrapped.extend(_wrap_text_to_px(ln, max_w, scale))
            text_h = pad + (len(head_lines) * line_h) + (len(body_wrapped) * line_h) + pad
            out = np.zeros((roi_h + text_h, out_w, 3), dtype=np.uint8)
            x_img = int(max(0, (out_w - roi_w) // 2))
            out[:roi_h, x_img : x_img + roi_w] = roi_disp
            y = roi_h + pad + 24
            for hl in head_lines:
                out = _draw_text_unicode(out, hl, (pad, y), head_scale, color, 1)
                y += line_h
            for bl in body_wrapped:
                if bl: out = _draw_text_unicode(out, bl, (pad, y), scale, color, 1)
                y += line_h

        exp_draw_lines = []
        try:
            if expected_lines:
                for ln in [str(x) for x in list(expected_lines) if str(x).strip()]:
                    exp_draw_lines.extend(_wrap_text_to_px(ln, max(120, int(out_w - 40)), 0.65))
        except Exception: pass

        try:
            header_h = 100
            summary_h = 120 + (max(0, len(exp_draw_lines)) * 32) + 16
            
            # Add extra vertical space if we have a red error message to display
            if error_msg:
                summary_h += 40

            header = np.zeros((header_h, out_w, 3), dtype=np.uint8)
            cv2.putText(header, f"OCR Result - {device_name}" if device_name else "OCR Result", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

            summary = np.zeros((summary_h, out_w, 3), dtype=np.uint8)
            cv2.putText(summary, f"Detected Language: {lang_label if lang_label else 'Unknown'}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
            
            v = str(verdict or "").strip().upper()
            if v:
                vcol = (0, 255, 0) if v == "PASS" else ((0, 0, 255) if v == "FAIL" else (0, 255, 255))
                cv2.putText(summary, f"Verdict: {v}", (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 1.1, vcol, 3)

            start_y = 118
            
            # 1. Draw the Expected lines first
            if exp_draw_lines:
                for i, ln in enumerate(exp_draw_lines):
                    summary = _draw_text_unicode(summary, ln, (24, start_y + (i * 32)), 0.8, (255, 255, 255), 1)
                
                # Push the starting Y coordinate down past the expected lines
                start_y += len(exp_draw_lines) * 32

            # 2. Draw the Overlap Error message below them
            if error_msg:
                # Add a 10px padding below the expected lines
                summary = _draw_text_unicode(summary, error_msg, (24, start_y + 10), 1.1, (0, 0, 255), 3)

            disp = np.vstack([header, summary, out])
        except Exception: disp = np.vstack([header, out])

        try:
            h0, w0, min_w, min_h = disp.shape[:2], disp.shape[:2][1], 1250, 900
            if w0 > 0 and h0 > 0 and (w0 < min_w or h0 < min_h):
                s = max(1.0, min(2.2, max(float(min_w) / float(w0), float(min_h) / float(h0))))
                disp = cv2.resize(disp, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_CUBIC)
        except Exception: pass

        cv2.namedWindow("OCR Result", cv2.WINDOW_NORMAL)
        try: cv2.resizeWindow("OCR Result", int(disp.shape[1]), int(disp.shape[0]))
        except Exception: pass
        cv2.imshow("OCR Result", disp)
        
        # --- AUTOMATION UPDATE: Auto-close after 5 seconds ---
        start_time = time.time()
        while True:
            try:
                if hasattr(cv2, "getWindowProperty") and hasattr(cv2, "WND_PROP_VISIBLE") and cv2.getWindowProperty("OCR Result", cv2.WND_PROP_VISIBLE) < 1: break
            except Exception: pass
            
            # Break if a key is pressed manually
            if cv2.waitKey(50) not in [None, -1]: break
            
            # Break automatically after 5 seconds
            if time.time() - start_time > 5.0: break
        # -----------------------------------------------------
        
        try: cv2.destroyWindow("OCR Result")
        except Exception: pass
    except Exception: return

def _find_sheet(xls: pd.ExcelFile, name: str) -> str:
    target = _norm_col(name)
    for s in xls.sheet_names:
        if _norm_col(s) == target: return s
    for s in xls.sheet_names:
        if target in _norm_col(s): return s
    raise ValueError(f"Sheet '{name}' not found. Available: {xls.sheet_names}")

def _pick_language_column(df: pd.DataFrame, region: str, language: str) -> str:
    want = _norm_col(language)
    cols = { _norm_col(c): c for c in df.columns }
    synonyms = {
        "japanese": ["japanese", "ja"], "korean": ["korean", "ko"], "simplified chinese": ["simplified chinese", "chinese simplified", "zh cn", "zh-hans"],
        "traditional chinese": ["traditional chinese", "chinese traditional", "zh tw", "zh-hant"], "french": ["french", "fr"], "spanish": ["spanish", "es"],
        "german": ["german", "de"], "italian": ["italian", "it"], "polish": ["polish", "pl"], "russian": ["russian", "ru"], "turkish": ["turkish", "tr"],
        "arabic": ["arabic", "ar"], "hungarian": ["hungarian", "hu"], "hebrew": ["hebrew", "iw", "he"], "czech": ["czech", "cs"], "portuguese": ["portuguese", "pt"],
    }
    candidates = [_norm_col(v) for v in synonyms.get(want, [want])] if want in synonyms else [want]
    for c in candidates:
        if c in cols: return cols[c]
    raise ValueError(f"Language column for region='{region}' language='{language}' not found. Columns: {list(df.columns)}")

def load_expected(excel_path: str, region: str, language: str, index: str = "", tag: str = "") -> dict:
    if not os.path.exists(str(excel_path)): raise FileNotFoundError(excel_path)
    xls = pd.ExcelFile(str(excel_path), engine="openpyxl")
    english_sheet, category_sheet = _find_sheet(xls, "english"), _find_sheet(xls, "category")
    region_norm = _norm_col(region)
    if region_norm in ["apac"]: region_sheet = _find_sheet(xls, "apac")
    elif region_norm in ["emea"]: region_sheet = _find_sheet(xls, "emea")
    elif region_norm in ["lacr", "latam", "latam\u0026caribbean", "la cr"]: region_sheet = _find_sheet(xls, "lacr")
    elif region_norm in ["english", "en", "global"]: region_sheet = english_sheet
    else: raise ValueError("Region must be one of: english, apac, emea, lacr")

    df_en, df_cat, df_reg = pd.read_excel(xls, sheet_name=english_sheet, engine="openpyxl").copy(), pd.read_excel(xls, sheet_name=category_sheet, engine="openpyxl").copy(), pd.read_excel(xls, sheet_name=region_sheet, engine="openpyxl").copy()

    def _coerce_index(v):
        try:
            if pd.isna(v): return ""
        except Exception: pass
        s = str(v).strip()
        if s.endswith(".0"):
            try: return str(int(float(s)))
            except Exception: pass
        return s

    if "index" in [_norm_col(c) for c in df_en.columns]: df_en["__index"] = df_en[next(c for c in df_en.columns if _norm_col(c) == "index")].apply(_coerce_index)
    else: raise ValueError("English sheet missing 'index' column")
    if "index" in [_norm_col(c) for c in df_reg.columns]: df_reg["__index"] = df_reg[next(c for c in df_reg.columns if _norm_col(c) == "index")].apply(_coerce_index)
    else: raise ValueError(f"Region sheet '{region_sheet}' missing 'index' column")
    if "index" in [_norm_col(c) for c in df_cat.columns]: df_cat["__index"] = df_cat[next(c for c in df_cat.columns if _norm_col(c) == "index")].apply(_coerce_index)
    else: raise ValueError("Category sheet missing 'index' column")

    idx = str(index).strip()
    if idx and idx.endswith(".0"):
        try: idx = str(int(float(idx)))
        except Exception: pass

    def _find_tag_column(df: pd.DataFrame) -> str:
        preferred, fallback = [], []
        for c in df.columns:
            low, n = str(c or "").strip().lower(), _norm_col(c)
            if ("tag" in low and "string" in low) or n in ["string tag", "stringtag"]: preferred.append(c)
            elif n == "tag" or "tag" in low: fallback.append(c)
        return preferred[0] if preferred else (fallback[0] if fallback else "")

    idx_region = ""
    if idx:
        row_en, row_reg, row_cat = df_en[df_en["__index"] == idx], df_reg[df_reg["__index"] == idx], df_cat[df_cat["__index"] == idx]
    elif tag:
        tag_norm, en_tag_col = str(tag).strip().lower(), _find_tag_column(df_en)
        if not en_tag_col: raise ValueError("English sheet missing 'string tag' column")
        row_en_all = df_en[df_en[en_tag_col].astype(str).str.strip().str.lower() == tag_norm]
        if row_en_all.empty: raise ValueError(f"No row found for tag '{tag}' in English sheet")
        if len(row_en_all) > 1: raise ValueError(f"Multiple rows found for tag '{tag}' in English sheet. Please rerun with --index.")
        row_en = row_en_all
        idx = str(row_en.iloc[0]["__index"]).strip()
        row_cat = df_cat[df_cat["__index"] == idx]
        reg_tag_col = _find_tag_column(df_reg)
        if not reg_tag_col: row_reg = df_reg[df_reg["__index"] == "__no_match__"]
        else:
            row_reg_all = df_reg[df_reg[reg_tag_col].astype(str).str.strip().str.lower() == tag_norm]
            if len(row_reg_all) == 1: row_reg, idx_region = row_reg_all, str(row_reg_all.iloc[0].get("__index", "")).strip()
            elif len(row_reg_all) > 1: raise ValueError(f"Multiple rows found for tag '{tag}' in region sheet '{region_sheet}'. Please rerun with --index")
            else: row_reg = row_reg_all
    else: raise ValueError("Provide --index or --tag")

    if row_en.empty: raise ValueError(f"No English row found for index '{idx}'")
    en_row, reg_row, cat_row = row_en.iloc[0].to_dict(), row_reg.iloc[0].to_dict() if not row_reg.empty else {}, row_cat.iloc[0].to_dict() if not row_cat.empty else {}
    en_text_col = next((c for c in df_en.columns if _norm_col(c) in ["string (english)", "string english", "english", "string"]), None)
    if en_text_col is None: raise ValueError("English sheet missing 'string (english)' column")

    expected_en = "" if pd.isna(en_row.get(en_text_col)) else str(en_row.get(en_text_col) or "")
    if _norm_col(region_sheet) == _norm_col(english_sheet): expected_local = expected_en
    else:
        lang_col = _pick_language_column(df_reg, region, language)
        expected_local = "" if pd.isna(reg_row.get(lang_col)) else str(reg_row.get(lang_col) or "")

    tag_val = next(("" if pd.isna(en_row.get(c)) else str(en_row.get(c) or "")) for c in df_en.columns if _norm_col(c) in ["string tag", "tag", "stringtag"]) if any(_norm_col(c) in ["string tag", "tag", "stringtag"] for c in df_en.columns) else ""
    cat_val = next(("" if pd.isna(en_row.get(c)) else str(en_row.get(c) or "")) for c in df_en.columns if _norm_col(c) in ["string category", "category"]) if any(_norm_col(c) in ["string category", "category"] for c in df_en.columns) else ""
    font_style = next(("" if pd.isna(cat_row.get(c)) else str(cat_row.get(c) or "")) for c in df_cat.columns if _norm_col(c) == "font style") if any(_norm_col(c) == "font style" for c in df_cat.columns) else ""
    font_size = next(("" if pd.isna(cat_row.get(c)) else str(cat_row.get(c) or "")) for c in df_cat.columns if _norm_col(c) == "font size") if any(_norm_col(c) == "font size" for c in df_cat.columns) else ""

    return {
        "index": idx, "index_region": idx_region, "tag": tag_val, "category": cat_val,
        "font_style": font_style, "font_size": font_size, "expected_en": expected_en, "expected_local": expected_local, "region_sheet": region_sheet,
    }

def capture_screen_roi(detector: FastDetector, camera_id: int, confidence: float, warmup_sec: float = 0.7):
    # New code using DirectShow backend
    cap = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
    
    if not cap.isOpened(): raise RuntimeError(f"Could not open camera {camera_id}")
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass
    try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception: pass
    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception: pass

    _apply_camera_env_tuning(cap)
    t_end = time.time() + float(warmup_sec or 0.0)
    last = None
    while time.time() < t_end:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0: last = frame
        time.sleep(0.03)

    ok, frame = cap.read()
    if ok and frame is not None and frame.size > 0: last = frame

    cap.release()
    if last is None: raise RuntimeError("Failed to capture frame")

    boxes, screens = detector.detect_with_screens(last, confidence)
    if not boxes: raise RuntimeError("No device detected")

    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    rois = []
    for (x1, y1, x2, y2) in boxes:
        screen_box = next((s for s in screens if s[0] >= x1 and s[1] >= y1 and s[2] <= x2 and s[3] <= y2), None)
        if screen_box is None: screen_box = (x1 + (x2 - x1) // 4, y1 + 10, x2 - (x2 - x1) // 4, y1 + (y2 - y1) // 3)
        sx1, sy1, sx2, sy2 = max(0, screen_box[0]), max(0, screen_box[1]), min(last.shape[1], screen_box[2]), min(last.shape[0], screen_box[3])
        if sx2 <= sx1 or sy2 <= sy1: continue
        rois.append(last[sy1:sy2, sx1:sx2])
        
    if not rois: raise RuntimeError("Invalid screen ROIs")
    return last, rois

def capture_screen_roi_preview(detector: FastDetector | None, camera_id: int, confidence: float = 0.25, model_path: str = "", window_name: str = "Verify Preview", profiles: dict = None):
    # New code using DirectShow backend
    cap = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
    if not cap.isOpened(): raise RuntimeError(f"Could not open camera {camera_id}")

    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass
    try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception: pass
    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception: pass

    det_holder = {"det": detector}
    if det_holder["det"] is None and model_path:
        def _load_det():
            try: det_holder["det"] = FastDetector(model_path)
            except Exception: det_holder["det"] = None
        threading.Thread(target=_load_det, daemon=True).start()

    last_frame, last_boxes, last_screens = None, [], []
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try: cv2.resizeWindow(window_name, 1280, 720)
    except Exception: pass

    overlay = _create_camera_overlay_state(cap) if cap else None
    zoom, did_tune = 1.0, False

    try:
        while True:
            if overlay is not None: _apply_camera_overlay_settings(cap, overlay)
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0: continue
            if not did_tune:
                try: _apply_camera_env_tuning(cap)
                except Exception: pass
                did_tune = True

            if zoom and float(zoom) > 1.0:
                try:
                    h, w = frame.shape[:2]
                    new_w, new_h = max(2, int(w / float(zoom))), max(2, int(h / float(zoom)))
                    x1, y1 = max(0, (w - new_w) // 2), max(0, (h - new_h) // 2)
                    frame = cv2.resize(frame[y1 : y1 + new_h, x1 : x1 + new_w], (w, h), interpolation=cv2.INTER_LINEAR)
                except Exception: pass
            last_frame = frame

            try:
                if det_holder.get("det") is not None: last_boxes, last_screens = det_holder.get("det").detect_with_screens(frame, confidence)
            except Exception: pass
            
            vis = frame.copy()
            for i, (x1, y1, x2, y2) in enumerate(sorted(last_boxes or [], key=lambda b: (b[0], b[1]))):
                dev_name = get_device_name(profiles, i + 1) if profiles else f"Device {i + 1}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, dev_name, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(vis, dev_name, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            for (sx1, sy1, sx2, sy2) in last_screens or []:
                cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (0, 0, 255), 1)

            cv2.putText(vis, "SPACE=capture  T=settings  +/-=zoom  X=cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if det_holder.get("det") is None: cv2.putText(vis, "Loading detector...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if overlay is not None: vis = _draw_camera_overlay(vis, overlay)
            
            cv2.imshow(window_name, vis)
            
            if hasattr(cv2, "getWindowProperty") and hasattr(cv2, "WND_PROP_VISIBLE") and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: raise RuntimeError("Cancelled")
            if overlay is not None and overlay.get("enabled"): _attach_camera_overlay_mouse(window_name, overlay)

            k = cv2.waitKey(1) & 0xFF
            if k in [ord('x'), ord('X')]: raise RuntimeError("Cancelled")
            if k in [ord('t'), ord('T')] and overlay is not None: overlay["enabled"], overlay["drag"] = not overlay["enabled"], None
            if k in [ord('+'), ord('=')]: zoom = min(4.0, float(zoom) + 0.1)
            if k in [ord('-'), ord('_')]: zoom = max(1.0, float(zoom) - 0.1)
            if k in [ord('z'), ord('Z')]: zoom = 1.0
            if k == ord(' ') and det_holder.get("det") is not None: break
    finally:
        cap.release()
        try: cv2.destroyWindow(window_name)
        except Exception: pass

    if last_frame is None: raise RuntimeError("Failed to capture frame")
    if not last_boxes: raise RuntimeError("No device detected. Try adjusting lighting/camera angle or lower --confidence (e.g. 0.15).")

    rois = []
    for (x1, y1, x2, y2) in sorted(last_boxes, key=lambda b: (b[0], b[1])):
        screen_box = next((s for s in last_screens if s[0] >= x1 and s[1] >= y1 and s[2] <= x2 and s[3] <= y2), None)
        if screen_box is None: screen_box = (x1 + (x2 - x1) // 4, y1 + 10, x2 - (x2 - x1) // 4, y1 + (y2 - y1) // 3)
        sx1, sy1, sx2, sy2 = max(0, screen_box[0]), max(0, screen_box[1]), min(last_frame.shape[1], screen_box[2]), min(last_frame.shape[0], screen_box[3])
        if sx2 > sx1 and sy2 > sy1: rois.append(last_frame[sy1:sy2, sx1:sx2])

    if not rois: raise RuntimeError("Invalid screen ROIs")
    return last_frame, rois

def main():
    t0_total = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--save-roi", default="")
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()
    try: args.region = _norm_col(args.region)
    except Exception: pass
    region_map = {"english": "english", "en": "english", "apac": "apac", "emea": "emea", "lacr": "lacr", "lac": "lacr", "latam": "lacr"}
    args.region = region_map.get(args.region)
    if not args.region: raise ValueError("--region must be one of: english, apac, emea, lacr")

    with open(args.config, "r", encoding="utf-8") as f: cfg = yaml.safe_load(f) or {}

    def _resolve_epoch_weights(epoch: int) -> str:
        fname = f"epoch{int(epoch)}.pt"
        for p in [Path("runs") / "detect" / "models" / "trained" / "walkie_detector" / "weights" / fname, Path("runs") / "detect" / "train" / "weights" / fname, Path("models") / "trained" / "walkie_detector" / "weights" / fname, Path("models") / "trained" / "walkie_detector" / fname]:
            if p.exists(): return str(p)
        for root in [Path("runs"), Path("models")]:
            if not root.exists(): continue
            try:
                for p in root.rglob(fname): return str(p)
            except Exception: pass
        return ""

    model_path = args.model_path or ""
    if not model_path and args.epoch is not None:
        model_path = _resolve_epoch_weights(args.epoch)
        if not model_path: raise ValueError(f"Could not find weights for epoch {args.epoch}")
    if not model_path: model_path = cfg.get("detector", {}).get("path", "")
    if not model_path: raise ValueError("Detector model path not provided")

    camera_id = args.camera_id if args.camera_id is not None else int(cfg.get("camera", {}).get("source", 0))
    confidence = float(cfg.get("detector", {}).get("confidence", 0.25))

    profiles = load_device_profiles()

    if args.preview:
        t0_preview = time.time()
        full_frame, rois = capture_screen_roi_preview(None, camera_id=camera_id, confidence=confidence, model_path=model_path, profiles=profiles)
        t1_preview = t0_cap = t1_cap = time.time()
    else:
        detector = FastDetector(model_path)
        t0_cap = time.time()
        full_frame, rois = capture_screen_roi(detector, camera_id=camera_id, confidence=confidence)
        t1_cap = time.time()

    expected = load_expected(args.excel, args.region, args.language, index=args.index, tag=args.tag)

    print("=" * 70)
    print(f"RADIO STRING VERIFICATION - DETECTED {len(rois)} DEVICES")
    print("=" * 70)
    print(f"Index: {expected['index']}")
    if expected.get("tag"): print(f"Tag: {expected['tag']}")
    print(f"Expected (English): {expected['expected_en']}")
    print(f"Expected ({args.region}/{args.language}): {expected['expected_local']}")

    ocr = MSIGenAIOCR()
    try: threading.Thread(target=lambda: ocr.get_or_init_session(), daemon=True).start()
    except Exception: pass

    total_ocr_time = 0.0
    all_results = []
    
    for idx, roi in enumerate(rois):
        device_id = idx + 1
        dev_name = get_device_name(profiles, device_id)
        
        print("\n" + "=" * 70)
        print(f"Device: {dev_name} (Extracting text...)")
        print("=" * 70)

        if args.save_roi:
            outp = Path(args.save_roi)
            new_outp = outp.with_name(f"{outp.stem}_D{device_id}{outp.suffix}")
            new_outp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(new_outp), roi)

        t0_ocr = time.time()
        text, _conf = ocr.extract_text(roi, expected_language=args.language)
        t1_ocr = time.time()
        total_ocr_time += (t1_ocr - t0_ocr)

        # -----------------------------------------------------------------
        # ERROR DETECTION WITH MSI GENAI LOGIC
        # -----------------------------------------------------------------
        parsed = _parse_structured_fields(text)
        
        parsed_doc_error = bool(parsed.get("error_red"))
        parsed_doc_type = str(parsed.get("error_type") or "").strip().lower()
        parsed_doc_evidence = str(parsed.get("error_evidence") or "").strip()

        parsed["error_red"] = False
        parsed["error_evidence"] = ""
        parsed["error_type"] = ""

        try:
            sep_count = _strict_count_vertical_separators(roi)
        except Exception:
            sep_count = 0

        try:
            bridge = bool(_strict_separator_bridge_error(roi))
        except Exception:
            bridge = False

        try:
            if int(sep_count) >= 2 and bool(bridge):
                src = parsed.get("original") or parsed.get("english") or text
                ln = _strict_pick_column_line(src)
                if not ln:
                    try:
                        candidates = [x.strip() for x in str(src or "").splitlines() if x.strip()]
                        candidates.sort(key=len, reverse=True)
                        ln = candidates[0] if candidates else ""
                    except Exception: ln = ""
                toks = _strict_pick_overlap_tokens_from_line(ln, top_k=3) if ln else []
                if ln and toks:
                    parsed["error_red"] = True
                    lines = ["Overlap (bridge)", f"Line: {ln}"]
                    for i, tok in enumerate(toks): lines.append(f"Token {i+1}: {tok}")
                    parsed["error_evidence"] = "\n".join(lines)
                    parsed["error_type"] = "overlap"
        except Exception: pass

        try:
            exp_softkeys = 0
            if profiles.get(device_id) and profiles[device_id].get("expected_softkeys"):
                exp_softkeys = int(profiles[device_id]["expected_softkeys"])
            if not exp_softkeys:
                es = os.getenv(f"WALKIE_EXPECT_SOFTKEYS_D{device_id}", "").strip()
                if es: exp_softkeys = int(es)

            exp_seps = int(exp_softkeys) - 1 if int(exp_softkeys) > 0 else 0
            if exp_seps > 0 and not bool(parsed.get("error_red")):
                got_seps = int(sep_count)
                src = parsed.get("original") or parsed.get("english") or text
                ln = _strict_pick_column_line(src)
                if ln: got_seps = max(int(got_seps), int(str(ln).count("|")))

                if int(got_seps) < int(exp_seps):
                    miss = int(exp_seps) - int(got_seps)
                    top_k = min(3, max(1, miss))
                    guesses = _strict_guess_overlap_from_missing_sep(ln, exp_softkeys, top_k=top_k) if ln else []
                    if not guesses: guesses = _strict_guess_overlap_from_text_no_sep(src, top_k=top_k)
                    if guesses:
                        parsed["error_red"] = True
                        lines = [f"Overlap: missing column (expected {int(exp_seps)} separators, got {int(got_seps)})"]
                        if ln: lines.append(f"Line: {ln}")
                        for i, g in enumerate(guesses): lines.append(f"Likely {i+1}: {g}")
                        parsed["error_evidence"] = "\n".join(lines).strip()
                        parsed["error_type"] = "overlap"
        except Exception: pass

        try:
            if not parsed.get("error_red") and int(sep_count) > 0:
                src = parsed.get("original") or parsed.get("english") or text
                mixed_list = _strict_mixed_script_merge_tokens(src, top_k=3)
                if mixed_list:
                    parsed["error_red"] = True
                    if not parsed.get("error_evidence"):
                        ln = _strict_pick_column_line(src)
                        lines = ["Overlap (mixed)"]
                        if ln: lines.append(f"Line: {ln}")
                        for i, tok in enumerate(mixed_list): lines.append(f"Token {i+1}: {tok}")
                        parsed["error_evidence"] = "\n".join(lines).strip()
                        parsed["error_type"] = "overlap"
        except Exception: pass

        final_error_red = bool(parsed.get("error_red"))
        final_error_evidence = str(parsed.get("error_evidence") or "").strip()
        final_error_type = str(parsed.get("error_type") or "").strip()

        if (not final_error_red) and parsed_doc_error:
            if parsed_doc_type in ["misalignment", "upside down", "overlap"]:
                final_error_red = True
                final_error_type = parsed_doc_type
                if not final_error_evidence: final_error_evidence = parsed_doc_evidence

        lang_detected = parsed.get("language") or ""
        orig_text = parsed.get("original") or text
        eng_text = parsed.get("english") or ""

        observed_n = _norm_text(orig_text)
        expected_local_n = _norm_text(expected["expected_local"])

        ok, warn = False, False
        if expected_local_n:
            ok = observed_n == expected_local_n or expected_local_n in observed_n
            if not ok:
                try: ok = "".join(observed_n.split()) == "".join(expected_local_n.split()) or "".join(expected_local_n.split()) in "".join(observed_n.split())
                except Exception: pass
            if not ok and args.language and str(args.language).strip().lower() in ["japanese", "ja"]:
                try: warn = _jp_strip_diacritics(observed_n) == _jp_strip_diacritics(expected_local_n)
                except Exception: pass

        verdict = "PASS" if ok else ("WARN" if warn else "FAIL")
        exp_lines = [
            f"Expected (English): {expected.get('expected_en','')}",
            f"Expected ({args.region}/{args.language}): {expected.get('expected_local','')}"
        ]
        print("-" * 70)
        print(f"Observed (normalized): {observed_n}")
        print("-" * 70)
        
        # --- Format the Error String (for ANY error) ---
        error_msg_display = ""
        if final_error_red:
            error_words = []
            for ln in final_error_evidence.splitlines():
                if ln.lower().startswith("likely") or ln.lower().startswith("token"):
                    error_words.append(ln.split(":")[-1].strip())
            
            err_type_str = str(final_error_type or "STRING").upper()
            
            # If specific tokens/words were found, display them. Otherwise, show the first line of the evidence.
            if error_words:
                words_str = ", ".join(error_words)
            else:
                words_str = final_error_evidence.splitlines()[0] if final_error_evidence else "Error detected"
            
            # Build the dynamic string: e.g. [MISALIGNMENT ERROR: Misalignment error detected]
            error_msg_display = f"[{err_type_str} ERROR: {words_str}]"
            print(error_msg_display)
        
        if ok: print("PASS")
        elif warn: print("WARN")
        else: print("FAIL")

        if not ok and not warn:
            print("Expected (normalized):")
            print(expected_local_n)

        all_results.append({
            "roi": roi, "orig_text": orig_text, "eng_text": eng_text, "lang_detected": lang_detected,
            "verdict": verdict, "exp_lines": exp_lines, "dev_name": dev_name, "error_msg": error_msg_display
        })

    for res in all_results:
        try: _show_ocr_result_window(res["roi"], res["orig_text"], res["eng_text"], res["lang_detected"], res["verdict"], expected_lines=res["exp_lines"], device_name=res["dev_name"], error_msg=res.get("error_msg", ""))
        except Exception: pass

    try:
        total_s = time.time() - t0_total
        cap_s = t1_cap - t0_cap
        if args.preview: print(f"\n[TIMING] Preview: {(t1_preview - t0_preview):.2f}s | Capture: {cap_s:.2f}s | OCR: {total_ocr_time:.2f}s | Total: {total_s:.2f}s")
        else: print(f"\n[TIMING] Capture: {cap_s:.2f}s | OCR: {total_ocr_time:.2f}s | Total: {total_s:.2f}s")
    except Exception: pass

if __name__ == "__main__":
    main()