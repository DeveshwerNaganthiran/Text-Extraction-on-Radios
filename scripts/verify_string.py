from __future__ import annotations

import argparse
import os
import sys
import time
import unicodedata
from pathlib import Path
import threading

import cv2
import numpy as np
import yaml

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fast_detector import FastDetector
from src.msi_genai_ocr import MSIGenAIOCR


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
        out.extend(
            [
                r"C:\\Windows\\Fonts\\malgun.ttf",
                r"C:\\Windows\\Fonts\\gulim.ttc",
                r"C:\\Windows\\Fonts\\batang.ttc",
            ]
        )
    elif has_cjk:
        out.extend(
            [
                r"C:\\Windows\\Fonts\\msyh.ttc",
                r"C:\\Windows\\Fonts\\simsun.ttc",
                r"C:\\Windows\\Fonts\\meiryo.ttc",
                r"C:\\Windows\\Fonts\\msgothic.ttc",
            ]
        )

    out.extend(
        [
            r"C:\\Windows\\Fonts\\tahoma.ttf",
            r"C:\\Windows\\Fonts\\segoeui.ttf",
            r"C:\\Windows\\Fonts\\arial.ttf",
            r"C:\\Windows\\Fonts\\arialbd.ttf",
            r"C:\\Windows\\Fonts\\arialuni.ttf",
            r"C:\\Windows\\Fonts\\times.ttf",
            r"C:\\Windows\\Fonts\\timesbd.ttf",
        ]
    )
    seen = set()
    dedup = []
    for p in out:
        if not p or p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    return dedup


def _draw_text_unicode(
    img_bgr: np.ndarray,
    text: str,
    org: tuple,
    font_scale: float,
    color_bgr: tuple,
    thickness: int = 1,
    font_path: str = "",
):
    try:
        s = str(text or "")
        if not s:
            return img_bgr
        needs_unicode = any(ord(ch) > 127 for ch in s)
        if not needs_unicode or Image is None or ImageDraw is None or ImageFont is None:
            cv2.putText(img_bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), tuple(int(x) for x in color_bgr), int(thickness), cv2.LINE_AA)
            return img_bgr

        size = max(12, int(round(28 * float(font_scale))))
        candidates = _font_candidates_for_text(s, font_path)
        font = None
        for c in candidates:
            if not c:
                continue
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
        if not s:
            return ""
        ell = "..."
        needs_unicode = any(ord(ch) > 127 for ch in s)

        if not needs_unicode or ImageFont is None:
            (tw, _th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), 1)
            if tw <= int(max_w):
                return s
            lo = 0
            hi = len(s)
            best = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = s[:mid].rstrip() + ell
                (cw, _ch), _ = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), 1)
                if cw <= int(max_w):
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best if best else ell

        size = max(12, int(round(28 * float(font_scale))))
        font = None
        for c in _font_candidates_for_text(s, ""):
            try:
                if os.path.exists(c):
                    font = ImageFont.truetype(c, size)
                    break
            except Exception:
                font = None

        if font is None:
            return s if len(s) <= 60 else (s[:57] + ell)

        def _w(t: str) -> int:
            try:
                if hasattr(font, "getlength"):
                    return int(round(float(font.getlength(t))))
            except Exception:
                pass
            try:
                bbox = font.getbbox(t)
                return int(bbox[2] - bbox[0])
            except Exception:
                return len(t) * size

        if _w(s) <= int(max_w):
            return s
        lo = 0
        hi = len(s)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = s[:mid].rstrip() + ell
            if _w(cand) <= int(max_w):
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best if best else ell
    except Exception:
        return str(text or "")


def _wrap_text_to_px(text: str, max_w: int, font_scale: float) -> list:
    try:
        s = str(text or "").strip()
        if not s:
            return [""]
        out = []
        rest = s
        while rest:
            if len(rest) <= 1:
                out.append(rest)
                break

            if _truncate_text_to_px(rest, max_w, font_scale) == rest:
                out.append(rest)
                break

            lo = 1
            hi = len(rest)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = rest[:mid].rstrip()
                if not cand:
                    hi = mid - 1
                    continue
                if _truncate_text_to_px(cand, max_w, font_scale) == cand:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            seg = rest[:best].rstrip()
            if not seg:
                seg = rest[:1]
                best = 1
            out.append(seg)
            rest = rest[best:].lstrip()
        return out if out else [""]
    except Exception:
        return [str(text or "")] 


def _create_camera_overlay_state(cap: cv2.VideoCapture) -> dict:
    state = {
        "enabled": False,
        "drag": None,
        "values": {
            "Brightness": 0,
            "Sharpness": 0,
            "Focus": 0,
        },
        "last_applied": {},
        "layout": {},
    }

    max_vals = {
        "Brightness": 255,
        "Sharpness": 255,
        "Focus": 50,
    }

    def _safe_get(prop):
        try:
            return cap.get(prop)
        except Exception:
            return None

    for label, prop in [
        ("Brightness", getattr(cv2, "CAP_PROP_BRIGHTNESS", None)),
        ("Sharpness", getattr(cv2, "CAP_PROP_SHARPNESS", None)),
        ("Focus", getattr(cv2, "CAP_PROP_FOCUS", None)),
    ]:
        if prop is None:
            continue
        v = _safe_get(prop)
        try:
            if v is None or (isinstance(v, float) and (v != v)):
                vv = 0
            else:
                vv = int(round(float(v)))
            vmax = int(max_vals.get(label, 255))
            vv = max(0, min(vmax, vv))
            state["values"][label] = vv
        except Exception:
            state["values"][label] = 0

    try:
        state["last_applied"] = dict(state["values"])
    except Exception:
        state["last_applied"] = {}
    return state


def _apply_camera_overlay_settings(cap: cv2.VideoCapture, overlay: dict) -> None:
    if cap is None or not overlay:
        return
    vals = overlay.get("values") or {}
    last = overlay.get("last_applied") or {}
    prop_autofocus = getattr(cv2, "CAP_PROP_AUTOFOCUS", None)
    props = {
        "Brightness": getattr(cv2, "CAP_PROP_BRIGHTNESS", None),
        "Sharpness": getattr(cv2, "CAP_PROP_SHARPNESS", None),
        "Focus": getattr(cv2, "CAP_PROP_FOCUS", None),
    }
    for k, prop in props.items():
        if prop is None or k not in vals:
            continue
        v = vals.get(k)
        if last.get(k) == v:
            continue
        try:
            if k == "Focus" and prop_autofocus is not None:
                try:
                    cap.set(prop_autofocus, 0)
                except Exception:
                    pass
            cap.set(prop, float(v))
            overlay.setdefault("last_applied", {})[k] = v
        except Exception:
            pass


def _draw_camera_overlay(img_bgr: np.ndarray, overlay: dict) -> np.ndarray:
    if img_bgr is None or not overlay or not overlay.get("enabled"):
        return img_bgr

    h, w = img_bgr.shape[:2]
    panel_w = min(360, max(260, int(w * 0.38)))
    panel_h = 150
    x0 = 10
    y0 = max(10, h - panel_h - 10)
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    out = img_bgr
    try:
        overlay_img = out.copy()
        cv2.rectangle(overlay_img, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay_img, 0.45, out, 0.55, 0, out)
        cv2.rectangle(out, (x0, y0), (x1, y1), (40, 40, 40), 1)
    except Exception:
        return img_bgr

    vals = overlay.get("values") or {}
    labels = ["Brightness", "Sharpness", "Focus"]
    max_vals = {
        "Brightness": 255,
        "Sharpness": 255,
        "Focus": 50,
    }
    slider_left = x0 + 120
    slider_right = x1 - 15
    row_y = [y0 + 35, y0 + 75, y0 + 115]
    overlay["layout"] = {}

    try:
        cv2.putText(out, "Camera Settings", (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    except Exception:
        pass

    for i, lab in enumerate(labels):
        y = row_y[i]
        v = int(vals.get(lab, 0) or 0)
        vmax = int(max_vals.get(lab, 255))
        v = max(0, min(vmax, v))
        try:
            cv2.putText(out, f"{lab}: {v}", (x0 + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        except Exception:
            pass
        try:
            cv2.line(out, (slider_left, y), (slider_right, y), (190, 190, 190), 2)
            denom = float(vmax) if float(vmax) > 0 else 1.0
            knob_x = int(slider_left + (slider_right - slider_left) * (float(v) / denom))
            cv2.circle(out, (knob_x, y), 7, (0, 200, 255), -1)
            cv2.circle(out, (knob_x, y), 7, (0, 0, 0), 1)
        except Exception:
            pass
        overlay["layout"][lab] = {
            "x1": int(slider_left),
            "x2": int(slider_right),
            "y": int(y),
        }

    return out


def _attach_camera_overlay_mouse(window_name: str, overlay: dict) -> None:
    if not window_name or not overlay:
        return

    def _set_value(label: str, x: int) -> None:
        try:
            lay = (overlay.get("layout") or {}).get(label) or {}
            x1 = int(lay.get("x1") or 0)
            x2 = int(lay.get("x2") or 0)
            if x2 <= x1:
                return
            xx = max(x1, min(x2, int(x)))
            max_vals = {
                "Brightness": 255,
                "Sharpness": 255,
                "Focus": 50,
            }
            vmax = float(max_vals.get(label, 255))
            v = int(round((float(xx - x1) / float(x2 - x1)) * vmax))
            v = max(0, min(int(vmax), v))
            overlay.setdefault("values", {})[label] = v
        except Exception:
            return

    def _hit_label(x: int, y: int) -> str:
        try:
            for lab, lay in (overlay.get("layout") or {}).items():
                yy = int(lay.get("y") or 0)
                x1 = int(lay.get("x1") or 0)
                x2 = int(lay.get("x2") or 0)
                if x1 <= x <= x2 and (yy - 12) <= y <= (yy + 12):
                    return str(lab)
        except Exception:
            return ""
        return ""

    def _on_mouse(event, x, y, flags, _userdata):
        try:
            if not overlay.get("enabled"):
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                lab = _hit_label(int(x), int(y))
                if lab:
                    overlay["drag"] = lab
                    _set_value(lab, int(x))
            elif event == cv2.EVENT_MOUSEMOVE:
                if (flags & cv2.EVENT_FLAG_LBUTTON) and overlay.get("drag"):
                    _set_value(str(overlay.get("drag")), int(x))
            elif event == cv2.EVENT_LBUTTONUP:
                overlay["drag"] = None
        except Exception:
            return

    try:
        cv2.setMouseCallback(window_name, _on_mouse)
    except Exception:
        pass


def _apply_camera_env_tuning(cap: cv2.VideoCapture) -> None:
    try:
        if cap is None:
            return
    except Exception:
        return

    def _env(name: str) -> str:
        try:
            return (os.getenv(name, "") or "").strip()
        except Exception:
            return ""

    focus = _env("WALKIE_CAMERA_FOCUS")

    try:
        if focus != "" and hasattr(cv2, "CAP_PROP_FOCUS"):
            cap.set(cv2.CAP_PROP_FOCUS, float(focus))
    except Exception:
        pass


def _norm_col(s: str) -> str:
    import re
    v = str(s or "").strip().lower().replace("_", " ")
    # remove common punctuation but keep letters/numbers/spaces
    v = re.sub(r"[\(\)\[\]\{\}:,;/\\\-]+", " ", v)
    v = " ".join(v.split())
    # normalize headers like "string (japanese)" -> "japanese"
    if v.startswith("string "):
        v = v[len("string "):].strip()
    if v.startswith("str "):
        v = v[len("str "):].strip()
    return v


def _norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    # Unicode normalization: makes half-width/full-width forms comparable (e.g., Japanese ｱ vs ア)
    try:
        s = unicodedata.normalize("NFKC", s)
    except Exception:
        pass
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(ln.strip().split()) for ln in s.split("\n")]
    lines = [ln for ln in lines if ln != ""]
    return "\n".join(lines).strip()


def _jp_strip_diacritics(s: str) -> str:
    """Remove dakuten/handakuten differences so バ/パ/ハ compare equal.

    This is used only for a *warning* classification when the only mismatch is diacritics.
    """
    s = "" if s is None else str(s)
    try:
        s = unicodedata.normalize("NFKC", s)
    except Exception:
        pass
    try:
        decomp = unicodedata.normalize("NFD", s)
        decomp = decomp.replace("\u3099", "").replace("\u309A", "")
        return unicodedata.normalize("NFC", decomp)
    except Exception:
        return s


def _parse_structured_original(block: str) -> str:
    if not block:
        return ""
    lines = [ln.rstrip("\r") for ln in str(block).splitlines()]
    mode = None
    buf_orig = []
    for raw in lines:
        s = raw.strip()
        low = s.lower()
        if low.startswith("detected text(original):"):
            mode = "orig"
            rest = s.split(":", 1)[-1].strip()
            if rest and rest.lower() != "detected text(original)":
                buf_orig.append(rest)
            continue
        if low.startswith("detected text(english translation):"):
            break
        if mode == "orig" and s and s not in ["<<<", ">>>"]:
            buf_orig.append(raw.rstrip())
    out = "\n".join([x for x in buf_orig if str(x).strip()]).strip()
    return out


def _parse_structured_language(block: str) -> str:
    if not block:
        return ""
    try:
        for raw in str(block).splitlines():
            s = raw.strip()
            low = s.lower()
            if low.startswith("detected language:") or low.startswith("detected languages:"):
                return s.split(":", 1)[-1].strip()
    except Exception:
        return ""
    return ""


def _parse_structured_english(block: str) -> str:
    if not block:
        return ""
    lines = [ln.rstrip("\r") for ln in str(block).splitlines()]
    mode = None
    buf = []
    for raw in lines:
        s = raw.strip()
        low = s.lower()
        if low.startswith("detected text(english translation):"):
            mode = "eng"
            rest = s.split(":", 1)[-1].strip()
            if rest and rest.lower() != "detected text(english translation)":
                buf.append(rest)
            continue
        if low.startswith("overlap error:") or low.startswith("vertical overlap error:") or low.startswith("ui render overlap error:"):
            break
        if mode == "eng" and s and s not in ["<<<", ">>>"]:
            buf.append(raw.rstrip())
    return "\n".join([x for x in buf if str(x).strip()]).strip()


def _show_ocr_result_window(
    roi: np.ndarray,
    original: str,
    english: str,
    language: str,
    verdict: str = "",
    expected_lines: list | None = None,
) -> None:
    try:
        if roi is None or getattr(roi, "size", 0) == 0:
            return
        lang_label = (language or "").strip()
        try:
            parts = [p.strip().lower() for p in lang_label.split(",") if p.strip()]
        except Exception:
            parts = []
        english_only = bool(parts) and all(p in ["english", "en"] for p in parts)
        show_english = bool(english and english.strip()) and (not english_only)

        o_lines = [ln.strip() for ln in str(original or "").splitlines() if ln.strip()]
        e_lines = [ln.strip() for ln in str(english or "").splitlines() if ln.strip()]

        pad = 22
        gap = 44
        line_h = 40
        roi_h0, roi_w0 = roi.shape[:2]
        out_w = int(max(roi_w0, 1250 if show_english else 1050))

        roi_scale = 1.0
        try:
            if roi_w0 < 520:
                roi_scale = 2.0
            if roi_w0 < 360:
                roi_scale = 3.0
        except Exception:
            roi_scale = 1.0

        roi_disp = roi
        try:
            if float(roi_scale) > 1.01:
                roi_disp = cv2.resize(
                    roi,
                    (int(round(roi_w0 * roi_scale)), int(round(roi_h0 * roi_scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
        except Exception:
            roi_disp = roi
        roi_h, roi_w = roi_disp.shape[:2]

        scale = 0.92 if out_w >= 1050 else 0.82
        head_scale = 1.05 if out_w >= 1050 else 0.95
        color = (255, 255, 255)

        if show_english:
            col_w = max(220, (out_w - (2 * pad) - gap) // 2)
            left_x = pad
            right_x = pad + col_w + gap
            left_head = f"Original ({lang_label})" if lang_label else "Original"
            right_head = "English"
            max_w = max(60, col_w - 10)

            left_head_lines = _wrap_text_to_px(left_head, max_w, head_scale)
            right_head_lines = _wrap_text_to_px(right_head, max_w, head_scale)
            header_rows = max(len(left_head_lines), len(right_head_lines), 1)

            left_wrapped = []
            right_wrapped = []
            max_rows = 0
            max_content_lines = max(len(o_lines), len(e_lines), 1)
            for i in range(max_content_lines):
                ltxt = o_lines[i] if i < len(o_lines) else ""
                rtxt = e_lines[i] if i < len(e_lines) else ""
                lw = _wrap_text_to_px(ltxt, max_w, scale)
                rw = _wrap_text_to_px(rtxt, max_w, scale)
                rows = max(len(lw), len(rw), 1)
                while len(lw) < rows:
                    lw.append("")
                while len(rw) < rows:
                    rw.append("")
                left_wrapped.extend(lw)
                right_wrapped.extend(rw)
                max_rows += rows

            text_h = pad + (header_rows * line_h) + (max_rows * line_h) + pad
            out = np.zeros((roi_h + text_h, out_w, 3), dtype=np.uint8)
            x_img = int(max(0, (out_w - roi_w) // 2))
            out[:roi_h, x_img : x_img + roi_w] = roi_disp

            y = roi_h + pad + 24
            for r in range(header_rows):
                l = left_head_lines[r] if r < len(left_head_lines) else ""
                rr = right_head_lines[r] if r < len(right_head_lines) else ""
                out = _draw_text_unicode(out, l, (left_x, y), head_scale, color, 1)
                out = _draw_text_unicode(out, rr, (right_x, y), head_scale, color, 1)
                y += line_h

            for idx in range(max_rows):
                l = left_wrapped[idx] if idx < len(left_wrapped) else ""
                rr = right_wrapped[idx] if idx < len(right_wrapped) else ""
                if l:
                    out = _draw_text_unicode(out, l, (left_x, y), scale, color, 1)
                if rr:
                    out = _draw_text_unicode(out, rr, (right_x, y), scale, color, 1)
                y += line_h

        else:
            max_w = max(80, out_w - (2 * pad) - 10)
            head = f"Original ({lang_label})" if lang_label else "Original"
            head_lines = _wrap_text_to_px(head, max_w, head_scale)
            body_wrapped = []
            for ln in (o_lines or [""]):
                body_wrapped.extend(_wrap_text_to_px(ln, max_w, scale))
            text_h = pad + (len(head_lines) * line_h) + (len(body_wrapped) * line_h) + pad
            out = np.zeros((roi_h + text_h, out_w, 3), dtype=np.uint8)
            x_img = int(max(0, (out_w - roi_w) // 2))
            out[:roi_h, x_img : x_img + roi_w] = roi_disp

            y = roi_h + pad + 24
            for hl in head_lines:
                out = _draw_text_unicode(out, hl, (pad, y), head_scale, color, 1)
                y += line_h
            for bl in body_wrapped:
                if bl:
                    out = _draw_text_unicode(out, bl, (pad, y), scale, color, 1)
                y += line_h

        exp_draw_lines = []
        try:
            v2 = str(verdict or "").strip().upper()
            if v2 == "FAIL" and expected_lines:
                raw_lines = [str(x) for x in list(expected_lines) if str(x).strip()]
                for ln in raw_lines:
                    exp_draw_lines.extend(_wrap_text_to_px(ln, max(120, int(out_w - 40)), 0.65))
        except Exception:
            exp_draw_lines = []

        try:
            header_h = 80
            summary_h = 120 + (max(0, len(exp_draw_lines)) * 32) + 16
            header = np.zeros((header_h, out_w, 3), dtype=np.uint8)
            cv2.putText(header, "OCR Result", (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)

            summary = np.zeros((summary_h, out_w, 3), dtype=np.uint8)
            lang_show = lang_label if lang_label else "Unknown"
            cv2.putText(summary, f"Detected Language: {lang_show}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

            v = str(verdict or "").strip().upper()
            if v:
                vcol = (255, 255, 255)
                if v == "PASS":
                    vcol = (0, 255, 0)
                elif v == "FAIL":
                    vcol = (0, 0, 255)
                elif v == "WARN":
                    vcol = (0, 255, 255)
                cv2.putText(summary, f"Verdict: {v}", (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 1.1, vcol, 3)

            try:
                if exp_draw_lines:
                    y0 = 118
                    for i, ln in enumerate(exp_draw_lines):
                        summary = _draw_text_unicode(summary, ln, (24, y0 + (i * 32)), 0.8, (255, 255, 255), 1)
            except Exception:
                pass

            combined = np.vstack([header, summary, out])
        except Exception:
            combined = np.vstack([header, out])

        disp = combined
        try:
            min_w = 1250
            min_h = 900
            h0, w0 = disp.shape[:2]
            if w0 > 0 and h0 > 0 and (w0 < min_w or h0 < min_h):
                s = max(float(min_w) / float(w0), float(min_h) / float(h0))
                s = max(1.0, min(2.2, float(s)))
                disp = cv2.resize(disp, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_CUBIC)
        except Exception:
            disp = combined

        cv2.namedWindow("OCR Result", cv2.WINDOW_NORMAL)
        try:
            hh, ww = disp.shape[:2]
            cv2.resizeWindow("OCR Result", int(ww), int(hh))
        except Exception:
            pass
        cv2.imshow("OCR Result", disp)
        while True:
            try:
                if hasattr(cv2, "getWindowProperty") and hasattr(cv2, "WND_PROP_VISIBLE"):
                    if cv2.getWindowProperty("OCR Result", cv2.WND_PROP_VISIBLE) < 1:
                        break
            except Exception:
                pass
            k = cv2.waitKey(50)
            if k is not None and int(k) != -1:
                break
        try:
            cv2.destroyWindow("OCR Result")
        except Exception:
            pass
    except Exception:
        return


def _find_sheet(xls: pd.ExcelFile, name: str) -> str:
    target = _norm_col(name)
    for s in xls.sheet_names:
        if _norm_col(s) == target:
            return s
    for s in xls.sheet_names:
        if target in _norm_col(s):
            return s
    raise ValueError(f"Sheet '{name}' not found. Available: {xls.sheet_names}")


def _pick_language_column(df: pd.DataFrame, region: str, language: str) -> str:
    want = _norm_col(language)
    cols = { _norm_col(c): c for c in df.columns }

    synonyms = {
        "japanese": ["japanese", "ja"],
        "korean": ["korean", "ko"],
        "simplified chinese": ["simplified chinese", "chinese simplified", "zh cn", "zh-hans"],
        "traditional chinese": ["traditional chinese", "chinese traditional", "zh tw", "zh-hant"],
        "french": ["french", "fr"],
        "spanish": ["spanish", "es"],
        "german": ["german", "de"],
        "italian": ["italian", "it"],
        "polish": ["polish", "pl"],
        "russian": ["russian", "ru"],
        "turkish": ["turkish", "tr"],
        "arabic": ["arabic", "ar"],
        "hungarian": ["hungarian", "hu"],
        "hebrew": ["hebrew", "iw", "he"],
        "czech": ["czech", "cs"],
        "portuguese": ["portuguese", "pt"],
    }

    candidates = [want]
    for k, vs in synonyms.items():
        if want == k or want in vs:
            candidates = [_norm_col(v) for v in vs]
            break

    for c in candidates:
        if c in cols:
            return cols[c]

    raise ValueError(
        f"Language column for region='{region}' language='{language}' not found. Columns: {list(df.columns)}"
    )


def load_expected(excel_path: str, region: str, language: str, index: str = "", tag: str = "") -> dict:
    excel_path = str(excel_path)
    if not os.path.exists(excel_path):
        raise FileNotFoundError(excel_path)

    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(f"pandas is required to read the Excel file: {e}")

    xls = pd.ExcelFile(excel_path, engine="openpyxl")

    english_sheet = _find_sheet(xls, "english")
    category_sheet = _find_sheet(xls, "category")

    region_norm = _norm_col(region)
    if region_norm in ["apac"]:
        region_sheet = _find_sheet(xls, "apac")
    elif region_norm in ["emea"]:
        region_sheet = _find_sheet(xls, "emea")
    elif region_norm in ["lacr", "latam", "latam\u0026caribbean", "la cr"]:
        region_sheet = _find_sheet(xls, "lacr")
    elif region_norm in ["english", "en", "global"]:
        region_sheet = english_sheet
    else:
        raise ValueError("Region must be one of: english, apac, emea, lacr")

    df_en = pd.read_excel(xls, sheet_name=english_sheet, engine="openpyxl")
    df_cat = pd.read_excel(xls, sheet_name=category_sheet, engine="openpyxl")
    df_reg = pd.read_excel(xls, sheet_name=region_sheet, engine="openpyxl")

    def _coerce_index(v):
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        s = str(v).strip()
        if s.endswith(".0"):
            try:
                s2 = str(int(float(s)))
                return s2
            except Exception:
                pass
        return s

    df_en = df_en.copy()
    df_cat = df_cat.copy()
    df_reg = df_reg.copy()

    if "index" in [_norm_col(c) for c in df_en.columns]:
        en_index_col = next(c for c in df_en.columns if _norm_col(c) == "index")
        df_en["__index"] = df_en[en_index_col].apply(_coerce_index)
    else:
        raise ValueError("English sheet missing 'index' column")

    if "index" in [_norm_col(c) for c in df_reg.columns]:
        reg_index_col = next(c for c in df_reg.columns if _norm_col(c) == "index")
        df_reg["__index"] = df_reg[reg_index_col].apply(_coerce_index)
    else:
        raise ValueError(f"Region sheet '{region_sheet}' missing 'index' column")

    if "index" in [_norm_col(c) for c in df_cat.columns]:
        cat_index_col = next(c for c in df_cat.columns if _norm_col(c) == "index")
        df_cat["__index"] = df_cat[cat_index_col].apply(_coerce_index)
    else:
        raise ValueError("Category sheet missing 'index' column")

    idx = str(index).strip()
    if idx and idx.endswith(".0"):
        try:
            idx = str(int(float(idx)))
        except Exception:
            pass

    def _find_tag_column(df: pd.DataFrame) -> str:
        # Prefer explicit 'String Tag' style headers.
        preferred = []
        fallback = []
        for c in df.columns:
            raw = str(c or "")
            low = raw.strip().lower()
            n = _norm_col(c)
            if "tag" in low and "string" in low:
                preferred.append(c)
            elif n in ["string tag", "stringtag"]:
                preferred.append(c)
            elif n == "tag" or "tag" in low:
                fallback.append(c)
        if preferred:
            return preferred[0]
        if fallback:
            return fallback[0]
        return ""

    idx_region = ""

    if idx:
        row_en = df_en[df_en["__index"] == idx]
        row_reg = df_reg[df_reg["__index"] == idx]
        row_cat = df_cat[df_cat["__index"] == idx]
    elif tag:
        tag_norm = str(tag).strip().lower()
        # Canonical lookup: resolve the row (and index) from the English sheet first.
        # Region sheets can have different index numbering, so we do NOT rely on them for index.
        en_tag_col = _find_tag_column(df_en)
        if not en_tag_col:
            raise ValueError("English sheet missing 'string tag' column")
        row_en_all = df_en[df_en[en_tag_col].astype(str).str.strip().str.lower() == tag_norm]
        if row_en_all.empty:
            raise ValueError(f"No row found for tag '{tag}' in English sheet")
        if len(row_en_all) > 1:
            cols_norm = {_norm_col(c): c for c in df_en.columns}
            en_col = None
            for k in ["string english", "string (english)", "english", "string"]:
                if k in cols_norm:
                    en_col = cols_norm[k]
                    break
            cat_col = None
            for k in ["string category", "category"]:
                if k in cols_norm:
                    cat_col = cols_norm[k]
                    break
            ver_col = None
            for k in ["version", "ver"]:
                if k in cols_norm:
                    ver_col = cols_norm[k]
                    break
            preview = []
            for _, r in row_en_all.iterrows():
                preview.append(
                    {
                        "index": str(r.get("__index", "")).strip(),
                        "tag": str(r.get(en_tag_col, "")).strip(),
                        "english": str(r.get(en_col, "")).strip() if en_col else "",
                        "category": str(r.get(cat_col, "")).strip() if cat_col else "",
                        "version": str(r.get(ver_col, "")).strip() if ver_col else "",
                    }
                )
            raise ValueError(
                "Multiple rows found for tag '"
                + str(tag)
                + "' in English sheet. Please rerun with --index to select the correct row. Candidates: "
                + str(preview)
            )

        row_en = row_en_all
        idx = str(row_en.iloc[0]["__index"]).strip()
        row_cat = df_cat[df_cat["__index"] == idx]

        # Localized lookup: find the row in the selected region sheet by tag.
        reg_tag_col = _find_tag_column(df_reg)
        if not reg_tag_col:
            row_reg = df_reg[df_reg["__index"] == "__no_match__"]
        else:
            row_reg_all = df_reg[df_reg[reg_tag_col].astype(str).str.strip().str.lower() == tag_norm]
            if len(row_reg_all) == 1:
                row_reg = row_reg_all
                idx_region = str(row_reg.iloc[0].get("__index", "")).strip()
            elif len(row_reg_all) > 1:
                preview = []
                for _, r in row_reg_all.iterrows():
                    preview.append(
                        {
                            "index": str(r.get("__index", "")).strip(),
                            "tag": str(r.get(reg_tag_col, "")).strip(),
                        }
                    )
                raise ValueError(
                    "Multiple rows found for tag '"
                    + str(tag)
                    + f"' in region sheet '{region_sheet}'. Please rerun with --index (English index) and I'll pick the localized row by index. Candidates: "
                    + str(preview)
                )
            else:
                row_reg = row_reg_all
    else:
        raise ValueError("Provide --index or --tag")

    if row_en.empty:
        raise ValueError(f"No English row found for index '{idx}'")

    en_row = row_en.iloc[0].to_dict()
    reg_row = row_reg.iloc[0].to_dict() if not row_reg.empty else {}
    cat_row = row_cat.iloc[0].to_dict() if not row_cat.empty else {}

    en_text_col = None
    for c in df_en.columns:
        if _norm_col(c) in ["string (english)", "string english", "english", "string"]:
            en_text_col = c
            break

    if en_text_col is None:
        raise ValueError("English sheet missing 'string (english)' column")

    expected_en = "" if pd.isna(en_row.get(en_text_col)) else str(en_row.get(en_text_col) or "")

    if _norm_col(region_sheet) == _norm_col(english_sheet):
        expected_local = expected_en
    else:
        lang_col = _pick_language_column(df_reg, region, language)
        expected_local = "" if pd.isna(reg_row.get(lang_col)) else str(reg_row.get(lang_col) or "")

    tag_val = ""
    for c in df_en.columns:
        if _norm_col(c) in ["string tag", "tag", "stringtag"]:
            tag_val = "" if pd.isna(en_row.get(c)) else str(en_row.get(c) or "")
            break

    cat_val = ""
    for c in df_en.columns:
        if _norm_col(c) in ["string category", "category"]:
            cat_val = "" if pd.isna(en_row.get(c)) else str(en_row.get(c) or "")
            break

    font_style = ""
    font_size = ""
    for c in df_cat.columns:
        if _norm_col(c) == "font style":
            font_style = "" if pd.isna(cat_row.get(c)) else str(cat_row.get(c) or "")
        if _norm_col(c) == "font size":
            font_size = "" if pd.isna(cat_row.get(c)) else str(cat_row.get(c) or "")

    return {
        "index": idx,
        "index_region": idx_region,
        "tag": tag_val,
        "category": cat_val,
        "font_style": font_style,
        "font_size": font_size,
        "expected_en": expected_en,
        "expected_local": expected_local,
        "region_sheet": region_sheet,
    }


def capture_screen_roi(detector: FastDetector, camera_id: int, confidence: float, warmup_sec: float = 0.7):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass

    _apply_camera_env_tuning(cap)
    _ts_log(t0, "camera env tuning applied")

    t_end = time.time() + float(warmup_sec or 0.0)
    last = None
    while time.time() < t_end:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            last = frame
        time.sleep(0.03)

    ok, frame = cap.read()
    if ok and frame is not None and frame.size > 0:
        last = frame

    cap.release()
    if last is None:
        raise RuntimeError("Failed to capture frame")

    boxes, screens = detector.detect_with_screens(last, confidence)
    if not boxes:
        raise RuntimeError("No device detected")

    x1, y1, x2, y2 = boxes[0]
    screen_box = None
    for s in screens:
        sx1, sy1, sx2, sy2 = s
        if sx1 >= x1 and sy1 >= y1 and sx2 <= x2 and sy2 <= y2:
            screen_box = s
            break

    if screen_box is None:
        device_w = x2 - x1
        device_h = y2 - y1
        screen_box = (
            x1 + device_w // 4,
            y1 + 10,
            x2 - device_w // 4,
            y1 + device_h // 3,
        )

    sx1, sy1, sx2, sy2 = screen_box
    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(last.shape[1], sx2), min(last.shape[0], sy2)
    if sx2 <= sx1 or sy2 <= sy1:
        raise RuntimeError("Invalid screen ROI")

    roi = last[sy1:sy2, sx1:sx2]
    return last, roi


def capture_screen_roi_preview(
    detector: FastDetector | None,
    camera_id: int,
    confidence: float = 0.25,
    model_path: str = "",
    window_name: str = "Verify Preview",
):
    t0 = time.time()
    _ts_log(t0, f"preview start (camera_id={int(camera_id)})")
    cap = None
    try:
        cap = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
    except Exception:
        cap = None
    if cap is None:
        cap = cv2.VideoCapture(int(camera_id))
    _ts_log(t0, f"VideoCapture created (opened={bool(cap.isOpened())})")
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass

    det_holder = {"det": detector}

    if det_holder["det"] is None and model_path:
        _ts_log(t0, f"starting detector load thread (model_path='{model_path}')")
        def _load_det():
            try:
                t_det = time.time()
                det_holder["det"] = FastDetector(model_path)
                _ts_log(t0, f"detector loaded in {(time.time()-t_det):.3f}s")
            except Exception:
                det_holder["det"] = None
                _ts_log(t0, "detector load FAILED")

        try:
            th = threading.Thread(target=_load_det, daemon=True)
            th.start()
        except Exception:
            pass

    last_frame = None
    last_boxes = []
    last_screens = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _ts_log(t0, f"window created ('{window_name}')")
    try:
        cv2.resizeWindow(window_name, 1280, 720)
    except Exception:
        pass
    overlay = None
    try:
        overlay = _create_camera_overlay_state(cap)
    except Exception:
        overlay = None
    zoom = 1.0
    preview_mode = True
    did_tune = False
    try:
        while True:
            if overlay is not None:
                _apply_camera_overlay_settings(cap, overlay)

            if not preview_mode:
                closed = False
                try:
                    if hasattr(cv2, "getWindowProperty") and hasattr(cv2, "WND_PROP_VISIBLE"):
                        closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
                except Exception:
                    closed = False
                if closed:
                    raise RuntimeError("Cancelled")
                k = cv2.waitKey(50) & 0xFF
                if k in [ord('x'), ord('X')]:
                    raise RuntimeError("Cancelled")
                if k in [ord('t'), ord('T')]:
                    try:
                        if overlay is None:
                            overlay = _create_camera_overlay_state(cap)
                        if overlay is not None:
                            overlay["enabled"] = not bool(overlay.get("enabled"))
                            overlay["drag"] = None
                    except Exception:
                        pass
                if k in [ord('+'), ord('=')]:
                    zoom = min(4.0, float(zoom) + 0.1)
                if k in [ord('-'), ord('_')]:
                    zoom = max(1.0, float(zoom) - 0.1)
                if k in [ord('z'), ord('Z')]:
                    zoom = 1.0
                continue

            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue

            if last_frame is None:
                _ts_log(t0, f"first frame received ({frame.shape[1]}x{frame.shape[0]})")
            if not did_tune:
                try:
                    _apply_camera_env_tuning(cap)
                except Exception:
                    pass
                did_tune = True
                _ts_log(t0, "camera env tuning applied")

            if zoom and float(zoom) > 1.0:
                try:
                    h, w = frame.shape[:2]
                    new_w = max(2, int(w / float(zoom)))
                    new_h = max(2, int(h / float(zoom)))
                    x1 = max(0, (w - new_w) // 2)
                    y1 = max(0, (h - new_h) // 2)
                    crop = frame[y1 : y1 + new_h, x1 : x1 + new_w]
                    frame = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    pass
            last_frame = frame

            try:
                det = det_holder.get("det")
                if det is not None:
                    boxes, screens = det.detect_with_screens(frame, confidence)
                    last_boxes = boxes or []
                    last_screens = screens or []
                else:
                    last_boxes = []
                    last_screens = []
            except Exception:
                last_boxes = []
                last_screens = []

            vis = frame.copy()
            for (x1, y1, x2, y2) in last_boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for (sx1, sy1, sx2, sy2) in last_screens:
                cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (0, 0, 255), 1)

            cv2.putText(
                vis,
                "SPACE=capture  T=settings  +/-=zoom  X=cancel",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            try:
                if det_holder.get("det") is None:
                    cv2.putText(
                        vis,
                        "Loading detector...",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
            except Exception:
                pass
            try:
                if overlay is not None:
                    vis = _draw_camera_overlay(vis, overlay)
            except Exception:
                pass
            cv2.imshow(window_name, vis)
            closed = False
            try:
                if hasattr(cv2, "getWindowProperty") and hasattr(cv2, "WND_PROP_VISIBLE"):
                    closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
            except Exception:
                closed = False
            if closed:
                raise RuntimeError("Cancelled")
            if last_frame is frame:
                pass
            try:
                if overlay is not None and overlay.get("enabled"):
                    _attach_camera_overlay_mouse(window_name, overlay)
            except Exception:
                pass

            k = cv2.waitKey(1) & 0xFF
            if k in [ord('x'), ord('X')]:
                raise RuntimeError("Cancelled")
            if k in [ord('t'), ord('T')]:
                try:
                    if overlay is None:
                        overlay = _create_camera_overlay_state(cap)
                    if overlay is not None:
                        overlay["enabled"] = not bool(overlay.get("enabled"))
                        overlay["drag"] = None
                except Exception:
                    pass
            if k in [ord('+'), ord('=')]:
                zoom = min(4.0, float(zoom) + 0.1)
            if k in [ord('-'), ord('_')]:
                zoom = max(1.0, float(zoom) - 0.1)
            if k in [ord('z'), ord('Z')]:
                zoom = 1.0
            if k == ord(' '):
                try:
                    if det_holder.get("det") is None:
                        continue
                except Exception:
                    pass
                break
    finally:
        cap.release()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

    if last_frame is None:
        raise RuntimeError("Failed to capture frame")
    if not last_boxes:
        raise RuntimeError(
            "No device detected. Try adjusting lighting/camera angle or lower --confidence (e.g. 0.15)."
        )

    x1, y1, x2, y2 = last_boxes[0]
    screen_box = None
    for s in last_screens:
        sx1, sy1, sx2, sy2 = s
        if sx1 >= x1 and sy1 >= y1 and sx2 <= x2 and sy2 <= y2:
            screen_box = s
            break

    if screen_box is None:
        device_w = x2 - x1
        device_h = y2 - y1
        screen_box = (
            x1 + device_w // 4,
            y1 + 10,
            x2 - device_w // 4,
            y1 + device_h // 3,
        )

    sx1, sy1, sx2, sy2 = screen_box
    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(last_frame.shape[1], sx2), min(last_frame.shape[0], sy2)
    if sx2 <= sx1 or sy2 <= sy1:
        raise RuntimeError("Invalid screen ROI")

    roi = last_frame[sy1:sy2, sx1:sx2]
    _ts_log(t0, "preview capture complete (returning ROI)")
    return last_frame, roi


def main():
    t0_total = time.time()
    _ts_log(t0_total, "process start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Path to Mackenzie_Radio_String_Translation_List.xlsm")
    parser.add_argument(
        "--region",
        required=True,
        help="Region sheet (english/apac/emea/lacr). Case-insensitive.",
    )
    parser.add_argument("--language", required=True, help="Language column within region sheet (e.g., Japanese, French)")
    parser.add_argument("--index", default="", help="String index")
    parser.add_argument("--tag", default="", help="String tag")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--model-path", default="", help="YOLO model path override")
    parser.add_argument("--epoch", type=int, default=None, help="Use YOLO checkpoint epochN.pt (e.g., 95)")
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--save-roi", default="", help="Optional path to save captured ROI image")
    parser.add_argument("--preview", action="store_true", help="Open live preview window; press SPACE to capture")

    args = parser.parse_args()
    _ts_log(t0_total, "args parsed")

    # Normalize region input (accept APAC/EMEA/LACR/English in any case)
    try:
        args.region = _norm_col(args.region)
    except Exception:
        pass
    region_map = {
        "english": "english",
        "en": "english",
        "apac": "apac",
        "emea": "emea",
        "lacr": "lacr",
        "lac": "lacr",
        "latam": "lacr",
    }
    if args.region in region_map:
        args.region = region_map[args.region]
    else:
        raise ValueError("--region must be one of: english, apac, emea, lacr")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    _ts_log(t0_total, f"config loaded ('{args.config}')")

    def _resolve_epoch_weights(epoch: int) -> str:
        fname = f"epoch{int(epoch)}.pt"
        candidates = [
            Path("runs") / "detect" / "models" / "trained" / "walkie_detector" / "weights" / fname,
            Path("runs") / "detect" / "train" / "weights" / fname,
            Path("models") / "trained" / "walkie_detector" / "weights" / fname,
            Path("models") / "trained" / "walkie_detector" / fname,
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        # fallback: search for the first matching epoch file
        for root in [Path("runs"), Path("models")]:
            if not root.exists():
                continue
            try:
                for p in root.rglob(fname):
                    return str(p)
            except Exception:
                pass
        return ""

    model_path = args.model_path or ""
    if not model_path and args.epoch is not None:
        model_path = _resolve_epoch_weights(args.epoch)
        if not model_path:
            raise ValueError(f"Could not find weights for epoch {args.epoch} (expected file: epoch{args.epoch}.pt)")
    if not model_path:
        model_path = cfg.get("detector", {}).get("path", "")
    if not model_path:
        raise ValueError("Detector model path not provided. Use --model-path or set detector.path in settings.yaml")
    _ts_log(t0_total, f"model resolved ('{model_path}')")

    camera_id = args.camera_id
    if camera_id is None:
        camera_id = int(cfg.get("camera", {}).get("source", 0))
    _ts_log(t0_total, f"camera_id resolved ({int(camera_id)})")

    confidence = float(cfg.get("detector", {}).get("confidence", 0.25))

    if args.preview:
        t0_preview = time.time()
        _ts_log(t0_total, "entering preview")
        full_frame, roi = capture_screen_roi_preview(None, camera_id=camera_id, confidence=confidence, model_path=model_path)
        t1_preview = time.time()
        _ts_log(t0_total, f"preview returned in {(t1_preview - t0_preview):.3f}s")
        t0_cap = t1_preview
        t1_cap = t1_preview
    else:
        # Non-preview capture needs detector synchronously.
        _ts_log(t0_total, "loading detector (non-preview)")
        detector = FastDetector(model_path)
        t0_cap = time.time()
        full_frame, roi = capture_screen_roi(detector, camera_id=camera_id, confidence=confidence)
        t1_cap = time.time()
        _ts_log(t0_total, f"capture complete in {(t1_cap - t0_cap):.3f}s")

    _ts_log(t0_total, "loading expected strings from excel")
    expected = load_expected(args.excel, args.region, args.language, index=args.index, tag=args.tag)
    _ts_log(t0_total, "expected strings loaded")

    print("=" * 70)
    print("RADIO STRING VERIFICATION (OPTION A)")
    print("=" * 70)
    print(f"Index: {expected['index']}")
    if expected.get("index_region"):
        print(f"Index (region sheet): {expected['index_region']}")
    if expected.get("tag"):
        print(f"Tag: {expected['tag']}")
    if expected.get("category"):
        print(f"Category: {expected['category']}")
    if expected.get("font_style") or expected.get("font_size"):
        print(f"Font: style='{expected.get('font_style','')}', size='{expected.get('font_size','')}'")
    print(f"Region sheet: {expected['region_sheet']}")
    print(f"Expected (English): {expected['expected_en']}")
    print(f"Expected ({args.region}/{args.language}): {expected['expected_local']}")

    # Warm GenAI session after we already captured ROI (or while user was in preview).
    # This reduces perceived camera-open delay.
    ocr = MSIGenAIOCR()
    try:
        def _warm():
            try:
                ocr.get_or_init_session()
            except Exception:
                pass

        th = threading.Thread(target=_warm, daemon=True)
        th.start()
    except Exception:
        pass

    if args.save_roi:
        outp = Path(args.save_roi)
        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), roi)
        print(f"Saved ROI: {outp}")

    t0_ocr = time.time()
    text, _conf = ocr.extract_text(roi, expected_language=args.language)
    t1_ocr = time.time()

    try:
        lang_detected = _parse_structured_language(text)
        orig_text = _parse_structured_original(text) or text
        eng_text = _parse_structured_english(text)
    except Exception:
        pass

    observed = _parse_structured_original(text) or text
    observed_n = _norm_text(observed)
    expected_local_n = _norm_text(expected["expected_local"])

    ok = False
    warn = False
    if expected_local_n:
        ok = observed_n == expected_local_n or expected_local_n in observed_n

        # Also ignore whitespace-only differences (common with CJK where OCR may omit spaces).
        if not ok:
            try:
                observed_ns = "".join(observed_n.split())
                expected_ns = "".join(expected_local_n.split())
                ok = observed_ns == expected_ns or expected_ns in observed_ns
            except Exception:
                pass

        # Japanese diacritic tolerance (dakuten/handakuten) as WARN (do not fail).
        # Example: バ vs パ, ば vs ぱ, ハ vs バ vs パ, etc.
        if not ok and args.language and str(args.language).strip().lower() in ["japanese", "ja"]:
            try:
                o_base = _jp_strip_diacritics(observed_n)
                e_base = _jp_strip_diacritics(expected_local_n)
                if o_base == e_base:
                    warn = True
            except Exception:
                pass

    try:
        verdict = "PASS" if ok else ("WARN" if warn else "FAIL")
    except Exception:
        verdict = ""

    try:
        exp_lines = []
        try:
            if expected.get("index_region"):
                exp_lines.append(f"Index (region sheet): {expected.get('index_region')}")
            if expected.get("tag"):
                exp_lines.append(f"Tag: {expected.get('tag')}")
            if expected.get("category"):
                exp_lines.append(f"Category: {expected.get('category')}")
            exp_lines.append(f"Region sheet: {expected.get('region_sheet','')}")
            exp_lines.append(f"Expected (English): {expected.get('expected_en','')}")
            exp_lines.append(f"Expected ({args.region}/{args.language}): {expected.get('expected_local','')}")
        except Exception:
            exp_lines = []

        _show_ocr_result_window(roi, orig_text, eng_text, lang_detected, verdict, expected_lines=exp_lines)
    except Exception:
        try:
            _show_ocr_result_window(roi, orig_text, eng_text, lang_detected)
        except Exception:
            pass

    print("-" * 70)
    print(f"Observed (normalized): {observed_n}")
    print("-" * 70)
    if ok:
        print("PASS")
    elif warn:
        print("WARN")
    else:
        print("FAIL")

    if not ok and not warn:
        print("Expected (normalized):")
        print(expected_local_n)

    try:
        total_s = time.time() - t0_total
        cap_s = t1_cap - t0_cap
        ocr_s = t1_ocr - t0_ocr
        if args.preview:
            try:
                preview_wait_s = t1_preview - t0_preview
            except Exception:
                preview_wait_s = 0.0
            print(
                f"[TIMING] Preview-wait: {preview_wait_s:.2f}s | Capture: {cap_s:.2f}s | OCR: {ocr_s:.2f}s | Total: {total_s:.2f}s"
            )
        else:
            print(f"[TIMING] Capture: {cap_s:.2f}s | OCR: {ocr_s:.2f}s | Total: {total_s:.2f}s")
    except Exception:
        pass


if __name__ == "__main__":
    main()
