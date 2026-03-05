import cv2
import sys
import os
from pathlib import Path
import numpy as np
import yaml
from datetime import datetime
import json
import time
from dotenv import load_dotenv
import argparse
import re
import atexit
import tempfile
import shutil

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None

def _strict_count_vertical_separators(screen_roi: np.ndarray) -> int:
    try:
        try:
            dbg_on = str(os.getenv("WALKIE_DEBUG_SEP", "") or "").strip().lower() in ["1", "true", "yes", "on"]
        except Exception:
            dbg_on = False

        _tmp_dir_holder = {"dir": None}

        def _get_debug_sep_dir() -> Path | None:
            try:
                if not bool(dbg_on):
                    return None
                out_dir_str = (os.getenv("WALKIE_DEBUG_SEP_DIR", "") or "").strip()
                if out_dir_str:
                    p = Path(out_dir_str)
                    p.mkdir(parents=True, exist_ok=True)
                    return p
                if _tmp_dir_holder.get("dir") is None:
                    d = tempfile.mkdtemp(prefix="walkie_debug_sep_")
                    _tmp_dir_holder["dir"] = d
                    try:
                        os.environ["WALKIE_DEBUG_SEP_DIR"] = str(d)
                    except Exception:
                        pass

                def _cleanup_tmp():
                    try:
                        dd = _tmp_dir_holder.get("dir")
                        if dd and Path(dd).exists():
                            shutil.rmtree(dd, ignore_errors=True)
                    except Exception:
                        pass
                try:
                    atexit.register(_cleanup_tmp)
                except Exception:
                    pass
                p = Path(str(_tmp_dir_holder.get("dir")))
                p.mkdir(parents=True, exist_ok=True)
                return p
            except Exception:
                return None
        ts = None
        out_dir = None
        if screen_roi is None or getattr(screen_roi, "size", 0) == 0:
            return 0
        h, w = screen_roi.shape[:2]
        if h < 20 or w < 30:
            if dbg_on:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    out_dir = _get_debug_sep_dir()
                    if out_dir is None:
                        return 0
                    cv2.imwrite(str(out_dir / f"sep_{ts}_screen_roi.jpg"), screen_roi)
                except Exception:
                    pass
            return 0

        # Separators we care about are typically in the bottom softkey row.
        y0 = int(max(0, h * 0.68))
        y1 = int(min(h, h * 0.98))
        roi = screen_roi[y0:y1, :]
        if roi is None or getattr(roi, "size", 0) == 0:
            return 0
        hh, ww_img = roi.shape[:2]
        if hh < 10 or ww_img < 30:
            if dbg_on:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    out_dir = _get_debug_sep_dir()
                    if out_dir is None:
                        return 0
                    cv2.imwrite(str(out_dir / f"sep_{ts}_screen_roi.jpg"), screen_roi)
                    cv2.imwrite(str(out_dir / f"sep_{ts}_roi.jpg"), roi)
                except Exception:
                    pass
            return 0

        # Robust separator detection: find thin vertical bright strokes via morphology.
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
            bw_adapt = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                31,
                -10,
            )
        except Exception:
            bw_adapt = bw_bright

        try:
            bw = cv2.max(cv2.max(bw_bright, bw_dark), bw_adapt)
        except Exception:
            bw = bw_bright

        # Extract vertical components.
        k_h = max(8, int(hh * 0.55))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(k_h)))
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

        try:
            if dbg_on:
                if ts is None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if out_dir is None:
                    out_dir = _get_debug_sep_dir()
                    if out_dir is None:
                        raise RuntimeError("debug_sep_dir unavailable")
                cv2.imwrite(str(out_dir / f"sep_{ts}_screen_roi.jpg"), screen_roi)
                cv2.imwrite(str(out_dir / f"sep_{ts}_roi.jpg"), roi)
                cv2.imwrite(str(out_dir / f"sep_{ts}_gray.png"), gray)
                cv2.imwrite(str(out_dir / f"sep_{ts}_bright.png"), bw_bright)
                cv2.imwrite(str(out_dir / f"sep_{ts}_dark.png"), bw_dark)
                cv2.imwrite(str(out_dir / f"sep_{ts}_adapt.png"), bw_adapt)
                cv2.imwrite(str(out_dir / f"sep_{ts}_bw.png"), bw)
                cv2.imwrite(str(out_dir / f"sep_{ts}_vert.png"), vert)
        except Exception:
            pass

        try:
            contours, _hier = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            _res = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = _res[0] if _res else []

        xs_w = []
        for c in contours or []:
            x, y, ww, hh0 = cv2.boundingRect(c)
            # Separator bars should span nearly the full softkey band height.
            if hh0 < int(hh * 0.55):
                continue
            if ww > max(10, int(ww_img * 0.05)):
                continue
            # Strongly vertical.
            if int(hh0) < int(ww) * 6:
                continue
            xc = int(x + (ww // 2))
            # Ignore near-border vertical edges/noise.
            if xc < int(ww_img * 0.08) or xc > int(ww_img * 0.92):
                continue
            xs_w.append((int(xc), int(max(1, ww))))

        if not xs_w:
            try:
                col = np.mean((bw_bright.astype(np.uint8) > 0).astype(np.float32), axis=0)
                try:
                    win = max(3, int(ww_img * 0.01) | 1)
                except Exception:
                    win = 5
                kernel_1d = np.ones((int(win),), dtype=np.float32) / float(max(1, int(win)))
                col_s = np.convolve(col, kernel_1d, mode="same")
                thr = float(max(0.55, min(0.90, np.percentile(col_s, 98) * 0.85)))
                mask = col_s >= thr
                groups = []
                start = None
                for xi, on in enumerate(mask.tolist()):
                    if on and start is None:
                        start = int(xi)
                    elif (not on) and start is not None:
                        groups.append((int(start), int(xi - 1)))
                        start = None
                if start is not None:
                    groups.append((int(start), int(len(mask) - 1)))

                max_w = max(2, int(ww_img * 0.020))
                for a, b in groups:
                    gw = int(b - a + 1)
                    if gw > int(max_w):
                        continue
                    xc = int((int(a) + int(b)) // 2)
                    if xc < int(ww_img * 0.08) or xc > int(ww_img * 0.92):
                        continue
                    xs_w.append((int(xc), int(gw)))
            except Exception:
                pass

        # Robust fallback: Canny + column projection peak detection.
        if not xs_w:
            try:
                can = cv2.Canny(blur, 50, 150)
                proj = np.mean((can.astype(np.uint8) > 0).astype(np.float32), axis=0)
                win = max(5, (int(ww_img * 0.015) | 1))
                ker = np.ones((int(win),), dtype=np.float32) / float(max(1, int(win)))
                proj_s = np.convolve(proj, ker, mode="same")
                try:
                    thrp = float(max(0.03, np.percentile(proj_s, 99) * 0.35))
                except Exception:
                    thrp = 0.05
                mask = proj_s >= thrp
                groups = []
                start = None
                for xi, on in enumerate(mask.tolist()):
                    if on and start is None:
                        start = int(xi)
                    elif (not on) and start is not None:
                        groups.append((int(start), int(xi - 1)))
                        start = None
                if start is not None:
                    groups.append((int(start), int(len(mask) - 1)))

                max_w = max(2, int(ww_img * 0.04))
                for a, b in groups:
                    gw = int(b - a + 1)
                    if gw > int(max_w):
                        continue
                    xc = int((int(a) + int(b)) // 2)
                    if xc < int(ww_img * 0.08) or xc > int(ww_img * 0.92):
                        continue
                    xs_w.append((int(xc), int(max(1, gw))))

                try:
                    if dbg_on:
                        if ts is None:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        if out_dir is None:
                            out_dir = _get_debug_sep_dir()
                            if out_dir is None:
                                raise RuntimeError("debug_sep_dir unavailable")
                        cv2.imwrite(str(out_dir / f"sep_{ts}_canny.png"), can)
                except Exception:
                    pass
            except Exception:
                pass

        if not xs_w:
            try:
                gx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
                gx = cv2.convertScaleAbs(gx)
                try:
                    thr_e = int(max(25, np.percentile(gx, 97)))
                except Exception:
                    thr_e = 40
                _t, bw_e = cv2.threshold(gx, int(thr_e), 255, cv2.THRESH_BINARY)
                k_h2 = max(8, int(hh * 0.65))
                ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(k_h2)))
                vert2 = cv2.morphologyEx(bw_e, cv2.MORPH_OPEN, ker2, iterations=1)
                try:
                    contours2, _hier2 = cv2.findContours(vert2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                except Exception:
                    _res2 = cv2.findContours(vert2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours2 = _res2[0] if _res2 else []
                max_w2 = max(2, int(ww_img * 0.020))
                for c in contours2 or []:
                    x, y, ww, hh0 = cv2.boundingRect(c)
                    if hh0 < int(hh * 0.70):
                        continue
                    if ww > int(max_w2):
                        continue
                    xc = int(x + (ww // 2))
                    if xc < int(ww_img * 0.08) or xc > int(ww_img * 0.92):
                        continue
                    xs_w.append((int(xc), int(max(1, ww))))
            except Exception:
                pass

        if not xs_w:
            try:
                mask = (bw_bright.astype(np.uint8) > 0)
                if mask.ndim == 2 and mask.shape[0] == hh and mask.shape[1] == ww_img:
                    min_run = int(max(6, hh * 0.70))
                    max_w = max(2, int(ww_img * 0.020))
                    cur_start = None
                    for x in range(int(ww_img)):
                        if x < int(ww_img * 0.08) or x > int(ww_img * 0.92):
                            continue

                        colm = mask[:, x]
                        run = 0
                        max_run = 0
                        for v in colm.tolist():
                            if v:
                                run += 1
                                if run > max_run:
                                    max_run = run
                            else:
                                run = 0
                        if int(max_run) >= int(min_run):
                            if cur_start is None:
                                cur_start = int(x)
                        else:
                            if cur_start is not None:
                                a = int(cur_start)
                                b = int(x - 1)
                                gw = int(b - a + 1)
                                if gw <= int(max_w):
                                    xc = int((a + b) // 2)
                                    xs_w.append((int(xc), int(gw)))
                                cur_start = None
                    if cur_start is not None:
                        a = int(cur_start)
                        b = int(ww_img - 1)
                        gw = int(b - a + 1)
                        if gw <= int(max_w):
                            xc = int((a + b) // 2)
                            xs_w.append((int(xc), int(gw)))
            except Exception:
                pass

        if not xs_w:
            return 0

        # Merge nearby detections.
        xs_w.sort(key=lambda t: t[0])
        merged = []
        # Larger gap to avoid double-counting thick or edge-detected separators.
        min_gap = max(6, int(ww_img * 0.035))
        for x, wline in xs_w:
            if not merged:
                merged.append([int(x), int(wline)])
                continue
            prev_x, prev_w = merged[-1]
            gap = max(int(min_gap), int(prev_w) + int(wline))
            if abs(int(x) - int(prev_x)) <= int(gap):
                # Merge: keep midpoint.
                merged[-1][0] = int((int(prev_x) + int(x)) // 2)
                merged[-1][1] = int(max(int(prev_w), int(wline)))
            else:
                merged.append([int(x), int(wline)])

        return int(len(merged))
    except Exception:
        return 0


def _strict_mixed_script_merge_tokens(text_block: str, top_k: int = 3) -> list[str]:
    try:
        s = str(text_block or "")
        if not s:
            return []
        toks = re.split(r"[\s\|]+", s)
        out: list[str] = []
        seen = set()
        k = max(1, int(top_k or 1))
        for tok in toks:
            t = (tok or "").strip().strip("'\"`.,;:!?()[]{}<>")
            if not t:
                continue
            if not (re.search(r"[A-Za-z]", t) and re.search(r"[\uAC00-\uD7A3]", t)):
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            if len(out) >= k:
                break
        return out
    except Exception:
        return []


def _strict_guess_overlap_from_text_no_sep(text_block: str, top_k: int = 2) -> list[str]:
    try:
        s = str(text_block or "").strip()
        if not s:
            return []
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if not lines:
            return []

        def _is_kr(ch: str) -> bool:
            try:
                o = ord(ch)
                return 0xAC00 <= o <= 0xD7A3
            except Exception:
                return False

        def _boundary_snippet(t: str) -> str:
            txt = str(t or "")
            best = ""
            for i in range(1, len(txt)):
                a, b = txt[i - 1], txt[i]
                # Only KR<->non-KR transitions are meaningful for overlap.
                if _is_kr(a) != _is_kr(b):
                    lo = max(0, i - 6)
                    hi = min(len(txt), i + 6)
                    cand = txt[lo:hi].strip().strip("'\"`.,;:!?()[]{}<>")
                    if len(cand) > len(best):
                        best = cand
            try:
                if " " in best:
                    best = max([p for p in str(best).split() if p], key=len)
            except Exception:
                pass
            return str(best)

        # Prefer explicit mixed-script tokens.
        out: list[str] = []
        seen = set()
        k = max(1, int(top_k or 1))

        mixed = _strict_mixed_script_merge_tokens(s, top_k=k)
        for m in mixed:
            key = str(m).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(str(m))
            if len(out) >= k:
                return out

        # Otherwise, look for script-boundary snippets within the longest lines.
        lines.sort(key=len, reverse=True)
        for ln in lines[:4]:
            sn = _boundary_snippet(ln)
            if not sn:
                continue
            key = sn.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(sn)
            if len(out) >= k:
                break

        return out
    except Exception:
        return []

def _strict_pick_overlap_tokens_from_line(line: str, top_k: int = 3) -> list[str]:
    try:
        ln = str(line or "").strip()
        if not ln:
            return []

        out: list[str] = []
        seen = set()

        # Only return truly overlapped tokens: mixed-script (Hangul + Latin).
        def _is_mixed_token(tok: str) -> bool:
            try:
                t = str(tok or "")
                if not t:
                    return False
                return bool(re.search(r"[A-Za-z]", t) and re.search(r"[\uAC00-\uD7A3]", t))
            except Exception:
                return False

        # First: pick the best mixed token from the whole line.
        toks_all = [t for t in re.split(r"[\s\|]+", ln) if t and str(t).strip()]
        cleaned: list[str] = []
        for t in toks_all:
            tt = str(t).strip().strip("'\"`.,;:!?()[]{}<>")
            if len(tt) < 2:
                continue
            if not _is_mixed_token(tt):
                continue
            cleaned.append(tt)

        cleaned.sort(key=lambda x: (len(x), x), reverse=True)
        k = max(1, int(top_k or 1))
        for tt in cleaned:
            key = tt.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tt)
            if len(out) >= k:
                break
        return out
    except Exception:
        return []


def _strict_guess_overlap_from_missing_sep(line: str, expected_cols: int, top_k: int = 1) -> list[str]:
    try:
        ln = str(line or "").strip()
        if not ln or "|" not in ln:
            return []
        parts = [p.strip() for p in ln.split("|")]
        parts = [p for p in parts if p]
        if expected_cols and len(parts) >= int(expected_cols):
            return []

        def _is_kr(ch: str) -> bool:
            try:
                o = ord(ch)
                return 0xAC00 <= o <= 0xD7A3
            except Exception:
                return False

        def _boundary_snippet(s: str) -> str:
            t = str(s or "")
            best = ""
            for i in range(1, len(t)):
                a, b = t[i - 1], t[i]
                # Only KR<->non-KR transitions are meaningful for overlap.
                if _is_kr(a) != _is_kr(b):
                    lo = max(0, i - 6)
                    hi = min(len(t), i + 6)
                    cand = t[lo:hi].strip()
                    if len(cand) > len(best):
                        best = cand
            best = best.strip().strip("'\"`.,;:!?()[]{}<>")
            try:
                if " " in best:
                    best = max([p for p in best.split() if p], key=len)
            except Exception:
                pass
            return best

        scored = []
        for p in parts:
            pp = str(p)
            base = len(pp)
            snip = _boundary_snippet(pp)
            bonus = 10 if snip else 0
            # Only consider tokens that look like a merged overlap.
            is_mixed = False
            try:
                is_mixed = bool(re.search(r"[A-Za-z]", pp) and re.search(r"[\uAC00-\uD7A3]", pp))
            except Exception:
                is_mixed = False
            if not snip and not is_mixed:
                continue
            scored.append((base + bonus, snip, pp))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return []

        out: list[str] = []
        seen = set()
        k = max(1, int(top_k or 1))
        for _score, snip, pp in scored:
            cand = snip if snip else pp
            cand = str(cand).strip().strip("'\"`.,;:!?()[]{}<>")
            if not cand:
                continue
            key = cand.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
            if len(out) >= k:
                break
        return out
    except Exception:
        return []


def _strict_pick_column_line(text_block: str) -> str:
    try:
        if not text_block:
            return ""
        best_ln = ""
        best_cols = 0
        for raw in str(text_block).splitlines():
            ln = str(raw).strip()
            if "|" not in ln:
                continue
            parts = [p.strip() for p in ln.split("|")]
            parts = [p for p in parts if p]
            cols = len(parts)
            if cols > best_cols:
                best_cols = cols
                best_ln = ln
        return best_ln.strip()
    except Exception:
        return ""


def _camera_tuning_path() -> Path:
    try:
        return Path(__file__).resolve().parent / ".walkie_camera_tuning.json"
    except Exception:
        return Path(".walkie_camera_tuning.json")


def _load_camera_tuning() -> dict:
    p = _camera_tuning_path()
    try:
        if not p.exists():
            return {}
    except Exception:
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_camera_tuning(values: dict) -> None:
    if not isinstance(values, dict):
        return
    p = _camera_tuning_path()
    try:
        data = {k: int(values.get(k, 0) or 0) for k in ["Brightness", "Sharpness", "Focus"]}
    except Exception:
        return
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        return


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
        saved = _load_camera_tuning()
        if saved:
            for k, vmax in max_vals.items():
                if k in saved:
                    v = int(saved.get(k) or 0)
                    v = max(0, min(int(vmax), int(v)))
                    state["values"][k] = v
    except Exception:
        pass

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
    panel_h = 200
    x0 = 10
    y0 = max(10, h - panel_h - 10)
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    out = img_bgr
    try:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), -1)
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
    row_text_y = [y0 + 48, y0 + 108, y0 + 168]
    row_slider_y = [y0 + 66, y0 + 126, y0 + 186]
    overlay["layout"] = {}

    try:
        cv2.putText(out, "Camera Settings", (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    except Exception:
        pass

    for i, lab in enumerate(labels):
        y_text = row_text_y[i]
        y_slider = row_slider_y[i]
        v = int(vals.get(lab, 0) or 0)
        vmax = int(max_vals.get(lab, 255))
        v = max(0, min(vmax, v))
        try:
            tx = x0 + 10
            ty = y_text
            bg_x1 = x0 + 6
            bg_y1 = y_text - 18
            bg_x2 = slider_left - 8
            bg_y2 = y_text + 8
            try:
                cv2.rectangle(out, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), (0, 0, 0), -1)
            except Exception:
                pass
            text = f"{lab}: {v}"
            try:
                cv2.putText(out, text, (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            except Exception:
                pass
            cv2.putText(out, text, (int(tx), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        except Exception:
            pass
        try:
            cv2.line(out, (slider_left, y_slider), (slider_right, y_slider), (190, 190, 190), 2)
            denom = float(vmax) if float(vmax) > 0 else 1.0
            knob_x = int(slider_left + (slider_right - slider_left) * (float(v) / denom))
            cv2.circle(out, (knob_x, y_slider), 7, (0, 200, 255), -1)
            cv2.circle(out, (knob_x, y_slider), 7, (0, 0, 0), 1)
        except Exception:
            pass
        overlay["layout"][lab] = {
            "x1": int(slider_left),
            "x2": int(slider_right),
            "y": int(y_slider),
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
            try:
                _save_camera_tuning(overlay.get("values") or {})
            except Exception:
                pass
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


def _strict_separator_bridge_error(screen_roi: np.ndarray) -> bool:
    try:
        if screen_roi is None or getattr(screen_roi, "size", 0) == 0:
            return False
        if len(screen_roi.shape) < 2:
            return False

        h, w = screen_roi.shape[:2]
        if h < 20 or w < 20:
            return False

        # Focus bridge analysis on the bottom softkey row where separators exist.
        y0 = int(max(0, h * 0.68))
        y1 = int(min(h, h * 0.98))
        roi = screen_roi[y0:y1, :]
        if roi is None or getattr(roi, "size", 0) == 0:
            return False
        hh, ww_img = roi.shape[:2]
        if hh < 10 or ww_img < 20:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass

        # Detect separator positions using the same robust morphology as the counter.
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
            bw_adapt = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                31,
                -10,
            )
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
            x0, y0, ww, hh0 = cv2.boundingRect(c)
            if hh0 < int(hh * 0.75):
                continue
            if int(y0) > int(hh * 0.20):
                continue
            if int(y0 + hh0) < int(hh * 0.92):
                continue
            if ww > max(6, int(ww_img * 0.03)):
                continue
            if int(hh0) < int(ww) * 6:
                continue
            xc = int(x0 + (ww // 2))
            if xc < int(ww_img * 0.12) or xc > int(ww_img * 0.88):
                continue
            xs_w.append((int(xc), int(max(1, ww))))

        if not xs_w:
            try:
                col = np.mean((bw_bright.astype(np.uint8) > 0).astype(np.float32), axis=0)
                try:
                    win = max(3, int(ww_img * 0.01) | 1)
                except Exception:
                    win = 5
                kernel_1d = np.ones((int(win),), dtype=np.float32) / float(max(1, int(win)))
                col_s = np.convolve(col, kernel_1d, mode="same")
                thr = float(max(0.55, min(0.90, np.percentile(col_s, 98) * 0.85)))
                mask = col_s >= thr
                groups = []
                start = None
                for xi, on in enumerate(mask.tolist()):
                    if on and start is None:
                        start = int(xi)
                    elif (not on) and start is not None:
                        groups.append((int(start), int(xi - 1)))
                        start = None
                if start is not None:
                    groups.append((int(start), int(len(mask) - 1)))

                max_w = max(2, int(ww_img * 0.020))
                for a, b in groups:
                    gw = int(b - a + 1)
                    if gw > int(max_w):
                        continue
                    xc = int((int(a) + int(b)) // 2)
                    if xc < int(ww_img * 0.12) or xc > int(ww_img * 0.88):
                        continue
                    xs_w.append((int(xc), int(gw)))
            except Exception:
                pass

        if not xs_w:
            try:
                mask = (bw_bright.astype(np.uint8) > 0)
                if mask.ndim == 2 and mask.shape[0] == hh and mask.shape[1] == ww_img:
                    min_run = int(max(6, hh * 0.70))
                    max_w = max(2, int(ww_img * 0.020))
                    cur_start = None
                    for x in range(int(ww_img)):
                        if x < int(ww_img * 0.12) or x > int(ww_img * 0.88):
                            continue
                        colm = mask[:, x]
                        run = 0
                        max_run = 0
                        for v in colm.tolist():
                            if v:
                                run += 1
                                if run > max_run:
                                    max_run = run
                            else:
                                run = 0
                        if int(max_run) >= int(min_run):
                            if cur_start is None:
                                cur_start = int(x)
                        else:
                            if cur_start is not None:
                                a = int(cur_start)
                                b = int(x - 1)
                                gw = int(b - a + 1)
                                if gw <= int(max_w):
                                    xc = int((a + b) // 2)
                                    xs_w.append((int(xc), int(gw)))
                                cur_start = None
                    if cur_start is not None:
                        a = int(cur_start)
                        b = int(ww_img - 1)
                        gw = int(b - a + 1)
                        if gw <= int(max_w):
                            xc = int((a + b) // 2)
                            xs_w.append((int(xc), int(gw)))
            except Exception:
                pass
        if not xs_w:
            return False

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

        # If there is dark text inside the separator band, it's likely an overlap/bridge.
        y_top = int(max(0, hh * 0.10))
        y_bot = int(min(hh, hh * 0.95))
        band_half = max(2, int(ww_img * 0.006))
        for x, _wline in merged:
            x1 = max(0, int(x) - band_half)
            x2 = min(ww_img, int(x) + band_half + 1)
            if x2 <= x1:
                continue
            band = gray[y_top:y_bot, x1:x2]
            if band.size == 0:
                continue
            dark = float(np.mean((band.astype(np.uint8) < 90).astype(np.float32)))
            if dark > 0.08:
                return True

        return False
    except Exception:
        return False


def _draw_text_overlay(
    img_bgr: np.ndarray,
    text: str,
    org: tuple,
    font_scale: float = 0.5,
    color_bgr: tuple = (255, 255, 255),
    thickness: int = 1,
):
    if img_bgr is None or text is None:
        return img_bgr

    s = str(text)
    try:
        # Check if text contains non-ASCII characters (Cyrillic, Asian, Arabic, etc.)
        needs_unicode = any(ord(ch) > 127 for ch in s)
    except Exception:
        needs_unicode = False

    # If Pillow is missing, it will force OpenCV to draw boxes for Unicode
    if Image is None or ImageDraw is None or ImageFont is None:
        if needs_unicode:
            print("[WARNING] Pillow is not installed. Non-English text will appear as boxes! (Run: pip install pillow)")
        try:
            cv2.putText(img_bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (0, 0, 0), int(max(2, int(thickness) + 2)))
        except Exception: pass
        cv2.putText(img_bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), tuple(int(x) for x in color_bgr), int(thickness))
        return img_bgr

    # If it's just English ASCII, OpenCV is much faster
    if not needs_unicode:
        try:
            cv2.putText(img_bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (0, 0, 0), int(max(2, int(thickness) + 2)))
        except Exception: pass
        cv2.putText(img_bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), tuple(int(x) for x in color_bgr), int(thickness))
        return img_bgr

    # Calculate PIL font size roughly equivalent to OpenCV font scale
    font_size = max(12, int(28 * float(font_scale)))
    
    # Detect the specific script to pick the best font
    has_hangul = any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in s)
    has_cjk = any(
        (0x3040 <= ord(ch) <= 0x30FF) or (0x4E00 <= ord(ch) <= 0x9FFF) or 
        (0x3400 <= ord(ch) <= 0x4DBF) or (0xF900 <= ord(ch) <= 0xFAFF)
        for ch in s
    )
    has_arabic = any(0x0600 <= ord(ch) <= 0x06FF for ch in s)
    has_cyrillic = any(0x0400 <= ord(ch) <= 0x04FF for ch in s)

    import platform
    system = platform.system()
    font_candidates = []

    # Map to valid OS-specific fonts (fixed string escaping)
    if system == "Windows":
        if has_hangul:
            font_candidates.extend([r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\gulim.ttc", r"C:\Windows\Fonts\batang.ttc"])
        if has_cjk:
            font_candidates.extend([r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\meiryo.ttc"])
        if has_arabic:
            font_candidates.extend([r"C:\Windows\Fonts\arabtype.ttf", r"C:\Windows\Fonts\tahoma.ttf"])
        
        # Universal Windows fallbacks (Arial supports Cyrillic, Greek, most symbols)
        font_candidates.extend([
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\tahoma.ttf",
            r"C:\Windows\Fonts\arialuni.ttf"
        ])
    elif system == "Darwin": # macOS
        if has_hangul:
            font_candidates.extend(["/System/Library/Fonts/AppleGothic.ttf", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"])
        if has_cjk:
            font_candidates.extend(["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"])
        if has_arabic:
            font_candidates.extend(["/System/Library/Fonts/GeezaPro.ttc"])
            
        font_candidates.extend([
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf"
        ])
    else: # Linux
        if has_hangul:
            font_candidates.extend(["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"])
        if has_cjk:
            font_candidates.extend(["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"])
        if has_arabic:
            font_candidates.extend(["/usr/share/fonts/truetype/kacst/KacstOne.ttf"])
            
        font_candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        ])

    # Deduplicate and test paths
    seen = set()
    dedup = []
    for fp in font_candidates:
        if not fp or fp in seen:
            continue
        seen.add(fp)
        dedup.append(fp)

    font = None
    for fp in dedup:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                font = None

    # Absolute fallback to OpenCV if no PIL font file was found
    if font is None:
        print("[WARNING] Could not find a suitable TrueType font. Non-English text may appear as boxes.")
        try:
            cv2.putText(img_bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), (0, 0, 0), int(max(2, int(thickness) + 2)))
        except Exception: pass
        cv2.putText(img_bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), tuple(int(x) for x in color_bgr), int(thickness))
        return img_bgr

    # Draw using PIL for flawless Unicode support
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    x, y = int(org[0]), int(org[1])
    y = max(0, y - int(font_size * 0.85))

    b, g, r = [int(v) for v in color_bgr]
    
    # Draw stroke/outline for better visibility
    stroke_w = max(1, int(thickness))
    draw.text((x, y), s, font=font, fill=(0, 0, 0), stroke_width=stroke_w, stroke_fill=(0, 0, 0))
    draw.text((x, y), s, font=font, fill=(r, g, b))

    out_rgb = np.asarray(pil_img)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr

# Load environment variables
load_dotenv(override=True)

sys.path.append(str(Path(__file__).parent))

from src.fast_detector import FastDetector
from src.msi_genai_ocr import MSIGenAIOCR

class WalkieMSIApp:
    def __init__(self, config_path="configs/settings.yaml", model_path_override: str = ""):
        self._init_start_perf = time.perf_counter()
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device_profiles = {}
        try:
            raw = str(os.getenv("WALKIE_DEVICE_PROFILES_JSON", "") or "").strip()
            if raw:
                obj = json.loads(raw)
                devices = obj.get("devices") if isinstance(obj, dict) else None
                if isinstance(devices, list):
                    for d in devices:
                        if not isinstance(d, dict):
                            continue
                        try:
                            did = int(d.get("id"))
                        except Exception:
                            continue
                        name = str(d.get("name") or "").strip()
                        try:
                            exp = int(d.get("expected_softkeys")) if d.get("expected_softkeys") is not None else None
                        except Exception:
                            exp = None
                        self.device_profiles[int(did)] = {
                            "name": name,
                            "expected_softkeys": exp,
                        }
            else:
                try:
                    cfgp = Path(__file__).resolve().parent / "configs" / "device_profiles.json"
                    if cfgp.exists():
                        with open(cfgp, "r", encoding="utf-8") as f:
                            obj = json.load(f) or {}
                        devices = obj.get("devices") if isinstance(obj, dict) else None
                        if isinstance(devices, list):
                            for d in devices:
                                if not isinstance(d, dict):
                                    continue
                                try:
                                    did = int(d.get("id"))
                                except Exception:
                                    continue
                                name = str(d.get("name") or "").strip()
                                try:
                                    exp = int(d.get("expected_softkeys")) if d.get("expected_softkeys") is not None else None
                                except Exception:
                                    exp = None
                                self.device_profiles[int(did)] = {
                                    "name": name,
                                    "expected_softkeys": exp,
                                }
                except Exception:
                    pass
        except Exception:
            self.device_profiles = {}
        
        print("=" * 70)
        print("WALKIE-TRACKER WITH MSI GENAI")
        print("=" * 70)
        
        # Initialize detector
        model_path = self.config['detector']['path']
        if model_path_override:
            model_path = model_path_override
        self.detector = FastDetector(model_path)
        
        # Get camera method from config
        self.camera_method = self.config.get('camera', {}).get('method', 'opencv')  # opencv or ffmpeg
        print(f"[INFO] Using camera method: {self.camera_method.upper()}")
        
        # Initialize MSI GenAI OCR with detailed error handling
        self.use_msi_genai = False  # Assume not available until proven otherwise
        self.ocr_fallback = None
        
        try:
            print("[INFO] Initializing MSI GenAI OCR...")
            print(f"[INFO] Host: {os.getenv('MSI_HOST')}")
            print(f"[INFO] User: {os.getenv('MSI_USER_ID')}")
            
            self.ocr = MSIGenAIOCR()
            print("[INFO] MSI GenAI object created")

            try:
                self.ocr.get_or_init_session()
                self.use_msi_genai = True
            except Exception as e:
                print(f"[WARNING] GenAI session init failed before camera open: {e}")
                self.use_msi_genai = False
                
        except ValueError as e:
            print(f"[ERROR] MSI GenAI configuration error: {e}")
            print("[ERROR] Please check your .env file and ensure all required variables are set:")
            print("  - MSI_HOST")
            print("  - MSI_API_KEY")
            print("  - MSI_USER_ID")
            print("  - MSI_DATASTORE_ID")
            self.use_msi_genai = False
        except Exception as e:
            print(f"[ERROR] Failed to initialize MSI GenAI: {e}")
            print("[ERROR] Possible reasons:")
            print("  1. Network connection issue")
            print("  2. Invalid API credentials")
            print("  3. MSI service is down (404 error)")
            print("  4. Firewall blocking the connection")
            self.use_msi_genai = False
        
        # Fallback OCR
        if not self.use_msi_genai:
            try:
                from src.simple_ocr import SimpleOCR
                self.ocr_fallback = SimpleOCR()
                print("[INFO] Using simple OCR as fallback")
            except Exception as e:
                print(f"[WARNING] No fallback OCR available: {e}")
                self.ocr_fallback = None
        
        # Camera settings
        self.camera_id = self.config['camera']['source']
        env_cam = os.getenv("WALKIE_CAMERA_ID", "").strip()
        if env_cam != "":
            try:
                self.camera_id = int(env_cam)
                print(f"[INFO] WALKIE_CAMERA_ID override: {self.camera_id}")
            except Exception:
                print(f"[WARN] Invalid WALKIE_CAMERA_ID='{env_cam}' (expected int). Using config value: {self.camera_id}")
        self.disable_camera_fallbacks = os.getenv("WALKIE_DISABLE_CAMERA_FALLBACKS", "0").strip().lower() in ["1", "true", "yes", "y"]
        self.cap = None
        
        # Initialize FFmpeg if selected
        if self.camera_method == 'ffmpeg':
            try:
                from src.capture_with_ffmpeg import FFmpegCapture
                self.ffmpeg_capture = FFmpegCapture()
                print("[INFO] FFmpeg capture initialized")
            except Exception as e:
                print(f"[ERROR] Failed to initialize FFmpeg: {e}")
                print("[INFO] Falling back to OpenCV")
                self.camera_method = 'opencv'
        
        # Detection
        self.confidence = self.config['detector']['confidence']
        
        # State
        self.preview_mode = True
        self.last_boxes = []
        self.last_screens = []
        
        # Output
        out_override = str(os.getenv("WALKIE_OUTPUT_DIR", "") or "").strip()
        if out_override:
            self.output_dir = Path(out_override)
        else:
            self.output_dir = Path(self.config['output']['save_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _device_name(self, device_id: int) -> str:
        try:
            prof = (self.device_profiles or {}).get(int(device_id)) or {}
            name = str(prof.get("name") or "").strip()
            return name if name else f"Device {int(device_id)}"
        except Exception:
            return f"Device {int(device_id)}"

    def _device_expected_softkeys(self, device_id: int) -> int:
        try:
            prof = (self.device_profiles or {}).get(int(device_id)) or {}
            v = prof.get("expected_softkeys")
            if v is not None:
                return int(v)
        except Exception:
            pass
        try:
            exp_softkeys = str(os.getenv(f"WALKIE_EXPECT_SOFTKEYS_D{int(device_id)}", "") or "").strip()
            if exp_softkeys:
                return int(exp_softkeys)
        except Exception:
            pass
        try:
            exp_softkeys = str(os.getenv("WALKIE_EXPECT_SOFTKEYS", "") or "").strip()
            return int(exp_softkeys) if exp_softkeys else 0
        except Exception:
            return 0
    
    def open_camera(self):
        """Open camera based on selected method"""
        print(f"\n[INFO] Opening camera with method: {self.camera_method.upper()}")

        try:
            if getattr(self, "use_msi_genai", False) and hasattr(self, "ocr") and hasattr(self.ocr, "prefetch_session_async"):
                self.ocr.prefetch_session_async()
        except Exception:
            pass
        
        if self.camera_method == 'ffmpeg':
            return self.open_camera_ffmpeg()
        else:
            return self.open_camera_opencv()
    
    def open_camera_opencv(self):
        """Open camera using OpenCV"""
        print("[INFO] Using OpenCV for camera capture")

        # Prefer trying the configured camera first, then a minimal fallback set
        cam_candidates = []
        if self.camera_id is not None:
            cam_candidates.append(self.camera_id)
        if not self.disable_camera_fallbacks:
            for fallback_id in [0, 1]:
                if fallback_id not in cam_candidates:
                    cam_candidates.append(fallback_id)
        else:
            print("[INFO] Camera fallbacks disabled (WALKIE_DISABLE_CAMERA_FALLBACKS=1)")

        # Prefer faster Windows backend when available
        backends = [None]
        if hasattr(cv2, "CAP_DSHOW"):
            backends = [cv2.CAP_DSHOW, None]

        for cam_id in cam_candidates:
            for backend in backends:
                backend_name = "DSHOW" if backend == getattr(cv2, "CAP_DSHOW", -1) else "DEFAULT"
                print(f"  Trying OpenCV camera {cam_id} ({backend_name})...")

                try:
                    if backend is None:
                        self.cap = cv2.VideoCapture(cam_id)
                    else:
                        self.cap = cv2.VideoCapture(cam_id, backend)
                
                    if not self.cap.isOpened():
                        print(f"    Failed to open camera {cam_id}")
                        self.cap.release()
                        continue

                    # Reduce internal buffering and prefer MJPG for faster negotiation
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    except Exception:
                        pass

                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                    ret = False
                    frame = None
                    for attempt in range(2):
                        ret, frame = self.cap.read()
                        if ret and frame is not None and frame.size > 0:
                            print(f"    Got valid frame (attempt {attempt + 1})")
                            break

                    if not ret or frame is None or frame.size == 0:
                        print(f"    Cannot read valid frames from camera {cam_id}")
                        self.cap.release()
                        continue

                    print(f"[OK] OpenCV camera {cam_id} ready")
                    return True

                except Exception as e:
                    print(f"    Error: {e}")
                    if self.cap:
                        self.cap.release()
                    continue
        
        print("[ERROR] Failed to open any camera with OpenCV")
        return False
    
    def open_camera_ffmpeg(self):
        """Open camera using FFmpeg"""
        print("[INFO] Using FFmpeg for camera capture")
        
        if not hasattr(self, 'ffmpeg_capture'):
            print("[ERROR] FFmpeg not initialized")
            return False
        
        # Test FFmpeg capture
        try:
            test_capture = self.ffmpeg_capture.capture_single(
                output_path="test_capture.jpg",
                camera_id=self.camera_id,
                timeout=3
            )
            
            if test_capture and os.path.exists(test_capture):
                print("[OK] FFmpeg camera ready")
                os.remove(test_capture)  # Clean up test file
                return True
            else:
                print("[ERROR] FFmpeg capture test failed")
                return False
                
        except Exception as e:
            print(f"[ERROR] FFmpeg test failed: {e}")
            return False
    
    def capture_frame(self):
        """Capture a frame using the selected method"""
        if self.camera_method == 'ffmpeg' and hasattr(self, 'ffmpeg_capture'):
            try:
                # Capture with FFmpeg
                temp_file = self.ffmpeg_capture.capture_single(
                    output_path=None,  # Let FFmpeg generate temp file
                    camera_id=self.camera_id,
                    timeout=2
                )
                
                if temp_file and os.path.exists(temp_file):
                    frame = cv2.imread(temp_file)
                    os.remove(temp_file)  # Clean up
                    
                    if frame is not None and frame.size > 0:
                        return True, frame
                    
                print("[WARNING] FFmpeg capture failed, trying OpenCV...")
                
            except Exception as e:
                print(f"[WARNING] FFmpeg error: {e}")
        
        # Fallback to OpenCV if FFmpeg fails or not selected
        if hasattr(self, 'cap') and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame
        
        return False, None
    
    def draw_live_preview(self, frame):
        """Draw live preview"""
        preview = frame.copy()
        height, width = preview.shape[:2]
        
        # Update detection periodically
        self.last_boxes, self.last_screens = self.detector.detect_with_screens(
            frame, self.confidence
        )

        # Draw device boxes
        try:
            boxes_sorted = sorted(list(self.last_boxes or []), key=lambda b: (int(b[0]), int(b[1])))
        except Exception:
            boxes_sorted = list(self.last_boxes or [])

        for i, box in enumerate(boxes_sorted):
            try:
                device_id = i + 1
                x1, y1, x2, y2 = box
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = self._device_name(device_id)
                cv2.putText(
                    preview,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            except Exception:
                pass

        # Draw screen boxes
        for box in self.last_screens:
            try:
                x1, y1, x2, y2 = box
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 1)
            except Exception:
                pass
        
        # Add info overlay
        cv2.putText(preview, f"Walkie-Tracker Preview [{self.camera_method.upper()}]", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(preview, "SPACE=capture  T=settings  +/-=zoom  X=exit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return preview
    
    def process_capture_msi(self, frame, device_boxes):
        """Process captured frame using MSI GenAI"""
        print("\n" + "=" * 60)
        print("PROCESSING WITH MSI GENAI...")
        print("=" * 60)
        
        if not self.use_msi_genai:
            print("[WARNING] MSI GenAI not available, using fallback OCR")
        
        annotated = frame.copy()
        results = []
        screen_rois = []  # Store screen ROIs for saving

        try:
            device_boxes = sorted(list(device_boxes or []), key=lambda b: (int(b[0]), int(b[1])))
        except Exception:
            pass

        for i, box in enumerate(device_boxes):
            device_id = i + 1
            x1, y1, x2, y2 = box
            
            print(f"\n[Device {device_id}]:")
            
            # Find screen region
            screen_box = None
            for screen in self.last_screens:
                sx1, sy1, sx2, sy2 = screen
                if (sx1 >= x1 and sy1 >= y1 and sx2 <= x2 and sy2 <= y2):
                    screen_box = screen
                    break
            
            # Estimate screen if not detected
            if not screen_box:
                device_width = x2 - x1
                device_height = y2 - y1
                screen_box = (
                    x1 + device_width // 4,
                    y1 + 10,
                    x2 - device_width // 4,
                    y1 + device_height // 3
                )
            
            # Extract and process screen region
            sx1, sy1, sx2, sy2 = screen_box
            sx1, sy1 = max(0, sx1), max(0, sy1)
            sx2, sy2 = min(frame.shape[1], sx2), min(frame.shape[0], sy2)
            
            if sx2 > sx1 and sy2 > sy1:
                screen_roi = frame[sy1:sy2, sx1:sx2]
                screen_rois.append(screen_roi)

                device_roi = None
                try:
                    device_roi = frame[y1:y2, x1:x2]
                except Exception:
                    device_roi = None
                
                print(f"  Extracting text...")
                
                # Use MSI GenAI or fallback
                if self.use_msi_genai:
                    img_for_ocr = screen_roi
                    try:
                        if str(os.getenv("WALKIE_OCR_USE_DEVICE_ROI", "") or "").strip().lower() in ["1", "true", "yes", "on"]:
                            if device_roi is not None and getattr(device_roi, "size", 0) > 0:
                                img_for_ocr = device_roi
                    except Exception:
                        pass

                    text, _confidence = self.ocr.extract_text(img_for_ocr)
                    ocr_method = "msi_genai"
                    
                    # Check for errors
                    if text.startswith("API_ERROR"):
                        print(f"  [ERROR] MSI GenAI failed: {text}")
                        if self.ocr_fallback:
                            print("  Trying fallback OCR...")
                            text, _confidence = self.ocr_fallback.extract_text_simple(screen_roi)
                            ocr_method = "simple_ocr_fallback"
                elif self.ocr_fallback:
                    text, _confidence = self.ocr_fallback.extract_text_simple(screen_roi)
                    ocr_method = "simple_ocr"
                else:
                    text = "NO_OCR_AVAILABLE"
                    ocr_method = "none"
                
                # Store result
                device_name = self._device_name(device_id)
                results.append({
                    'device_id': device_id,
                    'device_name': device_name,
                    'device_box': box,
                    'screen_box': screen_box,
                    'text': text,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'ocr_method': ocr_method,
                    'error_red': False,
                    'error_evidence': "",
                })
                
                def _parse_structured_fields(block: str):
                    info = {
                        "error_red": False,
                        "error_evidence": "",
                        "error_type": "",
                        "language": "",
                        "original": "",
                        "english": "",
                    }
                    if not block:
                        return info
                    lines = [ln.rstrip("\r") for ln in str(block).splitlines()]
                    compact = [ln.strip() for ln in lines if ln.strip()]
                    if compact and compact[0].strip().lower().startswith("error detected"):
                        info["error_red"] = True
                        try:
                            first_raw = compact[0].strip()
                            first_lower = first_raw.lower()
                            if "upside" in first_lower:
                                info["error_type"] = "upside down"
                            elif "overlap" in first_lower or "bridge" in first_lower:
                                info["error_type"] = "overlap"
                            elif "misalign" in first_lower:
                                info["error_type"] = "misalignment"
                            else:
                                info["error_type"] = "generic"
                                
                            ev_lines = []
                            if ":" in first_raw:
                                ev_lines.append(first_raw.split(":", 1)[-1].strip().strip('"'))
                            else:
                                ev_lines.append(first_raw)
                            
                            for ev_ln in compact[1:]:
                                ev_low = ev_ln.lower()
                                if ev_low.startswith("detected language") or ev_low.startswith("detected text"):
                                    break
                                if ev_low.startswith("likely") or ev_low.startswith("token"):
                                    ev_lines.append(ev_ln.strip())
                            info["error_evidence"] = "\n".join(ev_lines).strip()
                        except Exception:
                            pass
                    mode = None
                    buf_orig = []
                    buf_eng = []
                    for raw in lines:
                        s = raw.strip()
                        low = s.lower()
                        if low.startswith("error detected") or low.startswith("likely") or low.startswith("token"):
                            continue
                        if low.startswith("detected language:"):
                            info["language"] = s.split(":", 1)[-1].strip()
                            mode = None
                            continue
                        if low.startswith("detected text(original):"):
                            mode = "orig"
                            rest = s.split(":", 1)[-1].strip()
                            if rest and rest.lower() != "detected text(original)":
                                buf_orig.append(rest)
                            continue
                        if low.startswith("detected text(english translation):"):
                            mode = "eng"
                            rest = s.split(":", 1)[-1].strip()
                            if rest and rest.lower() != "detected text(english translation)":
                                buf_eng.append(rest)
                            continue

                        if mode == "orig" and s:
                            buf_orig.append(s)
                        elif mode == "eng" and s:
                            buf_eng.append(s)

                    info["original"] = "\n".join([x for x in buf_orig if x]).strip()
                    info["english"] = "\n".join([x for x in buf_eng if x]).strip()
                    return info

                parsed = _parse_structured_fields(text)

                # Preserve any GenAI-reported error so we can later
                # re-attach misalignment-only warnings even when no
                # visual overlap is detected.
                parsed_doc_error = bool(parsed.get("error_red"))
                parsed_doc_type = str(parsed.get("error_type") or "").strip().lower()
                parsed_doc_evidence = str(parsed.get("error_evidence") or "").strip()

                # Only visual overlap across '|' separators should be treated
                # as overlap. Reset the working flags before running
                # separator-based checks.
                parsed["error_red"] = False
                parsed["error_evidence"] = ""
                parsed["error_type"] = ""

                try:
                    sep_count = _strict_count_vertical_separators(screen_roi)
                except Exception:
                    sep_count = 0

                try:
                    bridge = bool(_strict_separator_bridge_error(screen_roi))
                except Exception:
                    bridge = False

                try:
                    results[-1]["debug_sep_count"] = int(sep_count)
                    results[-1]["debug_bridge"] = bool(bridge)
                except Exception:
                    pass
                try:
                    if str(os.getenv("WALKIE_DEBUG_OVERLAP", "") or "").strip() in ["1", "true", "yes", "on"]:
                        did_dbg = results[-1].get("device_id")
                        print(f"[DEBUG] D{did_dbg} overlap: sep_count={int(sep_count)} bridge={bool(bridge)}", flush=True)
                except Exception:
                    pass

                try:
                    # Only raise overlap errors when we detect at least one visual separator line.
                    # Require at least 2 separators to avoid false positives from UI borders/edges.
                    if int(sep_count) >= 2 and bool(bridge):
                        src = parsed.get("original") or parsed.get("english") or text
                        ln = _strict_pick_column_line(src)
                        if not ln:
                            try:
                                candidates = [x.strip() for x in str(src or "").splitlines() if x.strip()]
                                candidates.sort(key=len, reverse=True)
                                ln = candidates[0] if candidates else ""
                            except Exception:
                                ln = ""
                        toks = _strict_pick_overlap_tokens_from_line(ln, top_k=3) if ln else []
                        if ln and toks:
                            parsed["error_red"] = True
                            lines = ["Overlap (bridge)"]
                            if ln:
                                lines.append(f"Line: {ln}")
                            for i, tok in enumerate(toks):
                                lines.append(f"Token {i+1}: {tok}")
                            parsed["error_evidence"] = "\n".join(lines)
                except Exception:
                    pass

                try:
                    # Optional fallback: fixed expected on-screen columns.
                    exp_softkeys = int(self._device_expected_softkeys(device_id) or 0)
                    exp_seps = int(exp_softkeys) - 1 if int(exp_softkeys) > 0 else 0
                    if exp_seps > 0 and not bool(parsed.get("error_red")):
                        got_seps = int(sep_count)
                        try:
                            src = parsed.get("original") or parsed.get("english") or text
                            ln = _strict_pick_column_line(src)
                            if ln:
                                # Take the maximum of visual separator detection and OCR '|' count.
                                got_seps = max(int(got_seps), int(str(ln).count("|")))
                        except Exception:
                            pass

                        if int(got_seps) < int(exp_seps):
                            src = parsed.get("original") or parsed.get("english") or text
                            ln = _strict_pick_column_line(src)
                            miss = int(exp_seps) - int(got_seps)
                            try:
                                top_k = min(3, max(1, int(miss)))
                            except Exception:
                                top_k = 1
                            guesses = _strict_guess_overlap_from_missing_sep(ln, int(exp_softkeys), top_k=top_k) if ln else []
                            if not guesses:
                                guesses = _strict_guess_overlap_from_text_no_sep(src, top_k=top_k)

                            # Only flag as overlap if we can point to at least one likely merged token.
                            if guesses:
                                parsed["error_red"] = True
                                lines = [f"Overlap: missing column (expected {int(exp_seps)} separators, got {int(got_seps)})"]
                                if ln:
                                    lines.append(f"Line: {ln}")
                                for i, g in enumerate(guesses or []):
                                    lines.append(f"Likely {i+1}: {g}")
                                parsed["error_evidence"] = "\n".join(lines).strip()
                except Exception:
                    pass

                try:
                    if not parsed.get("error_red"):
                        # Mixed-script token is treated as overlap only when separators exist.
                        if int(sep_count) > 0:
                            src = parsed.get("original") or parsed.get("english") or text
                            mixed_list = _strict_mixed_script_merge_tokens(src, top_k=3)
                            if mixed_list:
                                parsed["error_red"] = True
                                if not parsed.get("error_evidence"):
                                    ln = _strict_pick_column_line(src)
                                    if not ln:
                                        try:
                                            candidates = [x.strip() for x in str(src or "").splitlines() if x.strip()]
                                            candidates.sort(key=len, reverse=True)
                                            ln = candidates[0] if candidates else ""
                                        except Exception:
                                            ln = ""
                                    lines = ["Overlap (mixed)"]
                                    if ln:
                                        lines.append(f"Line: {ln}")
                                    for i, tok in enumerate(mixed_list):
                                        lines.append(f"Token {i+1}: {tok}")
                                    parsed["error_evidence"] = "\n".join(lines).strip()
                except Exception:
                    pass

                try:
                    # Merge visual overlap detection with any GenAI-reported error
                    final_error_red = bool(parsed.get("error_red"))
                    final_error_evidence = str(parsed.get("error_evidence") or "").strip()
                    final_error_type = str(parsed.get("error_type") or "").strip()

                    if (not final_error_red) and parsed_doc_error:
                        if parsed_doc_type in ["misalignment", "upside down", "overlap"]:
                            final_error_red = True
                            final_error_type = parsed_doc_type
                            if not final_error_evidence:
                                final_error_evidence = parsed_doc_evidence

                    parsed["error_red"] = bool(final_error_red)
                    parsed["error_evidence"] = str(final_error_evidence or "")
                    parsed["error_type"] = str(final_error_type or "")

                    results[-1]["error_red"] = bool(final_error_red)
                    results[-1]["error_evidence"] = str(final_error_evidence or "").strip()
                    results[-1]["error_type"] = str(final_error_type or "")
                except Exception:
                    pass

                try:
                    if bool(parsed.get("error_red")):
                        existing = str(results[-1].get("error_evidence") or "").strip()
                        etype = str(results[-1].get("error_type") or "").strip().lower()
                        
                        if etype == "misalignment":
                            translations = {
                                "en": "Misalignment error detected",
                                "ru": "Обнаружена ошибка выравнивания",
                                "it": "Errore di allineamento rilevato",
                                "es": "Error de alineación detectado",
                                "fr": "Erreur d'alignement détectée",
                                "de": "Ausrichtungsfehler erkannt",
                                "pt": "Erro de alinhamento detectado",
                                "zh": "检测到对齐错误",
                            }
                        elif etype == "upside down":
                            translations = {
                                "en": "Upside down letters detected",
                            }
                        else:
                            translations = {
                                "en": "Overlap detected",
                                "ru": "Обнаружено наложение",
                                "it": "Sovrapposizione rilevata",
                                "es": "Solapamiento detectado",
                                "fr": "Chevauchement détecté",
                                "de": "Überlappung erkannt",
                                "pt": "Sobreposição detectada",
                                "zh": "检测到重叠",
                            }

                        # Build a compact multilingual footer
                        try:
                            footer_lines = [f"[{k.upper()}] {v}" for k, v in translations.items()]
                            footer = "\n" + "\n".join(footer_lines)
                            if existing:
                                new_evidence = existing + "\n\n" + footer
                            else:
                                new_evidence = footer
                            results[-1]["error_evidence"] = new_evidence
                            try:
                                parsed["error_evidence"] = new_evidence
                            except Exception:
                                pass
                        except Exception:
                            pass
                except Exception:
                    pass
                device_color = (0, 0, 255) if parsed["error_red"] else (0, 255, 0)

                # Draw on image
                cv2.rectangle(annotated, (x1, y1), (x2, y2), device_color, 3)
                cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)

                # Label the device box so it's always clear which is Device 1/2/3...
                label = self._device_name(device_id)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tag_x1 = max(0, x1)
                tag_y1 = max(0, y1 - th - 10)
                tag_x2 = min(frame.shape[1] - 1, tag_x1 + tw + 12)
                tag_y2 = min(frame.shape[0] - 1, tag_y1 + th + 10)
                cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), device_color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (tag_x1 + 6, tag_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )
                
                # Display text INSIDE the device box (at the bottom)
                if text and text not in ["NO_TEXT", "NO_OCR_AVAILABLE"] and not text.startswith(("API_ERROR", "CONNECTION")):
                    orig_block = (parsed.get("original") or "").strip()
                    eng_block = (parsed.get("english") or "").strip()

                    lang_label = (parsed.get("language") or "").strip()
                    try:
                        parts = [p.strip().lower() for p in lang_label.split(",") if p.strip()]
                    except Exception:
                        parts = []
                    english_only = bool(parts) and all(p in ["english", "en"] for p in parts)

                    has_two = bool(orig_block) and bool(eng_block) and (not english_only)

                    if parsed["language"] or orig_block or eng_block:
                        if has_two:
                            orig_lines = [ln.strip() for ln in orig_block.splitlines() if ln.strip()]
                            eng_lines = [ln.strip() for ln in eng_block.splitlines() if ln.strip()]
                            display_lines = []
                        else:
                            structured_lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
                            # Do not show GenAI internal error lines in the overlay when we only care about overlap.
                            try:
                                if not bool(parsed.get("error_red")):
                                    structured_lines = [
                                        ln
                                        for ln in structured_lines
                                        if not str(ln).strip().lower().startswith("error detected")
                                    ]
                            except Exception:
                                pass
                            display_lines = structured_lines
                    else:
                        flat_text = " ".join([ln.strip() for ln in str(text).split("\n") if ln.strip()])
                        display_lines = [f'Detected: "{flat_text}"']

                    device_w = x2 - x1
                    device_h = y2 - y1
                    line_height = 18
                    font_scale = 0.52
                    margin = 6
                    bottom_pad = 8

                    try:
                        if parsed["language"] or orig_block or eng_block:
                            if has_two:
                                content_line_count = max(len(orig_lines), len(eng_lines))
                                display_count = 1 + max(1, int(content_line_count))
                            else:
                                display_count = int(len(display_lines))
                        else:
                            display_count = int(len(display_lines))
                    except Exception:
                        display_count = int(len(display_lines)) if "display_lines" in locals() else 1

                    # Calculate how many lines fit inside the device box.
                    usable_h = max(20, device_h - (2 * margin) - bottom_pad - 6)
                    max_lines = max(1, min(int(display_count), int(usable_h // line_height)))
                    text_block_h = (max_lines * line_height) + 10

                    # Text area at bottom of device box
                    text_area_y1 = max(y1 + margin, y2 - text_block_h - margin)
                    text_area_y2 = y2 - margin - bottom_pad

                    # Draw semi-transparent background inside device box
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (x1 + margin, text_area_y1), (x2 - margin, text_area_y2), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

                    text_color = (0, 0, 255) if parsed["error_red"] else (255, 255, 255)

                    if parsed["language"] or orig_block or eng_block:
                        if has_two:
                            col_gap = 10
                            col_w = max(40, (device_w - 2 * margin - col_gap) // 2)
                            left_x = x1 + margin + 4
                            right_x = left_x + col_w + col_gap
                            left_chars = max(6, col_w // 7)
                            right_chars = max(6, col_w // 7)

                            head_y = text_area_y1 + 14
                            try:
                                left_header = f"Original ({lang_label})" if lang_label else "Original"
                            except Exception:
                                left_header = "Original"
                            annotated = _draw_text_overlay(
                                annotated,
                                left_header,
                                (left_x, head_y),
                                font_scale=font_scale,
                                color_bgr=text_color,
                                thickness=1,
                            )
                            annotated = _draw_text_overlay(
                                annotated,
                                "English",
                                (right_x, head_y),
                                font_scale=font_scale,
                                color_bgr=text_color,
                                thickness=1,
                            )

                            current_y = head_y + line_height
                            usable_lines = max(1, (max_lines - 1))
                            for i in range(usable_lines):
                                ltxt = orig_lines[i] if i < len(orig_lines) else ""
                                rtxt = eng_lines[i] if i < len(eng_lines) else ""
                                if len(ltxt) > left_chars:
                                    ltxt = ltxt[: left_chars - 2] + ".."
                                if len(rtxt) > right_chars:
                                    rtxt = rtxt[: right_chars - 2] + ".."
                                if ltxt:
                                    annotated = _draw_text_overlay(
                                        annotated,
                                        ltxt,
                                        (left_x, current_y),
                                        font_scale=font_scale,
                                        color_bgr=text_color,
                                        thickness=1,
                                    )
                                if rtxt:
                                    annotated = _draw_text_overlay(
                                        annotated,
                                        rtxt,
                                        (right_x, current_y),
                                        font_scale=font_scale,
                                        color_bgr=text_color,
                                        thickness=1,
                                    )
                                current_y += line_height
                        else:
                            current_y = text_area_y1 + 14
                            max_chars = max(8, (device_w - 2 * margin - 10) // 7)
                            try:
                                head = f"Original ({lang_label})" if lang_label else "Original"
                            except Exception:
                                head = "Original"
                            annotated = _draw_text_overlay(
                                annotated,
                                head,
                                (x1 + margin + 4, current_y),
                                font_scale=font_scale,
                                color_bgr=text_color,
                                thickness=1,
                            )
                            current_y += line_height
                            for line_text in display_lines[:max_lines]:
                                if len(line_text) > max_chars:
                                    line_text = line_text[: max_chars - 2] + ".."
                                annotated = _draw_text_overlay(
                                    annotated,
                                    line_text,
                                    (x1 + margin + 4, current_y),
                                    font_scale=font_scale,
                                    color_bgr=text_color,
                                    thickness=1,
                                )
                                current_y += line_height
                    else:
                        current_y = text_area_y1 + 14
                        max_chars = max(8, (device_w - 2 * margin - 10) // 7)
                        for line_text in display_lines[:max_lines]:
                            if len(line_text) > max_chars:
                                line_text = line_text[: max_chars - 2] + ".."
                            annotated = _draw_text_overlay(
                                annotated,
                                line_text,
                                (x1 + margin + 4, current_y),
                                font_scale=font_scale,
                                color_bgr=text_color,
                                thickness=1,
                            )
                            current_y += line_height

                else:
                    # Show error status
                    error_text = f"Device {device_id}: {text[:30]}..." if len(text) > 30 else f"Device {device_id}: {text}"
                    frame_height, frame_width = annotated.shape[:2]
                    error_y = min(y2 + 30, frame_height - 10)
                    cv2.putText(annotated, error_text, (max(10, x1), error_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        print(f"\n[OK] Processing complete!")
        print(f"   Devices: {len(results)}")
        print("=" * 60)
        
        return annotated, results, screen_rois


    
    def run(self):
        """Main application loop"""
        if not self.open_camera():
            print("[ERROR] Could not open camera. Exiting...")
            return

        try:
            autoexit = str(os.getenv("WALKIE_LAUNCHER_AUTOEXIT", "") or "").strip().lower() in ["1", "true", "yes", "on"]
        except Exception:
            autoexit = False

        overlay = None
        try:
            if self.camera_method == 'opencv' and getattr(self, 'cap', None) is not None:
                overlay = _create_camera_overlay_state(self.cap)
        except Exception:
            overlay = None
        zoom = 1.0
        preview_window_inited = False
        preview_window_name = None
        cancelled = False

        try:
            init_elapsed = time.perf_counter() - float(self._init_start_perf)
            print(f"[TIMING] Initialization: {init_elapsed:.3f}s")
        except Exception:
            pass
        
        print("\n🎮 CONTROLS:")
        print(f"  SPACE: Capture and extract text with {('MSI GenAI' if self.use_msi_genai else 'Simple OCR')}")
        print("  C: Switch camera method (OpenCV/FFmpeg)")
        print("  X: Exit")
        print("  T: Toggle camera settings (Brightness/Sharpness/Focus)")
        print("  +/-: Zoom in/out, Z: Reset zoom")
        print("\n[INFO] Camera method:", self.camera_method.upper())
        print("[INFO] MSI GenAI:", "✓ Available" if self.use_msi_genai else "✗ NOT available (using fallback)")
        if not self.use_msi_genai:
            print("[INFO] Reason: MSI GenAI initialization failed - check network and API credentials")
        print("=" * 70)
        
        while True:
            try:
                if overlay is not None and self.camera_method == 'opencv' and getattr(self, 'cap', None) is not None:
                    _apply_camera_overlay_settings(self.cap, overlay)
            except Exception:
                pass

            if not self.preview_mode:
                key = cv2.waitKey(50) & 0xFF

                if key in [ord('x'), ord('X')]:
                    break
                elif key in [ord('t'), ord('T')]:
                    try:
                        if overlay is None and self.camera_method == 'opencv' and getattr(self, 'cap', None) is not None:
                            overlay = _create_camera_overlay_state(self.cap)
                        if overlay is not None:
                            overlay["enabled"] = not bool(overlay.get("enabled"))
                            overlay["drag"] = None
                            print(f"[INFO] Camera Settings Overlay: {'ON' if overlay.get('enabled') else 'OFF'}")
                    except Exception:
                        pass
                elif key in [ord('+'), ord('=')]:
                    zoom = min(4.0, float(zoom) + 0.1)
                elif key in [ord('-'), ord('_')]:
                    zoom = max(1.0, float(zoom) - 0.1)
                elif key in [ord('z'), ord('Z')]:
                    zoom = 1.0

                continue

            ret, frame = self.capture_frame()
            
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame")
                time.sleep(0.1)
                continue

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
            
            if self.preview_mode:
                win_name = f"Walkie-Tracker [{self.camera_method.upper()}]"
                if preview_window_name != win_name:
                    preview_window_name = win_name
                    preview_window_inited = False

                if preview_window_inited:
                    try:
                        if hasattr(cv2, "getWindowProperty") and hasattr(cv2, "WND_PROP_VISIBLE"):
                            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                                cancelled = True
                                print("[INFO] Preview window closed. Exiting...", flush=True)
                                break
                    except Exception:
                        cancelled = True
                        print("[INFO] Preview window closed. Exiting...", flush=True)
                        break
                else:
                    try:
                        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    except Exception:
                        pass
                    preview_window_inited = True

                preview = self.draw_live_preview(frame)
                try:
                    if overlay is not None:
                        preview = _draw_camera_overlay(preview, overlay)
                except Exception:
                    pass
                cv2.imshow(win_name, preview)
                try:
                    if overlay is not None and overlay.get("enabled"):
                        _attach_camera_overlay_mouse(win_name, overlay)
                except Exception:
                    pass
            
            key = cv2.waitKey(1) & 0xFF
            
            if key in [ord('x'), ord('X')]:
                break

            elif key in [ord('t'), ord('T')]:
                try:
                    if overlay is None and self.camera_method == 'opencv' and getattr(self, 'cap', None) is not None:
                        overlay = _create_camera_overlay_state(self.cap)
                    if overlay is not None:
                        overlay["enabled"] = not bool(overlay.get("enabled"))
                        overlay["drag"] = None
                        print(f"[INFO] Camera Settings Overlay: {'ON' if overlay.get('enabled') else 'OFF'}")
                except Exception:
                    pass

            elif key in [ord('+'), ord('=')]:
                zoom = min(4.0, float(zoom) + 0.1)

            elif key in [ord('-'), ord('_')]:
                zoom = max(1.0, float(zoom) - 0.1)

            elif key in [ord('z'), ord('Z')]:
                zoom = 1.0
            
            elif key == ord('c'):
                # Switch camera method
                self.camera_method = 'ffmpeg' if self.camera_method == 'opencv' else 'opencv'
                print(f"\n[INFO] Switched to {self.camera_method.upper()} camera method")

                # Reinitialize camera
                if hasattr(self, 'cap'):
                    self.cap.release()
                self.open_camera()

                overlay = None
                try:
                    if self.camera_method == 'opencv' and getattr(self, 'cap', None) is not None:
                        overlay = _create_camera_overlay_state(self.cap)
                except Exception:
                    overlay = None

            elif key == ord(' ') and self.preview_mode:
                t_capture_start = time.perf_counter()
                captured = frame.copy()
                
                if len(self.last_boxes) == 0:
                    print("⚠ No devices detected!")
                    continue
                
                cv2.destroyWindow(f"Walkie-Tracker [{self.camera_method.upper()}]")
                self.preview_mode = False
                
                processed_image, results, screen_rois = self.process_capture_msi(captured, self.last_boxes)

                try:
                    t_capture_end = time.perf_counter()
                    print(f"[TIMING] Capture->OCR total: {(t_capture_end - t_capture_start):.3f}s")
                except Exception:
                    pass
                
                # Show results
                self.show_results(processed_image, results)

                try:
                    self.show_error_popup(results)
                except Exception:
                    pass
                
                # Ask to save
                print("\nS= save or any key to cont")
                key = self._wait_key_or_close("Results - Press any key to exit", 50)
                
                if key == ord('s'):
                    self.save_results(captured, processed_image, results, screen_rois)

                
                self.preview_mode = True
                try:
                    cv2.destroyWindow("Results - Press any key to exit")
                except Exception:
                    pass
                try:
                    cv2.destroyWindow(f"Walkie-Tracker [{self.camera_method.upper()}]")
                except Exception:
                    pass
        
        # Cleanup
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n🎉 Application closed")
        if cancelled:
            try:
                os._exit(0)
            except Exception:
                return

    def run_once(self, warmup_sec: float = 0.0):
        if not self.open_camera():
            print("Detected: ''", flush=True)
            return
        try:
            warmup = float(warmup_sec or 0.0)
            if warmup > 0:
                try:
                    if getattr(self, "use_msi_genai", False) and hasattr(self, "ocr") and hasattr(self.ocr, "prefetch_session_async"):
                        self.ocr.prefetch_session_async()
                except Exception:
                    pass
                print(f"[INFO] Warmup: waiting {warmup:.1f}s for camera autofocus...", flush=True)
                t_end = time.time() + warmup
                last_frame = None
                while time.time() < t_end:
                    r, f = self.capture_frame()
                    if r and f is not None and f.size > 0:
                        last_frame = f
                    time.sleep(0.03)
                if last_frame is not None:
                    ret, frame = True, last_frame
                else:
                    ret, frame = self.capture_frame()

            if not ret or frame is None:
                print("Detected: ''", flush=True)
                return

            try:
                boxes, screens = self.detector.detect_with_screens(frame, self.confidence)
            except Exception:
                boxes, screens = [], []

            if not boxes:
                print("Detected: ''", flush=True)
                return

            try:
                self.last_screens = screens or []
            except Exception:
                pass

            first_box = boxes[0]
            _annotated, results, _screen_rois = self.process_capture_msi(frame, [first_box])
            text_out = ""
            try:
                if results:
                    text_out = (results[0].get("text") or "").strip()
            except Exception:
                text_out = ""

            def _parse_structured_summary(block: str) -> str:
                if not block:
                    return ""
                english = ""
                original = ""
                for ln in str(block).splitlines():
                    s = ln.strip()
                    low = s.lower()
                    if low.startswith("detected text(english translation):"):
                        english = s.split(":", 1)[-1].strip()
                    elif low.startswith("detected text(original):"):
                        original = s.split(":", 1)[-1].strip()
                picked = english or original or block
                return " ".join([x.strip() for x in str(picked).splitlines() if x.strip()])

            summary = _parse_structured_summary(text_out)
            print(f"Detected: '{summary}'", flush=True)
        finally:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
    
    def show_results(self, image, results):
        """Display results window"""
        height, width = image.shape[:2]
        
        # Create header
        header = np.zeros((170, width, 3), dtype=np.uint8)
        cv2.putText(header, "MSI GenAI Results", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(header, f"Camera: {self.camera_method.upper()}", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(header, "S = Save", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
        cv2.putText(header, "Press any key to exit", 
                   (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
        
        # Create summary
        summary = np.zeros((90, width, 3), dtype=np.uint8)
        
        devices_with_text = sum(1 for r in results if r['text'] and r['text'] not in ["NO_TEXT", "NO_OCR_AVAILABLE"] and not r['text'].startswith(("API_ERROR", "CONNECTION")))
        
        cv2.putText(summary, f"Devices Detected: {len(results)}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(summary, f"Text Extracted: {devices_with_text}/{len(results)}", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 255, 0) if devices_with_text > 0 else (255, 255, 255), 2)
        
        # Combine and show
        combined = np.vstack([header, summary, image])
        cv2.imshow("Results - Press any key to exit", combined)

    def _wait_key_or_close(self, window_name: str, delay_ms: int = 50):
        while True:
            try:
                if hasattr(cv2, "getWindowProperty") and hasattr(cv2, "WND_PROP_VISIBLE"):
                    if cv2.getWindowProperty(str(window_name), cv2.WND_PROP_VISIBLE) < 1:
                        return None
            except Exception:
                pass
            try:
                k = cv2.waitKey(int(delay_ms))
            except Exception:
                k = -1
            if k is not None and int(k) != -1:
                return k

    def show_error_popup(self, results):
        try:
            items = []
            for r in results or []:
                try:
                    if not r.get("error_red"):
                        continue

                    did = r.get("device_id")
                    dname = str(r.get("device_name") or "").strip()
                    if not dname and did is not None:
                        try:
                            dname = self._device_name(int(did))
                        except Exception:
                            dname = ""
                    if not dname and did is not None:
                        dname = f"Device {did}"

                    ev = str(r.get("error_evidence") or "").strip()
                    ev = ev if ev else "Error Detected"
                    parts = [ln.strip() for ln in str(ev).splitlines() if ln.strip()]
                    first = parts[0] if parts else str(ev).strip()
                    try:
                        first = str(first).replace("Overlap (bridge)", "Overlap").replace("Overlap (mixed)", "Overlap")
                    except Exception:
                        pass

                    words = []
                    try:
                        for ln in parts[1:]:
                            # Find tokens/evidence text
                            m = re.match(r"^(Token|Likely)\s*\d+\s*:\s*(.+)$", str(ln).strip(), flags=re.IGNORECASE)
                            if not m:
                                continue
                            w = (m.group(2) or "").strip()
                            if not w:
                                continue
                            
                            # Deduplicate and append (We no longer artificially restrict it to Hangul+Latin!)
                            if w.lower() not in [x.lower() for x in words]:
                                words.append(w)
                                
                            if len(words) >= 3:
                                break
                    except Exception:
                        pass

                    if words:
                        msg = ", ".join(words)
                    else:
                        msg = str(first)
                        # Fallback parsing if regex failed but it has "Likely:" format
                        if not words:
                            for ln in parts[1:]:
                                if ":" in ln:
                                    msg = ln.split(":", 1)[1].strip()
                                    break

                    items.append({
                        "device": str(dname or ""),
                        "kind": str(first or ""),
                        "msg": str(msg or ""),
                    })
                except Exception:
                    continue
            if not items:
                return

            def _classify_kind(s: str) -> str:
                try:
                    low = str(s or "").strip().lower()
                    if "upside" in low: return "Upside Down"
                    if "overlap" in low: return "Overlap"
                    if "missing column" in low: return "Overlap"
                    if "misalign" in low: return "Misalignment"
                    return "Error"
                except Exception:
                    return "Error"

            def _truncate(s: str, max_chars: int) -> str:
                try:
                    t = " ".join(str(s or "").split())
                    if len(t) <= int(max_chars):
                        return t
                    return t[: max(0, int(max_chars) - 3)] + "..."
                except Exception:
                    return str(s or "")

            errs = []
            max_devices = 5
            for it in items[:max_devices]:
                dev = str(it.get("device") or "")
                msg = str(it.get("msg") or "")
                
                raw_kind = str(it.get("kind") or "")
                item_kind = _classify_kind(raw_kind)
                
                msg = _truncate(msg, 60)
                
                # Requested formatting logic here:
                if dev:
                    errs.append(f"{item_kind}")
                    errs.append(f"{dev}: {msg}")
                else:
                    errs.append(f"{item_kind}")
                    errs.append(f"{msg}")
                    
                errs.append("") # Blank line separator

            pad = 24
            title_scale = 1.4
            label_scale = 0.9
            body_scale = 0.9
            line_h = 48

            def _wrap_line(s: str, max_w: int, scale: float) -> list:
                try:
                    t = str(s or "").strip()
                    if not t:
                        return [""]
                    if cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, float(scale), 2)[0][0] <= int(max_w):
                        return [t]
                    words = t.split(" ")
                    out = []
                    cur = ""
                    for w0 in words:
                        cand = (cur + " " + w0).strip() if cur else w0
                        tw = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, float(scale), 2)[0][0]
                        if tw <= int(max_w):
                            cur = cand
                        else:
                            if cur:
                                out.append(cur)
                            cur = w0
                    if cur:
                        out.append(cur)
                    if out:
                        return out

                    chunk = max(8, int(len(t) * 0.33))
                    out2 = []
                    i = 0
                    while i < len(t):
                        out2.append(t[i : i + chunk])
                        i += chunk
                    return out2 if out2 else [t]
                except Exception:
                    return [str(s or "")]

            # First pass wrap to estimate needed width.
            max_text_w = 900
            wrapped = []
            for s in errs:
                wrapped.extend(_wrap_line(s, max_text_w, body_scale))

            try:
                max_line_px = 0
                for ln in wrapped:
                    if not ln:
                        continue
                    try:
                        (tw, _th), _ = cv2.getTextSize(str(ln), cv2.FONT_HERSHEY_SIMPLEX, float(body_scale), 2)
                        max_line_px = max(int(max_line_px), int(tw))
                    except Exception:
                        continue
                w = int(max(640, min(1200, max_line_px + (2 * pad) + 80)))
            except Exception:
                w = 900

            # Second pass wrap using final width.
            max_text_w = max(200, int(w - (2 * pad) - 16))
            wrapped = []
            for s in errs:
                wrapped.extend(_wrap_line(s, max_text_w, body_scale))

            footer_h = 56
            h = (pad + 70) + (pad + 40) + (len(wrapped) * line_h) + footer_h
            h = int(max(h, 320))

            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(img, "ERROR DETECTED", (pad, pad + 40), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 0, 255), 4)

            y = pad + 110
            for s in wrapped:
                if s:
                    img = _draw_text_overlay(img, str(s), (pad + 8, y), font_scale=body_scale, color_bgr=(255, 255, 255), thickness=2)
                y += line_h

            cv2.putText(img, "Press any key to close", (pad, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

            cv2.namedWindow("Error Detected", cv2.WINDOW_NORMAL)
            try:
                cv2.resizeWindow("Error Detected", int(w), int(h))
            except Exception:
                pass

            cv2.imshow("Error Detected", img)
            self._wait_key_or_close("Error Detected", 50)
            try:
                cv2.destroyWindow("Error Detected")
            except Exception:
                pass    
        except Exception:
            return
    
    def save_results(self, raw_image, annotated_image, results, screen_rois):
        """Save results to organized folder structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create main session folder
        session_folder = self.output_dir / f"session_{timestamp}"
        session_folder.mkdir(parents=True, exist_ok=True)
        
        # Save main images
        raw_path = session_folder / "raw_capture.jpg"
        annotated_path = session_folder / "annotated_result.jpg"
        
        cv2.imwrite(str(raw_path), raw_image)
        cv2.imwrite(str(annotated_path), annotated_image)
        
        # Create device-specific folders and save ROI images
        for i, result in enumerate(results):
            device_id = result['device_id']
            device_folder = session_folder / f"device_{device_id}"
            device_folder.mkdir(exist_ok=True)
            
            # Save device-specific information
            device_info = {
                'device_id': device_id,
                'device_name': str(result.get('device_name') or f"D{device_id}"),
                'device_box': result['device_box'],
                'screen_box': result['screen_box'],
                'detected_text': result['text'],
                'ocr_method': result['ocr_method'],
                'timestamp': result['timestamp']
            }
            
            # Save device info as JSON
            with open(device_folder / "device_info.json", 'w') as f:
                json.dump(device_info, f, indent=2)
            
            # Save screen ROI image if available
            if i < len(screen_rois):
                roi_path = device_folder / "screen_roi.jpg"
                cv2.imwrite(str(roi_path), screen_rois[i])
                
                # Also save the ROI image that was sent to GenAI
                # Extract ROI from original image
                sx1, sy1, sx2, sy2 = result['screen_box']
                sx1, sy1 = max(0, sx1), max(0, sy1)
                sx2, sy2 = min(raw_image.shape[1], sx2), min(raw_image.shape[0], sy2)
                
                if sx2 > sx1 and sy2 > sy1:
                    sent_roi = raw_image[sy1:sy2, sx1:sx2]
                    sent_roi_path = device_folder / "sent_to_genai.jpg"
                    cv2.imwrite(str(sent_roi_path), sent_roi)
        
        # Save overall session JSON
        session_data = {
            'timestamp': timestamp,
            'camera_method': self.camera_method,
            'msi_genai_available': self.use_msi_genai,
            'total_devices': len(results),
            'devices': [
                {
                    'device_id': r['device_id'],
                    'device_name': str(r.get('device_name') or f"D{r.get('device_id')}"),
                    'text': r['text'],
                    'ocr_method': r['ocr_method']
                } for r in results
            ]
        }
        
        json_path = session_folder / "session_summary.json"
        with open(json_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"\n[SAVED] Results organized in folder: {session_folder.name}")
        print(f"  Session folder: {session_folder}")
        print(f"  Device folders: {len(results)}")
        print(f"  Screen ROI images saved in each device folder")
        
        # Show extracted text
        print(f"\n📝 Extracted text:")
        for result in results:
            if result['text'] and result['text'] not in ["NO_TEXT", "NO_OCR_AVAILABLE"]:
                print(f"  Device {result['device_id']}: '{result['text']}'")
            elif result['text'].startswith(("API_ERROR", "CONNECTION")):
                print(f"  Device {result['device_id']}: ❌ Error: {result['text']}")

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--warmup-sec", type=float, default=float(os.getenv("WALKIE_WARMUP_SEC", "0") or 0))
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--use-last", action="store_true")
    args, _unknown = parser.parse_known_args()

    def _resolve_model_override(model_path: str, use_last: bool) -> str:
        model_override = (model_path or "").strip()
        if not model_override and bool(use_last):
            try:
                candidates = list(Path("runs").glob("detect/train*/weights/last.pt"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    model_override = str(candidates[0])
            except Exception:
                model_override = ""
        return model_override

    def _run_app(once: bool, warmup_sec: float, model_path: str, use_last: bool):
        model_override = _resolve_model_override(model_path=model_path, use_last=use_last)
        app = WalkieMSIApp(model_path_override=model_override)
        if once:
            app.run_once(warmup_sec=warmup_sec)
        else:
            app.run()

    def _launch_gui() -> None:
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
        except Exception as e:
            print(f"[ERROR] GUI not available (tkinter import failed): {e}")
            return

        import subprocess
        import threading

        root = tk.Tk()
        root.title("Walkie-Tracker Launcher")
        try:
            root.geometry("900x750")
        except Exception:
            pass
        try:
            root.minsize(900, 600)
        except Exception:
            pass
        root.resizable(True, True)

        frm = ttk.Frame(root, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        try:
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)
        except Exception:
            pass

        try:
            frm.columnconfigure(0, weight=1)
            frm.rowconfigure(2, weight=1)
        except Exception:
            pass

        maincol = ttk.Frame(frm)
        maincol.grid(row=2, column=0, sticky="nsew")
        try:
            maincol.columnconfigure(0, weight=1)
            maincol.rowconfigure(4, weight=1)
        except Exception:
            pass

        var_model_path = tk.StringVar(value="")
        var_camera = tk.StringVar(value="")
        var_save_dir = tk.StringVar(value="")

        device_rows = []

        try:
            cfg_path = Path(__file__).resolve().parent / "configs" / "settings.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        try:
            cfg_model = (((cfg.get("detector") or {}).get("path")) or "").strip()
        except Exception:
            cfg_model = ""
        if cfg_model:
            try:
                p = Path(cfg_model)
                if not p.is_absolute():
                    p = (Path(__file__).resolve().parent / p).resolve()
                var_model_path.set(str(p))
            except Exception:
                var_model_path.set(cfg_model)

        try:
            cfg_out = (((cfg.get("output") or {}).get("save_dir")) or "").strip()
        except Exception:
            cfg_out = ""
        try:
            env_out = (os.getenv("WALKIE_OUTPUT_DIR", "") or "").strip()
        except Exception:
            env_out = ""
        if env_out:
            var_save_dir.set(env_out)
        elif cfg_out:
            try:
                p = Path(cfg_out)
                if not p.is_absolute():
                    p = (Path(__file__).resolve().parent / p).resolve()
                var_save_dir.set(str(p))
            except Exception:
                var_save_dir.set(cfg_out)

        desired_camera_id = None
        try:
            env_cam = (os.getenv("WALKIE_CAMERA_ID", "") or "").strip()
            if env_cam != "":
                desired_camera_id = int(env_cam)
        except Exception:
            desired_camera_id = None
        if desired_camera_id is None:
            try:
                desired_camera_id = int(((cfg.get("camera") or {}).get("source")) or 0)
            except Exception:
                desired_camera_id = 0

        title = ttk.Label(frm, text="Walkie-Tracker Launcher", font=("Segoe UI", 12, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        ttk.Separator(frm).grid(row=1, column=0, sticky="ew", pady=10)

        lf_paths = ttk.LabelFrame(maincol, text="Paths", padding=10)
        lf_paths.grid(row=0, column=0, sticky="ew")

        ttk.Label(lf_paths, text="Model (.pt):").grid(row=0, column=0, sticky="w")
        ent_model = ttk.Entry(lf_paths, textvariable=var_model_path, width=46)
        ent_model.grid(row=0, column=1, sticky="ew", padx=(8, 8))

        def _browse_model() -> None:
            fp = filedialog.askopenfilename(
                title="Select model (.pt)",
                filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            )
            if fp:
                try:
                    var_model_path.set(str(Path(fp).resolve()))
                except Exception:
                    var_model_path.set(fp)

        btn_browse = ttk.Button(lf_paths, text="Browse...", command=_browse_model)
        btn_browse.grid(row=0, column=2, sticky="w")

        ttk.Label(lf_paths, text="Save folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ent_save = ttk.Entry(lf_paths, textvariable=var_save_dir, width=46)
        ent_save.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))

        def _browse_save_dir() -> None:
            d = filedialog.askdirectory(title="Select save folder")
            if d:
                try:
                    var_save_dir.set(str(Path(d).resolve()))
                except Exception:
                    var_save_dir.set(d)

        btn_browse_save = ttk.Button(lf_paths, text="Browse...", command=_browse_save_dir)
        btn_browse_save.grid(row=1, column=2, sticky="w", pady=(8, 0))

        try:
            lf_paths.columnconfigure(1, weight=1)
        except Exception:
            pass

        lf_cam = ttk.LabelFrame(maincol, text="Camera", padding=10)
        lf_cam.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        ttk.Label(lf_cam, text="Camera:").grid(row=0, column=0, sticky="w")
        cmb_camera = ttk.Combobox(lf_cam, textvariable=var_camera, width=43, state="readonly")
        cmb_camera.grid(row=0, column=1, sticky="ew", padx=(8, 8))

        cam_refresh_state = {"running": False}

        def _probe_cameras(max_id: int = 5) -> list:
            found = []
            for cam_id in range(int(max_id) + 1):
                cap = None
                try:
                    backend = getattr(cv2, "CAP_DSHOW", 0)
                    cap = cv2.VideoCapture(cam_id, backend)
                    if not cap.isOpened():
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = cv2.VideoCapture(cam_id)
                    if not cap.isOpened():
                        continue
                    ok, frame = cap.read()
                    if not ok or frame is None or getattr(frame, "size", 0) == 0:
                        continue
                    found.append(cam_id)
                except Exception:
                    continue
                finally:
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
            return found

        def _refresh_cameras() -> None:
            try:
                if bool(cam_refresh_state.get("running")):
                    return
                cam_refresh_state["running"] = True
            except Exception:
                pass

            try:
                btn_refresh_cam.configure(state="disabled")
            except Exception:
                pass

            def _apply(values: list[str]) -> None:
                try:
                    cmb_camera["values"] = values
                    if not var_camera.get():
                        if values:
                            var_camera.set(values[0])
                    elif var_camera.get() not in values and values:
                        var_camera.set(values[0])
                finally:
                    try:
                        cam_refresh_state["running"] = False
                    except Exception:
                        pass
                    try:
                        btn_refresh_cam.configure(state="normal")
                    except Exception:
                        pass

            def _worker() -> None:
                try:
                    try:
                        max_id = int(os.getenv("WALKIE_CAM_MAX_ID", "5") or 5)
                    except Exception:
                        max_id = 5
                    cams = _probe_cameras(max_id=max_id)
                    values = [f"Camera {i}" for i in cams]
                except Exception:
                    values = []
                try:
                    root.after(0, _apply, values)
                except Exception:
                    _apply(values)

            try:
                threading.Thread(target=_worker, daemon=True).start()
            except Exception:
                _apply([])

        btn_refresh_cam = ttk.Button(lf_cam, text="Refresh", command=_refresh_cameras)
        btn_refresh_cam.grid(row=0, column=2, sticky="w")

        _refresh_cameras()

        try:
            lf_cam.columnconfigure(1, weight=1)
        except Exception:
            pass

        lf_devices = ttk.LabelFrame(maincol, text="Devices", padding=10)
        lf_devices.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        selected_device_id = tk.IntVar(value=0)

        ttk.Label(lf_devices, text="").grid(row=0, column=0, sticky="w")
        ttk.Label(lf_devices, text="ID").grid(row=0, column=1, sticky="w")
        ttk.Label(lf_devices, text="Name").grid(row=0, column=2, sticky="w")
        ttk.Label(lf_devices, text="Columns").grid(row=0, column=3, sticky="w")

        try:
            lf_devices.columnconfigure(0, weight=0)
            lf_devices.columnconfigure(1, weight=0)
            lf_devices.columnconfigure(2, weight=1)
            lf_devices.columnconfigure(3, weight=0)
            lf_devices.columnconfigure(4, weight=0)
        except Exception:
            pass

        def _regrid_device_rows() -> None:
            try:
                for idx, row in enumerate(device_rows):
                    row_idx = idx + 1
                    widgets = row.get("widgets") or ()
                    if len(widgets) >= 4:
                        rb, lbl_id, ent_name, spn = widgets[:4]
                        rb.grid(row=row_idx, column=0, sticky="w", padx=(0, 6), pady=2)
                        lbl_id.grid(row=row_idx, column=1, sticky="w", padx=(0, 10), pady=2)
                        ent_name.grid(row=row_idx, column=2, sticky="ew", pady=2)
                        spn.grid(row=row_idx, column=3, sticky="w", padx=(10, 0), pady=2)
            except Exception:
                pass

        def _renumber_device_rows() -> None:
            try:
                for idx, row in enumerate(device_rows):
                    new_id = idx + 1
                    row["id"] = int(new_id)
                    widgets = row.get("widgets") or ()
                    if len(widgets) >= 2:
                        rb, lbl_id = widgets[0], widgets[1]
                        try:
                            rb.configure(value=int(new_id))
                        except Exception:
                            pass
                        try:
                            lbl_id.configure(text=f"D{int(new_id)}")
                        except Exception:
                            pass
            except Exception:
                pass

        def _add_device_row(did: int, name: str = "", expected_softkeys: int | None = None) -> None:
            row_idx = len(device_rows) + 1
            var_name = tk.StringVar(value=str(name or ""))
            var_exp = tk.StringVar(value="" if expected_softkeys is None else str(int(expected_softkeys)))

            rb = ttk.Radiobutton(lf_devices, variable=selected_device_id, value=int(did))
            lbl_id = ttk.Label(lf_devices, text=f"D{int(did)}")
            ent_name = ttk.Entry(lf_devices, textvariable=var_name, width=30)
            spn = ttk.Spinbox(lf_devices, from_=0, to=8, width=6, textvariable=var_exp)

            rb.grid(row=row_idx, column=0, sticky="w", padx=(0, 6), pady=2)
            lbl_id.grid(row=row_idx, column=1, sticky="w", padx=(0, 10), pady=2)
            ent_name.grid(row=row_idx, column=2, sticky="ew", pady=2)
            spn.grid(row=row_idx, column=3, sticky="w", padx=(10, 0), pady=2)

            device_rows.append({
                "id": int(did),
                "var_name": var_name,
                "var_expected": var_exp,
                "widgets": (rb, lbl_id, ent_name, spn),
            })

            try:
                if int(selected_device_id.get() or 0) == 0:
                    selected_device_id.set(int(did))
            except Exception:
                pass

        def _remove_selected_device_row() -> None:
            if not device_rows:
                return
            target_id = None
            try:
                target_id = int(selected_device_id.get())
            except Exception:
                target_id = None

            idx = None
            if target_id is not None:
                for i, r in enumerate(device_rows):
                    if int(r.get("id") or 0) == int(target_id):
                        idx = i
                        break
            if idx is None:
                idx = len(device_rows) - 1

            row = device_rows.pop(int(idx))
            try:
                for w in row.get("widgets") or []:
                    try:
                        w.destroy()
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                _renumber_device_rows()
                _regrid_device_rows()
            except Exception:
                pass

            try:
                if device_rows:
                    selected_device_id.set(int(device_rows[min(int(idx), len(device_rows) - 1)].get("id") or 0))
                else:
                    selected_device_id.set(0)
            except Exception:
                pass

        def _load_devices_from_env_or_defaults() -> None:
            try:
                raw = str(os.getenv("WALKIE_DEVICE_PROFILES_JSON", "") or "").strip()
                if raw:
                    obj = json.loads(raw)
                    devices = obj.get("devices") if isinstance(obj, dict) else None
                    if isinstance(devices, list) and devices:
                        for d in devices:
                            if not isinstance(d, dict):
                                continue
                            try:
                                did = int(d.get("id"))
                            except Exception:
                                continue
                            nm = str(d.get("name") or "")
                            try:
                                exp = int(d.get("expected_softkeys")) if d.get("expected_softkeys") is not None else None
                            except Exception:
                                exp = None
                            _add_device_row(did, nm, exp)
                        return
            except Exception:
                pass

            try:
                cfgp = Path(__file__).resolve().parent / "configs" / "device_profiles.json"
                if cfgp.exists():
                    with open(cfgp, "r", encoding="utf-8") as f:
                        obj = json.load(f) or {}
                    devices = obj.get("devices") if isinstance(obj, dict) else None
                    if isinstance(devices, list) and devices:
                        for d in devices:
                            if not isinstance(d, dict):
                                continue
                            try:
                                did = int(d.get("id"))
                            except Exception:
                                continue
                            nm = str(d.get("name") or "")
                            try:
                                exp = int(d.get("expected_softkeys")) if d.get("expected_softkeys") is not None else None
                            except Exception:
                                exp = None
                            _add_device_row(did, nm, exp)
                        return
            except Exception:
                pass

            try:
                default_n = int(((cfg.get("performance") or {}).get("max_devices")) or 2)
            except Exception:
                default_n = 2
            default_n = max(1, min(6, int(default_n)))
            for i in range(default_n):
                _add_device_row(i + 1, "", None)

        _load_devices_from_env_or_defaults()

        dev_btns = ttk.Frame(lf_devices)
        dev_btns.grid(row=0, column=4, rowspan=2, sticky="ne", padx=(12, 0))

        def _on_add_device() -> None:
            next_id = len(device_rows) + 1
            _add_device_row(next_id, "", None)

        btn_add_dev = ttk.Button(dev_btns, text="Add", command=_on_add_device)
        btn_add_dev.grid(row=0, column=0, sticky="ew")

        btn_rem_dev = ttk.Button(dev_btns, text="Remove", command=_remove_selected_device_row)
        btn_rem_dev.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        btn_row = ttk.Frame(maincol)
        btn_row.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        status_var = tk.StringVar(value="Idle")

        proc_holder = {"proc": None, "stopping": False}
        auto_start_holder = {"on": False}

        out_frame = ttk.LabelFrame(maincol, text="Idle", padding=8)
        out_frame.grid(row=4, column=0, sticky="nsew", pady=(10, 0))

        try:
            out_frame.columnconfigure(0, weight=1)
            out_frame.rowconfigure(0, weight=1)
        except Exception:
            pass

        txt_out = tk.Text(out_frame, height=22, wrap="word")
        txt_out.grid(row=0, column=0, sticky="nsew")
        scr = ttk.Scrollbar(out_frame, orient="vertical", command=txt_out.yview)
        scr.grid(row=0, column=1, sticky="ns")
        try:
            txt_out.configure(yscrollcommand=scr.set)
        except Exception:
            pass

        try:
            txt_out.tag_configure("pass", foreground="#1B5E20")
            txt_out.tag_configure("fail", foreground="#B71C1C")
            txt_out.tag_configure("warn", foreground="#E65100")
            txt_out.tag_configure("cmd", foreground="#1565C0")
            txt_out.tag_configure("error", foreground="#B71C1C")
        except Exception:
            pass

        def _pick_tag_for_line(line: str) -> str | None:
            try:
                s = str(line or "")
                stripped = s.strip()
                low = stripped.lower()
                if stripped.startswith("$"):
                    return "cmd"
                if low.startswith("finished") and "exit=" in low and "exit=0" not in low:
                    return "error"
                if stripped in ["PASS", "FAIL", "WARN"]:
                    return stripped.lower()
                if stripped.startswith("Traceback"):
                    return "error"
                if "[error]" in low or "[gui error]" in low:
                    return "error"
                if low.startswith("error:") or low.startswith("exception"):
                    return "error"
                if "typeerror" in low or "valueerror" in low or "runtimeerror" in low:
                    return "error"
                if "[warn" in low or low.startswith("warning"):
                    return "warn"
                return None
            except Exception:
                return None

        def _set_status(v: str) -> None:
            try:
                status_var.set(v)
            except Exception:
                pass
            try:
                out_frame.configure(text=str(v))
            except Exception:
                pass

        def _maybe_set_status_from_output(line: str) -> None:
            try:
                s = str(line or "").strip()
                low = s.lower()
                if not s:
                    return
                if "entering preview" in low or "controls:" in low:
                    _set_status("Preview")
                    return
                if "processing with msi genai" in low or "capture->ocr" in low:
                    _set_status("Process")
                    return
                if "processing complete" in low:
                    _set_status("Finished")
                    return
                if "application closed" in low:
                    _set_status("Finished")
                    return
            except Exception:
                return

        def _append_output(s: str, tag: str | None = None) -> None:
            if s is None:
                return
            txt_out.configure(state="normal")
            if tag:
                txt_out.insert("end", str(s), (tag,))
            else:
                txt_out.insert("end", str(s))
            try:
                txt_out.see("end")
            except Exception:
                pass
            try:
                txt_out.yview_moveto(1.0)
            except Exception:
                pass
            txt_out.configure(state="disabled")

        def _append_line(line: str) -> None:
            msg = line if line.endswith("\n") else (line + "\n")
            tag = _pick_tag_for_line(msg)
            root.after(0, _append_output, msg, tag)
            try:
                root.after(0, _maybe_set_status_from_output, msg)
            except Exception:
                pass

        def _stream_process(p: subprocess.Popen, on_done):
            try:
                while True:
                    b = p.stdout.readline()
                    if not b:
                        break
                    try:
                        line = b.decode("utf-8", errors="replace")
                    except Exception:
                        try:
                            line = b.decode(errors="replace")
                        except Exception:
                            line = str(b)
                    _append_line(line.rstrip("\n"))
            except Exception as e:
                _append_line(f"[GUI] Stream error: {e}")
            finally:
                try:
                    rc = p.wait(timeout=1)
                except Exception:
                    rc = None
                root.after(0, on_done, rc)

        def _run_subprocess(argv, on_done, extra_env: dict | None = None):
            try:
                if proc_holder.get("proc") is not None:
                    messagebox.showwarning("Busy", "A process is already running. Stop it first.")
                    return

                try:
                    _set_status("Process")
                except Exception:
                    pass

                _append_line(f"$ {' '.join(argv)}")
                env = os.environ.copy()
                env.setdefault("PYTHONIOENCODING", "utf-8")
                env.setdefault("PYTHONUTF8", "1")
                env.setdefault("PYTHONUNBUFFERED", "1")
                try:
                    if extra_env:
                        for k, v in dict(extra_env).items():
                            if v is None:
                                continue
                            env[str(k)] = str(v)
                except Exception:
                    pass
                

                cam_sel = (var_camera.get() or "").strip()
                try:
                    cam_id = int("".join([ch for ch in cam_sel if ch.isdigit()]))
                    env["WALKIE_CAMERA_ID"] = str(cam_id)
                except Exception:
                    pass
                p = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=False,
                    bufsize=0,
                    cwd=str(Path(__file__).resolve().parent),
                    env=env,
                )
                proc_holder["proc"] = p

                # Watchdog: ensure UI is re-enabled even if the stream thread blocks.
                def _watchdog():
                    try:
                        cur = proc_holder.get("proc")
                        if cur is None or cur is not p:
                            return
                        rc = p.poll()
                        if rc is not None:
                            proc_holder["proc"] = None
                            on_done(rc)
                            return
                    except Exception:
                        return
                    root.after(500, _watchdog)

                try:
                    root.after(500, _watchdog)
                except Exception:
                    pass

                th = threading.Thread(target=_stream_process, args=(p, on_done), daemon=True)
                th.start()
            except Exception as e:
                proc_holder["proc"] = None
                _append_line(f"[GUI] Failed to start process: {e}")
                on_done(None)

        def _stop_running() -> None:
            p = proc_holder.get("proc")
            if p is None:
                return
            try:
                proc_holder["stopping"] = True
                _append_line("[GUI] Stopping process...")
                p.terminate()
                try:
                    _set_status("Idle")
                except Exception:
                    pass
            except Exception as e:
                _append_line(f"[GUI] Stop failed: {e}")
                try:
                    proc_holder["stopping"] = False
                except Exception:
                    pass

        def _init_done(_rc):
            proc_holder["proc"] = None
            will_autostart = False
            try:
                will_autostart = bool(auto_start_holder.get("on")) and int(_rc or 0) == 0
            except Exception:
                will_autostart = False

            if will_autostart:
                btn_run.configure(state="disabled")
                btn_stop.configure(state="normal")
            else:
                btn_run.configure(state="normal")
                btn_stop.configure(state="disabled")

            stopping = False
            try:
                stopping = bool(proc_holder.get("stopping"))
                proc_holder["stopping"] = False
            except Exception:
                stopping = False

            rc = 1 if stopping else _rc
            try:
                if rc in [0, None]:
                    _append_line("Finished")
                else:
                    _append_line(f"Finished (exit={int(rc)})")
            except Exception:
                pass

            try:
                if will_autostart:
                    _set_status("Init GenAI finished, starting preview...")
                else:
                    _set_status("Init GenAI finished")
            except Exception:
                pass

            try:
                if will_autostart:
                    auto_start_holder["on"] = False
                    _start()
            except Exception:
                auto_start_holder["on"] = False

        def _start_done(_rc):
            proc_holder["proc"] = None
            btn_run.configure(state="normal")
            btn_stop.configure(state="disabled")

            stopping = False
            try:
                stopping = bool(proc_holder.get("stopping"))
                proc_holder["stopping"] = False
            except Exception:
                stopping = False

            rc = 1 if stopping else _rc
            try:
                if rc in [0, None]:
                    _append_line("Finished")
                else:
                    _append_line(f"Finished (exit={int(rc)})")
            except Exception:
                pass

            try:
                _set_status("Finished")
            except Exception:
                pass

        def _start() -> None:
            btn_run.configure(state="disabled")
            btn_stop.configure(state="normal")

            try:
                _set_status("Process")
            except Exception:
                pass

            argv = [sys.executable, str(Path(__file__).resolve())]

            model_path = (var_model_path.get() or "").strip()
            if model_path:
                argv.extend(["--model-path", model_path])

            extra_env = {}

            try:
                devices = []
                for r in device_rows:
                    did = int(r.get("id"))
                    nm = str(r.get("var_name").get() if r.get("var_name") is not None else "")
                    nm = nm.strip()
                    exp_raw = str(r.get("var_expected").get() if r.get("var_expected") is not None else "").strip()
                    exp_val = None
                    if exp_raw != "":
                        exp_val = int(exp_raw)
                    devices.append({
                        "id": did,
                        "name": nm,
                        "expected_softkeys": exp_val,
                    })
                extra_env["WALKIE_DEVICE_PROFILES_JSON"] = json.dumps({"devices": devices}, ensure_ascii=False)

                try:
                    cfgp = Path(__file__).resolve().parent / "configs" / "device_profiles.json"
                    cfgp.parent.mkdir(parents=True, exist_ok=True)
                    with open(cfgp, "w", encoding="utf-8") as f:
                        json.dump({"devices": devices}, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            except Exception:
                pass

            try:
                save_dir = (var_save_dir.get() or "").strip()
                if save_dir:
                    extra_env["WALKIE_OUTPUT_DIR"] = save_dir
            except Exception:
                pass

            _run_subprocess(argv, _start_done, extra_env=extra_env)

        def _init_session() -> None:
            btn_run.configure(state="disabled")
            btn_stop.configure(state="normal")
            try:
                _set_status("Process")
            except Exception:
                pass
            argv = [sys.executable, str(Path("scripts") / "init_genai_session.py")]
            _run_subprocess(argv, _init_done)

        def _init_and_start() -> None:
            try:
                auto_start_holder["on"] = True
            except Exception:
                pass
            _init_session()

        def _quit() -> None:
            _stop_running()
            root.destroy()

        btn_run = ttk.Button(btn_row, text="Start", command=_init_and_start)
        btn_run.grid(row=0, column=0, sticky="w")

        btn_stop = ttk.Button(btn_row, text="Stop", command=_stop_running, state="disabled")
        btn_stop.grid(row=0, column=1, sticky="w", padx=(10, 0))

        btn_quit = ttk.Button(btn_row, text="Close", command=_quit)
        btn_quit.grid(row=0, column=2, sticky="e", padx=(10, 0))

        root.mainloop()

    if bool(args.gui):
        _launch_gui()
        return

    _run_app(
        once=bool(args.once),
        warmup_sec=float(args.warmup_sec or 0.0),
        model_path=(args.model_path or ""),
        use_last=bool(args.use_last),
    )

if __name__ == "__main__":
    main()