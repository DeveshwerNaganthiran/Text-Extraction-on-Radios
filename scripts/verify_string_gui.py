import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import time
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import ttk

import cv2
import json
import re
import yaml

from openpyxl import load_workbook


_EVT_FINISHED = "__PROCESS_FINISHED__"


def _settings_path() -> Path:
    base = Path(os.getenv("APPDATA") or Path.home())
    return base / "walkie_tracker_verify_string_gui.json"


def _load_settings() -> dict:
    p = _settings_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}


def _save_settings(data: dict) -> None:
    p = _settings_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _probe_camera_ids(max_id: int = 5) -> list[str]:
    found = []
    for i in range(int(max_id) + 1):
        cap = None
        try:
            backend = getattr(cv2, "CAP_DSHOW", 0)
            cap = cv2.VideoCapture(i, backend)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                continue
            found.append(str(i))
        except Exception:
            continue
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return found


def _norm_col(s: str) -> str:
    v = str(s or "").strip().lower().replace("_", " ")
    v = re.sub(r"[\(\)\[\]\{\}:,;/\\\-]+", " ", v)
    v = " ".join(v.split())
    if v.startswith("string "):
        v = v[len("string ") :].strip()
    if v.startswith("str "):
        v = v[len("str ") :].strip()
    return v


def _sheet_name_for_region(xls_path: str, region: str) -> str:
    wb = load_workbook(filename=xls_path, read_only=True, data_only=True)
    try:
        want = _norm_col(region)
        # Canonical region names
        if want in ["en", "english", "global"]:
            want = "english"
        if want in ["latam", "lac", "lacr", "la cr"]:
            want = "lacr"

        # Exact match first
        for s in wb.sheetnames:
            if _norm_col(s) == want:
                return s
        # Partial match
        for s in wb.sheetnames:
            if want in _norm_col(s):
                return s
        raise ValueError(f"Sheet for region '{region}' not found")
    finally:
        try:
            wb.close()
        except Exception:
            pass


def _tag_options_from_excel(excel_path: str) -> list[str]:
    if not excel_path:
        return []
    p = Path(excel_path)
    if not p.exists():
        return []

    try:
        english_sheet = _sheet_name_for_region(str(p), "english")
    except Exception:
        english_sheet = "English"

    wb = load_workbook(filename=str(p), read_only=True, data_only=True)
    try:
        if english_sheet not in wb.sheetnames:
            return []
        ws = wb[english_sheet]

        headers = []
        for row in ws.iter_rows(min_row=1, max_row=1, values_only=True):
            headers = list(row)
            break

        tag_col_idx = None
        for i, h in enumerate(headers):
            n = _norm_col(h)
            if n in ["string tag", "stringtag", "tag"]:
                tag_col_idx = i
                break
        if tag_col_idx is None:
            return []

        tags = []
        seen = set()
        for row in ws.iter_rows(min_row=2, values_only=True):
            if tag_col_idx >= len(row):
                continue
            v = row[tag_col_idx]
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            # Sometimes the sheet can contain repeated header-like cells in the body; exclude them.
            try:
                if _norm_col(s) in ["string tag", "stringtag", "tag"]:
                    continue
            except Exception:
                pass
            if s not in seen:
                seen.add(s)
                tags.append(s)
        return tags
    finally:
        try:
            wb.close()
        except Exception:
            pass


def _language_options_from_excel(excel_path: str, region: str) -> list[str]:
    if not excel_path:
        return []
    p = Path(excel_path)
    if not p.exists():
        return []

    sheet = _sheet_name_for_region(str(p), region)
    wb = load_workbook(filename=str(p), read_only=True, data_only=True)
    try:
        ws = wb[sheet]
        headers = []
        for row in ws.iter_rows(min_row=1, max_row=1, values_only=True):
            headers = list(row)
            break

        # Normalize and filter obvious non-language columns
        ignore = {
            "index",
            "string tag",
            "tag",
            "string category",
            "category",
            "version",
            "ver",
            "english",
            "comment",
            "comments",
            "notes",
            "note",
            "description",
            "desc",
        }

        opts = []
        seen = set()
        for h in headers:
            n = _norm_col(h)
            if not n or n in ignore:
                continue
            # For headers like "String (Japanese)", _norm_col becomes "japanese".
            label = str(h).strip() if h is not None else ""
            # Prefer clean normalized language name when it looks like a language
            if n in [
                "japanese",
                "korean",
                "simplified chinese",
                "traditional chinese",
                "french",
                "spanish",
                "german",
                "italian",
                "polish",
                "russian",
                "turkish",
                "arabic",
                "hungarian",
                "hebrew",
                "czech",
                "portuguese",
                "english",
            ]:
                label = n.title()
            if label and label not in seen:
                seen.add(label)
                opts.append(label)

        return opts
    finally:
        try:
            wb.close()
        except Exception:
            pass


def _default_excel_path() -> str:
    # Try common locations / previously used path
    p = os.getenv("VERIFY_EXCEL", "").strip()
    if p:
        return p
    return ""


def _python_exe() -> str:
    return sys.executable or "python"


def _default_model_path() -> str:
    try:
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "settings.yaml"
        if not cfg_path.exists():
            return ""
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        p = (((cfg.get("detector") or {}).get("path")) or "").strip()
        if not p:
            return ""
        pp = Path(p)
        if not pp.is_absolute():
            pp = (Path(__file__).resolve().parents[1] / pp).resolve()
        return str(pp)
    except Exception:
        return ""


def _resolve_path(p: str) -> str:
    v = (p or "").strip()
    if not v:
        return ""
    try:
        pp = Path(v)
        if not pp.is_absolute():
            pp = (Path(__file__).resolve().parents[1] / pp).resolve()
        return str(pp)
    except Exception:
        return v


class VerifyStringGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Verify String (Walkie-Tracker)")
        self.root.minsize(900, 600)

        self._auto_start_verify = False
        self._last_run_is_verification = True
        self.proc = None
        self.q = queue.Queue()
        self.last_result = ""
        self.last_expected = ""
        self._pending_expected_norm = False

        self._settings = _load_settings()

        self.excel_var = tk.StringVar(value=str(self._settings.get("excel") or _default_excel_path()))
        self.region_var = tk.StringVar(value=str(self._settings.get("region") or "APAC"))
        self.language_var = tk.StringVar(value=str(self._settings.get("language") or "Japanese"))
        self.tag_var = tk.StringVar(value=str(self._settings.get("tag") or ""))
        self.index_var = tk.StringVar(value=str(self._settings.get("index") or ""))
        self.preview_var = tk.BooleanVar(value=bool(self._settings.get("preview", True)))
        saved_model = str(self._settings.get("model_path") or "")
        if not saved_model.strip():
            saved_model = _default_model_path()
        self.model_path_var = tk.StringVar(value=_resolve_path(saved_model))
        self.camera_id_var = tk.StringVar(value=str(self._settings.get("camera_id") or "1"))
        self.save_log_var = tk.BooleanVar(value=bool(self._settings.get("save_log", False)))
        self.log_path_var = tk.StringVar(value=str(self._settings.get("log_path") or ""))
        self._log_fp = None
        self._log_session_dir = None

        frm = tk.Frame(root, padx=10, pady=10)
        frm.pack(fill=tk.BOTH, expand=True)

        row = 0

        tk.Label(frm, text="Excel (.xlsm/.xlsx)").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.excel_var, width=60).grid(row=row, column=1, sticky="we", padx=(8, 8))
        tk.Button(frm, text="Browse...", command=self.browse_excel).grid(row=row, column=2, sticky="e")
        row += 1

        tk.Label(frm, text="Region").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.region_combo = ttk.Combobox(
            frm,
            textvariable=self.region_var,
            width=20,
            state="normal",
            values=["APAC", "EMEA", "LACR", "English"],
        )
        self.region_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        try:
            self.region_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_languages())
        except Exception:
            pass

        try:
            self.region_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_tags(), add=True)
        except Exception:
            pass
        row += 1

        tk.Label(frm, text="Language").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.language_combo = ttk.Combobox(
            frm,
            textvariable=self.language_var,
            width=30,
            state="normal",
            values=[
                "Japanese",
                "Korean",
                "Simplified Chinese",
                "Traditional Chinese",
                "French",
                "Spanish",
                "German",
                "Italian",
                "Polish",
                "Russian",
                "Turkish",
                "Arabic",
                "Hungarian",
                "Hebrew",
                "Czech",
                "Portuguese",
            ],
        )
        self.language_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        row += 1

        tk.Label(frm, text="String Tag").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.tag_combo = ttk.Combobox(frm, textvariable=self.tag_var, width=50, state="normal", values=[])
        self.tag_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        try:
            self.tag_combo.bind("<KeyRelease>", self._on_tag_typed)
        except Exception:
            pass
        try:
            self.tag_combo.bind("<Escape>", self._on_tag_escape)
            self.tag_combo.bind("<FocusOut>", self._on_tag_focus_out)
            self.tag_combo.bind("<Down>", self._on_tag_down)
        except Exception:
            pass
        row += 1

        tk.Label(frm, text="Index (optional)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        tk.Entry(frm, textvariable=self.index_var, width=20).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        row += 1

        extras = tk.Frame(frm)
        extras.grid(row=row, column=0, columnspan=3, sticky="we", pady=(8, 0))

        tk.Checkbutton(extras, text="Preview (OpenCV window)", variable=self.preview_var).pack(side=tk.LEFT)

        tk.Label(extras, text="Camera ID").pack(side=tk.LEFT, padx=(12, 2))

        self.camera_combo = ttk.Combobox(extras, textvariable=self.camera_id_var, width=8, state="readonly")
        self.camera_combo.pack(side=tk.LEFT)
        tk.Button(extras, text="Refresh", command=self.refresh_cameras).pack(side=tk.LEFT, padx=(6, 0))

        tk.Checkbutton(extras, text="Save log", variable=self.save_log_var).pack(side=tk.LEFT, padx=(12, 0))
        tk.Entry(extras, textvariable=self.log_path_var, width=28).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(extras, text="Browse...", command=self.browse_log).pack(side=tk.LEFT, padx=(6, 0))

        row += 1

        tk.Label(frm, text="Model Path (optional)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        tk.Entry(frm, textvariable=self.model_path_var, width=60).grid(row=row, column=1, sticky="we", padx=(8, 8), pady=(6, 0))
        tk.Button(frm, text="Browse...", command=self.browse_model).grid(row=row, column=2, sticky="e", pady=(6, 0))
        row += 1

        btns = tk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
        self.btn_run = tk.Button(btns, text="Start", command=self.init_and_run)
        self.btn_run.pack(side=tk.LEFT)
        self.btn_stop = tk.Button(btns, text="Stop", command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=(8, 0))
        self.btn_close = tk.Button(btns, text="Close", command=self.close)
        self.btn_close.pack(side=tk.LEFT, padx=(8, 0))
        row += 1

        self.status_var = tk.StringVar(value="Idle")
        self.status_label = tk.Label(frm, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=row, column=0, columnspan=3, sticky="we", pady=(8, 0))
        row += 1

        self.output = tk.Text(frm, height=22, wrap=tk.WORD)
        self.output.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

        self.output.tag_configure("pass", foreground="#1B5E20")
        self.output.tag_configure("fail", foreground="#B71C1C")
        self.output.tag_configure("warn", foreground="#E65100")
        self.output.tag_configure("cmd", foreground="#1565C0")
        self.output.tag_configure("error", foreground="#B71C1C")

        frm.grid_columnconfigure(1, weight=1)
        frm.grid_rowconfigure(row, weight=1)

        try:
            cur = (self.camera_id_var.get() or "").strip()
            if cur:
                self.camera_combo["values"] = [cur]
        except Exception:
            pass
        self.refresh_languages()
        self.refresh_tags()

        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

        self.root.after(50, self._drain_queue)

    def browse_excel(self):
        p = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel Files", "*.xlsm *.xlsx *.xls"), ("All Files", "*.*")],
        )
        if p:
            self.excel_var.set(p)
            try:
                self.refresh_languages()
                self.refresh_tags()
            except Exception:
                pass

    def browse_model(self):
        p = filedialog.askopenfilename(
            title="Select YOLO model weights",
            filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")],
        )
        if p:
            self.model_path_var.set(_resolve_path(p))

    def browse_log(self):
        d = filedialog.askdirectory(title="Select log folder")
        if d:
            self.log_path_var.set(d)

    def _append(self, s: str):
        self.output.insert(tk.END, s)
        self.output.see(tk.END)
        try:
            if self._log_fp is not None:
                self._log_fp.write(s)
                self._log_fp.flush()
        except Exception:
            pass

    def _append_cmd(self, s: str):
        self.output.insert(tk.END, s, ("cmd",))
        self.output.see(tk.END)
        try:
            if self._log_fp is not None:
                self._log_fp.write(s)
                self._log_fp.flush()
        except Exception:
            pass

    def _append_line_with_result_color(self, line: str):
        # Colorize PASS/FAIL lines from verify_string.py
        stripped = (line or "").strip()
        low = stripped.lower()

        if stripped.startswith("Expected (") and "):" in stripped:
            try:
                self.last_expected = stripped.split("):", 1)[1].strip()
            except Exception:
                pass
        elif stripped == "Expected (normalized):":
            self._pending_expected_norm = True
        elif self._pending_expected_norm and stripped and stripped not in ["PASS", "FAIL", "WARN"]:
            try:
                self.last_expected = stripped
            except Exception:
                pass
            self._pending_expected_norm = False

        is_error = False
        if stripped.startswith("Traceback"):
            is_error = True
        elif "[error]" in low or "[gui error]" in low:
            is_error = True
        elif low.startswith("error:") or low.startswith("exception"):
            is_error = True
        elif "typeerror" in low or "valueerror" in low or "runtimeerror" in low:
            is_error = True
        elif "❌" in stripped:
            is_error = True
        if stripped == "PASS":
            self.last_result = "PASS"
            self.output.insert(tk.END, line, ("pass",))
        elif stripped == "FAIL":
            self.last_result = "FAIL"
            self.output.insert(tk.END, line, ("fail",))
            try:
                if (self.last_expected or "").strip():
                    exp_line = f"Expected: {self.last_expected}\n"
                    self.output.insert(tk.END, exp_line)
            except Exception:
                pass
        elif stripped == "WARN":
            self.last_result = "WARN"
            self.output.insert(tk.END, line, ("warn",))
        elif is_error:
            self.output.insert(tk.END, line, ("error",))
        else:
            self.output.insert(tk.END, line)
        self.output.see(tk.END)
        try:
            if self._log_fp is not None:
                self._log_fp.write(line)
                if not line.endswith("\n"):
                    self._log_fp.write("\n")
                self._log_fp.flush()
        except Exception:
            pass

    def _current_settings(self) -> dict:
        return {
            "excel": (self.excel_var.get() or "").strip(),
            "region": (self.region_var.get() or "").strip(),
            "language": (self.language_var.get() or "").strip(),
            "tag": (self.tag_var.get() or "").strip(),
            "index": (self.index_var.get() or "").strip(),
            "preview": bool(self.preview_var.get()),
            "model_path": (self.model_path_var.get() or "").strip(),
            "camera_id": (self.camera_id_var.get() or "").strip(),
            "save_log": bool(self.save_log_var.get()),
            "log_path": (self.log_path_var.get() or "").strip(),
        }

    def _persist_settings(self):
        self._settings = self._current_settings()
        _save_settings(self._settings)

    def _on_close(self):
        try:
            self._persist_settings()
        except Exception:
            pass
        try:
            if self._log_fp is not None:
                self._log_fp.close()
                self._log_fp = None
        except Exception:
            pass
        try:
            self._log_session_dir = None
        except Exception:
            pass
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
        self.root.destroy()

    def refresh_cameras(self):
        cams = _probe_camera_ids(max_id=8)
        if not cams:
            cams = ["0", "1", "2"]
        try:
            self.camera_combo["values"] = cams
            cur = (self.camera_id_var.get() or "").strip()
            if cur not in cams:
                if "1" in cams:
                    self.camera_id_var.set("1")
                else:
                    self.camera_id_var.set(cams[0])
        except Exception:
            pass

    def refresh_languages(self):
        excel = (self.excel_var.get() or "").strip()
        region = (self.region_var.get() or "").strip() or "APAC"

        opts = []
        try:
            opts = _language_options_from_excel(excel, region)
        except Exception:
            opts = []

        if not opts:
            opts = [
                "Japanese",
                "Korean",
                "Simplified Chinese",
                "Traditional Chinese",
                "French",
                "Spanish",
                "German",
                "Italian",
                "Polish",
                "Russian",
                "Turkish",
                "Arabic",
                "Hungarian",
                "Hebrew",
                "Czech",
                "Portuguese",
            ]

        try:
            self.language_combo["values"] = opts
            cur = (self.language_var.get() or "").strip()
            if cur and cur in opts:
                return
            if cur and cur not in opts:
                return
            if opts:
                self.language_var.set(opts[0])
        except Exception:
            pass

    def refresh_tags(self):
        excel = (self.excel_var.get() or "").strip()
        tags = []
        try:
            tags = _tag_options_from_excel(excel)
        except Exception:
            tags = []

        try:
            self._all_tags = tags
        except Exception:
            pass

        try:
            self.tag_combo["values"] = tags
        except Exception:
            pass

    def _on_tag_typed(self, _evt=None):
        try:
            all_tags = list(getattr(self, "_all_tags", []) or [])
        except Exception:
            all_tags = []

        if not all_tags:
            return

        # Read directly from the widget; Tk can lag syncing the StringVar depending on event order.
        try:
            typed_raw = self.tag_combo.get()
        except Exception:
            typed_raw = self.tag_var.get()
        typed = (typed_raw or "").strip().lower()
        if not typed:
            filtered = all_tags
        else:
            filtered = [t for t in all_tags if typed in str(t).lower()]

        try:
            self.tag_combo["values"] = filtered
        except Exception:
            pass

    def _unpost_tag_dropdown(self):
        try:
            self.tag_combo.tk.call("ttk::combobox::Unpost")
        except Exception:
            pass

    def _on_tag_escape(self, _evt=None):
        self._unpost_tag_dropdown()

    def _on_tag_focus_out(self, _evt=None):
        self._unpost_tag_dropdown()

    def _on_tag_down(self, _evt=None):
        # Let the user open suggestions explicitly.
        try:
            if not list(self.tag_combo["values"] or []):
                self.refresh_tags()
        except Exception:
            pass

        # Ensure list is filtered to whatever is currently typed.
        try:
            self._on_tag_typed()
        except Exception:
            pass
        try:
            self.tag_combo.tk.call("ttk::combobox::Post", str(self.tag_combo))
        except Exception:
            pass

    def clear(self):
        self.output.delete("1.0", tk.END)
        self.last_result = ""
        self.status_var.set("Idle")
        try:
            self.status_label.configure(fg="black")
        except Exception:
            pass

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            try:
                self.root.quit()
            except Exception:
                pass

    def _set_running(self, running: bool):
        try:
            if running:
                self.btn_run.configure(state=tk.DISABLED)
                self.btn_stop.configure(state=tk.NORMAL)
            else:
                self.btn_run.configure(state=tk.NORMAL)
                self.btn_stop.configure(state=tk.DISABLED)
        except Exception:
            pass

    def _build_cmd(self):
        excel = self.excel_var.get().strip()
        if not excel:
            raise ValueError("Excel path is required")

        region = self.region_var.get().strip()
        language = self.language_var.get().strip()
        tag = self.tag_var.get().strip()
        idx = self.index_var.get().strip()

        if not region:
            raise ValueError("Region is required (e.g., APAC/EMEA/LACR/English)")
        if not language:
            raise ValueError("Language is required (e.g., Japanese)")
        if not tag and not idx:
            raise ValueError("Provide either String Tag or Index")

        script_path = Path(__file__).resolve().parent / "verify_string.py"
        cmd = [_python_exe(), str(script_path), "--excel", excel, "--region", region, "--language", language]

        if tag:
            cmd += ["--tag", tag]
        if idx:
            cmd += ["--index", idx]
        if self.preview_var.get():
            cmd += ["--preview"]

        model_path = _resolve_path(self.model_path_var.get())
        try:
            self.model_path_var.set(model_path)
        except Exception:
            pass
        if model_path:
            cmd += ["--model-path", model_path]

        camera_id = self.camera_id_var.get().strip()
        if camera_id:
            cmd += ["--camera-id", camera_id]

        return cmd

    def _run_subprocess(self, cmd, *, is_verification: bool = True):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Running", "A process is already running. Click Stop first.")
            return

        try:
            self._last_run_is_verification = bool(is_verification)
        except Exception:
            self._last_run_is_verification = True

        self.last_result = ""
        self.status_var.set("Running...")
        try:
            self.status_label.configure(fg="black")
        except Exception:
            pass
        self._set_running(True)

        try:
            if self._log_fp is not None:
                self._log_fp.close()
        except Exception:
            pass
        self._log_fp = None

        if self.save_log_var.get() and bool(is_verification):
            d = (self.log_path_var.get() or "").strip()
            if not d:
                d = str(Path.cwd())
                self.log_path_var.set(d)

            # If user pasted a file path, treat its parent as the folder.
            try:
                dp = Path(d)
                if dp.suffix.lower() in [".log", ".txt"]:
                    dp = dp.parent
                d = str(dp)
            except Exception:
                pass

            # Create one dated subfolder per RUN so each verification has its own folder.
            sess = time.strftime("%Y%m%d_%H%M%S")
            self._log_session_dir = str(Path(d) / f"verified_{sess}")
            try:
                Path(self._log_session_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                self._log_session_dir = d

            ts = time.strftime("%Y%m%d_%H%M%S")
            safe_tag = (self.tag_var.get() or "").strip()
            safe_tag = "".join([c for c in safe_tag if c.isalnum() or c in ["_", "-"]])[:40]
            name = f"verify_string_{ts}.log" if not safe_tag else f"verify_string_{safe_tag}_{ts}.log"
            p = str(Path(self._log_session_dir) / name)
            try:
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                self._log_fp = open(p, "a", encoding="utf-8")
                self._log_fp.write("\n" + ("=" * 72) + "\n")
                self._log_fp.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            except Exception:
                self._log_fp = None

        # Show the command in the GUI, but do not write it into the log file.
        try:
            self.output.insert(tk.END, "$ " + " ".join(cmd) + "\n\n", ("cmd",))
            self.output.see(tk.END)
        except Exception:
            pass

        # If this is a verification run and logging is enabled, ensure ROI is saved.
        # verify_string.py supports --save-roi <path>.
        try:
            if self.save_log_var.get() and bool(is_verification) and self._log_session_dir:
                has_save_roi = "--save-roi" in cmd
                if not has_save_roi:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    safe_tag = (self.tag_var.get() or "").strip()
                    safe_tag = "".join([c for c in safe_tag if c.isalnum() or c in ["_", "-"]])[:40]
                    idx = (self.index_var.get() or "").strip()
                    suffix = safe_tag or ("idx" + idx if idx else "roi")
                    roi_path = str(Path(self._log_session_dir) / f"roi_{suffix}_{ts}.jpg")
                    cmd = list(cmd) + ["--save-roi", roi_path]
        except Exception:
            pass

        def _worker():
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=False,
                    bufsize=0,
                )
                assert self.proc.stdout is not None
                for raw in iter(self.proc.stdout.readline, b""):
                    try:
                        line = raw.decode("utf-8", errors="replace")
                    except Exception:
                        try:
                            line = raw.decode(errors="replace")
                        except Exception:
                            line = str(raw)
                    self.q.put(line)

                try:
                    rc = self.proc.wait(timeout=1)
                except Exception:
                    rc = None
                self.q.put((_EVT_FINISHED, rc))
            except Exception as e:
                self.q.put(f"[GUI ERROR] {e}\n")
                self.q.put((_EVT_FINISHED, None))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def init_genai(self):
        script_path = Path(__file__).resolve().parent / "init_genai_session.py"
        cmd = [_python_exe(), str(script_path)]
        self._run_subprocess(cmd, is_verification=False)

    def init_and_run(self):
        try:
            self._auto_start_verify = True
        except Exception:
            pass
        self.init_genai()

    def run(self):
        try:
            cmd = self._build_cmd()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        try:
            self._persist_settings()
        except Exception:
            pass

        self._run_subprocess(cmd, is_verification=True)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self._append("\n[INFO] Stopping process...\n")
            except Exception:
                pass

    def _drain_queue(self):
        try:
            while True:
                s = self.q.get_nowait()
                if isinstance(s, tuple) and len(s) == 2 and s[0] == _EVT_FINISHED:
                    rc = s[1]
                    is_verify = bool(self._last_run_is_verification)

                    will_autostart = False
                    try:
                        will_autostart = bool(self._auto_start_verify) and (not is_verify) and int(rc or 0) == 0
                    except Exception:
                        will_autostart = False

                    if will_autostart:
                        self.status_var.set("Init GenAI finished, starting verification...")
                    else:
                        if is_verify:
                            msg = "Finished" if rc in [0, None] else f"Finished (exit={rc})"
                        else:
                            msg = "Init GenAI finished" if rc in [0, None] else f"Init GenAI finished (exit={rc})"
                        self.status_var.set(msg)

                    try:
                        if rc not in [0, None]:
                            self.status_label.configure(fg="#B71C1C")
                        else:
                            self.status_label.configure(fg="black")
                    except Exception:
                        pass

                    if not will_autostart:
                        self._set_running(False)

                    try:
                        if self._log_fp is not None:
                            self._log_fp.write(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            self._log_fp.flush()
                            self._log_fp.close()
                            self._log_fp = None
                    except Exception:
                        pass

                    try:
                        if will_autostart:
                            self._auto_start_verify = False
                            self.run()
                    except Exception:
                        self._auto_start_verify = False
                    continue

                if isinstance(s, str):
                    self._append_line_with_result_color(s)
                else:
                    self._append(str(s))
        except queue.Empty:
            pass

        self.root.after(50, self._drain_queue)


def main():
    root = tk.Tk()
    app = VerifyStringGUI(root)
    root.minsize(900, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
