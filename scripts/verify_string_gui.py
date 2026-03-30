import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import time
import tempfile
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import ttk

import cv2
import json
import re
import yaml
import pandas as pd

from openpyxl import load_workbook

import sys
sys.coinit_flags = 2  # Forces COM initialization to STA mode to prevent Tkinter crashes
# ---------------------------

try:
    from pywinauto import Desktop
    from pywinauto.application import Application
    from pywinauto.keyboard import send_keys  
    HAS_PYWINAUTO = True
except ImportError:
    Desktop = None
    Application = None
    send_keys = None
    HAS_PYWINAUTO = False

_EVT_FINISHED = "__PROCESS_FINISHED__"
_EVT_COMMG_DONE = "__COMMG_DONE__"


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


def _probe_camera_ids(max_id: int = 5) -> list[tuple[str, str]]:
    cam_names = []
    if os.name == 'nt':
        try:
            cmd = 'powershell -ExecutionPolicy Bypass -Command "Get-PnpDevice -PresentOnly | Where-Object { $_.Class -eq \'Camera\' -or $_.Class -eq \'Image\' } | Select-Object -ExpandProperty FriendlyName"'
            res = subprocess.check_output(cmd, shell=True, text=True, creationflags=0x08000000)
            cam_names = [line.strip() for line in res.strip().split('\n') if line.strip()]
        except Exception:
            pass

    found = []
    for i in range(int(max_id) + 1):
        cap = None
        try:
            backend = getattr(cv2, "CAP_DSHOW", 0)
            cap = cv2.VideoCapture(i, backend)
            if not cap.isOpened():
                try: cap.release()
                except Exception: pass
                cap = cv2.VideoCapture(i)
                
            if not cap.isOpened():
                continue
                
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                continue
            
            # Return a tuple of (ID, Name)
            name = cam_names[i] if i < len(cam_names) else f"Camera {i}"
            found.append((str(i), name))
            
        except Exception:
            continue
        finally:
            try:
                if cap is not None: cap.release()
            except Exception: pass
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
        if want in ["en", "english", "global"]:
            want = "english"
        if want in ["latam", "lac", "lacr", "la cr"]:
            want = "lacr"

        for s in wb.sheetnames:
            if _norm_col(s) == want:
                return s
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

def _build_index_to_tag_map(excel_path: str) -> dict:
    if not excel_path:
        return {}
    try:
        xls = pd.ExcelFile(excel_path, engine="openpyxl")
        
        # 1. Find the English sheet dynamically
        target = "english"
        sheet_name = None
        for s in xls.sheet_names:
            if _norm_col(s) in ["english", "en", "global"]:
                sheet_name = s
                break
            if target in _norm_col(s):
                sheet_name = s
                break
                
        if not sheet_name: 
            return {}
            
        df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")
        
        # 2. Find Index Column
        idx_col = next((c for c in df.columns if _norm_col(c) == "index"), None)
        
        # 3. Find Tag Column
        preferred, fallback = [], []
        for c in df.columns:
            low, n = str(c or "").strip().lower(), _norm_col(c)
            if ("tag" in low and "string" in low) or n in ["string tag", "stringtag"]: preferred.append(c)
            elif n == "tag" or "tag" in low: fallback.append(c)
        tag_col = preferred[0] if preferred else (fallback[0] if fallback else "")
        
        if not idx_col or not tag_col: 
            return {}
            
        # 4. Build Dictionary mapping Index -> Tag
        mapping = {}
        for _, row in df.iterrows():
            idx_val = row[idx_col]
            tag_val = row[tag_col]
            
            if pd.isna(idx_val) or pd.isna(tag_val): 
                continue
                
            iv = str(idx_val).strip()
            tv = str(tag_val).strip()
            
            if tv.lower() == 'nan' or not tv: 
                continue
                
            # Clean up index to match
            if iv.endswith(".0"): 
                try: iv = str(int(float(iv)))
                except Exception: pass
            if iv.isdigit(): 
                try: iv = str(int(iv))
                except Exception: pass
                
            # --- THE FIX: Only map the FIRST occurrence to match verify_string.py ---
            if iv not in mapping:
                mapping[iv] = tv
            
        return mapping
    except Exception:
        return {}

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

        ignore = {
            "index", "string tag", "tag", "string category", "category",
            "version", "ver", "english", "comment", "comments",
            "notes", "note", "description", "desc",
        }

        opts = []
        seen = set()
        for h in headers:
            n = _norm_col(h)
            if not n or n in ignore:
                continue
            label = str(h).strip() if h is not None else ""
            if n in [
                "japanese", "korean", "simplified chinese", "traditional chinese",
                "french", "spanish", "german", "italian", "polish",
                "russian", "turkish", "arabic", "hungarian", "hebrew",
                "czech", "portuguese", "english",
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
        self.root.title("Verify String (Walkie-Tracker + Automation Integration)")
        self.root.minsize(1000, 850)

        self._auto_start_verify = False
        self._last_run_is_verification = True
        self.proc = None
        self.q = queue.Queue()
        self.last_result = ""
        self.last_expected = ""
        self.last_actual = ""       # <-- NEW: Track actual extracted string
        self.last_error_msg = ""    # <-- NEW: Track error messages
        self._pending_expected_norm = False
        self._pending_actual_norm = False # <-- NEW: Flag for actual normalization
        self._batch_log_dir = None  # <-- NEW: Track single folder for batch runs

        self._settings = _load_settings()

        # CommG specific variables
        self.COMMG_PATH = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Motorola\CommG_LTD\CommG_LTD.lnk"
        self.WINDOW_SEARCH_TERM = "CommuniGATOR"
        self.commg_handles = []
        self.type_lock = threading.Lock()
        
        # Automation queues & windows
        self._commg_pending_queue = []
        self._commg_is_active_run = False
        self.active_cmd_windows = []

        # Build Main UI Frame
        frm = tk.Frame(root, padx=10, pady=10)
        frm.pack(fill=tk.BOTH, expand=True)

        row = 0

        # --- EXCEL & STRING CONFIG ---
        self.excel_var = tk.StringVar(value=str(self._settings.get("excel") or _default_excel_path()))
        self.region_var = tk.StringVar(value=str(self._settings.get("region") or "APAC"))
        self.language_var = tk.StringVar(value=str(self._settings.get("language") or "Japanese"))
        self.tag_var = tk.StringVar(value=str(self._settings.get("tag") or ""))
        self.index_var = tk.StringVar(value=str(self._settings.get("index") or ""))
        
        tk.Label(frm, text="Excel (.xlsm/.xlsx)").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.excel_var, width=60).grid(row=row, column=1, sticky="we", padx=(8, 8))
        tk.Button(frm, text="Browse...", command=self.browse_excel).grid(row=row, column=2, sticky="e")
        row += 1

        tk.Label(frm, text="Region").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.region_combo = ttk.Combobox(frm, textvariable=self.region_var, width=20, state="normal", values=["APAC", "EMEA", "LACR", "English"])
        self.region_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.region_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_languages())
        self.region_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_tags(), add=True)
        row += 1

        tk.Label(frm, text="Language").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.language_combo = ttk.Combobox(frm, textvariable=self.language_var, width=30, state="normal")
        self.language_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        row += 1

        tk.Label(frm, text="String Tag").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.tag_combo = ttk.Combobox(frm, textvariable=self.tag_var, width=50, state="normal")
        self.tag_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        self.tag_combo.bind("<KeyRelease>", self._on_tag_typed)
        self.tag_combo.bind("<Escape>", self._on_tag_escape)
        self.tag_combo.bind("<FocusOut>", self._on_tag_focus_out)
        self.tag_combo.bind("<Down>", self._on_tag_down)
        row += 1

        tk.Label(frm, text="Index (optional)").grid(row=row, column=0, sticky="w", pady=(6, 0))
        tk.Entry(frm, textvariable=self.index_var, width=20).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        row += 1

        # --- CAMERA, PREVIEW & LOGS ---
        self.camera_id_var = tk.StringVar(value=str(self._settings.get("camera_id") or "1"))
        self.save_log_var = tk.BooleanVar(value=bool(self._settings.get("save_log", False)))
        self.log_path_var = tk.StringVar(value=str(self._settings.get("log_path") or ""))
        self._log_fp = None
        self._log_session_dir = None

        extras = tk.Frame(frm)
        extras.grid(row=row, column=0, columnspan=3, sticky="we", pady=(8, 0))

        tk.Label(extras, text="Camera ID").pack(side=tk.LEFT, padx=(0, 2))
        self.camera_combo = ttk.Combobox(extras, textvariable=self.camera_id_var, width=35, state="readonly")
        self.camera_combo.pack(side=tk.LEFT)
        tk.Button(extras, text="Refresh", command=self.refresh_cameras).pack(side=tk.LEFT, padx=(6, 0))
        tk.Checkbutton(extras, text="Save log", variable=self.save_log_var).pack(side=tk.LEFT, padx=(12, 0))
        tk.Entry(extras, textvariable=self.log_path_var, width=28).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(extras, text="Browse...", command=self.browse_log).pack(side=tk.LEFT, padx=(6, 0))
        row += 1

        saved_model = str(self._settings.get("model_path") or "")
        if not saved_model.strip():
            saved_model = _default_model_path()
        self.model_path_var = tk.StringVar(value=_resolve_path(saved_model))
        
        tk.Label(frm, text="Model Path").grid(row=row, column=0, sticky="w", pady=(6, 0))
        tk.Entry(frm, textvariable=self.model_path_var, width=60).grid(row=row, column=1, sticky="we", padx=(8, 8), pady=(6, 0))
        tk.Button(frm, text="Browse...", command=self.browse_model).grid(row=row, column=2, sticky="e", pady=(6, 0))
        row += 1

        # -------------------------------------------------------------
        # ADDED COMMG/CMD INTEGRATION SECTION
        # -------------------------------------------------------------
        self.lf_commg = tk.LabelFrame(frm, text=" Automation Integration (CommG / CMD) ", padx=10, pady=5, fg="#00008B", font=('Arial', 10, 'bold'))
        self.lf_commg.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
        
        top_frame = tk.Frame(self.lf_commg)
        top_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))

        old_enable = self._settings.get("integration_enable")
        if old_enable is None:
            old_enable = self._settings.get("commg_enable", False)
        self.integration_enable_var = tk.BooleanVar(value=bool(old_enable))
        tk.Checkbutton(top_frame, text="Enable Automation Sequence", variable=self.integration_enable_var, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 15))

        self.integration_type_var = tk.StringVar(value=str(self._settings.get("integration_type", "CommG")))
        tk.Label(top_frame, text="Tool:").pack(side=tk.LEFT)
        tk.Radiobutton(top_frame, text="CommG", variable=self.integration_type_var, value="CommG").pack(side=tk.LEFT, padx=(5, 5))
        tk.Radiobutton(top_frame, text="CMD (Telnet)", variable=self.integration_type_var, value="CMD").pack(side=tk.LEFT, padx=(5, 5))

        self.telnet_port_var = tk.StringVar(value=str(self._settings.get("telnet_port", "23")))
        tk.Label(top_frame, text="Port (CMD):").pack(side=tk.LEFT, padx=(10, 2))
        tk.Entry(top_frame, textvariable=self.telnet_port_var, width=5).pack(side=tk.LEFT)

        # Target IPs
        ip_frame = tk.Frame(self.lf_commg)
        ip_frame.grid(row=1, column=0, sticky="nw", padx=(0, 20))
        tk.Label(ip_frame, text="Target IPs").pack(anchor="w")
        self.ip_listbox = tk.Listbox(ip_frame, height=3, width=20)
        self.ip_listbox.pack(side=tk.LEFT, fill="y")
        
        saved_ips = self._settings.get("commg_ips", ["192.168.10.1", "192.168.10.2"])
        for ip in saved_ips:
            self.ip_listbox.insert(tk.END, ip)

        ip_btns = tk.Frame(ip_frame)
        ip_btns.pack(side=tk.LEFT, padx=(5, 0), fill="y")
        self.new_ip_entry = tk.Entry(ip_btns, width=15)
        self.new_ip_entry.pack(pady=(0, 2))
        tk.Button(ip_btns, text="Add IP", command=self._commg_add_ip).pack(fill="x", pady=1)
        tk.Button(ip_btns, text="Remove", command=self._commg_remove_ip).pack(fill="x", pady=1)

        # Execution Type
        cmd_frame = tk.Frame(self.lf_commg)
        cmd_frame.grid(row=1, column=1, sticky="nw")
        
        self.commg_mode_var = tk.StringVar(value=str(self._settings.get("commg_mode", "Batch")))
        
        tk.Radiobutton(cmd_frame, text="Single Command", variable=self.commg_mode_var, value="Single").grid(row=0, column=0, sticky="w")
        self.commg_custom_cmd_var = tk.StringVar(value=str(self._settings.get("commg_custom_cmd", "03001101")))
        tk.Entry(cmd_frame, textvariable=self.commg_custom_cmd_var, width=35).grid(row=0, column=1, padx=(5, 0))

        tk.Radiobutton(cmd_frame, text="Batch (CSV/Excel)", variable=self.commg_mode_var, value="Batch").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.commg_batch_file_var = tk.StringVar(value=str(self._settings.get("commg_batch_file", "")))
        tk.Entry(cmd_frame, textvariable=self.commg_batch_file_var, width=35).grid(row=1, column=1, padx=(5, 0), pady=(8, 0))
        tk.Button(cmd_frame, text="Browse...", command=self._commg_browse_batch).grid(row=1, column=2, padx=(5, 0), pady=(8, 0))
        
        if not HAS_PYWINAUTO:
            tk.Label(self.lf_commg, text="⚠️ pywinauto not installed. Automation unavailable.", fg="red").grid(row=2, column=0, columnspan=3, sticky="w")
            self.integration_enable_var.set(False)
            for child in self.lf_commg.winfo_children():
                try: child.configure(state='disabled')
                except Exception: pass
        row += 1

        # -------------------------------------------------------------
        # ADDED DEVICES SECTION
        # -------------------------------------------------------------
        self.lf_devices = tk.LabelFrame(frm, text="Devices", padx=10, pady=5)
        self.lf_devices.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))

        self.device_rows = []
        self.selected_device_id = tk.IntVar(value=0)

        tk.Label(self.lf_devices, text="").grid(row=0, column=0, sticky="w")
        tk.Label(self.lf_devices, text="ID").grid(row=0, column=1, sticky="w")
        tk.Label(self.lf_devices, text="Name").grid(row=0, column=2, sticky="w")

        dev_btns = tk.Frame(self.lf_devices)
        dev_btns.grid(row=0, column=3, rowspan=5, sticky="ne", padx=(12, 0))
        tk.Button(dev_btns, text="Add", command=self._on_add_device).pack(fill="x", pady=(2, 2))
        tk.Button(dev_btns, text="Remove", command=self._on_remove_device).pack(fill="x", pady=(2, 2))

        self._load_devices_from_env_or_defaults()
        row += 1

        # -------------------------------------------------------------
        # ACTIONS & LOGS
        # -------------------------------------------------------------
        btns = tk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
        self.btn_run = tk.Button(btns, text="Start", command=self.init_and_run, bg="#90EE90", font=('Arial', 10, 'bold'))
        self.btn_run.pack(side=tk.LEFT)
        self.btn_stop = tk.Button(btns, text="Stop", command=self.stop, bg="#FFCCCB")
        self.btn_stop.pack(side=tk.LEFT, padx=(8, 0))
        self.btn_close = tk.Button(btns, text="Close", command=self.close)
        self.btn_close.pack(side=tk.LEFT, padx=(8, 0))
        row += 1

        self.status_var = tk.StringVar(value="Idle")
        self.status_label = tk.Label(frm, textvariable=self.status_var, anchor="w", font=('Arial', 10, 'bold'))
        self.status_label.grid(row=row, column=0, columnspan=3, sticky="we", pady=(8, 0))
        row += 1

        self.output = tk.Text(frm, height=18, wrap=tk.WORD)
        self.output.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

        self.output.tag_configure("pass", foreground="#1B5E20")
        self.output.tag_configure("fail", foreground="#B71C1C")
        self.output.tag_configure("warn", foreground="#E65100")
        self.output.tag_configure("cmd", foreground="#1565C0")
        self.output.tag_configure("error", foreground="#B71C1C")
        self.output.tag_configure("commg", foreground="#800080")

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

    # --- Automation / CommG / CMD Specific Methods ---
    def _commg_add_ip(self):
        new_ip = self.new_ip_entry.get().strip()
        if not new_ip:
            messagebox.showwarning("Empty Input", "Please enter an IP address before clicking Add.")
            return
            
        current_ips = list(self.ip_listbox.get(0, tk.END))
        if new_ip in current_ips:
            messagebox.showwarning("Duplicate IP", f"The IP address {new_ip} is already in the list!")
            return
            
        self.ip_listbox.insert(tk.END, new_ip)
        self.new_ip_entry.delete(0, tk.END)

    def _commg_remove_ip(self):
        selected = self.ip_listbox.curselection()
        if not selected:
            messagebox.showinfo("No Selection", "Please select an IP address from the list to remove.")
            return
            
        ip_to_remove = self.ip_listbox.get(selected[0])
        confirm = messagebox.askyesno(
            "Confirm Removal", 
            f"Are you sure you want to remove this IP address?\n\n{ip_to_remove}"
        )
        
        if confirm:
            self.ip_listbox.delete(selected[0])

    def _commg_browse_batch(self):
        p = filedialog.askopenfilename(
            title="Select Batch File",
            filetypes=[("Excel/CSV Files", "*.xlsx *.xls *.csv"), ("All Files", "*.*")]
        )
        if p:
            self.commg_batch_file_var.set(p)

    def _commg_ping_ip(self, ip):
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            response = subprocess.call(['ping', '-n', '1', '-w', '2000', ip], startupinfo=startupinfo)
            return response == 0
        except Exception:
            return False

    def _commg_find_handles(self):
        if not HAS_PYWINAUTO: return []
        handles = []
        desktop = Desktop(backend="uia")
        for win in desktop.windows():
            text = win.window_text()
            if text and self.WINDOW_SEARCH_TERM in text:
                handles.append(win.handle)
        return handles

    def _commg_ensure_connection(self):
        if not HAS_PYWINAUTO: return False
        if self.commg_handles:
            try:
                app = Application(backend="uia").connect(handle=self.commg_handles[0])
                app.window(handle=self.commg_handles[0]).exists()
                return True
            except:
                self.commg_handles = [] 

        self.q.put("[CommG] CommuniGATOR not found. Launching...\n")
        found_handles = self._commg_find_handles()
        
        if not found_handles:
            if not os.path.exists(self.COMMG_PATH):
                self.q.put(f"[CommG] ERROR: Cannot find shortcut at {self.COMMG_PATH}\n")
                return False
                
            os.startfile(self.COMMG_PATH) 
            time.sleep(3.5) 
            found_handles = self._commg_find_handles()
            
        if not found_handles:
            self.q.put("[CommG] ERROR: Software launched but couldn't attach.\n")
            return False

        self.commg_handles = found_handles[:1]
        self.q.put("[CommG] Successfully hooked into CommuniGATOR.\n")
        return True

    def _commg_send_command_thread_impl(self, payload):
        ips = list(self.ip_listbox.get(0, tk.END))
        if not ips:
            self.q.put("[CommG] No IPs configured!\n")
            self.q.put((_EVT_COMMG_DONE, "ABORT"))
            return

        if not self._commg_ensure_connection():
            self.q.put("[CommG] Failed to connect to CommuniGATOR.\n")
            self.q.put((_EVT_COMMG_DONE, "ABORT"))
            return

        handle = self.commg_handles[0]
        try:
            app = Application(backend="uia").connect(handle=handle)
            main_win = app.window(handle=handle)
            toolbar = main_win.child_window(auto_id="59392", control_type="ToolBar")
            input_field = main_win.child_window(auto_id="1004", control_type="Edit")
            
            for ip in ips:
                self.q.put(f"[CommG] Ping check {ip}...\n")
                if not self._commg_ping_ip(ip):
                    self.q.put(f"[CommG] WARNING: {ip} is offline.\n")
                    proceed = messagebox.askyesno("IP Not Found", f"Could not reach IP: {ip}\n\nDo you want to skip this one and proceed?")
                    if not proceed:
                        self.q.put("[CommG] Sequence aborted by user.\n")
                        self.q.put((_EVT_COMMG_DONE, "ABORT"))
                        return
                    continue

                try:
                    with self.type_lock:
                        main_win.set_focus()
                        
                        toolbar.button(0).click() 
                        time.sleep(1.0)
                        popup = app.top_window() 
                        popup.type_keys(ip + "{ENTER}", set_foreground=True)
                        time.sleep(1.5) 
                        
                        toolbar.button(1).click()
                        time.sleep(1.0)
                        
                        input_field.set_focus()
                        input_field.type_keys("^a{BACKSPACE}0121FF{ENTER}", set_foreground=True)
                        time.sleep(1.5)
                        
                        input_field.set_focus()
                        input_field.type_keys("^a{BACKSPACE}" + payload + "{ENTER}", set_foreground=True)
                        self.q.put(f"[CommG] Sent {payload} to {ip}\n")
                
                except Exception as inner_e:
                    self.q.put(f"[CommG] ERROR on {ip}: {inner_e}\n")
                    proceed = messagebox.askyesno("Automation Error", f"An error occurred while controlling the UI for {ip}.\n\nDo you want to skip and proceed?")
                    if not proceed:
                        self.q.put("[CommG] Sequence aborted by user due to UI error.\n")
                        self.q.put((_EVT_COMMG_DONE, "ABORT"))
                        return
                    
            self.q.put("[CommG] Waiting 5s for UI to update...\n")
            time.sleep(5.0) 
            
        except Exception as e:
            self.q.put(f"[CommG] FATAL ERROR: {e}\n")
            self.q.put((_EVT_COMMG_DONE, "ABORT"))
            return

        self.q.put((_EVT_COMMG_DONE, None))

    def _cmd_telnet_thread(self, payload):
        import socket
        
        ips = [ip.strip() for ip in self.ip_listbox.get(0, tk.END) if ip.strip()]
        if not ips:
            self.q.put("[CMD] No IPs configured!\n")
            self.q.put((_EVT_COMMG_DONE, "ABORT"))
            return

        # Get port safely
        try:
            port = int(self.telnet_port_var.get().strip() or 23)
        except ValueError:
            port = 23
            
        # Prepare list to track background sockets
        if not hasattr(self, "active_sockets"):
            self.active_sockets = []

        for ip in ips:
            self.q.put(f"[CMD] Ping check {ip}...\n")
            if not self._commg_ping_ip(ip):
                self.q.put(f"[CMD] WARNING: {ip} is offline.\n")
                proceed = messagebox.askyesno("IP Not Found", f"Could not reach IP: {ip}\n\nDo you want to skip this one and proceed?")
                if not proceed:
                    self.q.put("[CMD] Sequence aborted by user.\n")
                    self.q.put((_EVT_COMMG_DONE, "ABORT"))
                    return
                continue

            try:
                self.q.put(f"[CMD] Opening background connection to {ip}...\n")
                
                # 1. Connect natively in the background (No CMD window opens)
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((ip, port))
                
                # Wait briefly for the prompt to be ready
                time.sleep(1.0)
                
                # 2. Inject payload directly over the network
                s.sendall(f"{payload}\r\n".encode('ascii'))
                self.q.put(f"[CMD] Sent {payload} to {ip} (Session running invisibly)\n")
                
                # Track the socket so we can close it later
                self.active_sockets.append(s)
            
            except Exception as inner_e:
                self.q.put(f"[CMD] ERROR on {ip}: {inner_e}\n")
                proceed = messagebox.askyesno("Automation Error", f"An error occurred while connecting to {ip}.\n\nDo you want to skip and proceed?")
                if not proceed:
                    self.q.put("[CMD] Sequence aborted by user.\n")
                    self.q.put((_EVT_COMMG_DONE, "ABORT"))
                    return

        self.q.put("[CMD] Waiting 3s for devices to process...\n")
        time.sleep(3.0) 
        self.q.put((_EVT_COMMG_DONE, None))

    def _integration_send_command_thread(self, payload):
        integration_type = self.integration_type_var.get()
        if integration_type == "CMD":
            self._cmd_telnet_thread(payload)
        else:
            self._commg_send_command_thread_impl(payload)

    def _run_next_commg_step(self):
        if not self._commg_pending_queue:
            self._commg_is_active_run = False
            self._set_running(False)
            self.status_var.set("Automation Batch Complete")
            self.status_label.configure(fg="green")
            return

        item = self._commg_pending_queue.pop(0)
        cmd = item
        tag = ""
        if isinstance(item, tuple):
            cmd, tag = item
            
        self._current_batch_cmd = str(cmd).strip()  # <-- ADD THIS LINE to track the current command
            
        # Clean up tag (pandas might read empty cells as 'nan')
        tag = str(tag).strip()
        if tag.lower() == 'nan':
            tag = ""
        
        if tag.upper() in ["SKIP", "NO_VERIFY", "NONE"]:
            self.tag_var.set("SKIP_VERIFY")
            self.index_var.set("SKIP_VERIFY")
            tool = self.integration_type_var.get()
            self.q.put(f"\n[{tool}] Non-verification command detected: {cmd}\n", ("commg",))
        elif tag:
            self.tag_var.set(tag)
            self.index_var.set("")
            tool = self.integration_type_var.get()
            self.q.put(f"\n[{tool}] Switching String Tag to: {tag}\n", ("commg",))
        else:
            # Auto-extract index from the command (e.g., STR_TEST:FIX:0052:0030 -> 0030)
            if ":" in cmd:
                extracted_idx = cmd.split(":")[-1].strip()
            else:
                extracted_idx = cmd.strip()
            
            # Clean index for dictionary lookup (strip leading zeros)
            clean_idx = extracted_idx
            if clean_idx.isdigit():
                clean_idx = str(int(clean_idx))
                
            # Lookup the tag dynamically from the Excel file
            if not getattr(self, "_index_to_tag_cache", None):
                self.q.put("[GUI] Caching Excel index-to-tag mapping...\n", ("commg",))
                try:
                    self._index_to_tag_cache = _build_index_to_tag_map(self.excel_var.get().strip())
                except Exception:
                    self._index_to_tag_cache = {}

            found_tag = self._index_to_tag_cache.get(clean_idx, "")

            self.index_var.set(extracted_idx)
            tool = self.integration_type_var.get()
            
            if found_tag:
                self.tag_var.set(found_tag)
                self.q.put(f"\n[{tool}] Extracted Index '{extracted_idx}' (Found Tag: {found_tag})\n", ("commg",))
            else:
                self.tag_var.set("")  # Clear it only if no tag exists for that index
                self.q.put(f"\n[{tool}] Extracted Index '{extracted_idx}'\n", ("commg",))

        self.status_var.set(f"Automation Running: {cmd}...")
        self.status_label.configure(fg="purple")
        self._set_running(True)

        threading.Thread(target=self._integration_send_command_thread, args=(cmd,), daemon=True).start()

    # --- Device List Methods ---
    def _regrid_device_rows(self):
        for idx, row in enumerate(self.device_rows):
            row_idx = idx + 1
            widgets = row.get("widgets") or ()
            if len(widgets) >= 3:
                rb, lbl_id, ent_name = widgets[:3]
                rb.grid(row=row_idx, column=0, sticky="w", padx=(0, 6), pady=2)
                lbl_id.grid(row=row_idx, column=1, sticky="w", padx=(0, 10), pady=2)
                ent_name.grid(row=row_idx, column=2, sticky="ew", pady=2)

    def _renumber_device_rows(self):
        for idx, row in enumerate(self.device_rows):
            new_id = idx + 1
            row["id"] = int(new_id)
            widgets = row.get("widgets") or ()
            if len(widgets) >= 2:
                rb, lbl_id = widgets[0], widgets[1]
                try:
                    rb.configure(value=int(new_id))
                    lbl_id.configure(text=f"D{int(new_id)}")
                except Exception:
                    pass

    def _add_device_row(self, did: int, name: str = ""):
        row_idx = len(self.device_rows) + 1
        var_name = tk.StringVar(value=str(name or ""))

        rb = tk.Radiobutton(self.lf_devices, variable=self.selected_device_id, value=int(did))
        lbl_id = tk.Label(self.lf_devices, text=f"D{int(did)}")
        ent_name = tk.Entry(self.lf_devices, textvariable=var_name, width=30)

        rb.grid(row=row_idx, column=0, sticky="w", padx=(0, 6), pady=2)
        lbl_id.grid(row=row_idx, column=1, sticky="w", padx=(0, 10), pady=2)
        ent_name.grid(row=row_idx, column=2, sticky="ew", pady=2)

        self.device_rows.append({
            "id": int(did),
            "var_name": var_name,
            "widgets": (rb, lbl_id, ent_name),
        })

        if int(self.selected_device_id.get() or 0) == 0:
            self.selected_device_id.set(int(did))

    def _on_remove_device(self):
        if not self.device_rows: return
        target_id = None
        try: target_id = int(self.selected_device_id.get())
        except Exception: target_id = None

        idx = None
        if target_id is not None:
            for i, r in enumerate(self.device_rows):
                if int(r.get("id") or 0) == int(target_id):
                    idx = i
                    break
        if idx is None: idx = len(self.device_rows) - 1

        row = self.device_rows.pop(int(idx))
        for w in row.get("widgets") or []:
            try: w.destroy()
            except Exception: pass

        self._renumber_device_rows()
        self._regrid_device_rows()

        if self.device_rows:
            self.selected_device_id.set(int(self.device_rows[min(int(idx), len(self.device_rows) - 1)].get("id") or 0))
        else:
            self.selected_device_id.set(0)

    def _on_add_device(self):
        next_id = len(self.device_rows) + 1
        self._add_device_row(next_id, "")

    def _load_devices_from_env_or_defaults(self):
        try:
            raw = str(os.getenv("WALKIE_DEVICE_PROFILES_JSON", "") or "").strip()
            if raw:
                obj = json.loads(raw)
                devices = obj.get("devices") if isinstance(obj, dict) else None
                if isinstance(devices, list) and devices:
                    for d in devices:
                        if not isinstance(d, dict): continue
                        try: did = int(d.get("id"))
                        except Exception: continue
                        nm = str(d.get("name") or "")
                        self._add_device_row(did, nm)
                    return
        except Exception: pass

        try:
            cfgp = Path(__file__).resolve().parents[1] / "configs" / "device_profiles.json"
            if cfgp.exists():
                with open(cfgp, "r", encoding="utf-8") as f:
                    obj = json.load(f) or {}
                devices = obj.get("devices") if isinstance(obj, dict) else None
                if isinstance(devices, list) and devices:
                    for d in devices:
                        if not isinstance(d, dict): continue
                        try: did = int(d.get("id"))
                        except Exception: continue
                        nm = str(d.get("name") or "")
                        self._add_device_row(did, nm)
                    return
        except Exception: pass

        for i in range(2):
            self._add_device_row(i + 1, "")

    # --- Directory Handlers ---
    def browse_excel(self):
        p = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel Files", "*.xlsm *.xlsx *.xls"), ("All Files", "*.*")],
        )
        if p:
            self.excel_var.set(p)
            self._index_to_tag_cache = {}  # <-- ADD THIS LINE to clear the cache
            try:
                self.refresh_languages()
                self.refresh_tags()
            except Exception: pass

    def browse_model(self):
        p = filedialog.askopenfilename(
            title="Select YOLO model weights",
            filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")],
        )
        if p: self.model_path_var.set(_resolve_path(p))

    def browse_log(self):
        d = filedialog.askdirectory(title="Select Log Folder")
        if d:
            # Convert to absolute path and normalize to prevent crashes during mkdir
            normalized_path = str(Path(d).resolve())
            self.log_path_var.set(normalized_path)

    # --- Append Logs ---
    def _append(self, s: str):
        self.output.insert(tk.END, s)
        self.output.see(tk.END)
        # We removed the self._log_fp.write() from here to keep logs clean

    def _append_cmd(self, s: str):
        self.output.insert(tk.END, s, ("cmd",))
        self.output.see(tk.END)
        # We removed the self._log_fp.write() from here to keep logs clean

    def _append_line_with_result_color(self, line: str):
        stripped = (line or "").strip()
        low = stripped.lower()

        if stripped.startswith("Expected (") and "):" in stripped:
            try: self.last_expected = stripped.split("):", 1)[1].strip()
            except Exception: pass
        elif stripped == "Expected (normalized):":
            self._pending_expected_norm = True
        elif self._pending_expected_norm and stripped and stripped not in ["PASS", "FAIL", "WARN"]:
            try: self.last_expected = stripped
            except Exception: pass
            self._pending_expected_norm = False

        is_error = False
        if stripped.startswith("Traceback"): is_error = True
        elif "[error]" in low or "[gui error]" in low: is_error = True
        elif low.startswith("error:") or low.startswith("exception"): is_error = True
        elif "typeerror" in low or "valueerror" in low or "runtimeerror" in low: is_error = True
        elif "❌" in stripped: is_error = True

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
            except Exception: pass
        elif stripped == "WARN":
            self.last_result = "WARN"
            self.output.insert(tk.END, line, ("warn",))
        elif is_error:
            self.output.insert(tk.END, line, ("error",))
        elif "[CommG]" in stripped or "[CMD]" in stripped:
            self.output.insert(tk.END, line, ("commg",))
        else:
            self.output.insert(tk.END, line)
            
        self.output.see(tk.END)
        
        # --- NEW LOGIC: Only write to the log file if it's pure script output ---
        is_gui_msg = stripped.startswith("[CommG]") or stripped.startswith("[CMD]") or stripped.startswith("[GUI")
        # Trigger recording only when the main verification block starts
        if "RADIO STRING VERIFICATION - DETECTED" in stripped:
            
            # (OPTIONAL) If you STRICTLY want to save logs ONLY when exactly 3 devices are found, uncomment the next 4 lines:
            # import re
            # match = re.search(r"DETECTED (\d+) DEVICES", stripped)
            # if match and int(match.group(1)) != 3: 
            #     return 
            
            self._is_recording_log = True
            try:
                if self._log_fp is not None:
                    # Manually write the top border line that precedes this trigger
                    self._log_fp.write("=" * 70 + "\n")
            except Exception: pass

        # Write to the file only if the recording flag is True
        try:
            if self._log_fp is not None and getattr(self, "_is_recording_log", False) and not is_gui_msg:
                self._log_fp.write(line)
                if not line.endswith("\n"):
                    self._log_fp.write("\n")
                self._log_fp.flush()
        except Exception: pass

    def _current_settings(self) -> dict:
        camera_name = (self.camera_id_var.get() or "").strip()
        # Save the numeric ID so it works cleanly upon next application startup
        camera_id = getattr(self, "_camera_map", {}).get(camera_name, camera_name)
        
        return {
            "excel": (self.excel_var.get() or "").strip(),
            "region": (self.region_var.get() or "").strip(),
            "language": (self.language_var.get() or "").strip(),
            "tag": (self.tag_var.get() or "").strip(),
            "index": (self.index_var.get() or "").strip(),
            "model_path": (self.model_path_var.get() or "").strip(),
            "camera_id": camera_id,
            "save_log": bool(self.save_log_var.get()),
            "log_path": (self.log_path_var.get() or "").strip(),
            "integration_enable": bool(self.integration_enable_var.get()),
            "integration_type": (self.integration_type_var.get() or "CommG"),
            "telnet_port": (self.telnet_port_var.get() or "23"),
            "commg_enable": bool(self.integration_enable_var.get()), 
            "commg_ips": list(self.ip_listbox.get(0, tk.END)),
            "commg_mode": (self.commg_mode_var.get() or "Batch"),
            "commg_custom_cmd": (self.commg_custom_cmd_var.get() or "").strip(),
            "commg_batch_file": (self.commg_batch_file_var.get() or "").strip(),
        }

    def _persist_settings(self):
        self._settings = self._current_settings()
        _save_settings(self._settings)

    def _on_close(self, skip_prompt=False):
        # -- NEW: Confirmation warning for closing --
        if not skip_prompt:
            if not messagebox.askyesno("Confirm Exit", "Are you sure you want to close the application?"):
                return
        
        # Call stop silently to clean up background ports/sockets properly before exit
        try: 
            self.stop(skip_prompt=True)
        except Exception: 
            pass

        try: self._persist_settings()
        except Exception: pass
        try:
            if self._log_fp is not None:
                self._log_fp.close()
                self._log_fp = None
        except Exception: pass
        try: self._log_session_dir = None
        except Exception: pass
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception: pass
        
        self.root.destroy()

    def refresh_cameras(self):
        try: self.btn_run.configure(state=tk.DISABLED)
        except Exception: pass
        self.root.update()

        cam_data = _probe_camera_ids(max_id=8)
        
        if not cam_data: 
            cam_data = [("0", "Default Camera"), ("1", "External Camera"), ("2", "Virtual Camera")]
            
        self._camera_map = {}
        display_names = []
        
        for cid, cname in cam_data:
            # If two cameras have the exact same name, add the ID just to tell them apart
            if cname in self._camera_map:
                cname = f"{cname} (ID: {cid})"
            
            # Save to hidden dictionary
            self._camera_map[cname] = cid
            display_names.append(cname)
            
        try:
            self.camera_combo["values"] = display_names
            cur = (self.camera_id_var.get() or "").strip()
            
            if cur in display_names:
                self.camera_id_var.set(cur)
            elif display_names:
                self.camera_id_var.set(display_names[0])
                
            messagebox.showinfo("Camera Search", f"Successfully completed - Found {len(display_names)} camera(s)!")
            
        except Exception: 
            pass
        finally:
            try: self.btn_run.configure(state=tk.NORMAL)
            except Exception: pass

    def refresh_languages(self):
        excel = (self.excel_var.get() or "").strip()
        region = (self.region_var.get() or "").strip() or "APAC"
        
        # --- NEW LOGIC: Force English language if English region is selected ---
        if region.lower() == "english":
            opts = ["English"]
        else:
            try: opts = _language_options_from_excel(excel, region)
            except Exception: opts = []

            if not opts:
                opts = ["Japanese", "Korean", "Simplified Chinese", "Traditional Chinese", "French", "Spanish", "German", "Italian", "Polish", "Russian", "Turkish", "Arabic", "Hungarian", "Hebrew", "Czech", "Portuguese"]

        try:
            self.language_combo["values"] = opts
            cur = (self.language_var.get() or "").strip()
            
            # If current language is valid for this region, keep it
            if cur and cur in opts: 
                return
                
            # Otherwise, auto-select the first available language (which will be 'English' for the English region)
            if opts: 
                self.language_var.set(opts[0])
        except Exception: pass

    def refresh_tags(self):
        excel = (self.excel_var.get() or "").strip()
        try: tags = _tag_options_from_excel(excel)
        except Exception: tags = []
        try: self._all_tags = tags
        except Exception: pass
        try: self.tag_combo["values"] = tags
        except Exception: pass

    def _on_tag_typed(self, _evt=None):
        try: all_tags = list(getattr(self, "_all_tags", []) or [])
        except Exception: all_tags = []

        if not all_tags: return

        try: typed_raw = self.tag_combo.get()
        except Exception: typed_raw = self.tag_var.get()
        typed = (typed_raw or "").strip().lower()
        if not typed: filtered = all_tags
        else: filtered = [t for t in all_tags if typed in str(t).lower()]

        try: self.tag_combo["values"] = filtered
        except Exception: pass

    def _unpost_tag_dropdown(self):
        try: self.tag_combo.tk.call("ttk::combobox::Unpost")
        except Exception: pass

    def _on_tag_escape(self, _evt=None):
        self._unpost_tag_dropdown()

    def _on_tag_focus_out(self, _evt=None):
        self._unpost_tag_dropdown()

    def _on_tag_down(self, _evt=None):
        try:
            if not list(self.tag_combo["values"] or []): self.refresh_tags()
        except Exception: pass
        try: self._on_tag_typed()
        except Exception: pass
        try: self.tag_combo.tk.call("ttk::combobox::Post", str(self.tag_combo))
        except Exception: pass

    def clear(self):
        self.output.delete("1.0", tk.END)
        self.last_result = ""
        self.status_var.set("Idle")
        try: self.status_label.configure(fg="black")
        except Exception: pass

    def close(self):
        self._on_close()

    def _set_running(self, running: bool):
        try:
            if running:
                self.btn_run.configure(state=tk.DISABLED)
                self.btn_stop.configure(state=tk.NORMAL)
            else:
                self.btn_run.configure(state=tk.NORMAL)
                self.btn_stop.configure(state=tk.DISABLED)
        except Exception: pass

    def _build_cmd(self):
        excel = self.excel_var.get().strip()
        if not excel: raise ValueError("Excel path is required")

        region = self.region_var.get().strip()
        language = self.language_var.get().strip()
        tag = self.tag_var.get().strip()
        idx = self.index_var.get().strip()

        if not region: raise ValueError("Region is required (e.g., APAC/EMEA/LACR/English)")
        if not language: raise ValueError("Language is required (e.g., Japanese)")
        if not tag and not idx: raise ValueError("Provide either String Tag or Index")

        script_path = Path(__file__).resolve().parent / "verify_string.py"
        cmd = [_python_exe(), str(script_path), "--excel", excel, "--region", region, "--language", language]

        if tag: cmd += ["--tag", tag]
        if idx: cmd += ["--index", idx]

        model_path = _resolve_path(self.model_path_var.get())
        try: self.model_path_var.set(model_path)
        except Exception: pass
        if model_path: cmd += ["--model-path", model_path]

        # -- NEW: Look up the real numeric ID from the selected name --
        camera_name = self.camera_id_var.get().strip()
        if camera_name:
            # Fallback to whatever is in the box if it isn't in the map (e.g., user typed manually)
            camera_id = getattr(self, "_camera_map", {}).get(camera_name, camera_name)
            cmd += ["--camera-id", str(camera_id)]

        return cmd

    def _run_subprocess(self, cmd, *, is_verification: bool = True):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Running", "A process is already running. Click Stop first.")
            return

        try: self._last_run_is_verification = bool(is_verification)
        except Exception: self._last_run_is_verification = True

        self.last_result = ""
        self.last_expected = ""   
        self.last_actual = ""     
        self.last_error_msg = ""  
        self._is_recording_log = False # <--- ADD THIS LINE HERE
        
        self.status_var.set("Running Verify...")
        try: self.status_label.configure(fg="black")
        except Exception: pass
        self._set_running(True)

        try:
            if self._log_fp is not None: self._log_fp.close()
        except Exception: pass
        self._log_fp = None

        if self.save_log_var.get() and bool(is_verification):
            # --- NEW: Check if part of a batch, reuse folder if so ---
            if getattr(self, "_commg_is_active_run", False) and getattr(self, "_batch_log_dir", None):
                import re
                # Get the command name, default to "Unknown" if missing
                cmd_name = getattr(self, "_current_batch_cmd", "Unknown_Command")
                # Clean invalid Windows folder characters (like colons) into underscores
                safe_folder_name = re.sub(r'[\\/*?:"<>|]', '_', cmd_name)
                
                # Append the safe command name as a subfolder inside the batch directory
                self._log_session_dir = str(Path(self._batch_log_dir) / safe_folder_name)
                
                try: 
                    Path(self._log_session_dir).mkdir(parents=True, exist_ok=True)
                except Exception: 
                    self._log_session_dir = self._batch_log_dir
            else:
                d = (self.log_path_var.get() or "").strip()
                if not d:
                    d = str(Path.cwd())
                    self.log_path_var.set(d)

                try:
                    dp = Path(d)
                    if dp.suffix.lower() in [".log", ".txt"]: dp = dp.parent
                    d = str(dp)
                except Exception: pass

                sess = time.strftime("%Y%m%d_%H%M%S")
                self._log_session_dir = str(Path(d) / f"verified_{sess}")
                try: Path(self._log_session_dir).mkdir(parents=True, exist_ok=True)
                except Exception: self._log_session_dir = d
            # ---------------------------------------------------------

            ts = time.strftime("%Y%m%d_%H%M%S")
            safe_tag = (self.tag_var.get() or "").strip()
            safe_tag = "".join([c for c in safe_tag if c.isalnum() or c in ["_", "-"]])[:40]
            name = f"verify_string_{ts}.log" if not safe_tag else f"verify_string_{safe_tag}_{ts}.log"
            p = str(Path(self._log_session_dir) / name)
            try:
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                self._log_fp = open(p, "a", encoding="utf-8")
                # Deleted the hardcoded headers here so the file starts exactly at "RADIO STRING VERIFICATION..."
            except Exception: self._log_fp = None

        try:
            self.output.insert(tk.END, "$ " + " ".join(cmd) + "\n\n", ("cmd",))
            self.output.see(tk.END)
        except Exception: pass

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
        except Exception as e:
            self.q.put(f"[GUI ERROR] Failed to initialize log: {e}\n")
            self._log_fp = None

        env = os.environ.copy()
        try:
            devices = []
            for r in self.device_rows:
                did = int(r.get("id"))
                nm = str(r.get("var_name").get() if r.get("var_name") else "").strip()
                devices.append({"id": did, "name": nm})
            
            env["WALKIE_DEVICE_PROFILES_JSON"] = json.dumps({"devices": devices}, ensure_ascii=False)
            cfgp = Path(__file__).resolve().parents[1] / "configs" / "device_profiles.json"
            cfgp.parent.mkdir(parents=True, exist_ok=True)
            with open(cfgp, "w", encoding="utf-8") as f:
                json.dump({"devices": devices}, f, indent=2, ensure_ascii=False)
        except Exception: pass

        def _worker():
            try:
                # --- AUTOMATION UPDATE: Hide subprocess terminal window ---
                kwargs = {}
                if os.name == 'nt':
                    kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                # ----------------------------------------------------------
                
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False, bufsize=0, env=env, **kwargs
                )
                assert self.proc.stdout is not None
                for raw in iter(self.proc.stdout.readline, b""):
                    try: line = raw.decode("utf-8", errors="replace")
                    except Exception:
                        try: line = raw.decode(errors="replace")
                        except Exception: line = str(raw)
                    self.q.put(line)

                try: rc = self.proc.wait(timeout=1)
                except Exception: rc = None
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
        # -- NEW: Immediately disable start button --
        self.btn_run.configure(state=tk.DISABLED)
        
        # Setup Automation Batch if enabled
        if self.integration_enable_var.get() and HAS_PYWINAUTO:
            
            if not list(self.ip_listbox.get(0, tk.END)):
                messagebox.showwarning("No IPs", "Please add at least one IP address to the list before starting!")
                self.btn_run.configure(state=tk.NORMAL) # Re-enable on error
                return

            if self.commg_mode_var.get() == "Single":
                c = self.commg_custom_cmd_var.get().strip()
                if not c:
                    messagebox.showerror("Error", "Please enter a Custom Command.")
                    self.btn_run.configure(state=tk.NORMAL) # Re-enable on error
                    return
                self._commg_pending_queue = [c]
            else:
                bf = self.commg_batch_file_var.get().strip()
                if not bf or not os.path.exists(bf):
                    messagebox.showerror("Error", "Please select a valid Batch File.")
                    self.btn_run.configure(state=tk.NORMAL) # Re-enable on error
                    return
                try:
                    if bf.lower().endswith('.csv'): df = pd.read_csv(bf, header=None)
                    else: df = pd.read_excel(bf, header=None)
                    
                    cmds = df.iloc[:, 0].dropna().astype(str).tolist()
                    self._commg_pending_queue = [c.strip() for c in cmds if c.strip()]
                    
                    if len(df.columns) > 1:
                        tags = df.iloc[:, 1].fillna("").astype(str).tolist()
                        self._commg_pending_queue = list(zip(self._commg_pending_queue, tags))
                        
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load Batch File: {e}")
                    self.btn_run.configure(state=tk.NORMAL) # Re-enable on error
                    return
                    
            if not self._commg_pending_queue:
                messagebox.showwarning("Warning", "No commands found in the batch file.")
                self.btn_run.configure(state=tk.NORMAL) # Re-enable on error
                return
            
            self._commg_is_active_run = True
            tool = self.integration_type_var.get()
            self.q.put(f"[{tool}] Initialized batch with {len(self._commg_pending_queue)} commands.\n", ("commg",))

            # Create a single log folder for the whole batch
            if self.save_log_var.get():
                d = (self.log_path_var.get() or "").strip()
                if not d:
                    d = str(Path.cwd())
                    self.log_path_var.set(d)
                sess = time.strftime("%Y%m%d_%H%M%S")
                self._batch_log_dir = str(Path(d) / f"batch_run_{sess}")
                try:
                    Path(self._batch_log_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    self._batch_log_dir = d
        else:
            self._commg_pending_queue = []
            self._commg_is_active_run = False
            self._batch_log_dir = None

        self._auto_start_verify = True
        self.init_genai()

    def run(self):
        # -- NEW: Immediately disable start button --
        self.btn_run.configure(state=tk.DISABLED)
        
        try: cmd = self._build_cmd()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            self._commg_is_active_run = False
            self._set_running(False)
            return

        try: self._persist_settings()
        except Exception: pass

        self._run_subprocess(cmd, is_verification=True)

    def stop(self, skip_prompt=False):
        # -- NEW: Confirmation warning for stopping --
        if not skip_prompt:
            if not messagebox.askyesno("Confirm Stop", "Are you sure you want to stop the current process?"):
                return
                
        self._commg_pending_queue = []
        self._commg_is_active_run = False
        
        # Cleanup lingering background sockets if stopped midway
        if getattr(self, "active_sockets", []):
            for s in self.active_sockets:
                try:
                    # 1. Send the device closing command
                    s.sendall(b"STR_TEST:CLOSE\r\n")
                    time.sleep(0.2)
                    # 2. Close gracefully
                    s.close()
                except Exception:
                    pass
            self.active_sockets = []

        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self._append("\n[INFO] Stopping process...\n")
            except Exception: pass
            
    def _drain_queue(self):
        try:
            while True:
                s = self.q.get_nowait()
                
                # Intercept Automation Command Done
                if isinstance(s, tuple) and len(s) == 2 and s[0] == _EVT_COMMG_DONE:
                    if s[1] == "ABORT":
                        # If user aborted, cancel the active run and reset UI
                        self._commg_is_active_run = False
                        self._commg_pending_queue = []
                        self.status_var.set("Automation Sequence Aborted")
                        self.status_label.configure(fg="red")
                        self._set_running(False)
                    else:
                        # Check if skipping verification
                        if self.index_var.get() == "SKIP_VERIFY" or self.tag_var.get() == "SKIP_VERIFY":
                            self.q.put("[GUI] Skipping verification phase as requested.\n", ("commg",))
                            if self._commg_pending_queue:
                                self._run_next_commg_step()
                            else:
                                self._commg_is_active_run = False
                                self.status_var.set("Automation Batch Complete")
                                self.status_label.configure(fg="green")
                                self._set_running(False)
                        else:
                            # After successfully sending the command, run Verify
                            self.run()
                    continue

                if isinstance(s, tuple) and len(s) == 2 and s[0] == _EVT_FINISHED:
                    rc = s[1]
                    is_verify = bool(self._last_run_is_verification)

                    # --- Close background sockets after verification completes ---
                    if is_verify and getattr(self, "active_sockets", []):
                        self.q.put("[CMD] Verification complete. Sending close command and terminating sessions...\n")
                        for s in self.active_sockets:
                            try:
                                s.sendall(b"STR_TEST:CLOSE\r\n")
                                time.sleep(0.2)
                                s.close()
                            except Exception:
                                pass
                        self.active_sockets = []
                    # --------------------------------------------------------------------

                    will_autostart = False
                    try:
                        will_autostart = bool(self._auto_start_verify) and (not is_verify) and int(rc or 0) == 0
                    except Exception:
                        will_autostart = False

                    # Hook Integration iteration logic
                    if self._commg_is_active_run:
                        if is_verify:
                            if self._commg_pending_queue:
                                self._run_next_commg_step()
                            else:
                                self._commg_is_active_run = False
                                self.status_var.set("Automation Batch Complete")
                                self.status_label.configure(fg="green")
                                self._set_running(False)
                            continue
                        elif will_autostart:
                            self._auto_start_verify = False 
                            self._run_next_commg_step()
                            continue

                    if will_autostart:
                        self.status_var.set("Init GenAI finished, starting verification...")
                    else:
                        if is_verify: msg = "Finished" if rc in [0, None] else f"Finished (exit={rc})"
                        else: msg = "Init GenAI finished" if rc in [0, None] else f"Init GenAI finished (exit={rc})"
                        self.status_var.set(msg)

                    try:
                        if rc not in [0, None]: self.status_label.configure(fg="#B71C1C")
                        else: self.status_label.configure(fg="black")
                    except Exception: pass

                    if not will_autostart: self._set_running(False)

                    try:
                        if self._log_fp is not None:
                            # self._log_fp.write(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            self._log_fp.flush()
                            self._log_fp.close()
                            self._log_fp = None
                    except Exception: pass

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
    root.mainloop()


if __name__ == "__main__":
    main()