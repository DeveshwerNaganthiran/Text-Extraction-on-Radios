import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
from pywinauto import Desktop
from pywinauto.application import Application
import threading
import time
import os
import subprocess 
import json
import pandas as pd # <-- NEW: Required for reading CSV and Excel files

# --- 1. SET YOUR SHORTCUT PATH HERE ---
COMMG_PATH = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Motorola\CommG_LTD\CommG_LTD.lnk"
WINDOW_SEARCH_TERM = "CommuniGATOR" 

# --- 2. CONFIG PATH FOR SAVED IPs ---
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
IP_CONFIG_FILE = os.path.join(CONFIG_DIR, "communigator_ips.json")

class CommG_Ultimate_Controller:
    def __init__(self, root):
        self.root = root
        self.root.title("CommG Pro - Auto-Sequencer")
        self.root.geometry("650x800") # <-- Increased height to fit the new batch section
        
        self.running = True
        self.commg_handles = [] 
        self.log_counts = []
        self.type_lock = threading.Lock()

        # --- UI LAYOUT ---
        
        # 1. IP Address Setup (Dynamic List)
        self.ip_frame = tk.LabelFrame(root, text=" IP Address ", font=('Arial', 10, 'bold'))
        self.ip_frame.pack(pady=10, padx=15, fill="x")
        
        self.ip_listbox = tk.Listbox(self.ip_frame, height=4, width=30, font=('Arial', 10))
        self.ip_listbox.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        
        self.load_ips()

        self.ip_controls_frame = tk.Frame(self.ip_frame)
        self.ip_controls_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=10, sticky="n")
        
        self.new_ip_entry = tk.Entry(self.ip_controls_frame, width=20, font=('Arial', 10))
        self.new_ip_entry.pack(pady=(0, 5))
        
        tk.Button(self.ip_controls_frame, text="Add IP", bg="#90EE90", width=17, command=self.add_ip).pack(pady=2)
        tk.Button(self.ip_controls_frame, text="Remove Selected", bg="#FFCCCB", width=17, command=self.remove_ip).pack(pady=2)

        self.status_label = tk.Label(root, text="STATUS: IDLE - READY FOR COMMANDS", fg="blue", font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=5)

        # 2. Execution Commands
        self.mode_frame = tk.LabelFrame(root, text=" Execution Commands ", font=('Arial', 10, 'bold'))
        self.mode_frame.pack(pady=5, padx=15, fill="x")
        
        tk.Button(self.mode_frame, text="TEST MODE (03000C)", width=28, command=lambda: self.run_full_sequence("03000C")).grid(row=0, column=0, padx=10, pady=10)
        tk.Button(self.mode_frame, text="REBOOT (03000D)", width=28, bg="#FFCCCB", command=lambda: self.run_full_sequence("03000D")).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.mode_frame, text="BOOTMODE (030200)", width=28, bg="#DDA0DD", command=lambda: self.run_full_sequence("030200")).grid(row=1, column=0, padx=10, pady=10)
        tk.Button(self.mode_frame, text="MUX TO AP (03011001)", width=28, bg="#ADD8E6", command=self.mux_to_ap).grid(row=1, column=1, padx=10, pady=10)

        # 3. Custom Raw Command
        self.custom_frame = tk.LabelFrame(root, text=" Send Custom Raw Command ", font=('Arial', 10, 'bold'))
        self.custom_frame.pack(pady=5, padx=15, fill="x")
        
        tk.Label(self.custom_frame, text="Enter Command:").pack(side="left", padx=10, pady=10)
        self.custom_entry = tk.Entry(self.custom_frame, width=35, font=('Arial', 10))
        self.custom_entry.insert(0, "03001101") 
        self.custom_entry.pack(side="left", padx=5)
        tk.Button(self.custom_frame, text="EXECUTE", bg="#D8BFD8", width=15, command=self.send_custom_raw).pack(side="left", padx=10)

        # --- 4. NEW: BATCH EXECUTION FRAME ---
        self.batch_frame = tk.LabelFrame(root, text=" Batch Execution (CSV / Excel) ", font=('Arial', 10, 'bold'))
        self.batch_frame.pack(pady=5, padx=15, fill="x")
        
        self.file_path_var = tk.StringVar()
        self.file_entry = tk.Entry(self.batch_frame, textvariable=self.file_path_var, width=38, font=('Arial', 10), state='readonly')
        self.file_entry.pack(side="left", padx=10, pady=10)
        
        tk.Button(self.batch_frame, text="BROWSE", width=10, command=self.browse_file).pack(side="left", padx=5)
        tk.Button(self.batch_frame, text="EXECUTE BATCH", bg="#F0E68C", width=15, command=self.run_batch_sequence).pack(side="left", padx=10)

        # 5. Live Log
        tk.Label(root, text="LIVE LOG", font=('Arial', 9, 'bold')).pack(pady=(10,0))
        self.log_display = scrolledtext.ScrolledText(root, height=10, state='disabled', bg="black", fg="lime", font=('Consolas', 9))
        self.log_display.pack(pady=5, padx=15, fill="x")

        threading.Thread(target=self.bg_monitor, daemon=True).start()

    # --- SAVE / LOAD IP LOGIC ---
    def load_ips(self):
        default_ips = ["192.168.10.1", "192.168.10.2"]
        if os.path.exists(IP_CONFIG_FILE):
            try:
                with open(IP_CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    ips = data.get("ips", default_ips)
            except Exception as e:
                print(f"Error loading IPs: {e}")
                ips = default_ips
        else:
            ips = default_ips
            
        for ip in ips:
            self.ip_listbox.insert(tk.END, ip)

    def save_ips(self):
        ips = self.get_target_ips()
        os.makedirs(CONFIG_DIR, exist_ok=True)
        try:
            with open(IP_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump({"ips": ips}, f, indent=4)
        except Exception as e:
            print(f"Error saving IPs: {e}")

    # --- UI HELPERS ---
    def add_ip(self):
        new_ip = self.new_ip_entry.get().strip()
        if not new_ip:
            messagebox.showwarning("Empty Input", "Please enter an IP address before clicking Add.")
            return
            
        current_ips = self.get_target_ips()
        if new_ip in current_ips:
            messagebox.showwarning("Duplicate IP", f"The IP address {new_ip} is already in the list!")
            return
            
        self.ip_listbox.insert(tk.END, new_ip)
        self.new_ip_entry.delete(0, tk.END)
        self.save_ips()

    def remove_ip(self):
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
            self.save_ips()

    def ui_log(self, message):
        self.log_display.config(state='normal')
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.log_display.config(state='disabled')

    def get_target_ips(self):
        return list(self.ip_listbox.get(0, tk.END))

    # --- PING HELPER ---
    def ping_ip(self, ip):
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            response = subprocess.call(['ping', '-n', '1', '-w', '2000', ip], startupinfo=startupinfo)
            return response == 0
        except Exception:
            return False

    # --- AUTO-CONNECT LOGIC ---
    def _find_commg_handles(self):
        handles = []
        desktop = Desktop(backend="uia")
        for win in desktop.windows():
            text = win.window_text()
            if text and WINDOW_SEARCH_TERM in text:
                handles.append(win.handle)
        return handles

    def ensure_connection(self):
        if self.commg_handles:
            try:
                app = Application(backend="uia").connect(handle=self.commg_handles[0])
                app.window(handle=self.commg_handles[0]).exists()
                return True
            except:
                self.commg_handles = [] 

        self.root.after(0, self.status_label.config, {"text": "STATUS: LAUNCHING COMMG...", "fg": "orange"})
        self.root.after(0, self.ui_log, "SYSTEM: CommuniGATOR not found. Launching it now...")
        
        found_handles = self._find_commg_handles()
        
        if not found_handles:
            if not os.path.exists(COMMG_PATH):
                messagebox.showerror("File Not Found", f"Cannot find shortcut at:\n{COMMG_PATH}")
                self.root.after(0, self.status_label.config, {"text": "STATUS: LAUNCH FAILED", "fg": "red"})
                return False
                
            os.startfile(COMMG_PATH) 
            time.sleep(3.5) 
            found_handles = self._find_commg_handles()
            
        if not found_handles:
            self.root.after(0, self.status_label.config, {"text": "STATUS: WINDOW TITLE ERROR", "fg": "red"})
            self.root.after(0, self.ui_log, "ERROR: Software launched but couldn't attach.")
            return False

        self.commg_handles = found_handles[:1]
        self.log_counts = [0]
        self.root.after(0, self.ui_log, "SYSTEM: Successfully hooked into CommuniGATOR.")
        return True

    # --- ORIGINAL SINGLE COMMAND SEQUENCE ---
    def _execute_sequence_thread(self, payload):
        ips = self.get_target_ips()
        if not ips:
            messagebox.showwarning("No IPs", "Please add at least one IP address to the list!")
            return
            
        if not self.ensure_connection():
            return
            
        handle = self.commg_handles[0]
        
        try:
            app = Application(backend="uia").connect(handle=handle)
            main_win = app.window(handle=handle)
            toolbar = main_win.child_window(auto_id="59392", control_type="ToolBar")
            input_field = main_win.child_window(auto_id="1004", control_type="Edit")
            
            for ip in ips:
                self.root.after(0, self.status_label.config, {"text": f"STATUS: CHECKING {ip}...", "fg": "purple"})
                
                if not self.ping_ip(ip):
                    self.root.after(0, self.ui_log, f"WARNING: {ip} is offline/unreachable.")
                    proceed = messagebox.askyesno("IP Not Found", f"Could not reach IP: {ip}\n\nDo you want to skip this one and proceed to the next IP?")
                    if proceed: continue 
                    else:
                        self.root.after(0, self.ui_log, "SYSTEM: Sequence aborted by user.")
                        break

                self.root.after(0, self.status_label.config, {"text": f"STATUS: PROCESSING {ip}...", "fg": "purple"})
                
                try:
                    with self.type_lock:
                        main_win.set_focus()
                        
                        toolbar.button(0).click() 
                        time.sleep(1.0)
                        popup = app.top_window() 
                        popup.type_keys(ip + "{ENTER}", set_foreground=True)
                        self.root.after(0, self.ui_log, f"\n--- TARGET: {ip} ---")
                        time.sleep(1.5) 
                        
                        toolbar.button(1).click()
                        self.root.after(0, self.ui_log, f"UI: Auto-Clicked NO PROTO")
                        time.sleep(1.0)
                        
                        input_field.set_focus()
                        input_field.type_keys("^a{BACKSPACE}0121FF{ENTER}", set_foreground=True)
                        self.root.after(0, self.ui_log, f"UI: Auto-Sent HANDSHAKE")
                        time.sleep(1.5)
                        
                        input_field.set_focus()
                        input_field.type_keys("^a{BACKSPACE}" + payload + "{ENTER}", set_foreground=True)
                        self.root.after(0, self.ui_log, f"UI: Sent COMMAND ({payload})")
                            
                    self.root.after(0, self.status_label.config, {"text": f"STATUS: WAITING ON {ip}...", "fg": "orange"})
                    time.sleep(5.0) 
                
                except Exception as inner_e:
                    self.root.after(0, self.ui_log, f"ERROR on {ip}: {inner_e}")
                    proceed = messagebox.askyesno("Automation Error", f"An error occurred while controlling the UI for {ip}.\n\nDo you want to skip and proceed to the next IP?")
                    if not proceed:
                        self.root.after(0, self.ui_log, "SYSTEM: Sequence aborted by user due to UI error.")
                        break
            else: 
                self.root.after(0, self.status_label.config, {"text": "STATUS: SEQUENCE COMPLETE ON ALL DEVICES", "fg": "green"})

        except Exception as e:
            self.root.after(0, self.ui_log, f"FATAL ERROR: {e}")
            self.root.after(0, self.status_label.config, {"text": "STATUS: ERROR OCCURRED", "fg": "red"})

    # --- NEW: BATCH SEQUENCE LOGIC (ITERATES COMMANDS FIRST, THEN IPs) ---
    def _execute_batch_thread(self, commands):
        ips = self.get_target_ips()
        if not ips:
            messagebox.showwarning("No IPs", "Please add at least one IP address to the list!")
            return
            
        if not self.ensure_connection():
            return
            
        handle = self.commg_handles[0]
        
        try:
            app = Application(backend="uia").connect(handle=handle)
            main_win = app.window(handle=handle)
            toolbar = main_win.child_window(auto_id="59392", control_type="ToolBar")
            input_field = main_win.child_window(auto_id="1004", control_type="Edit")
            
            # Loop 1: Commands
            for cmd_idx, payload in enumerate(commands, 1):
                self.root.after(0, self.ui_log, f"\n=== BATCH COMMAND {cmd_idx}/{len(commands)}: {payload} ===")
                
                # Loop 2: Targets (all IPs execute the current command before moving to the next command)
                for ip in ips:
                    self.root.after(0, self.status_label.config, {"text": f"STATUS: CHECKING {ip}...", "fg": "purple"})
                    
                    if not self.ping_ip(ip):
                        self.root.after(0, self.ui_log, f"WARNING: {ip} is offline/unreachable.")
                        proceed = messagebox.askyesno("IP Not Found", f"Could not reach IP: {ip}\n\nDo you want to skip this one and proceed?")
                        if proceed: continue 
                        else:
                            self.root.after(0, self.ui_log, "SYSTEM: Batch sequence aborted by user.")
                            return # Exits the entire batch thread

                    self.root.after(0, self.status_label.config, {"text": f"STATUS: PROCESSING {ip} (CMD {cmd_idx}/{len(commands)})...", "fg": "purple"})
                    
                    try:
                        with self.type_lock:
                            main_win.set_focus()
                            
                            toolbar.button(0).click() 
                            time.sleep(1.0)
                            popup = app.top_window() 
                            popup.type_keys(ip + "{ENTER}", set_foreground=True)
                            self.root.after(0, self.ui_log, f"\n--- TARGET: {ip} ---")
                            time.sleep(1.5) 
                            
                            toolbar.button(1).click()
                            self.root.after(0, self.ui_log, f"UI: Auto-Clicked NO PROTO")
                            time.sleep(1.0)
                            
                            input_field.set_focus()
                            input_field.type_keys("^a{BACKSPACE}0121FF{ENTER}", set_foreground=True)
                            self.root.after(0, self.ui_log, f"UI: Auto-Sent HANDSHAKE")
                            time.sleep(1.5)
                            
                            input_field.set_focus()
                            input_field.type_keys("^a{BACKSPACE}" + payload + "{ENTER}", set_foreground=True)
                            self.root.after(0, self.ui_log, f"UI: Sent COMMAND ({payload})")
                                
                        self.root.after(0, self.status_label.config, {"text": f"STATUS: WAITING ON {ip}...", "fg": "orange"})
                        time.sleep(5.0) 
                    
                    except Exception as inner_e:
                        self.root.after(0, self.ui_log, f"ERROR on {ip}: {inner_e}")
                        proceed = messagebox.askyesno("Automation Error", f"An error occurred while controlling the UI for {ip}.\n\nDo you want to skip and proceed?")
                        if not proceed:
                            self.root.after(0, self.ui_log, "SYSTEM: Batch sequence aborted by user due to UI error.")
                            return

            # Finished all commands for all IPs
            self.root.after(0, self.status_label.config, {"text": "STATUS: BATCH SEQUENCE COMPLETE", "fg": "green"})

        except Exception as e:
            self.root.after(0, self.ui_log, f"FATAL ERROR: {e}")
            self.root.after(0, self.status_label.config, {"text": "STATUS: ERROR OCCURRED", "fg": "red"})


    # --- BUTTON ENDPOINTS ---
    def run_full_sequence(self, cmd):
        threading.Thread(target=self._execute_sequence_thread, args=(cmd,), daemon=True).start()

    def mux_to_ap(self):
        if messagebox.askyesno("Mux Warning", "Switching to AP will drop connection. Proceed?"):
            self.run_full_sequence("03011001")

    def send_custom_raw(self):
        cmd = self.custom_entry.get().strip()
        if cmd:
            self.run_full_sequence(cmd)

    # --- BATCH FILE HANDLERS ---
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Command File",
            filetypes=(("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if file_path:
            self.file_path_var.set(file_path)

    def run_batch_sequence(self):
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showwarning("No File", "Please select a CSV or Excel file first.")
            return
            
        try:
            # Load the file based on its extension
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path, header=None)
            else:
                df = pd.read_excel(file_path, header=None)
                
            # Assume commands are in the first column (index 0). 
            # Drop empty rows and convert numbers/data to string type
            commands = df.iloc[:, 0].dropna().astype(str).tolist()
            
            # Remove any empty strings just in case
            commands = [cmd.strip() for cmd in commands if cmd.strip()]
            
            if not commands:
                messagebox.showwarning("Empty File", "No commands found in the first column of the file.")
                return
                
            # Pass the list of commands to the new background thread
            threading.Thread(target=self._execute_batch_thread, args=(commands,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to read file:\n{str(e)}\n\n(Make sure 'pandas' and 'openpyxl' are installed via pip!)")


    # --- BACKGROUND LOGIC ---
    def bg_monitor(self):
        while self.running:
            if self.commg_handles:
                handle = self.commg_handles[0]
                try:
                    app = Application(backend="uia").connect(handle=handle)
                    dlg = app.window(handle=handle)
                    list_box = dlg.child_window(auto_id="1058", control_type="List")
                    items = list_box.items()
                    current_count = len(items)

                    if current_count > self.log_counts[0]:
                        new_items = items[self.log_counts[0]:]
                        for item in new_items:
                            text = item.window_text().strip()
                            if text:
                                self.root.after(0, self.ui_log, f"> {text}")
                        self.log_counts[0] = current_count
                except:
                    pass 
            time.sleep(0.5)

if __name__ == "__main__":
    root = tk.Tk()
    app = CommG_Ultimate_Controller(root)
    root.mainloop()