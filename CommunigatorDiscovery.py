import tkinter as tk
from tkinter import messagebox, scrolledtext
from pywinauto import Desktop
from pywinauto.application import Application
import threading
import time
import os

# --- 1. SET YOUR SHORTCUT PATH HERE ---
COMMG_PATH = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Motorola\CommG_LTD\CommG_LTD.lnk"
WINDOW_SEARCH_TERM = "CommuniGATOR" 

class CommG_Ultimate_Controller:
    def __init__(self, root):
        self.root = root
        self.root.title("CommG Pro - Auto-Sequencer")
        self.root.geometry("650x720") # Adjusted height to perfectly fit the removed frame
        
        self.running = True
        self.commg_handles = [] 
        self.log_counts = []
        self.type_lock = threading.Lock()

        # --- UI LAYOUT ---
        
        # 1. IP Address Setup (Dynamic List)
        self.ip_frame = tk.LabelFrame(root, text=" IP Address ", font=('Arial', 10, 'bold'))
        self.ip_frame.pack(pady=10, padx=15, fill="x")
        
        # Listbox for IPs
        self.ip_listbox = tk.Listbox(self.ip_frame, height=4, width=30, font=('Arial', 10))
        self.ip_listbox.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        self.ip_listbox.insert(tk.END, "192.168.10.1")
        self.ip_listbox.insert(tk.END, "192.168.10.2")

        # Controls for Add/Remove
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

        # 4. Live Log
        tk.Label(root, text="LIVE LOG", font=('Arial', 9, 'bold')).pack(pady=(10,0))
        self.log_display = scrolledtext.ScrolledText(root, height=13, state='disabled', bg="black", fg="lime", font=('Consolas', 9))
        self.log_display.pack(pady=5, padx=15, fill="x")

        threading.Thread(target=self.bg_monitor, daemon=True).start()

    # --- UI HELPERS ---
    def add_ip(self):
        new_ip = self.new_ip_entry.get().strip()
        if new_ip:
            self.ip_listbox.insert(tk.END, new_ip)
            self.new_ip_entry.delete(0, tk.END)

    def remove_ip(self):
        selected = self.ip_listbox.curselection()
        if selected:
            self.ip_listbox.delete(selected[0])

    def ui_log(self, message):
        self.log_display.config(state='normal')
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.log_display.config(state='disabled')

    def get_target_ips(self):
        return list(self.ip_listbox.get(0, tk.END))

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
        """Checks if CommuniGATOR is open. If not, launches it instantly."""
        if self.commg_handles:
            # Verify the window didn't get closed by the user
            try:
                app = Application(backend="uia").connect(handle=self.commg_handles[0])
                app.window(handle=self.commg_handles[0]).exists()
                return True
            except:
                self.commg_handles = [] # Window was closed, clear it and proceed to relaunch

        self.root.after(0, self.status_label.config, {"text": "STATUS: LAUNCHING COMMG...", "fg": "orange"})
        self.root.after(0, self.ui_log, "SYSTEM: CommuniGATOR not found. Launching it now...")
        
        found_handles = self._find_commg_handles()
        
        if not found_handles:
            if not os.path.exists(COMMG_PATH):
                messagebox.showerror("File Not Found", f"Cannot find shortcut at:\n{COMMG_PATH}")
                self.root.after(0, self.status_label.config, {"text": "STATUS: LAUNCH FAILED", "fg": "red"})
                return False
                
            os.startfile(COMMG_PATH) 
            time.sleep(3.5) # Wait for it to draw on screen
            found_handles = self._find_commg_handles()
            
        if not found_handles:
            self.root.after(0, self.status_label.config, {"text": "STATUS: WINDOW TITLE ERROR", "fg": "red"})
            self.root.after(0, self.ui_log, "ERROR: Software launched but couldn't attach.")
            return False

        self.commg_handles = found_handles[:1]
        self.log_counts = [0]
        self.root.after(0, self.ui_log, "SYSTEM: Successfully hooked into CommuniGATOR.")
        return True

    # --- THE MAGIC WORKFLOW SEQUENCE ---
    def _execute_sequence_thread(self, payload):
        ips = self.get_target_ips()
        if not ips:
            messagebox.showwarning("No IPs", "Please add at least one IP address to the list!")
            return
            
        # Guarantee we are connected before starting
        if not self.ensure_connection():
            return
            
        handle = self.commg_handles[0]
        
        try:
            app = Application(backend="uia").connect(handle=handle)
            main_win = app.window(handle=handle)
            toolbar = main_win.child_window(auto_id="59392", control_type="ToolBar")
            input_field = main_win.child_window(auto_id="1004", control_type="Edit")
            
            for ip in ips:
                self.root.after(0, self.status_label.config, {"text": f"STATUS: PROCESSING {ip}...", "fg": "purple"})
                
                with self.type_lock:
                    main_win.set_focus()
                    
                    # 1. TCP/IP Switch
                    toolbar.button(0).click() 
                    time.sleep(1.0)
                    popup = app.top_window() 
                    popup.type_keys(ip + "{ENTER}", set_foreground=True)
                    self.root.after(0, self.ui_log, f"\n--- TARGET: {ip} ---")
                    time.sleep(1.5) 
                    
                    # 2. Click NO PROTO
                    toolbar.button(1).click()
                    self.root.after(0, self.ui_log, f"UI: Auto-Clicked NO PROTO")
                    time.sleep(1.0)
                    
                    # 3. Send HANDSHAKE
                    input_field.set_focus()
                    input_field.type_keys("^a{BACKSPACE}0121FF{ENTER}", set_foreground=True)
                    self.root.after(0, self.ui_log, f"UI: Auto-Sent HANDSHAKE")
                    time.sleep(1.5)
                    
                    # 4. Send the target command
                    input_field.set_focus()
                    input_field.type_keys("^a{BACKSPACE}" + payload + "{ENTER}", set_foreground=True)
                    self.root.after(0, self.ui_log, f"UI: Sent COMMAND ({payload})")
                        
                # 5. Wait for the radio to process
                self.root.after(0, self.status_label.config, {"text": f"STATUS: WAITING ON {ip}...", "fg": "orange"})
                time.sleep(5.0) 
                
            self.root.after(0, self.status_label.config, {"text": "STATUS: SEQUENCE COMPLETE ON ALL DEVICES", "fg": "green"})

        except Exception as e:
            self.root.after(0, self.ui_log, f"ERROR: {e}")
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