import tkinter as tk
from tkinter import messagebox, scrolledtext
from pywinauto.application import Application
import threading
import time

class CommG_Ultimate_Controller:
    def __init__(self, root):
        self.root = root
        self.root.title("CommuniGATOR Pro Monitor & Control")
        self.root.geometry("550x850")
        
        # State Tracking
        self.initialized = False
        self.last_cmd = ""
        self.last_log_count = 0
        self.running = True

        # --- UI LAYOUT ---
        # 1. Status Indicator
        self.status_label = tk.Label(root, text="STATUS: DISCONNECTED", fg="red", font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=10)

        # 2. Initialization Frame
        self.init_frame = tk.LabelFrame(root, text=" 1. Initialization ", font=('Arial', 10, 'bold'))
        self.init_frame.pack(pady=5, padx=10, fill="x")
        
        tk.Button(self.init_frame, text="1. NO PROTO", bg="#FFD700", width=20, command=self.no_proto).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(self.init_frame, text="2. HANDSHAKE", bg="#90EE90", width=20, command=self.handshake).grid(row=0, column=1, padx=5, pady=5)

        # 3. Modes & Actions (Includes Reboot & Bootmode)
        self.mode_frame = tk.LabelFrame(root, text=" 2. Modes & Actions ", font=('Arial', 10, 'bold'))
        self.mode_frame.pack(pady=5, padx=10, fill="x")
        
        tk.Button(self.mode_frame, text="TEST MODE (03000C)", width=25, command=lambda: self.send_cmd("03000C")).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(self.mode_frame, text="REBOOT (03000D)", width=25, bg="#FFCCCB", command=lambda: self.send_cmd("03000D")).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.mode_frame, text="BOOTMODE (030200)", width=25, command=lambda: self.send_cmd("030200")).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(self.mode_frame, text="MUX TO AP (03011001)", width=25, bg="#ADD8E6", command=self.mux_to_ap).grid(row=1, column=1, padx=5, pady=5)

        # 4. Hardware Utilities (KeyUp, DeKey, Eject)
        self.util_frame = tk.LabelFrame(root, text=" 3. Hardware Utilities ", font=('Arial', 10, 'bold'))
        self.util_frame.pack(pady=5, padx=10, fill="x")
        
        # Toolbar indices: 3=KeyUp, 4=DeKey, 5=Eject
        tk.Button(self.util_frame, text="KEYUP", width=15, command=lambda: self.click_toolbar(3)).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(self.util_frame, text="DEKEY", width=15, command=lambda: self.click_toolbar(4)).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.util_frame, text="EJECT", width=15, bg="#E0E0E0", command=lambda: self.click_toolbar(5)).grid(row=0, column=2, padx=5, pady=5)

        # 5. Custom Raw Command (Replaces SN Converter)
        self.custom_frame = tk.LabelFrame(root, text=" 4. Send Raw Command ", font=('Arial', 10, 'bold'))
        self.custom_frame.pack(pady=5, padx=10, fill="x")
        
        tk.Label(self.custom_frame, text="Enter Hex:").pack(side="left", padx=5)
        self.custom_entry = tk.Entry(self.custom_frame, width=35)
        self.custom_entry.insert(0, "03001101...") # Example placeholder
        self.custom_entry.pack(side="left", padx=5)
        tk.Button(self.custom_frame, text="SEND", bg="#D8BFD8", command=self.send_custom_raw).pack(side="left", padx=5)

        # 6. Live Log
        tk.Label(root, text="LIVE LOG PREVIEW", font=('Arial', 9, 'bold')).pack(pady=(10,0))
        self.log_display = scrolledtext.ScrolledText(root, height=12, width=60, state='disabled', bg="black", fg="lime", font=('Consolas', 9))
        self.log_display.pack(pady=5, padx=10)

        # Start Monitor
        threading.Thread(target=self.bg_monitor, daemon=True).start()

    # --- CORE CONNECTIVITY ---
    def connect_app(self):
        try:
            return Application(backend="uia").connect(title="CommuniGATOR").window(title="CommuniGATOR")
        except:
            return None

    def ui_log(self, message):
        """Safely updates the GUI log from background threads."""
        self.log_display.config(state='normal')
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.log_display.config(state='disabled')

    # --- BUTTON HELPERS ---
    def send_cmd(self, cmd):
        self.last_cmd = cmd
        dlg = self.connect_app()
        if dlg:
            try:
                # auto_id 1004 is the Input Box
                input_field = dlg.child_window(auto_id="1004", control_type="Edit")
                input_field.set_focus()
                input_field.type_keys(cmd + "{ENTER}")
            except Exception as e:
                self.ui_log(f"Error sending {cmd}: {e}")

    def click_toolbar(self, index):
        """Clicks toolbar button by index. 1=NoProto, 3=KeyUp, 5=Eject, etc."""
        dlg = self.connect_app()
        if dlg:
            try:
                # auto_id 59392 is the Toolbar
                toolbar = dlg.child_window(auto_id="59392", control_type="ToolBar")
                toolbar.button(index).click()
                labels = ["TCP/IP", "NO PROTO", "TEST", "KEYUP", "DEKEY", "EJECT"]
                if index < len(labels):
                    self.ui_log(f"UI: Manual Click - {labels[index]}")
            except Exception as e:
                self.ui_log(f"Error clicking toolbar: {e}")

    # --- COMMAND FUNCTIONS ---
    def no_proto(self):
        self.click_toolbar(1) # Index 1 is NO PROTO
        self.initialized = True
        self.status_label.config(text="STATUS: PROTOCOL BYPASSED", fg="orange")

    def handshake(self):
        if not self.initialized:
            messagebox.showwarning("Warning", "Click 'NO PROTO' first to avoid Error 016!")
            return
        self.send_cmd("0121FF") # Handshake Hex

    def mux_to_ap(self):
        if messagebox.askyesno("Mux Warning", "Switching to AP will drop the connection. Proceed?"):
            self.send_cmd("03011001") # Mux Hex
            self.status_label.config(text="STATUS: MUXING...", fg="blue")

    def send_custom_raw(self):
        """Sends whatever is in the Custom Entry box directly."""
        cmd = self.custom_entry.get().strip()
        if not cmd:
            messagebox.showwarning("Error", "Please enter a hex command.")
            return
        self.send_cmd(cmd)
        self.ui_log(f"UI: Sent Custom Raw Command: {cmd}")

    # --- BACKGROUND LOGIC ---
    def bg_monitor(self):
        while self.running:
            dlg = self.connect_app()
            if dlg:
                try:
                    # auto_id 1058 is the ListBox (Log)
                    list_box = dlg.child_window(auto_id="1058", control_type="List")
                    items = list_box.items()
                    current_count = len(items)

                    if current_count > self.last_log_count:
                        new_items = items[self.last_log_count:]
                        for item in new_items:
                            text = item.window_text().strip()
                            if text:
                                self.root.after(0, self.ui_log, f"> {text}")
                                self.root.after(0, self.process_logic, text)
                        self.last_log_count = current_count
                except:
                    pass
            time.sleep(0.5)

    def process_logic(self, text):
        """Analyzes logs to trigger updates or recovery."""
        # Check for Mux Success (Error 138/General Error after Mux command)
        if "138-" in text or "GENERAL ERROR" in text:
            if self.last_cmd == "03011001":
                self.status_label.config(text="MUX SUCCESS! RECOVERING IN 15s...", fg="purple")
                threading.Thread(target=self.auto_recovery_sequence, daemon=True).start()
            else:
                self.status_label.config(text="STATUS: ERROR 138 (CONNECTION LOST)", fg="red")

        elif "000-" in text:
            self.status_label.config(text="STATUS: COMMAND SUCCESS", fg="green")
            
        elif "NO PROTOCOL INITIALIZED" in text:
            self.status_label.config(text="ERROR: CLICK 'NO PROTO' FIRST", fg="orange")

    def auto_recovery_sequence(self):
        """Automates the Eject -> No Proto -> TCP/IP sequence."""
        self.last_cmd = "RECOVERING" # Stop duplicate triggers
        
        # 1. Wait for AP Boot
        time.sleep(15)
        self.ui_log("UI: Auto-Recovery - Step 1: EJECT")
        
        # 2. Click EJECT (Index 5)
        self.click_toolbar(5)
        time.sleep(2)
        
        # 3. Click NO PROTO (Index 1) - CRITICAL FIX
        self.ui_log("UI: Auto-Recovery - Step 2: NO PROTO")
        self.click_toolbar(1)
        self.initialized = True
        time.sleep(1)
        
        # 4. Click TCP/IP (Index 0)
        self.ui_log("UI: Auto-Recovery - Step 3: TCP/IP")
        self.click_toolbar(0)
        
        self.status_label.config(text="RECOVERY COMPLETE. READY.", fg="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = CommG_Ultimate_Controller(root)
    root.mainloop()