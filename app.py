#!/usr/bin/env python3
"""
Codespaces Desktop Test Application
A modern, standalone desktop GUI to verify the display pipeline in GitHub Codespaces.
"""

import os
import sys
import platform
import datetime

# Check for tkinter
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("\n[!] Error: 'tkinter' is not installed in this Python environment.")
    print("    To fix, run:")
    print("       sudo apt update && sudo apt install -y python3-tk\n")
    sys.exit(1)


class DesktopTestApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Codespaces Desktop Test")
        self.root.geometry("720x520")
        self.root.minsize(600, 400)

        self.bg_color = "#12141a"
        self.card_bg = "#1f2430"
        self.border_color = "#2d3446"
        self.accent_color = "#3b82f6"
        self.success_color = "#10b981"
        self.text_primary = "#f3f4f6"
        self.text_secondary = "#9ca3af"

        self.root.configure(bg=self.bg_color)
        self.click_count = 0

        self.build_ui()
        self.log(f"Display active: {os.environ.get('DISPLAY', 'Not set')}")
        self.log(f"Resolution: {self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

    def build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#1a1f2c", padx=20, pady=12)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="🖥️  Codespaces Desktop Test",
            bg="#1a1f2c",
            fg=self.text_primary,
            font=("Segoe UI", 15, "bold")
        ).pack(side=tk.LEFT)

        tk.Label(
            header,
            text="● GUI ONLINE",
            bg="#064e3b",
            fg="#34d399",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4
        ).pack(side=tk.RIGHT)

        # Body
        body = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=16)
        body.pack(fill=tk.BOTH, expand=True)

        # Info Box
        info_card = tk.Frame(body, bg=self.card_bg, padx=16, pady=12, highlightthickness=1, highlightbackground=self.border_color)
        info_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(info_card, text="Diagnostics & Status", bg=self.card_bg, fg="#93c5fd", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))

        for label, val in [
            ("DISPLAY:", os.environ.get("DISPLAY", "None")),
            ("Resolution:", f"{self.root.winfo_screenwidth()} x {self.root.winfo_screenheight()}"),
            ("System:", f"{platform.system()} {platform.release()}"),
            ("Python:", sys.version.split()[0]),
        ]:
            row = tk.Frame(info_card, bg=self.card_bg)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, bg=self.card_bg, fg=self.text_secondary, font=("Segoe UI", 9), width=14, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=self.card_bg, fg=self.text_primary, font=("Segoe UI", 9, "bold"), anchor="w").pack(side=tk.LEFT)

        # Actions
        actions_card = tk.Frame(body, bg=self.card_bg, padx=16, pady=12, highlightthickness=1, highlightbackground=self.border_color)
        actions_card.pack(fill=tk.X, pady=(0, 12))

        self.btn_click = tk.Button(
            actions_card,
            text=f"Click Me to Test! (Clicks: {self.click_count})",
            bg=self.accent_color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=12,
            pady=6,
            command=self.on_click
        )
        self.btn_click.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            actions_card,
            text="Open Popup Dialog",
            bg="#374151",
            fg="white",
            font=("Segoe UI", 10),
            relief="flat",
            padx=12,
            pady=6,
            command=self.on_popup
        ).pack(side=tk.LEFT)

        # Log
        log_card = tk.Frame(body, bg=self.card_bg, padx=16, pady=12, highlightthickness=1, highlightbackground=self.border_color)
        log_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(log_card, text="Activity Log", bg=self.card_bg, fg="#93c5fd", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        self.log_box = tk.Text(log_card, bg="#0d1117", fg="#e5e7eb", font=("Consolas", 9), height=8, relief="flat", padx=8, pady=6)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def log(self, text: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{now}] {text}\n")
        self.log_box.see(tk.END)

    def on_click(self):
        self.click_count += 1
        self.btn_click.config(text=f"Click Me to Test! (Clicks: {self.click_count})")
        self.log(f"Button clicked! Total: {self.click_count}")

    def on_popup(self):
        self.log("Opening popup dialog...")
        ans = messagebox.askyesno("GUI Test", "Is this dialog window rendering properly on your screen?")
        self.log(f"Dialog response: {'Yes' if ans else 'No'}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DesktopTestApp(root)
    root.mainloop()
