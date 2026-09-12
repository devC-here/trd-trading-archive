#!/usr/bin/env python3
"""
TRD Desktop Runner - Codespace GUI Test Application
A modern, dependency-free desktop application to test and verify
the desktop GUI pipeline inside GitHub Codespaces.
"""

import os
import sys
import platform
import datetime
import tkinter as tk
from tkinter import ttk, messagebox


class ModernDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TRD Desktop Runner - Codespace GUI")
        self.root.geometry("780x560")
        self.root.minsize(640, 480)

        # Color Palette (Dark Theme)
        self.bg_color = "#12141a"
        self.surface_color = "#1b1e27"
        self.card_bg = "#232733"
        self.border_color = "#32384a"
        self.primary_color = "#3b82f6"
        self.success_color = "#10b981"
        self.text_primary = "#f3f4f6"
        self.text_secondary = "#9ca3af"

        self.root.configure(bg=self.bg_color)
        self.click_count = 0

        self.setup_styles()
        self.create_widgets()
        self.log_event("Application initialized successfully.")
        self.log_event(f"Detected DISPLAY: {os.environ.get('DISPLAY', 'Not Set')}")
        self.log_event(f"Screen Resolution: {self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

    def setup_styles(self):
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        # Configure TTK styles
        self.style.configure(".", background=self.bg_color, foreground=self.text_primary)
        self.style.configure(
            "TFrame",
            background=self.bg_color
        )
        self.style.configure(
            "Card.TFrame",
            background=self.card_bg,
            relief="flat"
        )
        self.style.configure(
            "TLabel",
            background=self.card_bg,
            foreground=self.text_primary,
            font=("Segoe UI", 10)
        )
        self.style.configure(
            "Header.TLabel",
            background=self.surface_color,
            foreground=self.text_primary,
            font=("Segoe UI", 16, "bold")
        )
        self.style.configure(
            "SubHeader.TLabel",
            background=self.surface_color,
            foreground=self.success_color,
            font=("Segoe UI", 10, "bold")
        )
        self.style.configure(
            "Title.TLabel",
            background=self.card_bg,
            foreground="#93c5fd",
            font=("Segoe UI", 11, "bold")
        )

    def create_widgets(self):
        # 1. Header Bar
        header_frame = tk.Frame(self.root, bg=self.surface_color, height=70, padx=20, pady=12)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = ttk.Label(header_frame, text="🖥️  TRD Desktop Runner", style="Header.TLabel")
        title_label.pack(side=tk.LEFT)

        status_badge = tk.Label(
            header_frame,
            text="● CODESPACE GUI ONLINE",
            bg="#064e3b",
            fg="#34d399",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
            relief="flat"
        )
        status_badge.pack(side=tk.RIGHT)

        # 2. Main Container
        main_container = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=16)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top Row: Diagnostics Card & Controls Card
        top_row = tk.Frame(main_container, bg=self.bg_color)
        top_row.pack(fill=tk.X, pady=(0, 14))

        # Diagnostics Card
        diag_card = tk.Frame(top_row, bg=self.card_bg, bd=1, relief="solid", highlightbackground=self.border_color, padx=16, pady=14)
        diag_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        ttk.Label(diag_card, text="📊 System & Display Diagnostics", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 10))

        diag_items = [
            ("Display Port:", os.environ.get("DISPLAY", "None (Headless)")),
            ("Resolution:", f"{self.root.winfo_screenwidth()} x {self.root.winfo_screenheight()}"),
            ("Operating System:", f"{platform.system()} {platform.release()}"),
            ("Python Version:", f"{sys.version.split()[0]} ({platform.architecture()[0]})"),
            ("Node / Host:", platform.node()),
        ]

        for label, val in diag_items:
            row = tk.Frame(diag_card, bg=self.card_bg)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, bg=self.card_bg, fg=self.text_secondary, font=("Segoe UI", 9), width=16, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=self.card_bg, fg=self.text_primary, font=("Segoe UI", 9, "bold"), anchor="w").pack(side=tk.LEFT)

        # Controls Card
        ctrl_card = tk.Frame(top_row, bg=self.card_bg, bd=1, relief="solid", highlightbackground=self.border_color, padx=16, pady=14)
        ctrl_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        ttk.Label(ctrl_card, text="⚡ Interactive Test Controls", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 10))

        # Counter Button
        self.btn_counter = tk.Button(
            ctrl_card,
            text=f"Click Me! Count: {self.click_count}",
            bg=self.primary_color,
            fg="white",
            activebackground="#2563eb",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=6,
            command=self.on_counter_click
        )
        self.btn_counter.pack(fill=tk.X, pady=4)

        # Dialog Test Button
        btn_modal = tk.Button(
            ctrl_card,
            text="Test Modal Dialog Window",
            bg="#475569",
            fg="white",
            activebackground="#334155",
            activeforeground="white",
            font=("Segoe UI", 10),
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=6,
            command=self.on_open_modal
        )
        btn_modal.pack(fill=tk.X, pady=4)

        # Input & Send Row
        input_frame = tk.Frame(ctrl_card, bg=self.card_bg)
        input_frame.pack(fill=tk.X, pady=(6, 0))

        self.input_entry = tk.Entry(
            input_frame,
            bg="#12141a",
            fg=self.text_primary,
            insertbackground="white",
            relief="flat",
            font=("Segoe UI", 9),
            highlightthickness=1,
            highlightbackground=self.border_color
        )
        self.input_entry.insert(0, "Type a test message...")
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=4)
        self.input_entry.bind("<Return>", lambda e: self.on_send_input())

        btn_send = tk.Button(
            input_frame,
            text="Log",
            bg=self.success_color,
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=4,
            command=self.on_send_input
        )
        btn_send.pack(side=tk.RIGHT)

        # 3. Bottom Row: Activity / Event Log Card
        log_card = tk.Frame(main_container, bg=self.card_bg, bd=1, relief="solid", highlightbackground=self.border_color, padx=16, pady=12)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_header = tk.Frame(log_card, bg=self.card_bg)
        log_header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(log_header, text="📝 Live Event Stream", style="Title.TLabel").pack(side=tk.LEFT)

        btn_clear = tk.Button(
            log_header,
            text="Clear Log",
            bg="#334155",
            fg=self.text_secondary,
            font=("Segoe UI", 8),
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=2,
            command=self.on_clear_log
        )
        btn_clear.pack(side=tk.RIGHT)

        # Text area with scrollbar
        text_container = tk.Frame(log_card, bg="#0d1117")
        text_container.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            text_container,
            bg="#0d1117",
            fg="#e5e7eb",
            font=("Consolas", 9),
            wrap=tk.WORD,
            relief="flat",
            padx=10,
            pady=8,
            insertbackground="white"
        )
        scrollbar = tk.Scrollbar(text_container, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def log_event(self, message: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{now}] {message}\n")
        self.log_text.see(tk.END)

    def on_counter_click(self):
        self.click_count += 1
        self.btn_counter.config(text=f"Click Me! Count: {self.click_count}")
        self.log_event(f"User clicked action button (Total clicks: {self.click_count})")

    def on_open_modal(self):
        self.log_event("Triggered modal test dialog.")
        response = messagebox.askyesno(
            "Codespace GUI Verification",
            "Are you able to see and interact with this popup window inside your browser?"
        )
        if response:
            self.log_event("User answered YES to dialog prompt! GUI pipeline confirmed working.")
        else:
            self.log_event("User answered NO to dialog prompt.")

    def on_send_input(self):
        text = self.input_entry.get().strip()
        if text:
            self.log_event(f"User input received: '{text}'")
            self.input_entry.delete(0, tk.END)

    def on_clear_log(self):
        self.log_text.delete("1.0", tk.END)
        self.log_event("Log cleared.")


def main():
    root = tk.Tk()
    app = ModernDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
