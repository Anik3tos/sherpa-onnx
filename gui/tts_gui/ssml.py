#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox


class TTSGuiSSMLMixin:
    def on_ssml_toggle(self):
        """Handle SSML enable/disable toggle"""
        if self.ssml_enabled.get():
            self.log_status(
                "🎭 SSML mode enabled - Use SSML markup for professional speech control"
            )
            self.ssml_info_label.config(
                text="SSML active! Use tags like <break>, <emphasis>, <prosody>, <say-as>",
                foreground=self.colors["accent_green"],
            )
        else:
            self.log_status("🎭 SSML mode disabled - Plain text mode")
            self.ssml_info_label.config(
                text="SSML enables: <emphasis>, <break>, <prosody>, <say-as>, and more",
                foreground=self.colors["accent_cyan"],
            )

    def show_ssml_templates(self):
        """Show SSML templates in a dialog"""
        templates_window = tk.Toplevel(self.root)
        templates_window.title("SSML Templates")
        templates_window.geometry("700x600")
        templates_window.configure(bg=self.colors["bg_primary"])

        # Make window modal
        templates_window.transient(self.root)
        templates_window.grab_set()

        # Title
        title_label = tk.Label(
            templates_window,
            text="📋 SSML Templates",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_primary"],
        )
        title_label.pack(pady=(15, 10))

        # Description
        desc_label = tk.Label(
            templates_window,
            text="Click a template to insert it into the text editor",
            font=("Segoe UI", 10),
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_muted"],
        )
        desc_label.pack(pady=(0, 10))

        # Template buttons frame
        templates_frame = ttk.Frame(templates_window, style="Dark.TFrame")
        templates_frame.pack(fill=tk.X, padx=20, pady=10)

        template_buttons = [
            ("Basic", "basic"),
            ("Emphasis", "emphasis"),
            ("Prosody (Rate/Pitch)", "prosody"),
            ("Say-As (Pronunciation)", "say_as"),
            ("Full Example", "full_example"),
        ]

        for i, (name, template_id) in enumerate(template_buttons):
            btn = ttk.Button(
                templates_frame,
                text=name,
                command=lambda tid=template_id: self._insert_ssml_template(
                    tid, templates_window
                ),
                style="Dark.TButton",
            )
            btn.grid(row=0, column=i, padx=5, pady=5)

        # Template preview
        preview_label = tk.Label(
            templates_window,
            text="Template Preview:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_primary"],
        )
        preview_label.pack(pady=(15, 5), anchor=tk.W, padx=20)

        self.template_preview = scrolledtext.ScrolledText(
            templates_window,
            width=80,
            height=20,
            bg=self.colors["bg_secondary"],
            fg=self.colors["fg_primary"],
            font=("Consolas", 10),
            wrap=tk.WORD,
        )
        self.template_preview.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Show basic template by default
        self.template_preview.insert(
            tk.END, self.ssml_processor.get_ssml_template("basic")
        )

        # Template selection callback
        def show_template(template_id):
            self.template_preview.delete(1.0, tk.END)
            self.template_preview.insert(
                tk.END, self.ssml_processor.get_ssml_template(template_id)
            )

        # Update buttons to show preview
        for i, (name, template_id) in enumerate(template_buttons):
            btn = templates_frame.grid_slaves(row=0, column=i)[0]
            btn.configure(command=lambda tid=template_id: show_template(tid))

        # Insert button
        insert_btn = ttk.Button(
            templates_window,
            text="📥 Insert Selected Template",
            command=lambda: self._insert_template_from_preview(templates_window),
            style="Primary.TButton",
        )
        insert_btn.pack(pady=15)

        # Close button
        close_btn = ttk.Button(
            templates_window,
            text="Close",
            command=templates_window.destroy,
            style="Dark.TButton",
        )
        close_btn.pack(pady=(0, 15))

    def _insert_ssml_template(self, template_id, window=None):
        """Insert an SSML template into the text editor"""
        template = self.ssml_processor.get_ssml_template(template_id)

        # Clear and insert
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, template)

        # Enable SSML mode
        self.ssml_enabled.set(True)
        self.on_ssml_toggle()

        # Update stats
        self.on_text_change(None)

        self.log_status(f"📋 Inserted SSML template: {template_id}")

        if window:
            window.destroy()

    def _insert_template_from_preview(self, window):
        """Insert template from preview text widget"""
        if hasattr(self, "template_preview"):
            template_text = self.template_preview.get(1.0, tk.END).strip()
            if template_text:
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, template_text)
                self.ssml_enabled.set(True)
                self.on_ssml_toggle()
                self.on_text_change(None)
                self.log_status("📋 Inserted SSML template from preview")
        window.destroy()

    def show_ssml_reference(self):
        """Show SSML reference documentation in a dialog"""
        ref_window = tk.Toplevel(self.root)
        ref_window.title("SSML Reference Guide")
        ref_window.geometry("750x700")
        ref_window.configure(bg=self.colors["bg_primary"])

        # Make window modal
        ref_window.transient(self.root)
        ref_window.grab_set()

        # Title
        title_label = tk.Label(
            ref_window,
            text="❓ SSML Quick Reference Guide",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_primary"],
        )
        title_label.pack(pady=(15, 10))

        # Subtitle
        subtitle_label = tk.Label(
            ref_window,
            text="Speech Synthesis Markup Language (W3C Standard)",
            font=("Segoe UI", 10),
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_muted"],
        )
        subtitle_label.pack(pady=(0, 10))

        # Reference text
        ref_text = scrolledtext.ScrolledText(
            ref_window,
            width=85,
            height=35,
            bg=self.colors["bg_secondary"],
            fg=self.colors["fg_primary"],
            font=("Consolas", 10),
            wrap=tk.WORD,
        )
        ref_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Insert reference content
        ref_text.insert(tk.END, self.ssml_processor.get_ssml_reference())
        ref_text.config(state=tk.DISABLED)

        # Close button
        close_btn = ttk.Button(
            ref_window, text="Close", command=ref_window.destroy, style="Dark.TButton"
        )
        close_btn.pack(pady=15)

    def validate_ssml_input(self):
        """Validate the current text as SSML"""
        text = self.text_widget.get(1.0, tk.END).strip()

        if not text:
            messagebox.showinfo("SSML Validation", "No text to validate.")
            return

        # Check if it looks like SSML
        if not self.ssml_processor.is_ssml(text):
            response = messagebox.askyesno(
                "SSML Validation",
                "The text doesn't appear to contain SSML markup.\n\n"
                "Would you like to wrap it in <speak> tags?",
            )
            if response:
                wrapped_text = f"<speak>\n    {text}\n</speak>"
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, wrapped_text)
                self.ssml_enabled.set(True)
                self.on_ssml_toggle()
                self.log_status("✓ Text wrapped in SSML <speak> tags")
            return

        # Parse and validate
        result = self.ssml_processor.parse_ssml(text)

        if result["errors"]:
            error_msg = "SSML Validation Errors:\n\n" + "\n".join(result["errors"])
            messagebox.showerror("SSML Validation Failed", error_msg)
            self.log_status(f"✗ SSML validation failed: {result['errors'][0]}")
        else:
            info_msg = f"✓ SSML is valid!\n\n"
            info_msg += f"Extracted text length: {len(result['text'])} characters\n"
            info_msg += f"Number of segments: {len(result['segments'])}\n"

            if result["has_prosody_changes"]:
                info_msg += f"Average rate adjustment: {result['rate']:.2f}x\n"

            # Show segment info
            if result["segments"]:
                unique_features = set()
                for seg in result["segments"]:
                    if seg.get("is_break"):
                        unique_features.add("breaks")
                    if seg.get("emphasis"):
                        unique_features.add(f"emphasis: {seg['emphasis']}")
                    if seg.get("interpret_as"):
                        unique_features.add(f"say-as: {seg['interpret_as']}")
                    if seg.get("phoneme"):
                        unique_features.add("phoneme hints")
                    if seg.get("voice_hint"):
                        unique_features.add("voice hints")

                if unique_features:
                    info_msg += f"\nFeatures detected: {', '.join(unique_features)}"

            messagebox.showinfo("SSML Validation", info_msg)
            self.log_status("✓ SSML validation passed")

            # Enable SSML mode if not already enabled
            if not self.ssml_enabled.get():
                self.ssml_enabled.set(True)
                self.on_ssml_toggle()

    def process_ssml_text(self, text):
        """
        Process SSML text and return plain text with prosody adjustments.

        Returns:
            dict with 'text' (processed plain text) and 'rate' (suggested speed multiplier)
        """
        # Check if SSML processing is enabled or auto-detect is on
        should_process = self.ssml_enabled.get() or (
            self.ssml_auto_detect.get() and self.ssml_processor.is_ssml(text)
        )

        if not should_process:
            return {"text": text, "rate": 1.0, "was_processed": False}

        # Parse SSML
        result = self.ssml_processor.parse_ssml(text)

        if result["errors"]:
            self.log_status(f"⚠ SSML parsing warnings: {result['errors'][0]}")

        if result["text"] != text:
            self.log_status(
                f"🎭 SSML processed: {len(text)} chars → {len(result['text'])} chars"
            )
            if result["has_prosody_changes"]:
                self.log_status(f"   Rate adjustment: {result['rate']:.2f}x")

        result["was_processed"] = True
        return result
