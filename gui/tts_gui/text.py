#!/usr/bin/env python3

import os
import tkinter as tk
from tkinter import filedialog, messagebox


class TTSGuiTextMixin:
    def on_paste(self, event):
        try:
            clipboard_text = self.root.clipboard_get()

            if not clipboard_text:
                return None

            lines = clipboard_text.split("\n")
            cleaned_lines = []
            prev_line = None
            duplicates_removed = 0
            garbage_removed = 0
            garbage_tokens = {"copy", "explain"}

            for line in lines:
                stripped = line.strip()
                if stripped and stripped.lower() in garbage_tokens:
                    garbage_removed += 1
                    continue
                if stripped and stripped == prev_line:
                    duplicates_removed += 1
                    continue
                cleaned_lines.append(line)
                prev_line = stripped

            cleaned_text = "\n".join(cleaned_lines)

            self.text_widget.insert(tk.INSERT, cleaned_text)

            if duplicates_removed > 0:
                self.log_status(
                    f"✓ Removed {duplicates_removed} duplicate line(s) from pasted text"
                )
            if garbage_removed > 0:
                self.log_status(
                    f"✓ Removed {garbage_removed} UI label line(s) from pasted text"
                )

            self.on_text_change(None)

            return "break"

        except tk.TclError:
            return None
        except Exception as e:
            self.log_status(f"⚠ Error processing paste: {str(e)}")
            return None

    def on_text_change(self, event):
        text = self.text_widget.get(1.0, tk.END).strip()

        stats = self.text_processor.get_text_stats(text)
        stats_text = f"Characters: {stats['chars']} | Words: {stats['words']} | Lines: {stats['lines']} | Sentences: {stats['sentences']}"
        self.stats_label.config(text=stats_text)

        if self.text_processor.needs_chunking(text):
            if self.selected_voice_config:
                model_type = self.selected_voice_config[1]["model_type"]
            else:
                model_type = "matcha"
            chunks = self.text_processor.split_text_into_chunks(text, model_type)
            chunk_info = (
                f"Will split into {len(chunks)} chunks for {model_type.upper()} model"
            )
            self.chunking_info_label.config(
                text=chunk_info, foreground=self.colors["accent_orange"]
            )
        else:
            self.chunking_info_label.config(
                text="Single chunk processing", foreground=self.colors["accent_green"]
            )

        is_valid, error_msg = self.text_processor.validate_text(text)
        if is_valid:
            self.validation_label.config(
                text="✓ Ready", foreground=self.colors["accent_green"]
            )
        else:
            self.validation_label.config(
                text=f"⚠ {error_msg}", foreground=self.colors["accent_orange"]
            )

        if text and self.ssml_auto_detect.get():
            if self.ssml_processor.is_ssml(text):
                self.ssml_info_label.config(
                    text="🎭 SSML markup detected - will be processed automatically",
                    foreground=self.colors["accent_green"],
                )
            elif self.ssml_enabled.get():
                self.ssml_info_label.config(
                    text="SSML mode enabled but no markup detected",
                    foreground=self.colors["accent_orange"],
                )
            else:
                self.ssml_info_label.config(
                    text="SSML enables: <emphasis>, <break>, <prosody>, <say-as>, and more",
                    foreground=self.colors["accent_cyan"],
                )

    def import_text(self):
        file_path = filedialog.askopenfilename(
            title="Import Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, content)
                self.on_text_change(None)
                self.log_status(f"📁 Text imported from: {os.path.basename(file_path)}")

            except Exception as e:
                self.log_status(f"✗ Error importing text: {str(e)}")
                messagebox.showerror(
                    "Import Error", f"Failed to import text:\n{str(e)}"
                )

    def export_text(self):
        text = self.text_widget.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Export Warning", "No text to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Text File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)

                self.log_status(f"💾 Text exported to: {os.path.basename(file_path)}")

            except Exception as e:
                self.log_status(f"✗ Error exporting text: {str(e)}")
                messagebox.showerror(
                    "Export Error", f"Failed to export text:\n{str(e)}"
                )

    def clear_text(self):
        if messagebox.askyesno(
            "Clear Text", "Are you sure you want to clear all text?"
        ):
            self.text_widget.delete(1.0, tk.END)
            self.on_text_change(None)
            self.log_status("🧹 Text cleared")
