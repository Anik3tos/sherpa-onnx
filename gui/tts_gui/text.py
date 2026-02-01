#!/usr/bin/env python3

import os
import tkinter as tk
from tkinter import filedialog, messagebox


class TTSGuiTextMixin:
    def on_paste(self, event):
        """Handle paste operation to remove duplicate consecutive lines"""
        try:
            # Get clipboard content
            clipboard_text = self.root.clipboard_get()

            if not clipboard_text:
                return None  # Allow default behavior for empty clipboard

            # Remove duplicate consecutive lines
            lines = clipboard_text.split("\n")
            cleaned_lines = []
            prev_line = None
            duplicates_removed = 0

            for line in lines:
                stripped = line.strip()
                # Only skip if the stripped line is non-empty and matches the previous
                if stripped and stripped == prev_line:
                    duplicates_removed += 1
                    continue  # Skip duplicate
                cleaned_lines.append(line)
                prev_line = stripped

            cleaned_text = "\n".join(cleaned_lines)

            # Insert cleaned text at cursor position
            self.text_widget.insert(tk.INSERT, cleaned_text)

            # Log if duplicates were removed
            if duplicates_removed > 0:
                self.log_status(
                    f"✓ Removed {duplicates_removed} duplicate line(s) from pasted text"
                )

            # Update text stats
            self.on_text_change(None)

            # Return "break" to prevent default paste behavior
            return "break"

        except tk.TclError:
            # Clipboard is empty or inaccessible, allow default behavior
            return None
        except Exception as e:
            # Log any other errors but allow default paste
            self.log_status(f"⚠ Error processing paste: {str(e)}")
            return None

    def on_text_change(self, event):
        """Handle text changes for real-time validation and stats"""
        text = self.text_widget.get(1.0, tk.END).strip()

        # Update text statistics
        stats = self.text_processor.get_text_stats(text)
        stats_text = f"Characters: {stats['chars']} | Words: {stats['words']} | Lines: {stats['lines']} | Sentences: {stats['sentences']}"
        self.stats_label.config(text=stats_text)

        # Update chunking information
        if self.text_processor.needs_chunking(text):
            # Use current model type for chunking preview
            if self.selected_voice_config:
                model_type = self.selected_voice_config[1]["model_type"]
            else:
                model_type = "matcha"  # Default
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

        # Update validation status
        is_valid, error_msg = self.text_processor.validate_text(text)
        if is_valid:
            self.validation_label.config(
                text="✓ Ready", foreground=self.colors["accent_green"]
            )
        else:
            self.validation_label.config(
                text=f"⚠ {error_msg}", foreground=self.colors["accent_orange"]
            )

        # Update SSML detection status
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
        """Import text from file"""
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
        """Export text to file"""
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
        """Clear text widget"""
        if messagebox.askyesno(
            "Clear Text", "Are you sure you want to clear all text?"
        ):
            self.text_widget.delete(1.0, tk.END)
            self.on_text_change(None)
            self.log_status("🧹 Text cleared")
