#!/usr/bin/env python3
"""
Text handling mixin for the TTS GUI using PySide6 (Qt).
"""

import os
from tts_gui.common import QFileDialog, QMessageBox, QColor


class TTSGuiTextMixin:
    """Mixin class providing text handling functionality."""

    def on_paste(self, text):
        """Handle paste event - remove duplicate lines and garbage tokens."""
        if not text:
            return text

        lines = text.split("\n")
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

        if duplicates_removed > 0:
            self.log_status(
                f"✓ Removed {duplicates_removed} duplicate line(s) from pasted text"
            )
        if garbage_removed > 0:
            self.log_status(
                f"✓ Removed {garbage_removed} UI label line(s) from pasted text"
            )

        return cleaned_text

    def on_text_change(self):
        """Handle text content change."""
        if not hasattr(self, "stats_label"):
            return
        text = self.text_widget.toPlainText().strip()

        # Update stats
        stats = self.text_processor.get_text_stats(text)
        stats_text = f"Characters: {stats['chars']} | Words: {stats['words']} | Lines: {stats['lines']} | Sentences: {stats['sentences']}"
        self.stats_label.setText(stats_text)

        # Check chunking needs
        if self.text_processor.needs_chunking(text):
            if self.selected_voice_config:
                model_type = self.selected_voice_config[1]["model_type"]
            else:
                model_type = "matcha"
            chunks = self.text_processor.split_text_into_chunks(text, model_type)
            chunk_info = (
                f"Will split into {len(chunks)} chunks for {model_type.upper()} model"
            )
            self.chunking_info_label.setText(chunk_info)
            self.chunking_info_label.setStyleSheet(
                f"color: {self.colors['accent_orange']};"
            )
        else:
            self.chunking_info_label.setText("Single chunk processing")
            self.chunking_info_label.setStyleSheet(
                f"color: {self.colors['accent_green']};"
            )

        # Validate text
        is_valid, error_msg = self.text_processor.validate_text(text)
        if is_valid:
            self.validation_label.setText("✓ Ready")
            self.validation_label.setStyleSheet(
                f"color: {self.colors['accent_green']};"
            )
        else:
            self.validation_label.setText(f"⚠ {error_msg}")
            self.validation_label.setStyleSheet(
                f"color: {self.colors['accent_orange']};"
            )

        # Check SSML
        if text and self.ssml_auto_detect:
            if self.ssml_processor.is_ssml(text):
                self.ssml_info_label.setText(
                    "🎭 SSML markup detected - will be processed automatically"
                )
                self.ssml_info_label.setStyleSheet(
                    f"color: {self.colors['accent_green']};"
                )
            elif self.ssml_enabled:
                self.ssml_info_label.setText("SSML mode enabled but no markup detected")
                self.ssml_info_label.setStyleSheet(
                    f"color: {self.colors['accent_orange']};"
                )
            else:
                self.ssml_info_label.setText(
                    "SSML enables: <emphasis>, <break>, <prosody>, <say-as>, and more"
                )
                self.ssml_info_label.setStyleSheet(
                    f"color: {self.colors['accent_cyan']};"
                )

    def import_text(self):
        """Import text from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Import Text File",
            "",
            "Text files (*.txt);;All files (*.*)",
        )

        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                self.text_widget.setPlainText(content)
                self.on_text_change()
                self.log_status(f"📁 Text imported from: {os.path.basename(file_path)}")

            except Exception as e:
                self.log_status(f"✗ Error importing text: {str(e)}")
                QMessageBox.critical(
                    self.main_window,
                    "Import Error",
                    f"Failed to import text:\n{str(e)}",
                )

    def export_text(self):
        """Export text to file."""
        text = self.text_widget.toPlainText().strip()
        if not text:
            QMessageBox.warning(self.main_window, "Export Warning", "No text to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Export Text File",
            "",
            "Text files (*.txt);;All files (*.*)",
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)

                self.log_status(f"💾 Text exported to: {os.path.basename(file_path)}")

            except Exception as e:
                self.log_status(f"✗ Error exporting text: {str(e)}")
                QMessageBox.critical(
                    self.main_window,
                    "Export Error",
                    f"Failed to export text:\n{str(e)}",
                )

    def clear_text(self):
        """Clear all text."""
        reply = QMessageBox.question(
            self.main_window,
            "Clear Text",
            "Are you sure you want to clear all text?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.text_widget.clear()
            self.on_text_change()
            self.log_status("🧹 Text cleared")
