#!/usr/bin/env python3
"""
SSML handling mixin for the TTS GUI using PySide6 (Qt).
"""

from tts_gui.common import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QWidget,
    QMessageBox,
    Qt,
)


class TTSGuiSSMLMixin:
    """Mixin class providing SSML functionality."""

    def on_ssml_toggle(self):
        """Handle SSML enable/disable toggle"""
        if self.ssml_enabled:
            self.log_status(
                "🎭 SSML mode enabled - Use SSML markup for professional speech control"
            )
            self.ssml_info_label.setText(
                "SSML active! Use tags like <break>, <emphasis>, <prosody>, <say-as>"
            )
            self.ssml_info_label.setStyleSheet(f"color: {self.colors['accent_green']};")
        else:
            self.log_status("🎭 SSML mode disabled - Plain text mode")
            self.ssml_info_label.setText(
                "SSML enables: <emphasis>, <break>, <prosody>, <say-as>, and more"
            )
            self.ssml_info_label.setStyleSheet(f"color: {self.colors['accent_cyan']};")
        if hasattr(self, "schedule_config_save"):
            self.schedule_config_save()

    def show_ssml_templates(self):
        """Show SSML templates in a dialog"""
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("SSML Templates")
        dialog.resize(700, 600)
        dialog.setStyleSheet(f"background-color: {self.colors['bg_primary']};")

        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 15, 20, 15)

        # Title
        title_label = QLabel("📋 SSML Templates")
        title_label.setStyleSheet(
            f"""
            font-size: 14pt;
            font-weight: bold;
            color: {self.colors['fg_primary']};
        """
        )
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel("Click a template to insert it into the text editor")
        desc_label.setStyleSheet(f"color: {self.colors['fg_muted']};")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Template buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)

        template_preview = QTextEdit()
        template_preview.setStyleSheet(
            f"""
            background-color: {self.colors['bg_secondary']};
            color: {self.colors['fg_primary']};
            font-family: Consolas;
            font-size: 10pt;
        """
        )

        def show_template(template_id):
            template_preview.setPlainText(
                self.ssml_processor.get_ssml_template(template_id)
            )

        template_buttons = [
            ("Basic", "basic"),
            ("Emphasis", "emphasis"),
            ("Prosody (Rate/Pitch)", "prosody"),
            ("Say-As (Pronunciation)", "say_as"),
            ("Full Example", "full_example"),
        ]

        for name, template_id in template_buttons:
            btn = QPushButton(name)
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.colors['bg_tertiary']};
                    color: {self.colors['fg_primary']};
                    border: none;
                    padding: 8px 12px;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background-color: {self.colors['accent_purple']};
                }}
            """
            )
            btn.clicked.connect(lambda checked, tid=template_id: show_template(tid))
            buttons_layout.addWidget(btn)

        layout.addLayout(buttons_layout)

        # Template preview label
        preview_label = QLabel("Template Preview:")
        preview_label.setStyleSheet(
            f"""
            font-weight: bold;
            color: {self.colors['fg_primary']};
        """
        )
        layout.addWidget(preview_label)

        layout.addWidget(template_preview, 1)

        # Show basic template by default
        template_preview.setPlainText(self.ssml_processor.get_ssml_template("basic"))

        # Insert button
        def insert_template():
            template_text = template_preview.toPlainText().strip()
            if template_text:
                self.text_widget.setPlainText(template_text)
                self.ssml_enabled = True
                self.on_ssml_toggle()
                self.on_text_change()
                self.log_status("📋 Inserted SSML template from preview")
            dialog.accept()

        insert_btn = QPushButton("📥 Insert Selected Template")
        insert_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['accent_pink']};
                color: {self.colors['bg_primary']};
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_purple']};
            }}
        """
        )
        insert_btn.clicked.connect(insert_template)
        layout.addWidget(insert_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_purple']};
            }}
        """
        )
        close_btn.clicked.connect(dialog.reject)
        layout.addWidget(close_btn)

        dialog.exec()

    def _insert_ssml_template(self, template_id, window=None):
        """Insert an SSML template into the text editor"""
        template = self.ssml_processor.get_ssml_template(template_id)

        # Clear and insert
        self.text_widget.setPlainText(template)

        # Enable SSML mode
        self.ssml_enabled = True
        self.on_ssml_toggle()

        # Update stats
        self.on_text_change()

        self.log_status(f"📋 Inserted SSML template: {template_id}")

        if window:
            window.accept()

    def show_ssml_reference(self):
        """Show SSML reference documentation in a dialog"""
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("SSML Reference Guide")
        dialog.resize(750, 700)
        dialog.setStyleSheet(f"background-color: {self.colors['bg_primary']};")

        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 15, 20, 15)

        # Title
        title_label = QLabel("❓ SSML Quick Reference Guide")
        title_label.setStyleSheet(
            f"""
            font-size: 14pt;
            font-weight: bold;
            color: {self.colors['fg_primary']};
        """
        )
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Speech Synthesis Markup Language (W3C Standard)")
        subtitle_label.setStyleSheet(f"color: {self.colors['fg_muted']};")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)

        # Reference text
        ref_text = QTextEdit()
        ref_text.setStyleSheet(
            f"""
            background-color: {self.colors['bg_secondary']};
            color: {self.colors['fg_primary']};
            font-family: Consolas;
            font-size: 10pt;
        """
        )
        ref_text.setPlainText(self.ssml_processor.get_ssml_reference())
        ref_text.setReadOnly(True)
        layout.addWidget(ref_text, 1)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_purple']};
            }}
        """
        )
        close_btn.clicked.connect(dialog.reject)
        layout.addWidget(close_btn)

        dialog.exec()

    def validate_ssml_input(self):
        """Validate the current text as SSML"""
        text = self.text_widget.toPlainText().strip()

        if not text:
            QMessageBox.information(
                self.main_window, "SSML Validation", "No text to validate."
            )
            return

        # Check if it looks like SSML
        if not self.ssml_processor.is_ssml(text):
            reply = QMessageBox.question(
                self.main_window,
                "SSML Validation",
                "The text doesn't appear to contain SSML markup.\n\n"
                "Would you like to wrap it in <speak> tags?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                wrapped_text = f"<speak>\n    {text}\n</speak>"
                self.text_widget.setPlainText(wrapped_text)
                self.ssml_enabled = True
                self.on_ssml_toggle()
                self.log_status("✓ Text wrapped in SSML <speak> tags")
            return

        # Parse and validate
        result = self.ssml_processor.parse_ssml(text)

        if result["errors"]:
            error_msg = "SSML Validation Errors:\n\n" + "\n".join(result["errors"])
            QMessageBox.critical(self.main_window, "SSML Validation Failed", error_msg)
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

            QMessageBox.information(self.main_window, "SSML Validation", info_msg)
            self.log_status("✓ SSML validation passed")

            # Enable SSML mode if not already enabled
            if not self.ssml_enabled:
                self.ssml_enabled = True
                self.on_ssml_toggle()

    def process_ssml_text(self, text):
        """
        Process SSML text and return plain text with prosody adjustments.

        Returns:
            dict with 'text' (processed plain text) and 'rate' (suggested speed multiplier)
        """
        # Check if SSML processing is enabled or auto-detect is on
        should_process = self.ssml_enabled or (
            self.ssml_auto_detect and self.ssml_processor.is_ssml(text)
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
