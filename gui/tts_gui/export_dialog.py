#!/usr/bin/env python3
"""
Export options dialog for the TTS GUI using PySide6 (Qt).
"""

from tts_gui.common import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QRadioButton,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QSpinBox,
    QListWidget,
    QFrame,
    QWidget,
    QMessageBox,
    Qt,
    QButtonGroup,
)


class ExportOptionsDialog(QDialog):
    """Dialog for advanced audio export options."""

    def __init__(
        self, parent, audio_exporter, audio_data, sample_rate, colors, original_text=""
    ):
        super().__init__(parent)
        self.exporter = audio_exporter
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.colors = colors
        self.original_text = original_text
        self.result = None
        self.detected_chapters = []

        self.setWindowTitle("Advanced Export Options")
        self.resize(650, 850)
        self.setModal(True)

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        """Setup the dialog UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title_label = QLabel("🎵 Advanced Audio Export")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("dialogTitle")
        main_layout.addWidget(title_label)

        # === Format Selection Frame ===
        format_group = QGroupBox("📀 Output Format")
        format_layout = QVBoxLayout(format_group)

        self.format_button_group = QButtonGroup(self)
        self.format_buttons = {}
        available_formats = self.exporter.get_available_formats()

        for fmt in available_formats:
            fmt_config = self.exporter.FORMATS[fmt]
            rb = QRadioButton(fmt_config["name"])
            rb.setObjectName("formatRadio")
            self.format_buttons[fmt] = rb
            self.format_button_group.addButton(rb)
            rb.toggled.connect(self._on_format_change)
            format_layout.addWidget(rb)

            # Description
            desc_label = QLabel(f"    {fmt_config['description']}")
            desc_label.setObjectName("formatDesc")
            format_layout.addWidget(desc_label)

        # Select WAV by default
        if "wav" in self.format_buttons:
            self.format_buttons["wav"].setChecked(True)

        # Format availability note
        if "mp3" not in available_formats or "ogg" not in available_formats:
            note_label = QLabel("ℹ️ Install ffmpeg for MP3/OGG support")
            note_label.setObjectName("noteLabel")
            format_layout.addWidget(note_label)

        main_layout.addWidget(format_group)

        # === Quality Settings Frame ===
        quality_group = QGroupBox("⚙️ Quality Settings")
        quality_layout = QVBoxLayout(quality_group)

        # Sample Rate
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel("Sample Rate:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems([str(sr) for sr in self.exporter.SAMPLE_RATES])
        sample_rate_str = str(self.sample_rate)
        if self.sample_rate_combo.findText(sample_rate_str) >= 0:
            self.sample_rate_combo.setCurrentText(sample_rate_str)
        else:
            self.sample_rate_combo.setCurrentIndex(0)
        sr_layout.addWidget(self.sample_rate_combo)
        sr_layout.addWidget(QLabel("Hz"))
        sr_layout.addStretch()
        quality_layout.addLayout(sr_layout)

        # Bitrate (for lossy formats)
        self.bitrate_widget = QWidget()
        bitrate_layout = QHBoxLayout(self.bitrate_widget)
        bitrate_layout.setContentsMargins(0, 0, 0, 0)
        bitrate_layout.addWidget(QLabel("Bitrate:"))
        self.bitrate_combo = QComboBox()
        self.bitrate_combo.addItems(
            ["64", "96", "128", "160", "192", "224", "256", "320"]
        )
        self.bitrate_combo.setCurrentText("192")
        bitrate_layout.addWidget(self.bitrate_combo)
        bitrate_layout.addWidget(QLabel("kbps"))
        bitrate_layout.addStretch()
        quality_layout.addWidget(self.bitrate_widget)
        self.bitrate_widget.hide()  # Initially hidden for WAV

        # Normalize checkbox
        self.normalize_check = QCheckBox(
            "Normalize audio (maximize volume without clipping)"
        )
        quality_layout.addWidget(self.normalize_check)

        main_layout.addWidget(quality_group)

        # === Split Options Frame ===
        split_group = QGroupBox("✂️ Split Options")
        split_layout = QVBoxLayout(split_group)

        self.split_button_group = QButtonGroup(self)

        # No split
        self.split_none_radio = QRadioButton("Export as single file")
        self.split_none_radio.setChecked(True)
        self.split_button_group.addButton(self.split_none_radio)
        self.split_none_radio.toggled.connect(self._on_split_mode_change)
        split_layout.addWidget(self.split_none_radio)

        # Split by silence
        self.split_silence_radio = QRadioButton(
            "Split by silence (automatic track detection)"
        )
        self.split_button_group.addButton(self.split_silence_radio)
        self.split_silence_radio.toggled.connect(self._on_split_mode_change)
        split_layout.addWidget(self.split_silence_radio)

        # Split by chapters
        self.split_chapters_radio = QRadioButton(
            "Split by chapters/sections (from text markers)"
        )
        self.split_button_group.addButton(self.split_chapters_radio)
        self.split_chapters_radio.toggled.connect(self._on_split_mode_change)
        split_layout.addWidget(self.split_chapters_radio)

        # Silence detection settings widget
        self.silence_settings_widget = QWidget()
        silence_layout = QVBoxLayout(self.silence_settings_widget)
        silence_layout.setContentsMargins(20, 10, 0, 0)

        # Min silence length
        sl_layout = QHBoxLayout()
        sl_layout.addWidget(QLabel("Min silence length:"))
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setRange(100, 5000)
        self.min_silence_spin.setSingleStep(100)
        self.min_silence_spin.setValue(500)
        sl_layout.addWidget(self.min_silence_spin)
        sl_layout.addWidget(QLabel("ms"))
        sl_layout.addStretch()
        silence_layout.addLayout(sl_layout)

        # Silence threshold
        st_layout = QHBoxLayout()
        st_layout.addWidget(QLabel("Silence threshold:"))
        self.silence_thresh_spin = QSpinBox()
        self.silence_thresh_spin.setRange(-60, -20)
        self.silence_thresh_spin.setSingleStep(5)
        self.silence_thresh_spin.setValue(-40)
        st_layout.addWidget(self.silence_thresh_spin)
        st_layout.addWidget(QLabel("dB"))
        st_layout.addStretch()
        silence_layout.addLayout(st_layout)

        # Preview button
        preview_btn = QPushButton("🔍 Preview Splits")
        preview_btn.clicked.connect(self._preview_silence_splits)
        silence_layout.addWidget(preview_btn)

        split_layout.addWidget(self.silence_settings_widget)
        self.silence_settings_widget.hide()

        # Detected chapters widget
        self.chapters_widget = QWidget()
        chapters_layout = QVBoxLayout(self.chapters_widget)
        chapters_layout.setContentsMargins(20, 10, 0, 0)

        chapters_label = QLabel("Detected chapters from text:")
        chapters_layout.addWidget(chapters_label)

        self.chapters_listbox = QListWidget()
        self.chapters_listbox.setMaximumHeight(100)
        chapters_layout.addWidget(self.chapters_listbox)

        split_layout.addWidget(self.chapters_widget)
        self.chapters_widget.hide()

        # Detect chapters
        self._detect_chapters()

        main_layout.addWidget(split_group)

        # === Metadata Frame ===
        metadata_group = QGroupBox("🏷️ Metadata (Optional)")
        metadata_layout = QVBoxLayout(metadata_group)

        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.meta_title_edit = QLineEdit()
        title_layout.addWidget(self.meta_title_edit)
        metadata_layout.addLayout(title_layout)

        # Artist
        artist_layout = QHBoxLayout()
        artist_layout.addWidget(QLabel("Artist:"))
        self.meta_artist_edit = QLineEdit()
        self.meta_artist_edit.setText("Sherpa-ONNX TTS")
        artist_layout.addWidget(self.meta_artist_edit)
        metadata_layout.addLayout(artist_layout)

        # Album
        album_layout = QHBoxLayout()
        album_layout.addWidget(QLabel("Album:"))
        self.meta_album_edit = QLineEdit()
        album_layout.addWidget(self.meta_album_edit)
        metadata_layout.addLayout(album_layout)

        main_layout.addWidget(metadata_group)

        # === Info Display ===
        duration = len(self.audio_data) / self.sample_rate if self.sample_rate > 0 else 0
        info_text = f"Duration: {duration:.1f}s | Sample Rate: {self.sample_rate} Hz | Samples: {len(self.audio_data):,}"
        self.info_label = QLabel(info_text)
        self.info_label.setObjectName("infoLabel")
        self.info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.info_label)

        # === Buttons ===
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        export_btn = QPushButton("💾 Export")
        export_btn.setObjectName("primaryButton")
        export_btn.clicked.connect(self._export)
        btn_layout.addWidget(export_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        main_layout.addLayout(btn_layout)

    def _apply_styles(self):
        """Apply stylesheet to the dialog"""
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {self.colors['bg_primary']};
            }}
            QLabel {{
                color: {self.colors['fg_primary']};
            }}
            QLabel#dialogTitle {{
                font-size: 14pt;
                font-weight: bold;
            }}
            QLabel#formatDesc {{
                color: {self.colors['fg_muted']};
                font-size: 9pt;
            }}
            QLabel#noteLabel {{
                color: {self.colors['accent_orange']};
                font-size: 9pt;
            }}
            QLabel#infoLabel {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['accent_cyan']};
                font-family: Consolas;
                padding: 8px;
                border-radius: 4px;
            }}
            QGroupBox {{
                background-color: {self.colors['bg_secondary']};
                border: 1px solid {self.colors['bg_tertiary']};
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                color: {self.colors['fg_primary']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QRadioButton, QCheckBox {{
                color: {self.colors['fg_primary']};
            }}
            QRadioButton::indicator, QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QComboBox {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['bg_tertiary']};
                border-radius: 4px;
                padding: 4px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                selection-background-color: {self.colors['selection']};
            }}
            QSpinBox {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['bg_tertiary']};
                border-radius: 4px;
                padding: 4px;
            }}
            QLineEdit {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['bg_tertiary']};
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget {{
                background-color: {self.colors['bg_primary']};
                color: {self.colors['fg_primary']};
                border: 1px solid {self.colors['bg_tertiary']};
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {self.colors['selection']};
            }}
            QPushButton {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['fg_primary']};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['accent_purple']};
            }}
            QPushButton#primaryButton {{
                background-color: {self.colors['accent_pink']};
                color: {self.colors['bg_primary']};
                font-weight: bold;
            }}
            QPushButton#primaryButton:hover {{
                background-color: {self.colors['accent_purple']};
            }}
        """
        )

    def _get_selected_format(self):
        """Get the currently selected format"""
        for fmt, btn in self.format_buttons.items():
            if btn.isChecked():
                return fmt
        return "wav"

    def _on_format_change(self):
        """Handle format selection change"""
        fmt = self._get_selected_format()
        fmt_config = self.exporter.FORMATS.get(fmt, {})

        if fmt_config.get("supports_bitrate", False):
            self.bitrate_widget.show()
            # Set default bitrate for format
            default_br = fmt_config.get("default_bitrate", 192)
            self.bitrate_combo.setCurrentText(str(default_br))
            # Update available bitrates
            if "bitrates" in fmt_config:
                self.bitrate_combo.clear()
                self.bitrate_combo.addItems([str(br) for br in fmt_config["bitrates"]])
        else:
            self.bitrate_widget.hide()

    def _on_split_mode_change(self):
        """Handle split mode change"""
        # Hide all split settings first
        self.silence_settings_widget.hide()
        self.chapters_widget.hide()

        if self.split_silence_radio.isChecked():
            self.silence_settings_widget.show()
        elif self.split_chapters_radio.isChecked():
            self.chapters_widget.show()

    def _detect_chapters(self):
        """Detect chapters from original text"""
        if not self.original_text:
            return

        chapters = self.exporter.detect_chapters_from_text(self.original_text)

        self.chapters_listbox.clear()
        self.detected_chapters = chapters

        if chapters:
            for i, ch in enumerate(chapters, 1):
                self.chapters_listbox.addItem(f"{i}. {ch['title']}")
        else:
            self.chapters_listbox.addItem("(No chapters detected)")
            self.detected_chapters = []

    def _preview_silence_splits(self):
        """Preview silence detection results"""
        try:
            min_silence = self.min_silence_spin.value()
            threshold = self.silence_thresh_spin.value()

            silence_regions = self.exporter.detect_silence(
                self.audio_data,
                self.sample_rate,
                min_silence_len=min_silence,
                silence_thresh=threshold,
            )

            if silence_regions:
                msg = f"Found {len(silence_regions)} silence region(s):\n\n"
                for i, (start, end) in enumerate(silence_regions[:10], 1):
                    duration = end - start
                    msg += (
                        f"  {i}. {start/1000:.1f}s - {end/1000:.1f}s ({duration}ms)\n"
                    )
                if len(silence_regions) > 10:
                    msg += f"\n  ... and {len(silence_regions) - 10} more"
                msg += f"\n\nThis would create {len(silence_regions) + 1} track(s)."
            else:
                msg = "No silence regions detected with current settings.\n\nTry adjusting:\n- Lower threshold (more sensitive)\n- Shorter minimum silence length"

            QMessageBox.information(self, "Silence Detection Preview", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preview failed: {str(e)}")

    def _export(self):
        """Perform export with selected options"""
        # Determine split mode
        if self.split_silence_radio.isChecked():
            split_mode = "silence"
        elif self.split_chapters_radio.isChecked():
            split_mode = "chapters"
        else:
            split_mode = "none"

        # Build options dict
        self.result = {
            "format": self._get_selected_format(),
            "target_sample_rate": int(self.sample_rate_combo.currentText()),
            "normalize": self.normalize_check.isChecked(),
            "split_mode": split_mode,
            "metadata": {
                "title": self.meta_title_edit.text(),
                "artist": self.meta_artist_edit.text(),
                "album": self.meta_album_edit.text(),
            },
        }

        # Add format-specific options
        fmt = self._get_selected_format()
        if self.exporter.FORMATS.get(fmt, {}).get("supports_bitrate", False):
            self.result["bitrate"] = int(self.bitrate_combo.currentText())

        # Add split options
        if self.result["split_mode"] == "silence":
            self.result["silence_settings"] = {
                "min_silence_len": self.min_silence_spin.value(),
                "silence_thresh": self.silence_thresh_spin.value(),
            }
        elif self.result["split_mode"] == "chapters":
            self.result["chapters"] = self.detected_chapters

        self.accept()

    def show(self):
        """Show dialog and wait for result"""
        self.exec()
        return self.result
