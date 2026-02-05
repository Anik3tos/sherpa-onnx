#!/usr/bin/env python3
"""
UI setup mixin for the TTS GUI using PySide6 (Qt).
"""

from tts_gui.common import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    QGroupBox,
    QTextEdit,
    QProgressBar,
    QFrame,
    QSizePolicy,
    Qt,
    QFont,
    QMessageBox,
    QColor,
    QTextCursor,
    QTextCharFormat,
    QThread,
    QTimer,
)
import time


class TTSGuiUiMixin:
    """Mixin class providing UI setup functionality."""

    def setup_ui(self):
        """Setup the main user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.main_window.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # === Voice Selection Frame ===
        voice_group = QGroupBox("🎤 Enhanced Voice Selection")
        voice_layout = QVBoxLayout(voice_group)
        voice_layout.setSpacing(10)

        # Voice Model Selection
        model_row = QHBoxLayout()
        model_label = QLabel("🤖 Voice Model:")
        self.voice_model_combo = QComboBox()
        self.voice_model_combo.setMinimumWidth(400)
        self.voice_model_combo.currentIndexChanged.connect(self.on_voice_model_changed)
        model_row.addWidget(model_label)
        model_row.addWidget(self.voice_model_combo, 1)
        model_row.addStretch()
        voice_layout.addLayout(model_row)

        # Speaker Selection
        speaker_row = QHBoxLayout()
        speaker_label = QLabel("👤 Voice/Speaker:")
        self.speaker_combo = QComboBox()
        self.speaker_combo.setMinimumWidth(400)
        self.speaker_combo.currentIndexChanged.connect(self.on_speaker_changed)
        self.preview_btn = QPushButton("🎵 Preview Voice")
        self.preview_btn.clicked.connect(self.preview_voice)
        speaker_row.addWidget(speaker_label)
        speaker_row.addWidget(self.speaker_combo, 1)
        speaker_row.addWidget(self.preview_btn)
        voice_layout.addLayout(speaker_row)

        # Voice Info
        info_frame = QFrame()
        info_frame.setObjectName("cardFrame")
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_icon = QLabel("ℹ️ Voice Info:")
        self.voice_info_label = QLabel("Select a voice model to see details")
        self.voice_info_label.setObjectName("timeLabel")
        self.voice_info_label.setWordWrap(True)
        info_layout.addWidget(info_icon)
        info_layout.addWidget(self.voice_info_label, 1)
        voice_layout.addWidget(info_frame)

        # Speed Control
        speed_row = QHBoxLayout()
        speed_label = QLabel("⚡ Generation Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(50)
        self.speed_slider.setMaximum(300)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickInterval(25)
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        self.speed_value_label = QLabel("1.0x")
        speed_row.addWidget(speed_label)
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(self.speed_value_label)
        voice_layout.addLayout(speed_row)

        # GPU Acceleration Toggle
        provider_row = QHBoxLayout()
        self.gpu_checkbox = QCheckBox("🚀 Use GPU if available")
        self.gpu_checkbox.setChecked(getattr(self, "use_gpu", False))
        self.gpu_checkbox.stateChanged.connect(self.on_gpu_toggle)
        self.provider_label = QLabel("Provider: CPU")
        self.provider_label.setObjectName("timeLabel")
        provider_row.addWidget(self.gpu_checkbox)
        provider_row.addWidget(self.provider_label)
        provider_row.addStretch()
        voice_layout.addLayout(provider_row)

        main_layout.addWidget(voice_group)

        # === Text Input Frame ===
        text_group = QGroupBox("📝 Enhanced Text Input")
        text_layout = QVBoxLayout(text_group)
        text_layout.setSpacing(10)

        # Text Controls
        text_controls = QHBoxLayout()
        self.import_btn = QPushButton("📁 Import Text")
        self.import_btn.setObjectName("utilityButton")
        self.import_btn.clicked.connect(self.import_text)
        self.export_text_btn = QPushButton("💾 Export Text")
        self.export_text_btn.setObjectName("utilityButton")
        self.export_text_btn.clicked.connect(self.export_text)
        self.clear_btn = QPushButton("🧹 Clear")
        self.clear_btn.setObjectName("warningButton")
        self.clear_btn.clicked.connect(self.clear_text)
        text_controls.addWidget(self.import_btn)
        text_controls.addWidget(self.export_text_btn)
        text_controls.addWidget(self.clear_btn)
        text_controls.addStretch()
        text_layout.addLayout(text_controls)

        # Text Processing Options
        preprocess_group = QGroupBox("🔧 Text Processing Options")
        preprocess_layout = QGridLayout(preprocess_group)
        preprocess_layout.setSpacing(10)

        # Create checkboxes for text options
        row = 0
        col = 0
        options_list = [
            ("normalize_whitespace", "Normalize whitespace"),
            ("normalize_punctuation", "Normalize punctuation"),
            ("remove_urls", "Remove URLs"),
            ("remove_duplicates", "Remove duplicate lines"),
            (
                "remove_word_dashes",
                "Remove dashes between words (high-quality → high quality)",
            ),
            ("numbers_to_words", "Numbers to words (123→one hundred...)"),
            ("expand_abbreviations", "Expand abbreviations (Dr.→Doctor)"),
            ("handle_acronyms", "Pronounce acronyms (NASA→N A S A)"),
            ("add_pauses", "Add natural pauses"),
        ]

        for key, label in options_list:
            cb = QCheckBox(label)
            cb.setChecked(self.text_options[key])
            cb.stateChanged.connect(
                lambda state, k=key: self._update_text_option(k, state)
            )
            preprocess_layout.addWidget(cb, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1

        text_layout.addWidget(preprocess_group)

        # SSML Support Frame
        ssml_group = QGroupBox("🎭 SSML Support (Professional Speech Control)")
        ssml_layout = QGridLayout(ssml_group)
        ssml_layout.setSpacing(10)

        self.ssml_enabled_cb = QCheckBox("Enable SSML parsing")
        self.ssml_enabled_cb.setChecked(False)
        self.ssml_enabled_cb.stateChanged.connect(self.on_ssml_toggle)

        self.ssml_auto_detect_cb = QCheckBox("Auto-detect SSML markup")
        self.ssml_auto_detect_cb.setChecked(True)
        self.ssml_auto_detect_cb.stateChanged.connect(self._on_user_setting_changed)

        self.ssml_templates_btn = QPushButton("📋 SSML Templates")
        self.ssml_templates_btn.clicked.connect(self.show_ssml_templates)

        self.ssml_reference_btn = QPushButton("❓ SSML Reference")
        self.ssml_reference_btn.clicked.connect(self.show_ssml_reference)

        self.ssml_validate_btn = QPushButton("✓ Validate SSML")
        self.ssml_validate_btn.clicked.connect(self.validate_ssml_input)

        ssml_layout.addWidget(self.ssml_enabled_cb, 0, 0)
        ssml_layout.addWidget(self.ssml_auto_detect_cb, 0, 1)
        ssml_layout.addWidget(self.ssml_templates_btn, 0, 2)
        ssml_layout.addWidget(self.ssml_reference_btn, 0, 3)
        ssml_layout.addWidget(self.ssml_validate_btn, 0, 4)

        self.ssml_info_label = QLabel(
            "SSML enables: <emphasis>, <break>, <prosody>, <say-as>, and more"
        )
        self.ssml_info_label.setObjectName("timeLabel")
        ssml_layout.addWidget(self.ssml_info_label, 1, 0, 1, 5)

        text_layout.addWidget(ssml_group)

        # Chunking Info Frame
        chunking_frame = QFrame()
        chunking_frame.setObjectName("cardFrame")
        chunking_layout = QHBoxLayout(chunking_frame)
        chunking_layout.setContentsMargins(8, 8, 8, 8)
        chunking_icon = QLabel("📄 Long Text Handling:")
        self.chunking_info_label = QLabel(
            "Texts over 8,000 chars will be automatically split and stitched"
        )
        self.chunking_info_label.setObjectName("timeLabel")
        chunking_layout.addWidget(chunking_icon)
        chunking_layout.addWidget(self.chunking_info_label, 1)
        text_layout.addWidget(chunking_frame)

        # Text Widget
        self.text_widget = QTextEdit()
        self.text_widget.setMinimumHeight(150)
        self.text_widget.setPlaceholderText("Enter text to synthesize...")
        self.text_widget.textChanged.connect(self.on_text_change)

        # Set sample text
        sample_text = (
            "Welcome to the enhanced high-quality English text-to-speech system! "
        )
        self.text_widget.setPlainText(sample_text)
        text_layout.addWidget(self.text_widget)

        # Text Stats Frame
        stats_frame = QFrame()
        stats_frame.setObjectName("cardFrame")
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_icon = QLabel("📊 Text Stats:")
        self.stats_label = QLabel("Characters: 0 | Words: 0 | Lines: 0 | Sentences: 0")
        self.stats_label.setObjectName("timeLabel")
        self.validation_label = QLabel("✓ Ready")
        stats_layout.addWidget(stats_icon)
        stats_layout.addWidget(self.stats_label, 1)
        stats_layout.addWidget(self.validation_label)
        text_layout.addWidget(stats_frame)

        main_layout.addWidget(text_group)

        # === Controls Frame ===
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.generate_btn = QPushButton("🎵 Generate Speech")
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.clicked.connect(self.generate_speech)

        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        self.cancel_btn.hide()

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setObjectName("successButton")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)

        self.stop_btn = QPushButton("⏸ Pause")
        self.stop_btn.setObjectName("warningButton")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)

        self.save_btn = QPushButton("💾 Save As...")
        self.save_btn.clicked.connect(self.save_audio)
        self.save_btn.setEnabled(False)

        self.shortcuts_btn = QPushButton("⌨️ Shortcuts (F1)")
        self.shortcuts_btn.clicked.connect(self.show_keyboard_shortcuts)

        controls_layout.addWidget(self.generate_btn)
        controls_layout.addWidget(self.cancel_btn)
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.shortcuts_btn)

        main_layout.addLayout(controls_layout)

        # === Playback Controls Frame ===
        playback_group = QGroupBox("🎛️ Audio Playback Controls")
        playback_layout = QVBoxLayout(playback_group)
        playback_layout.setSpacing(10)

        # Time Display
        time_frame = QFrame()
        time_frame.setObjectName("cardFrame")
        time_layout = QHBoxLayout(time_frame)
        time_layout.setContentsMargins(8, 8, 8, 8)
        time_icon = QLabel("⏱️ Time:")
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("timeLabel")
        time_layout.addWidget(time_icon)
        time_layout.addWidget(self.time_label)
        time_layout.addStretch()
        playback_layout.addWidget(time_frame)

        # Seek Bar
        seek_row = QHBoxLayout()
        seek_label = QLabel("🎯 Position:")
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(1000)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.valueChanged.connect(self.on_seek)
        seek_row.addWidget(seek_label)
        seek_row.addWidget(self.seek_slider, 1)
        playback_layout.addLayout(seek_row)

        # Playback Speed and Volume Row
        speed_volume_row = QHBoxLayout()

        # Playback Speed
        pb_speed_label = QLabel("🚀 Playback Speed (pitch preserved):")
        self.playback_speed_slider = QSlider(Qt.Horizontal)
        self.playback_speed_slider.setMinimum(50)
        self.playback_speed_slider.setMaximum(200)
        self.playback_speed_slider.setValue(100)
        self.playback_speed_slider.valueChanged.connect(
            self.update_playback_speed_label
        )
        self.playback_speed_label = QLabel("1.0x")

        speed_volume_row.addWidget(pb_speed_label)
        speed_volume_row.addWidget(self.playback_speed_slider)
        speed_volume_row.addWidget(self.playback_speed_label)
        speed_volume_row.addSpacing(20)

        # Volume
        volume_label = QLabel("🔊 Volume:")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.update_volume_label)
        self.volume_label = QLabel("70%")

        speed_volume_row.addWidget(volume_label)
        speed_volume_row.addWidget(self.volume_slider)
        speed_volume_row.addWidget(self.volume_label)

        playback_layout.addLayout(speed_volume_row)

        # Follow-Along Frame
        follow_frame = QFrame()
        follow_frame.setObjectName("cardFrame")
        follow_layout = QHBoxLayout(follow_frame)
        follow_layout.setContentsMargins(8, 8, 8, 8)

        self.follow_along_cb = QCheckBox("📖 Follow Along (highlight current word)")
        self.follow_along_cb.setChecked(True)
        self.follow_along_cb.stateChanged.connect(self._on_user_setting_changed)

        current_label = QLabel("Current:")
        self.follow_along_word_label = QLabel("---")
        self.follow_along_word_label.setObjectName("timeLabel")
        self.follow_along_word_label.setStyleSheet(
            f"color: {self.colors['accent_pink']}; font-weight: bold;"
        )

        progress_label = QLabel("Progress:")
        self.follow_along_progress_label = QLabel("0 / 0 words")
        self.follow_along_progress_label.setObjectName("timeLabel")

        follow_layout.addWidget(self.follow_along_cb)
        follow_layout.addSpacing(20)
        follow_layout.addWidget(current_label)
        follow_layout.addWidget(self.follow_along_word_label)
        follow_layout.addSpacing(20)
        follow_layout.addWidget(progress_label)
        follow_layout.addWidget(self.follow_along_progress_label)
        follow_layout.addStretch()

        playback_layout.addWidget(follow_frame)

        main_layout.addWidget(playback_group)

        # === Status Frame ===
        status_group = QGroupBox("📊 Status & Performance")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(10)

        # Performance Info
        perf_frame = QFrame()
        perf_frame.setObjectName("cardFrame")
        perf_layout = QHBoxLayout(perf_frame)
        perf_layout.setContentsMargins(8, 8, 8, 8)
        perf_icon = QLabel("🚀 Performance:")
        self.perf_label = QLabel("Cache: 0 items | Avg RTF: N/A")
        self.perf_label.setObjectName("timeLabel")
        self.clear_cache_btn = QPushButton("🗑️ Clear Cache")
        self.clear_cache_btn.setObjectName("warningButton")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        perf_layout.addWidget(perf_icon)
        perf_layout.addWidget(self.perf_label, 1)
        perf_layout.addWidget(self.clear_cache_btn)
        status_layout.addWidget(perf_frame)

        # Status Text
        self.status_text = QTextEdit()
        self.status_text.setObjectName("statusText")
        self.status_text.setMaximumHeight(120)
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)

        main_layout.addWidget(status_group)

        # Progress Bar
        progress_frame = QFrame()
        progress_frame.setObjectName("cardFrame")
        progress_layout = QHBoxLayout(progress_frame)
        progress_layout.setContentsMargins(8, 8, 8, 8)
        progress_icon = QLabel("⏳ Generation:")
        self.progress_label = QLabel("Idle")
        self.progress_label.setObjectName("timeLabel")
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)  # Indeterminate
        self.progress.setTextVisible(False)
        progress_layout.addWidget(progress_icon)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress, 1)
        progress_frame.hide()
        self.progress_frame = progress_frame
        main_layout.addWidget(progress_frame)

        # Initial text stats update
        self.on_text_change()

    def _update_text_option(self, key, state):
        """Update text processing option."""
        self.text_options[key] = bool(state)
        self._on_user_setting_changed()

    def _on_user_setting_changed(self, *_args):
        """Schedule settings persistence for UI-driven changes."""
        if hasattr(self, "schedule_config_save"):
            self.schedule_config_save()

    @property
    def speed_var(self):
        """Get speed value (0.5 - 3.0)."""
        if hasattr(self, "speed_slider"):
            return self.speed_slider.value() / 100.0
        return getattr(self, "_speed_var", 1.0)

    @speed_var.setter
    def speed_var(self, value):
        """Set speed value (0.5 - 3.0)."""
        self._speed_var = float(value)
        if hasattr(self, "speed_slider"):
            self.speed_slider.setValue(int(self._speed_var * 100))

    @property
    def playback_speed_var(self):
        """Get playback speed value."""
        if hasattr(self, "playback_speed_slider"):
            return self.playback_speed_slider.value() / 100.0
        return getattr(self, "_playback_speed_var", 1.0)

    @playback_speed_var.setter
    def playback_speed_var(self, value):
        """Set playback speed value."""
        self._playback_speed_var = float(value)
        if hasattr(self, "playback_speed_slider"):
            self.playback_speed_slider.setValue(int(self._playback_speed_var * 100))

    @property
    def volume_var(self):
        """Get volume value."""
        if hasattr(self, "volume_slider"):
            return self.volume_slider.value()
        return getattr(self, "_volume_var", 70)

    @volume_var.setter
    def volume_var(self, value):
        """Set volume value."""
        self._volume_var = int(value)
        if hasattr(self, "volume_slider"):
            self.volume_slider.setValue(self._volume_var)

    @property
    def seek_var(self):
        """Get seek position value (0-100)."""
        if hasattr(self, "seek_slider"):
            return self.seek_slider.value() / 10.0
        return getattr(self, "_seek_var", 0.0)

    @seek_var.setter
    def seek_var(self, value):
        """Set seek position value (0-100)."""
        self._seek_var = float(value)
        if hasattr(self, "seek_slider"):
            self.seek_slider.setValue(int(self._seek_var * 10))

    @property
    def ssml_enabled(self):
        """Get SSML enabled state."""
        if hasattr(self, "ssml_enabled_cb"):
            return self.ssml_enabled_cb.isChecked()
        return getattr(self, "_ssml_enabled", False)

    @ssml_enabled.setter
    def ssml_enabled(self, value):
        """Set SSML enabled state."""
        self._ssml_enabled = bool(value)
        if hasattr(self, "ssml_enabled_cb"):
            self.ssml_enabled_cb.setChecked(self._ssml_enabled)

    @property
    def ssml_auto_detect(self):
        """Get SSML auto-detect state."""
        if hasattr(self, "ssml_auto_detect_cb"):
            return self.ssml_auto_detect_cb.isChecked()
        return getattr(self, "_ssml_auto_detect", True)

    @ssml_auto_detect.setter
    def ssml_auto_detect(self, value):
        """Set SSML auto-detect state."""
        self._ssml_auto_detect = bool(value)
        if hasattr(self, "ssml_auto_detect_cb"):
            self.ssml_auto_detect_cb.setChecked(self._ssml_auto_detect)

    @property
    def follow_along_enabled(self):
        """Get follow-along enabled state."""
        if hasattr(self, "follow_along_cb"):
            return self.follow_along_cb.isChecked()
        return getattr(self, "_follow_along_enabled", True)

    @follow_along_enabled.setter
    def follow_along_enabled(self, value):
        """Set follow-along enabled state."""
        self._follow_along_enabled = bool(value)
        if hasattr(self, "follow_along_cb"):
            self.follow_along_cb.setChecked(self._follow_along_enabled)

    def update_speed_label(self, value):
        """Update speed label when slider changes."""
        speed = value / 100.0
        self.speed_value_label.setText(f"{speed:.1f}x")
        self._on_user_setting_changed()

    def update_playback_speed_label(self, value):
        """Update playback speed label when slider changes."""
        speed = value / 100.0
        self.playback_speed_label.setText(f"{speed:.1f}x")
        self._on_user_setting_changed()

    def update_volume_label(self, value):
        """Update volume label when slider changes."""
        self.volume_label.setText(f"{value}%")
        if self.current_sound and self.is_playing:
            self.current_sound.set_volume(value / 100.0)
        self._on_user_setting_changed()

    def update_performance_display(self):
        """Update performance information display."""
        cache_size = len(self.audio_cache.cache)
        avg_rtf = self.performance_monitor.get_average_rtf()

        perf_text = f"Cache: {cache_size} items"
        if avg_rtf > 0:
            perf_text += f" | Avg RTF: {avg_rtf:.3f}"
        else:
            perf_text += " | Avg RTF: N/A"

        self.perf_label.setText(perf_text)

    def on_seek(self, value):
        """Handle seek slider changes."""
        if self.audio_duration > 0 and not self.is_playing:
            seek_position = (value / 1000.0) * self.audio_duration
            self.pause_position = seek_position
            self.update_time_display(seek_position)

    def format_time(self, seconds):
        """Format time in MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def update_time_display(self, current_time=None):
        """Update the time display."""
        if current_time is None:
            if self.is_playing:
                elapsed = time.time() - self.playback_start_time
                current_time = self.pause_position + elapsed * self.playback_speed_var
            else:
                current_time = self.pause_position

        current_time = min(current_time, self.audio_duration)
        current_str = self.format_time(current_time)
        total_str = self.format_time(self.audio_duration)
        self.time_label.setText(f"{current_str} / {total_str}")

        if self.audio_duration > 0:
            progress = (current_time / self.audio_duration) * 1000
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(int(progress))
            self.seek_slider.blockSignals(False)

    def log_status(self, message, level="info"):
        """Add message to status text widget with color coding."""
        if QThread.currentThread() != self.main_window.thread():
            QTimer.singleShot(0, lambda: self.log_status(message, level))
            return

        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        # Determine color based on message type
        if "✓" in message or "successfully" in message.lower():
            color = self.colors["accent_green"]
        elif "⚠" in message or "warning" in message.lower():
            color = self.colors["accent_orange"]
        elif "✗" in message or "error" in message.lower():
            color = self.colors["accent_red"]
        else:
            color = self.colors["fg_secondary"]

        # Append with color
        cursor = self.status_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Insert colored text
        char_format = cursor.charFormat()
        char_format.setForeground(QColor(color))
        cursor.setCharFormat(char_format)
        cursor.insertText(formatted_message + "\n")

        # Scroll to bottom
        self.status_text.setTextCursor(cursor)
        self.status_text.ensureCursorVisible()

    def clear_cache(self):
        """Clear audio cache."""
        reply = QMessageBox.question(
            self.main_window,
            "Clear Cache",
            "Are you sure you want to clear the audio cache?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.audio_cache.clear()
            self.update_performance_display()
            self.log_status("🗑️ Audio cache cleared")
