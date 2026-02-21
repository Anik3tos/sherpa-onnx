#!/usr/bin/env python3
"""
Keyboard shortcuts mixin for the TTS GUI using PySide6 (Qt).
"""

from tts_gui.common import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
    QWidget,
    Qt,
    QShortcut,
    QKeySequence,
)
import os


class TTSGuiShortcutsMixin:
    """Mixin class providing keyboard shortcuts functionality."""

    def setup_keyboard_shortcuts(self):
        """
        Setup keyboard shortcuts for power user productivity.

        Shortcuts:
            Space         - Play/Pause audio
            Ctrl+Enter    - Generate speech
            Ctrl+G        - Generate speech (alternative)
            Ctrl+Shift+T  - Transcribe selected audio
            1, 2, 3       - Speed presets (0.75x, 1.0x, 1.5x) when not in text
            4, 5          - Speed presets (2.0x, 0.5x) when not in text
            Alt+Up        - Previous voice model
            Alt+Down      - Next voice model
            Shift+Alt+Up  - Previous speaker
            Shift+Alt+Down - Next speaker
            Ctrl+A        - Select all text
            Ctrl+Shift+C  - Clear text
            Escape        - Cancel generation / Stop playback
            F1 / Ctrl+/   - Show keyboard shortcuts help
        """
        self._shortcuts = []

        # F1 - Show shortcuts
        shortcut_f1 = QShortcut(QKeySequence("F1"), self.main_window)
        shortcut_f1.activated.connect(self.show_keyboard_shortcuts)
        self._shortcuts.append(shortcut_f1)

        # Ctrl+/ - Show shortcuts (alternative)
        shortcut_help = QShortcut(QKeySequence("Ctrl+/"), self.main_window)
        shortcut_help.activated.connect(self.show_keyboard_shortcuts)
        self._shortcuts.append(shortcut_help)

        # Space - Play/Pause (when not in text widget)
        shortcut_space = QShortcut(QKeySequence("Space"), self.main_window)
        shortcut_space.activated.connect(self._on_space_key)
        self._shortcuts.append(shortcut_space)

        # Ctrl+Enter - Generate speech
        shortcut_generate = QShortcut(QKeySequence("Ctrl+Return"), self.main_window)
        shortcut_generate.activated.connect(self._on_ctrl_enter)
        self._shortcuts.append(shortcut_generate)

        # Ctrl+G - Generate speech (alternative)
        shortcut_generate_g = QShortcut(QKeySequence("Ctrl+G"), self.main_window)
        shortcut_generate_g.activated.connect(self._on_ctrl_enter)
        self._shortcuts.append(shortcut_generate_g)

        # Ctrl+Shift+T - Transcribe selected audio
        shortcut_transcribe = QShortcut(
            QKeySequence("Ctrl+Shift+T"), self.main_window
        )
        shortcut_transcribe.activated.connect(self._on_ctrl_shift_t)
        self._shortcuts.append(shortcut_transcribe)

        # Number keys for speed presets
        shortcut_1 = QShortcut(QKeySequence("1"), self.main_window)
        shortcut_1.activated.connect(lambda: self._apply_speed_preset(0.75))
        self._shortcuts.append(shortcut_1)

        shortcut_2 = QShortcut(QKeySequence("2"), self.main_window)
        shortcut_2.activated.connect(lambda: self._apply_speed_preset(1.0))
        self._shortcuts.append(shortcut_2)

        shortcut_3 = QShortcut(QKeySequence("3"), self.main_window)
        shortcut_3.activated.connect(lambda: self._apply_speed_preset(1.5))
        self._shortcuts.append(shortcut_3)

        shortcut_4 = QShortcut(QKeySequence("4"), self.main_window)
        shortcut_4.activated.connect(lambda: self._apply_speed_preset(2.0))
        self._shortcuts.append(shortcut_4)

        shortcut_5 = QShortcut(QKeySequence("5"), self.main_window)
        shortcut_5.activated.connect(lambda: self._apply_speed_preset(0.5))
        self._shortcuts.append(shortcut_5)

        # Alt+Arrow keys for voice switching
        shortcut_prev_voice = QShortcut(QKeySequence("Alt+Up"), self.main_window)
        shortcut_prev_voice.activated.connect(self._previous_voice_model)
        self._shortcuts.append(shortcut_prev_voice)

        shortcut_next_voice = QShortcut(QKeySequence("Alt+Down"), self.main_window)
        shortcut_next_voice.activated.connect(self._next_voice_model)
        self._shortcuts.append(shortcut_next_voice)

        # Shift+Alt+Arrow keys for speaker switching
        shortcut_prev_speaker = QShortcut(
            QKeySequence("Shift+Alt+Up"), self.main_window
        )
        shortcut_prev_speaker.activated.connect(self._previous_speaker)
        self._shortcuts.append(shortcut_prev_speaker)

        shortcut_next_speaker = QShortcut(
            QKeySequence("Shift+Alt+Down"), self.main_window
        )
        shortcut_next_speaker.activated.connect(self._next_speaker)
        self._shortcuts.append(shortcut_next_speaker)

        # Escape - Cancel/Stop
        shortcut_escape = QShortcut(QKeySequence("Escape"), self.main_window)
        shortcut_escape.activated.connect(self._on_escape)
        self._shortcuts.append(shortcut_escape)

        # Ctrl+Shift+C - Clear text
        shortcut_clear = QShortcut(QKeySequence("Ctrl+Shift+C"), self.main_window)
        shortcut_clear.activated.connect(self.clear_text)
        self._shortcuts.append(shortcut_clear)

        self.log_status("⌨️ Keyboard shortcuts enabled (press F1 for help)")

    def _on_space_key(self):
        """Handle Space key for play/pause toggle."""
        # Check if focus is in the text widget
        focused = self.main_window.focusWidget()
        if focused == self.text_widget:
            return  # Let the text widget handle it

        # Toggle play/pause
        if self.is_playing:
            self.stop_audio()
        elif self.current_audio_file and os.path.exists(self.current_audio_file):
            self.play_audio()

    def _on_ctrl_enter(self):
        """Handle Ctrl+Enter for speech generation."""
        if self.generate_btn.isEnabled():
            self.generate_speech()

    def _on_ctrl_shift_t(self):
        """Handle Ctrl+Shift+T for audio transcription."""
        if hasattr(self, "start_transcription"):
            self.start_transcription()

    def _apply_speed_preset(self, speed):
        """Apply a speed preset if not in text widget."""
        # Check if focus is in the text widget
        focused = self.main_window.focusWidget()
        if focused == self.text_widget:
            return  # Let the text widget handle it

        # Apply the speed preset
        self.speed_slider.setValue(int(speed * 100))
        self.log_status(f"⚡ Speed preset: {speed}x")

    def _previous_voice_model(self):
        """Switch to previous voice model."""
        count = self.voice_model_combo.count()
        if count == 0:
            return

        current_idx = self.voice_model_combo.currentIndex()
        new_idx = (current_idx - 1) % count
        self.voice_model_combo.setCurrentIndex(new_idx)

        model_name = self.voice_model_combo.currentText()
        self.log_status(f"🎤 Voice model: {model_name[:50]}...")

    def _next_voice_model(self):
        """Switch to next voice model."""
        count = self.voice_model_combo.count()
        if count == 0:
            return

        current_idx = self.voice_model_combo.currentIndex()
        new_idx = (current_idx + 1) % count
        self.voice_model_combo.setCurrentIndex(new_idx)

        model_name = self.voice_model_combo.currentText()
        self.log_status(f"🎤 Voice model: {model_name[:50]}...")

    def _previous_speaker(self):
        """Switch to previous speaker."""
        count = self.speaker_combo.count()
        if count == 0:
            return

        current_idx = self.speaker_combo.currentIndex()
        new_idx = (current_idx - 1) % count
        self.speaker_combo.setCurrentIndex(new_idx)

        speaker_name = self.speaker_combo.currentText()
        self.log_status(f"👤 Speaker: {speaker_name[:40]}...")

    def _next_speaker(self):
        """Switch to next speaker."""
        count = self.speaker_combo.count()
        if count == 0:
            return

        current_idx = self.speaker_combo.currentIndex()
        new_idx = (current_idx + 1) % count
        self.speaker_combo.setCurrentIndex(new_idx)

        speaker_name = self.speaker_combo.currentText()
        self.log_status(f"👤 Speaker: {speaker_name[:40]}...")

    def _on_escape(self):
        """Handle Escape key - cancel generation or stop playback."""
        if getattr(self, "transcription_in_progress", False):
            self.cancel_transcription()
        elif self.generation_thread and self.generation_thread.is_alive():
            self.cancel_generation()
        elif self.is_playing:
            self.stop_audio()

    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts help dialog."""
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setFixedSize(550, 600)
        dialog.setStyleSheet(f"background-color: {self.colors['bg_primary']};")

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Title
        title = QLabel("⌨️ Keyboard Shortcuts")
        title.setStyleSheet(
            f"""
            font-size: 16px;
            font-weight: bold;
            color: {self.colors['fg_primary']};
        """
        )
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Power user productivity features")
        subtitle.setStyleSheet(f"color: {self.colors['fg_muted']}; font-size: 10px;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Scroll area for shortcuts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(5)

        # Define shortcut categories
        shortcuts_data = [
            (
                "🎵 Playback Controls",
                [
                    ("Space", "Play / Pause audio"),
                    ("Escape", "Stop playback"),
                ],
            ),
            (
                "🎤 Speech Generation",
                [
                    ("Ctrl+Enter", "Generate speech"),
                    ("Ctrl+G", "Generate speech (alternative)"),
                    ("Ctrl+Shift+T", "Transcribe selected audio"),
                    ("Escape", "Cancel generation (during processing)"),
                ],
            ),
            (
                "⚡ Speed Presets",
                [
                    ("1", "Set speed to 0.75x (slower)"),
                    ("2", "Set speed to 1.0x (normal)"),
                    ("3", "Set speed to 1.5x (faster)"),
                    ("4", "Set speed to 2.0x (fast)"),
                    ("5", "Set speed to 0.5x (slowest)"),
                ],
            ),
            (
                "🔊 Voice Switching",
                [
                    ("Alt+↑", "Previous voice model"),
                    ("Alt+↓", "Next voice model"),
                    ("Shift+Alt+↑", "Previous speaker"),
                    ("Shift+Alt+↓", "Next speaker"),
                ],
            ),
            (
                "📝 Text Editing",
                [
                    ("Ctrl+A", "Select all text"),
                    ("Ctrl+Shift+C", "Clear all text"),
                    ("Ctrl+V", "Paste (auto-removes duplicate lines)"),
                ],
            ),
            (
                "❓ Help",
                [
                    ("F1", "Show this help dialog"),
                    ("Ctrl+/", "Show this help dialog (alternative)"),
                ],
            ),
        ]

        for category, shortcuts in shortcuts_data:
            # Category header
            cat_label = QLabel(category)
            cat_label.setStyleSheet(
                f"""
                font-size: 11px;
                font-weight: bold;
                color: {self.colors['accent_cyan']};
                background-color: {self.colors['bg_secondary']};
                padding: 5px;
                border-radius: 4px;
            """
            )
            scroll_layout.addWidget(cat_label)

            # Shortcuts in this category
            for key, description in shortcuts:
                row = QHBoxLayout()
                row.setSpacing(10)

                key_label = QLabel(key)
                key_label.setFixedWidth(120)
                key_label.setStyleSheet(
                    f"""
                    font-family: Consolas;
                    font-size: 10px;
                    font-weight: bold;
                    color: {self.colors['accent_pink']};
                    background-color: {self.colors['bg_tertiary']};
                    padding: 4px 8px;
                    border-radius: 4px;
                """
                )

                desc_label = QLabel(description)
                desc_label.setStyleSheet(
                    f"color: {self.colors['fg_primary']}; font-size: 10px;"
                )

                row.addWidget(key_label)
                row.addWidget(desc_label, 1)
                scroll_layout.addLayout(row)

            scroll_layout.addSpacing(5)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)

        # Note
        note_frame = QFrame()
        note_frame.setObjectName("cardFrame")
        note_frame.setStyleSheet(
            f"background-color: {self.colors['bg_tertiary']}; border-radius: 4px;"
        )
        note_layout = QVBoxLayout(note_frame)
        note_layout.setContentsMargins(10, 10, 10, 10)

        note_label = QLabel(
            "💡 Note: Number keys (1-5) and Space only work when focus is outside the text editor.\n"
            "     Use Ctrl+Enter to generate while typing in the text editor."
        )
        note_label.setStyleSheet(f"color: {self.colors['fg_muted']}; font-size: 9px;")
        note_label.setWordWrap(True)
        note_layout.addWidget(note_label)
        layout.addWidget(note_frame)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()
