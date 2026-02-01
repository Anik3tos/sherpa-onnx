#!/usr/bin/env python3

from tts_gui.common import tk, ttk, scrolledtext, os


class TTSGuiShortcutsMixin:
    def setup_keyboard_shortcuts(self):
        """
        Setup keyboard shortcuts for power user productivity.

        Shortcuts:
            Space         - Play/Pause audio
            Ctrl+Enter    - Generate speech
            Ctrl+G        - Generate speech (alternative)
            1, 2, 3       - Speed presets (0.75x, 1.0x, 1.5x)
            4, 5          - Speed presets (2.0x, 0.5x)
            Alt+Up        - Previous voice model
            Alt+Down      - Next voice model
            Shift+Alt+Up  - Previous speaker
            Shift+Alt+Down - Next speaker
            Ctrl+A        - Select all text
            Ctrl+Shift+C  - Clear text
            Escape        - Cancel generation / Stop playback
            F1 / Ctrl+/   - Show keyboard shortcuts help
        """
        # NOTE: Some shortcuts only work when focus is not in text widget
        # to avoid conflicts with text editing

        # Global shortcuts (work anywhere in the window)
        self.root.bind("<F1>", self.show_keyboard_shortcuts)
        self.root.bind("<Control-slash>", self.show_keyboard_shortcuts)
        self.root.bind("<Control-question>", self.show_keyboard_shortcuts)

        # Play/Pause with Space (only when not in text widget)
        self.root.bind("<space>", self._on_space_key)

        # Generate with Ctrl+Enter or Ctrl+G
        self.root.bind("<Control-Return>", self._on_ctrl_enter)
        self.root.bind("<Control-g>", self._on_ctrl_enter)
        self.root.bind("<Control-G>", self._on_ctrl_enter)

        # Speed presets with number keys (only when not in text widget)
        self.root.bind("<Key-1>", lambda e: self._apply_speed_preset(e, 0.75))
        self.root.bind("<Key-2>", lambda e: self._apply_speed_preset(e, 1.0))
        self.root.bind("<Key-3>", lambda e: self._apply_speed_preset(e, 1.5))
        self.root.bind("<Key-4>", lambda e: self._apply_speed_preset(e, 2.0))
        self.root.bind("<Key-5>", lambda e: self._apply_speed_preset(e, 0.5))

        # Voice switching with Alt+Arrow keys
        self.root.bind("<Alt-Up>", self._previous_voice_model)
        self.root.bind("<Alt-Down>", self._next_voice_model)

        # Speaker switching with Shift+Alt+Arrow keys
        self.root.bind("<Shift-Alt-Up>", self._previous_speaker)
        self.root.bind("<Shift-Alt-Down>", self._next_speaker)

        # Cancel/Stop with Escape
        self.root.bind("<Escape>", self._on_escape)

        # Clear text with Ctrl+Shift+C (different from Ctrl+C copy)
        self.root.bind("<Control-Shift-c>", self._on_clear_text_shortcut)
        self.root.bind("<Control-Shift-C>", self._on_clear_text_shortcut)

        # Text widget specific bindings (Ctrl+A select all already built-in)
        # Add Ctrl+Enter in text widget to still generate
        self.text_widget.bind("<Control-Return>", self._on_ctrl_enter)

        self.log_status("⌨️ Keyboard shortcuts enabled (press F1 for help)")

    def _on_space_key(self, event):
        """Handle Space key for play/pause toggle"""
        # Check if focus is in a text entry widget
        focused = self.root.focus_get()
        if isinstance(
            focused, (tk.Text, ttk.Entry, tk.Entry, scrolledtext.ScrolledText)
        ):
            return  # Let the text widget handle it normally

        # Toggle play/pause
        if self.is_playing:
            self.stop_audio()
        elif self.current_audio_file and os.path.exists(self.current_audio_file):
            self.play_audio()

        return "break"  # Prevent default behavior

    def _on_ctrl_enter(self, event):
        """Handle Ctrl+Enter for speech generation"""
        # Only generate if button is enabled (not already generating)
        if str(self.generate_btn["state"]) != "disabled":
            self.generate_speech()
        return "break"

    def _apply_speed_preset(self, event, speed):
        """Apply a speed preset if not in text widget"""
        # Check if focus is in a text entry widget
        focused = self.root.focus_get()
        if isinstance(
            focused, (tk.Text, ttk.Entry, tk.Entry, scrolledtext.ScrolledText)
        ):
            return  # Let the text widget handle it normally

        # Apply the speed preset
        self.speed_var.set(speed)
        self.update_speed_label(speed)
        self.log_status(f"⚡ Speed preset: {speed}x")
        return "break"

    def _previous_voice_model(self, event):
        """Switch to previous voice model"""
        values = self.voice_model_combo["values"]
        if not values:
            return "break"

        current_idx = self.voice_model_combo.current()
        new_idx = (current_idx - 1) % len(values)
        self.voice_model_combo.current(new_idx)
        self.on_voice_model_changed(None)

        # Get the model name for logging
        model_name = values[new_idx] if new_idx < len(values) else "Unknown"
        self.log_status(f"🎤 Voice model: {model_name[:50]}...")
        return "break"

    def _next_voice_model(self, event):
        """Switch to next voice model"""
        values = self.voice_model_combo["values"]
        if not values:
            return "break"

        current_idx = self.voice_model_combo.current()
        new_idx = (current_idx + 1) % len(values)
        self.voice_model_combo.current(new_idx)
        self.on_voice_model_changed(None)

        # Get the model name for logging
        model_name = values[new_idx] if new_idx < len(values) else "Unknown"
        self.log_status(f"🎤 Voice model: {model_name[:50]}...")
        return "break"

    def _previous_speaker(self, event):
        """Switch to previous speaker"""
        values = self.speaker_combo["values"]
        if not values:
            return "break"

        current_idx = self.speaker_combo.current()
        new_idx = (current_idx - 1) % len(values)
        self.speaker_combo.current(new_idx)
        self.on_speaker_changed(None)

        # Get the speaker name for logging
        speaker_name = values[new_idx] if new_idx < len(values) else "Unknown"
        self.log_status(f"👤 Speaker: {speaker_name[:40]}...")
        return "break"

    def _next_speaker(self, event):
        """Switch to next speaker"""
        values = self.speaker_combo["values"]
        if not values:
            return "break"

        current_idx = self.speaker_combo.current()
        new_idx = (current_idx + 1) % len(values)
        self.speaker_combo.current(new_idx)
        self.on_speaker_changed(None)

        # Get the speaker name for logging
        speaker_name = values[new_idx] if new_idx < len(values) else "Unknown"
        self.log_status(f"👤 Speaker: {speaker_name[:40]}...")
        return "break"

    def _on_escape(self, event):
        """Handle Escape key - cancel generation or stop playback"""
        if self.generation_thread and self.generation_thread.is_alive():
            # Cancel ongoing generation
            self.cancel_generation()
        elif self.is_playing:
            # Stop audio playback
            self.stop_audio()
        return "break"

    def _on_clear_text_shortcut(self, event):
        """Handle Ctrl+Shift+C for clearing text"""
        self.clear_text()
        return "break"

    def show_keyboard_shortcuts(self, event=None):
        """Show keyboard shortcuts help dialog"""
        shortcuts_window = tk.Toplevel(self.root)
        shortcuts_window.title("Keyboard Shortcuts")
        shortcuts_window.geometry("550x600")
        shortcuts_window.configure(bg=self.colors["bg_primary"])

        # Make window modal
        shortcuts_window.transient(self.root)
        shortcuts_window.grab_set()

        # Title
        title_label = tk.Label(
            shortcuts_window,
            text="⌨️ Keyboard Shortcuts",
            font=("Segoe UI", 16, "bold"),
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_primary"],
        )
        title_label.pack(pady=(20, 5))

        # Subtitle
        subtitle_label = tk.Label(
            shortcuts_window,
            text="Power user productivity features",
            font=("Segoe UI", 10),
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_muted"],
        )
        subtitle_label.pack(pady=(0, 15))

        # Shortcuts content frame with scrollbar
        content_frame = ttk.Frame(shortcuts_window, style="Dark.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

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

        # Create shortcuts display
        for category, shortcuts in shortcuts_data:
            # Category header
            cat_label = tk.Label(
                content_frame,
                text=category,
                font=("Segoe UI", 11, "bold"),
                bg=self.colors["bg_secondary"],
                fg=self.colors["accent_cyan"],
                anchor="w",
            )
            cat_label.pack(fill=tk.X, pady=(10, 5))

            # Shortcuts in this category
            for key, description in shortcuts:
                shortcut_frame = ttk.Frame(content_frame, style="Dark.TFrame")
                shortcut_frame.pack(fill=tk.X, pady=2)

                # Key label (fixed width, highlighted)
                key_label = tk.Label(
                    shortcut_frame,
                    text=key,
                    font=("Consolas", 10, "bold"),
                    bg=self.colors["bg_tertiary"],
                    fg=self.colors["accent_pink"],
                    width=18,
                    anchor="w",
                    padx=8,
                    pady=2,
                )
                key_label.pack(side=tk.LEFT, padx=(10, 10))

                # Description label
                desc_label = tk.Label(
                    shortcut_frame,
                    text=description,
                    font=("Segoe UI", 10),
                    bg=self.colors["bg_secondary"],
                    fg=self.colors["fg_primary"],
                    anchor="w",
                )
                desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Note about text widget
        note_frame = ttk.Frame(shortcuts_window, style="Card.TFrame", padding=10)
        note_frame.pack(fill=tk.X, padx=20, pady=10)

        note_label = tk.Label(
            note_frame,
            text="💡 Note: Number keys (1-5) and Space only work when focus is outside the text editor.\n"
            "     Use Ctrl+Enter to generate while typing in the text editor.",
            font=("Segoe UI", 9),
            bg=self.colors["bg_tertiary"],
            fg=self.colors["fg_muted"],
            justify=tk.LEFT,
        )
        note_label.pack()

        # Close button
        close_btn = ttk.Button(
            shortcuts_window,
            text="Close",
            command=shortcuts_window.destroy,
            style="Dark.TButton",
        )
        close_btn.pack(pady=15)

        # Allow Escape to close the window
        shortcuts_window.bind("<Escape>", lambda e: shortcuts_window.destroy())
        shortcuts_window.bind("<F1>", lambda e: shortcuts_window.destroy())

        return "break"
