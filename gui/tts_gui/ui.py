#!/usr/bin/env python3

from tts_gui.common import tk, ttk, scrolledtext, messagebox
import time


class TTSGuiUiMixin:
    def setup_ui(self):
        # Main frame with dark theme
        main_frame = ttk.Frame(self.root, style="Dark.TFrame", padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.configure(style="Dark.TFrame")

        # Enhanced Voice Selection Frame
        voice_frame = ttk.LabelFrame(
            main_frame,
            text="🎤 Enhanced Voice Selection",
            style="Dark.TLabelframe",
            padding="15",
        )
        voice_frame.grid(
            row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        # Voice Model Selection
        model_selection_frame = ttk.Frame(voice_frame, style="Dark.TFrame")
        model_selection_frame.grid(
            row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        ttk.Label(
            model_selection_frame, text="🤖 Voice Model:", style="Dark.TLabel"
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        self.voice_model_var = tk.StringVar()
        self.voice_model_combo = ttk.Combobox(
            model_selection_frame,
            textvariable=self.voice_model_var,
            state="readonly",
            width=50,
            style="Dark.TSpinbox",
        )
        self.voice_model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.voice_model_combo.bind("<<ComboboxSelected>>", self.on_voice_model_changed)

        # Voice/Speaker Selection
        speaker_selection_frame = ttk.Frame(voice_frame, style="Dark.TFrame")
        speaker_selection_frame.grid(
            row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        ttk.Label(
            speaker_selection_frame, text="👤 Voice/Speaker:", style="Dark.TLabel"
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        self.speaker_var = tk.StringVar()
        self.speaker_combo = ttk.Combobox(
            speaker_selection_frame,
            textvariable=self.speaker_var,
            state="readonly",
            width=50,
            style="Dark.TSpinbox",
        )
        self.speaker_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.speaker_combo.bind("<<ComboboxSelected>>", self.on_speaker_changed)

        # Voice Preview Button
        self.preview_btn = ttk.Button(
            speaker_selection_frame,
            text="🎵 Preview Voice",
            command=self.preview_voice,
            style="Dark.TButton",
        )
        self.preview_btn.grid(row=0, column=2, padx=(10, 0))

        # Voice Information Display
        info_frame = ttk.Frame(voice_frame, style="Card.TFrame", padding="8")
        info_frame.grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        ttk.Label(info_frame, text="ℹ️ Voice Info:", style="Dark.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        self.voice_info_label = ttk.Label(
            info_frame,
            text="Select a voice model to see details",
            style="Time.TLabel",
            wraplength=600,
        )
        self.voice_info_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Configure grid weights for voice frame
        model_selection_frame.columnconfigure(1, weight=1)
        speaker_selection_frame.columnconfigure(1, weight=1)
        info_frame.columnconfigure(1, weight=1)

        # Speed control
        speed_frame = ttk.Frame(voice_frame, style="Dark.TFrame")
        speed_frame.grid(
            row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(15, 0)
        )

        ttk.Label(speed_frame, text="⚡ Generation Speed:", style="Dark.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(
            speed_frame,
            from_=0.5,
            to=3.0,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
            style="Dark.Horizontal.TScale",
        )
        speed_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.speed_label = ttk.Label(speed_frame, text="1.0x", style="Dark.TLabel")
        self.speed_label.grid(row=0, column=2, padx=(5, 0))

        # Update speed label when scale changes
        speed_scale.configure(command=self.update_speed_label)

        # Configure speed frame grid weights
        speed_frame.columnconfigure(1, weight=1)

        # Text input frame with dark theme
        text_frame = ttk.LabelFrame(
            main_frame,
            text="📝 Enhanced Text Input",
            style="Dark.TLabelframe",
            padding="15",
        )
        text_frame.grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15)
        )

        # Text controls frame
        text_controls_frame = ttk.Frame(text_frame, style="Dark.TFrame")
        text_controls_frame.grid(
            row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        # Import/Export buttons with distinctive colors
        ttk.Button(
            text_controls_frame,
            text="📁 Import Text",
            command=self.import_text,
            style="Utility.TButton",
        ).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(
            text_controls_frame,
            text="💾 Export Text",
            command=self.export_text,
            style="Utility.TButton",
        ).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(
            text_controls_frame,
            text="🧹 Clear",
            command=self.clear_text,
            style="Warning.TButton",
        ).grid(row=0, column=2, padx=(0, 10))

        # Text preprocessing options
        preprocess_frame = ttk.LabelFrame(
            text_frame,
            text="🔧 Text Processing Options",
            style="Dark.TLabelframe",
            padding="10",
        )
        preprocess_frame.grid(
            row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        ttk.Checkbutton(
            preprocess_frame,
            text="Normalize whitespace",
            variable=self.text_options["normalize_whitespace"],
            style="Dark.TRadiobutton",
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(
            preprocess_frame,
            text="Normalize punctuation",
            variable=self.text_options["normalize_punctuation"],
            style="Dark.TRadiobutton",
        ).grid(row=0, column=1, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(
            preprocess_frame,
            text="Remove URLs",
            variable=self.text_options["remove_urls"],
            style="Dark.TRadiobutton",
        ).grid(row=0, column=2, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(
            preprocess_frame,
            text="Remove emails",
            variable=self.text_options["remove_emails"],
            style="Dark.TRadiobutton",
        ).grid(row=1, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(
            preprocess_frame,
            text="Remove duplicate lines",
            variable=self.text_options["remove_duplicates"],
            style="Dark.TRadiobutton",
        ).grid(row=1, column=1, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(
            preprocess_frame,
            text="Numbers to words (123→one hundred...)",
            variable=self.text_options["numbers_to_words"],
            style="Dark.TRadiobutton",
        ).grid(row=1, column=2, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(
            preprocess_frame,
            text="Expand abbreviations (Dr.→Doctor)",
            variable=self.text_options["expand_abbreviations"],
            style="Dark.TRadiobutton",
        ).grid(row=2, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(
            preprocess_frame,
            text="Pronounce acronyms (NASA→N A S A)",
            variable=self.text_options["handle_acronyms"],
            style="Dark.TRadiobutton",
        ).grid(row=2, column=1, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(
            preprocess_frame,
            text="Add natural pauses",
            variable=self.text_options["add_pauses"],
            style="Dark.TRadiobutton",
        ).grid(row=2, column=2, sticky=tk.W, padx=(0, 15))

        # SSML Support Frame - Professional-grade speech control
        ssml_frame = ttk.LabelFrame(
            text_frame,
            text="🎭 SSML Support (Professional Speech Control)",
            style="Dark.TLabelframe",
            padding="10",
        )
        ssml_frame.grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        # SSML controls row 1
        ttk.Checkbutton(
            ssml_frame,
            text="Enable SSML parsing",
            variable=self.ssml_enabled,
            command=self.on_ssml_toggle,
            style="Dark.TRadiobutton",
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(
            ssml_frame,
            text="Auto-detect SSML markup",
            variable=self.ssml_auto_detect,
            style="Dark.TRadiobutton",
        ).grid(row=0, column=1, sticky=tk.W, padx=(0, 15))

        # SSML control buttons
        ttk.Button(
            ssml_frame,
            text="📋 SSML Templates",
            command=self.show_ssml_templates,
            style="Dark.TButton",
        ).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(
            ssml_frame,
            text="❓ SSML Reference",
            command=self.show_ssml_reference,
            style="Dark.TButton",
        ).grid(row=0, column=3, padx=(0, 10))
        ttk.Button(
            ssml_frame,
            text="✓ Validate SSML",
            command=self.validate_ssml_input,
            style="Dark.TButton",
        ).grid(row=0, column=4, padx=(0, 10))

        # SSML info label
        self.ssml_info_label = ttk.Label(
            ssml_frame,
            text="SSML enables: <emphasis>, <break>, <prosody>, <say-as>, and more",
            style="Time.TLabel",
        )
        self.ssml_info_label.grid(
            row=1, column=0, columnspan=5, sticky=tk.W, pady=(5, 0)
        )

        # Chunking info frame
        chunking_frame = ttk.Frame(text_frame, style="Card.TFrame", padding="8")
        chunking_frame.grid(
            row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        ttk.Label(
            chunking_frame, text="📄 Long Text Handling:", style="Dark.TLabel"
        ).grid(row=0, column=0, sticky=tk.W)
        self.chunking_info_label = ttk.Label(
            chunking_frame,
            text="Texts over 8,000 chars will be automatically split and stitched",
            style="Time.TLabel",
        )
        self.chunking_info_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Create custom text widget with Dracula theme
        self.text_widget = scrolledtext.ScrolledText(
            text_frame,
            width=75,
            height=8,
            wrap=tk.WORD,
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_primary"],
            insertbackground=self.colors["accent_cyan"],
            selectbackground=self.colors["selection"],
            selectforeground=self.colors["fg_primary"],
            font=("Segoe UI", 11),
            borderwidth=1,
            relief="solid",
        )
        self.text_widget.grid(
            row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        # Bind text change events for real-time validation and stats
        self.text_widget.bind("<KeyRelease>", self.on_text_change)
        self.text_widget.bind("<Button-1>", self.on_text_change)

        # Bind paste events to remove duplicate lines (multiple methods for compatibility)
        self.text_widget.bind("<<Paste>>", self.on_paste)
        self.text_widget.bind("<Control-v>", self.on_paste)
        self.text_widget.bind("<Control-V>", self.on_paste)
        self.text_widget.bind("<Shift-Insert>", self.on_paste)

        # Configure text tags for follow-along word highlighting
        self.text_widget.tag_configure(
            "current_word",
            background=self.colors["accent_pink"],
            foreground=self.colors["bg_primary"],
            font=("Segoe UI", 11, "bold"),
        )
        self.text_widget.tag_configure(
            "spoken_word", foreground=self.colors["fg_muted"]
        )
        self.text_widget.tag_configure(
            "upcoming_word", foreground=self.colors["fg_primary"]
        )

        # Text statistics frame
        stats_frame = ttk.Frame(text_frame, style="Card.TFrame", padding="8")
        stats_frame.grid(
            row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        ttk.Label(stats_frame, text="📊 Text Stats:", style="Dark.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        self.stats_label = ttk.Label(
            stats_frame,
            text="Characters: 0 | Words: 0 | Lines: 0 | Sentences: 0",
            style="Time.TLabel",
        )
        self.stats_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Validation status
        self.validation_label = ttk.Label(
            stats_frame, text="✓ Ready", style="Dark.TLabel"
        )
        self.validation_label.grid(row=0, column=2, sticky=tk.E, padx=(20, 0))

        # Add sample text with better guidance
        sample_text = (
            "Welcome to the enhanced high-quality English text-to-speech system! "
            "This version features improved text processing, performance optimizations, and audio caching. "
            "Try editing this text, importing your own content, or adjusting the processing options above. "
            "The system will provide real-time feedback on text statistics and validation."
        )
        self.text_widget.insert(tk.END, sample_text)

        # Initial text stats update
        self.on_text_change(None)

        # Controls frame with better spacing
        controls_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        controls_frame.grid(row=3, column=0, columnspan=3, pady=(0, 15))

        # Generate button (primary action)
        self.generate_btn = ttk.Button(
            controls_frame,
            text="🎵 Generate Speech",
            command=self.generate_speech,
            style="Primary.TButton",
        )
        self.generate_btn.grid(row=0, column=0, padx=(0, 10))

        # Cancel button (initially hidden)
        self.cancel_btn = ttk.Button(
            controls_frame,
            text="⏹ Cancel",
            command=self.cancel_generation,
            style="Danger.TButton",
        )
        self.cancel_btn.grid(row=0, column=1, padx=(0, 15))
        self.cancel_btn.grid_remove()  # Hide initially

        # Play button
        self.play_btn = ttk.Button(
            controls_frame,
            text="▶ Play",
            command=self.play_audio,
            state=tk.DISABLED,
            style="Success.TButton",
        )
        self.play_btn.grid(row=0, column=2, padx=(0, 10))

        # Stop button
        self.stop_btn = ttk.Button(
            controls_frame,
            text="⏸ Pause",
            command=self.stop_audio,
            state=tk.DISABLED,
            style="Warning.TButton",
        )
        self.stop_btn.grid(row=0, column=3, padx=(0, 10))

        # Save button
        self.save_btn = ttk.Button(
            controls_frame,
            text="💾 Save As...",
            command=self.save_audio,
            state=tk.DISABLED,
            style="Dark.TButton",
        )
        self.save_btn.grid(row=0, column=4, padx=(0, 10))

        # Keyboard shortcuts help button
        self.shortcuts_btn = ttk.Button(
            controls_frame,
            text="⌨️ Shortcuts (F1)",
            command=self.show_keyboard_shortcuts,
            style="Dark.TButton",
        )
        self.shortcuts_btn.grid(row=0, column=5, padx=(10, 0))

        # Audio Playback Controls Frame with enhanced styling
        playback_frame = ttk.LabelFrame(
            main_frame,
            text="🎛️ Audio Playback Controls",
            style="Dark.TLabelframe",
            padding="15",
        )
        playback_frame.grid(
            row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        # Time display with modern styling
        time_frame = ttk.Frame(playback_frame, style="Card.TFrame", padding="8")
        time_frame.grid(
            row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        ttk.Label(time_frame, text="⏱️ Time:", style="Dark.TLabel").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )
        self.time_label = ttk.Label(
            time_frame, text="00:00 / 00:00", style="Time.TLabel"
        )
        self.time_label.grid(row=0, column=1, sticky=tk.W)

        # Seek bar with enhanced styling
        seek_frame = ttk.Frame(playback_frame, style="Dark.TFrame")
        seek_frame.grid(
            row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        ttk.Label(seek_frame, text="🎯 Position:", style="Dark.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(
            seek_frame,
            from_=0.0,
            to=100.0,
            variable=self.seek_var,
            orient=tk.HORIZONTAL,
            command=self.on_seek,
            style="Dark.Horizontal.TScale",
        )
        self.seek_scale.grid(row=0, column=1, padx=(15, 0), sticky=(tk.W, tk.E))
        self.seek_scale.config(state=tk.DISABLED)

        # Playback speed control with enhanced styling
        playback_speed_frame = ttk.Frame(playback_frame, style="Dark.TFrame")
        playback_speed_frame.grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0)
        )

        ttk.Label(
            playback_speed_frame,
            text="🚀 Playback Speed (pitch preserved):",
            style="Dark.TLabel",
        ).grid(row=0, column=0, sticky=tk.W)
        self.playback_speed_var = tk.DoubleVar(value=1.0)
        playback_speed_scale = ttk.Scale(
            playback_speed_frame,
            from_=0.5,
            to=2.0,
            variable=self.playback_speed_var,
            orient=tk.HORIZONTAL,
            command=self.update_playback_speed_label,
            style="Dark.Horizontal.TScale",
        )
        playback_speed_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.playback_speed_label = ttk.Label(
            playback_speed_frame, text="1.0x", style="Dark.TLabel"
        )
        self.playback_speed_label.grid(row=0, column=2, padx=(5, 0))

        # Volume control with enhanced styling
        volume_frame = ttk.Frame(playback_frame, style="Dark.TFrame")
        volume_frame.grid(
            row=2, column=2, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0)
        )

        ttk.Label(volume_frame, text="🔊 Volume:", style="Dark.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        self.volume_var = tk.DoubleVar(value=70.0)
        volume_scale = ttk.Scale(
            volume_frame,
            from_=0.0,
            to=100.0,
            variable=self.volume_var,
            orient=tk.HORIZONTAL,
            command=self.update_volume_label,
            style="Dark.Horizontal.TScale",
        )
        volume_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.volume_label = ttk.Label(volume_frame, text="70%", style="Dark.TLabel")
        self.volume_label.grid(row=0, column=2, padx=(5, 0))

        # Follow-along word highlighting frame
        follow_along_frame = ttk.Frame(playback_frame, style="Card.TFrame", padding="8")
        follow_along_frame.grid(
            row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        # Enable/disable checkbox
        ttk.Checkbutton(
            follow_along_frame,
            text="📖 Follow Along (highlight current word)",
            variable=self.follow_along_enabled,
            style="Dark.TRadiobutton",
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))

        # Current word display
        ttk.Label(follow_along_frame, text="Current:", style="Dark.TLabel").grid(
            row=0, column=1, sticky=tk.W, padx=(10, 5)
        )
        self.follow_along_word_label = tk.Label(
            follow_along_frame,
            text="---",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["bg_tertiary"],
            fg=self.colors["accent_pink"],
            padx=10,
            pady=2,
        )
        self.follow_along_word_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 20))

        # Word progress display
        ttk.Label(follow_along_frame, text="Progress:", style="Dark.TLabel").grid(
            row=0, column=3, sticky=tk.W, padx=(10, 5)
        )
        self.follow_along_progress_label = ttk.Label(
            follow_along_frame, text="0 / 0 words", style="Time.TLabel"
        )
        self.follow_along_progress_label.grid(row=0, column=4, sticky=tk.W)

        follow_along_frame.columnconfigure(2, weight=1)

        # Status frame with dark theme
        status_frame = ttk.LabelFrame(
            main_frame,
            text="📊 Status & Performance",
            style="Dark.TLabelframe",
            padding="15",
        )
        status_frame.grid(
            row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        # Performance info frame
        perf_info_frame = ttk.Frame(status_frame, style="Card.TFrame", padding="8")
        perf_info_frame.grid(
            row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        ttk.Label(perf_info_frame, text="🚀 Performance:", style="Dark.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        self.perf_label = ttk.Label(
            perf_info_frame, text="Cache: 0 items | Avg RTF: N/A", style="Time.TLabel"
        )
        self.perf_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Cache management buttons
        ttk.Button(
            perf_info_frame,
            text="🗑️ Clear Cache",
            command=self.clear_cache,
            style="Warning.TButton",
        ).grid(row=0, column=2, sticky=tk.E, padx=(20, 0))

        # Status text with Dracula theme
        self.status_text = scrolledtext.ScrolledText(
            status_frame,
            width=75,
            height=6,
            wrap=tk.WORD,
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_secondary"],
            insertbackground=self.colors["accent_cyan"],
            selectbackground=self.colors["selection"],
            selectforeground=self.colors["fg_primary"],
            font=("Consolas", 9),
            borderwidth=1,
            relief="solid",
        )
        self.status_text.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # Progress bar with dark theme (can switch between indeterminate and determinate)
        self.progress = ttk.Progressbar(
            main_frame, mode="indeterminate", style="Dark.Horizontal.TProgressbar"
        )
        self.progress.grid(
            row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15)
        )

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        voice_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)
        playback_frame.columnconfigure(1, weight=1)
        playback_frame.columnconfigure(3, weight=1)
        seek_frame.columnconfigure(1, weight=1)
        playback_speed_frame.columnconfigure(1, weight=1)
        volume_frame.columnconfigure(1, weight=1)

    def update_speed_label(self, value):
        """Update speed label when scale changes"""
        self.speed_label.config(text=f"{float(value):.1f}x")

    def update_playback_speed_label(self, value):
        """Update playback speed label when scale changes"""
        self.playback_speed_label.config(text=f"{float(value):.1f}x")

    def update_volume_label(self, value):
        """Update volume label when scale changes"""
        self.volume_label.config(text=f"{int(float(value))}%")
        if self.current_sound and self.is_playing:
            self.current_sound.set_volume(float(value) / 100.0)

    def update_performance_display(self):
        """Update performance information display"""
        cache_size = len(self.audio_cache.cache)
        avg_rtf = self.performance_monitor.get_average_rtf()

        perf_text = f"Cache: {cache_size} items"
        if avg_rtf > 0:
            perf_text += f" | Avg RTF: {avg_rtf:.3f}"
        else:
            perf_text += " | Avg RTF: N/A"

        self.perf_label.config(text=perf_text)

    def on_seek(self, value):
        """Handle seek bar changes"""
        if self.audio_duration > 0 and not self.is_playing:
            seek_position = (float(value) / 100.0) * self.audio_duration
            self.pause_position = seek_position
            self.update_time_display(seek_position)

    def format_time(self, seconds):
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def update_time_display(self, current_time=None):
        """Update the time display"""
        if current_time is None:
            if self.is_playing:
                elapsed = time.time() - self.playback_start_time
                current_time = (
                    self.pause_position + elapsed * self.playback_speed_var.get()
                )
            else:
                current_time = self.pause_position

        current_time = min(current_time, self.audio_duration)
        current_str = self.format_time(current_time)
        total_str = self.format_time(self.audio_duration)
        self.time_label.config(text=f"{current_str} / {total_str}")

        if self.audio_duration > 0:
            progress = (current_time / self.audio_duration) * 100
            self.seek_var.set(progress)

    def log_status(self, message, level="info"):
        """Add message to status text widget with color coding"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # Configure text tags for different message types with Dracula colors
        if not hasattr(self, "_tags_configured"):
            self.status_text.tag_configure(
                "info", foreground=self.colors["fg_secondary"]
            )
            self.status_text.tag_configure(
                "success", foreground=self.colors["accent_green"]
            )
            self.status_text.tag_configure(
                "warning", foreground=self.colors["accent_orange"]
            )
            self.status_text.tag_configure(
                "error", foreground=self.colors["accent_red"]
            )
            self.status_text.tag_configure(
                "timestamp", foreground=self.colors["fg_muted"]
            )
            self._tags_configured = True

        # Determine message type based on content
        if "✓" in message or "successfully" in message.lower():
            tag = "success"
        elif "⚠" in message or "warning" in message.lower():
            tag = "warning"
        elif "✗" in message or "error" in message.lower():
            tag = "error"
        else:
            tag = "info"

        self.status_text.insert(tk.END, formatted_message, tag)
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def clear_cache(self):
        """Clear audio cache"""
        if messagebox.askyesno(
            "Clear Cache", "Are you sure you want to clear the audio cache?"
        ):
            self.audio_cache.clear()
            self.update_performance_display()
            self.log_status("🗑️ Audio cache cleared")
