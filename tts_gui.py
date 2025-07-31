#!/usr/bin/env python3

"""
High-Quality English TTS GUI
A user-friendly interface for sherpa-onnx text-to-speech with premium English models
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import time
import uuid
import pygame
import sherpa_onnx
import soundfile as sf
from pathlib import Path
import numpy as np

class TTSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("High-Quality English TTS - Sherpa-ONNX")
        self.root.geometry("950x850")

        # Dracula theme color scheme
        self.colors = {
            'bg_primary': '#282a36',      # Main background (Dracula background)
            'bg_secondary': '#44475a',    # Secondary background (Dracula selection)
            'bg_tertiary': '#6272a4',     # Tertiary background (Dracula comment)
            'bg_accent': '#44475a',       # Accent background
            'fg_primary': '#f8f8f2',      # Primary text (Dracula foreground)
            'fg_secondary': '#bd93f9',    # Secondary text (Dracula purple)
            'fg_muted': '#6272a4',        # Muted text (Dracula comment)
            'accent_pink': '#ff79c6',     # Dracula pink (primary accent)
            'accent_pink_hover': '#ff92d0', # Pink hover state
            'accent_cyan': '#8be9fd',     # Dracula cyan
            'accent_green': '#50fa7b',    # Dracula green (success)
            'accent_orange': '#ffb86c',   # Dracula orange (warning)
            'accent_red': '#ff5555',      # Dracula red (error/danger)
            'accent_purple': '#bd93f9',   # Dracula purple
            'selection': '#44475a',       # Selection background
            'border': '#6272a4',          # Border color
            'border_light': '#bd93f9',    # Light border (purple)
        }

        # Configure root window
        self.root.configure(bg=self.colors['bg_primary'])

        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

        # TTS model instances
        self.matcha_tts = None
        self.kokoro_tts = None
        self.current_audio_file = None

        # Model availability flags
        self.matcha_available = False
        self.kokoro_available = False

        # Audio playback control variables
        self.current_sound = None
        self.audio_duration = 0.0
        self.playback_start_time = 0.0
        self.is_playing = False
        self.is_paused = False
        self.pause_position = 0.0
        self.audio_data = None
        self.sample_rate = 22050

        # Setup theme and UI
        self.setup_theme()
        self.setup_ui()
        self.check_models()

    def setup_theme(self):
        """Configure the dark theme for ttk widgets"""
        style = ttk.Style()

        # Configure the theme
        style.theme_use('clam')

        # Configure Frame styles
        style.configure('Dark.TFrame',
                       background=self.colors['bg_secondary'],
                       borderwidth=1,
                       relief='solid',
                       bordercolor=self.colors['border'])

        style.configure('Card.TFrame',
                       background=self.colors['bg_tertiary'],
                       borderwidth=1,
                       relief='solid',
                       bordercolor=self.colors['border_light'])

        # Configure LabelFrame styles
        style.configure('Dark.TLabelframe',
                       background=self.colors['bg_secondary'],
                       borderwidth=2,
                       relief='solid',
                       bordercolor=self.colors['border'])

        style.configure('Dark.TLabelframe.Label',
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 10, 'bold'))

        # Configure Label styles
        style.configure('Dark.TLabel',
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 9))

        style.configure('Title.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 18, 'bold'))

        style.configure('Time.TLabel',
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['accent_cyan'],
                       font=('Consolas', 11, 'bold'))

        # Configure Button styles with Dracula colors
        style.configure('Primary.TButton',
                       background=self.colors['accent_pink'],
                       foreground=self.colors['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(15, 8))

        style.map('Primary.TButton',
                 background=[('active', self.colors['accent_pink_hover']),
                           ('pressed', '#ff66c4'),
                           ('disabled', self.colors['bg_accent'])])

        style.configure('Success.TButton',
                       background=self.colors['accent_green'],
                       foreground=self.colors['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10),
                       padding=(12, 6))

        style.map('Success.TButton',
                 background=[('active', '#45e070'),
                           ('pressed', '#3dd164'),
                           ('disabled', self.colors['bg_accent'])])

        style.configure('Warning.TButton',
                       background=self.colors['accent_orange'],
                       foreground=self.colors['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10),
                       padding=(12, 6))

        style.map('Warning.TButton',
                 background=[('active', '#ffad5c'),
                           ('pressed', '#ffa04d'),
                           ('disabled', self.colors['bg_accent'])])

        style.configure('Danger.TButton',
                       background=self.colors['accent_red'],
                       foreground=self.colors['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10),
                       padding=(12, 6))

        style.map('Danger.TButton',
                 background=[('active', '#ff4444'),
                           ('pressed', '#ff3333'),
                           ('disabled', self.colors['bg_accent'])])

        style.configure('Dark.TButton',
                       background=self.colors['bg_accent'],
                       foreground=self.colors['fg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10),
                       padding=(12, 6))

        style.map('Dark.TButton',
                 background=[('active', self.colors['bg_tertiary']),
                           ('pressed', self.colors['border']),
                           ('disabled', self.colors['bg_secondary'])])

        # Configure Radiobutton styles
        style.configure('Dark.TRadiobutton',
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['fg_primary'],
                       focuscolor='none',
                       font=('Segoe UI', 10))

        style.map('Dark.TRadiobutton',
                 background=[('active', self.colors['bg_accent'])])

        # Configure Scale styles with Dracula colors
        style.configure('Dark.Horizontal.TScale',
                       background=self.colors['bg_secondary'],
                       troughcolor=self.colors['bg_accent'],
                       borderwidth=0,
                       lightcolor=self.colors['accent_purple'],
                       darkcolor=self.colors['accent_purple'])

        # Configure Spinbox styles
        style.configure('Dark.TSpinbox',
                       fieldbackground=self.colors['bg_tertiary'],
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['fg_primary'],
                       bordercolor=self.colors['border'],
                       arrowcolor=self.colors['fg_secondary'],
                       font=('Segoe UI', 9))

        # Configure Progressbar styles with Dracula colors
        style.configure('Dark.Horizontal.TProgressbar',
                       background=self.colors['accent_pink'],
                       troughcolor=self.colors['bg_accent'],
                       borderwidth=0,
                       lightcolor=self.colors['accent_pink'],
                       darkcolor=self.colors['accent_pink'])

    def setup_ui(self):
        # Main frame with dark theme
        main_frame = ttk.Frame(self.root, style='Dark.TFrame', padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.configure(style='Dark.TFrame')

        # Title with modern styling
        title_label = ttk.Label(main_frame, text="🎵 High-Quality English Text-to-Speech", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 25))

        # Model selection frame with dark theme
        model_frame = ttk.LabelFrame(main_frame, text="🤖 TTS Model Selection", style='Dark.TLabelframe', padding="15")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        self.model_var = tk.StringVar(value="matcha")

        self.matcha_radio = ttk.Radiobutton(model_frame, text="🎯 Matcha-TTS LJSpeech (High Quality)",
                                          variable=self.model_var, value="matcha", style='Dark.TRadiobutton')
        self.matcha_radio.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.kokoro_radio = ttk.Radiobutton(model_frame, text="🗣️ Kokoro English (Multiple Speakers)",
                                          variable=self.model_var, value="kokoro", style='Dark.TRadiobutton')
        self.kokoro_radio.grid(row=1, column=0, sticky=tk.W, pady=5)

        # Speaker selection (for Kokoro)
        speaker_frame = ttk.Frame(model_frame, style='Dark.TFrame')
        speaker_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 0))

        ttk.Label(speaker_frame, text="👤 Speaker ID (Kokoro only):", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.speaker_var = tk.StringVar(value="0")
        speaker_spinbox = ttk.Spinbox(speaker_frame, from_=0, to=10, width=8, textvariable=self.speaker_var, style='Dark.TSpinbox')
        speaker_spinbox.grid(row=0, column=1, padx=(15, 0))

        # Speed control
        speed_frame = ttk.Frame(model_frame, style='Dark.TFrame')
        speed_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(15, 0))

        ttk.Label(speed_frame, text="⚡ Generation Speed:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.5, to=2.0, variable=self.speed_var,
                               orient=tk.HORIZONTAL, style='Dark.Horizontal.TScale')
        speed_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.speed_label = ttk.Label(speed_frame, text="1.0x", style='Dark.TLabel')
        self.speed_label.grid(row=0, column=2, padx=(5, 0))

        # Update speed label when scale changes
        speed_scale.configure(command=self.update_speed_label)

        # Text input frame with dark theme
        text_frame = ttk.LabelFrame(main_frame, text="📝 Text to Synthesize", style='Dark.TLabelframe', padding="15")
        text_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))

        # Create custom text widget with Dracula theme
        self.text_widget = scrolledtext.ScrolledText(text_frame, width=75, height=8, wrap=tk.WORD,
                                                    bg=self.colors['bg_primary'],
                                                    fg=self.colors['fg_primary'],
                                                    insertbackground=self.colors['accent_cyan'],
                                                    selectbackground=self.colors['selection'],
                                                    selectforeground=self.colors['fg_primary'],
                                                    font=('Segoe UI', 11),
                                                    borderwidth=1,
                                                    relief='solid')
        self.text_widget.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Add sample text
        sample_text = ("Welcome to the high-quality English text-to-speech system. "
                      "This demonstration showcases the advanced neural TTS capabilities "
                      "with natural pronunciation and excellent voice quality. "
                      "You can edit this text or replace it with your own content.")
        self.text_widget.insert(tk.END, sample_text)

        # Controls frame with better spacing
        controls_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        controls_frame.grid(row=3, column=0, columnspan=3, pady=(0, 15))

        # Generate button (primary action)
        self.generate_btn = ttk.Button(controls_frame, text="🎵 Generate Speech",
                                     command=self.generate_speech, style="Primary.TButton")
        self.generate_btn.grid(row=0, column=0, padx=(0, 15))

        # Play button
        self.play_btn = ttk.Button(controls_frame, text="▶ Play", command=self.play_audio,
                                 state=tk.DISABLED, style="Success.TButton")
        self.play_btn.grid(row=0, column=1, padx=(0, 10))

        # Stop button
        self.stop_btn = ttk.Button(controls_frame, text="⏸ Pause", command=self.stop_audio,
                                 state=tk.DISABLED, style="Warning.TButton")
        self.stop_btn.grid(row=0, column=2, padx=(0, 10))

        # Save button
        self.save_btn = ttk.Button(controls_frame, text="💾 Save As...", command=self.save_audio,
                                 state=tk.DISABLED, style="Dark.TButton")
        self.save_btn.grid(row=0, column=3, padx=(0, 10))

        # Audio Playback Controls Frame with enhanced styling
        playback_frame = ttk.LabelFrame(main_frame, text="🎛️ Audio Playback Controls",
                                       style='Dark.TLabelframe', padding="15")
        playback_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        # Time display with modern styling
        time_frame = ttk.Frame(playback_frame, style='Card.TFrame', padding="8")
        time_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(time_frame, text="⏱️ Time:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.time_label = ttk.Label(time_frame, text="00:00 / 00:00", style='Time.TLabel')
        self.time_label.grid(row=0, column=1, sticky=tk.W)

        # Seek bar with enhanced styling
        seek_frame = ttk.Frame(playback_frame, style='Dark.TFrame')
        seek_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 15))

        ttk.Label(seek_frame, text="🎯 Position:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(seek_frame, from_=0.0, to=100.0, variable=self.seek_var,
                                   orient=tk.HORIZONTAL, command=self.on_seek, style='Dark.Horizontal.TScale')
        self.seek_scale.grid(row=0, column=1, padx=(15, 0), sticky=(tk.W, tk.E))
        self.seek_scale.config(state=tk.DISABLED)

        # Playback speed control with enhanced styling
        playback_speed_frame = ttk.Frame(playback_frame, style='Dark.TFrame')
        playback_speed_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Label(playback_speed_frame, text="🚀 Playback Speed (pitch preserved):",
                 style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.playback_speed_var = tk.DoubleVar(value=1.0)
        playback_speed_scale = ttk.Scale(playback_speed_frame, from_=0.5, to=2.0,
                                        variable=self.playback_speed_var, orient=tk.HORIZONTAL,
                                        command=self.update_playback_speed_label, style='Dark.Horizontal.TScale')
        playback_speed_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.playback_speed_label = ttk.Label(playback_speed_frame, text="1.0x", style='Dark.TLabel')
        self.playback_speed_label.grid(row=0, column=2, padx=(5, 0))

        # Volume control with enhanced styling
        volume_frame = ttk.Frame(playback_frame, style='Dark.TFrame')
        volume_frame.grid(row=2, column=2, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Label(volume_frame, text="🔊 Volume:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.volume_var = tk.DoubleVar(value=70.0)
        volume_scale = ttk.Scale(volume_frame, from_=0.0, to=100.0,
                                variable=self.volume_var, orient=tk.HORIZONTAL,
                                command=self.update_volume_label, style='Dark.Horizontal.TScale')
        volume_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.volume_label = ttk.Label(volume_frame, text="70%", style='Dark.TLabel')
        self.volume_label.grid(row=0, column=2, padx=(5, 0))

        # Status frame with dark theme
        status_frame = ttk.LabelFrame(main_frame, text="📊 Status & Logs",
                                     style='Dark.TLabelframe', padding="15")
        status_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        # Status text with Dracula theme
        self.status_text = scrolledtext.ScrolledText(status_frame, width=75, height=6, wrap=tk.WORD,
                                                    bg=self.colors['bg_primary'],
                                                    fg=self.colors['fg_secondary'],
                                                    insertbackground=self.colors['accent_cyan'],
                                                    selectbackground=self.colors['selection'],
                                                    selectforeground=self.colors['fg_primary'],
                                                    font=('Consolas', 9),
                                                    borderwidth=1,
                                                    relief='solid')
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Progress bar with dark theme
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', style='Dark.Horizontal.TProgressbar')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        model_frame.columnconfigure(0, weight=1)
        speaker_frame.columnconfigure(1, weight=1)
        speed_frame.columnconfigure(1, weight=1)
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
                current_time = self.pause_position + elapsed * self.playback_speed_var.get()
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
        timestamp = time.strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}\n"

        # Configure text tags for different message types with Dracula colors
        if not hasattr(self, '_tags_configured'):
            self.status_text.tag_configure("info", foreground=self.colors['fg_secondary'])
            self.status_text.tag_configure("success", foreground=self.colors['accent_green'])
            self.status_text.tag_configure("warning", foreground=self.colors['accent_orange'])
            self.status_text.tag_configure("error", foreground=self.colors['accent_red'])
            self.status_text.tag_configure("timestamp", foreground=self.colors['fg_muted'])
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

    def check_models(self):
        """Check if models are available"""
        self.log_status("Checking available models...")

        # Check Matcha-TTS
        matcha_path = Path("matcha-icefall-en_US-ljspeech")
        vocoder_path = Path("vocos-22khz-univ.onnx")
        if (matcha_path.exists() and vocoder_path.exists() and
            (matcha_path / "model-steps-3.onnx").exists()):
            self.matcha_available = True
            self.log_status("✓ Matcha-TTS LJSpeech model found")
        else:
            self.matcha_radio.config(state=tk.DISABLED)
            self.log_status("✗ Matcha-TTS model not found")

        # Check Kokoro
        kokoro_path = Path("kokoro-en-v0_19")
        if (kokoro_path.exists() and
            (kokoro_path / "model.onnx").exists() and
            (kokoro_path / "voices.bin").exists()):
            self.kokoro_available = True
            self.log_status("✓ Kokoro English model found")
        else:
            self.kokoro_radio.config(state=tk.DISABLED)
            self.log_status("✗ Kokoro English model not found")
        if self.matcha_available:
            self.model_var.set("matcha")
        elif self.kokoro_available:
            self.model_var.set("kokoro")
        else:
            self.log_status("⚠ No models available! Please download models first.")
            self.generate_btn.config(state=tk.DISABLED)

    def load_matcha_model(self):
        """Load Matcha-TTS model"""
        if self.matcha_tts is None:
            self.log_status("Loading Matcha-TTS model...")

            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                        "./matcha-icefall-en_US-ljspeech/model-steps-3.onnx",  # acoustic_model
                        "./vocos-22khz-univ.onnx",  # vocoder
                        "",  # lexicon (empty for English)
                        "./matcha-icefall-en_US-ljspeech/tokens.txt",  # tokens
                        "./matcha-icefall-en_US-ljspeech/espeak-ng-data",  # data_dir
                    ),
                    num_threads=2,
                    debug=False,
                    provider="cpu",
                ),
                max_num_sentences=1,
            )

            self.matcha_tts = sherpa_onnx.OfflineTts(config)
            self.log_status("✓ Matcha-TTS model loaded")

    def load_kokoro_model(self):
        """Load Kokoro model"""
        if self.kokoro_tts is None:
            self.log_status("Loading Kokoro English model...")

            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        "./kokoro-en-v0_19/model.onnx",  # model
                        "./kokoro-en-v0_19/voices.bin",  # voices
                        "./kokoro-en-v0_19/tokens.txt",  # tokens
                        "./kokoro-en-v0_19/espeak-ng-data",  # data_dir
                    ),
                    num_threads=2,
                    debug=False,
                    provider="cpu",
                ),
                max_num_sentences=1,
            )

            self.kokoro_tts = sherpa_onnx.OfflineTts(config)
            self.log_status("✓ Kokoro English model loaded")

    def generate_speech_thread(self):
        """Generate speech in separate thread"""
        try:
            text = self.text_widget.get(1.0, tk.END).strip()
            if not text:
                self.log_status("⚠ Please enter some text to synthesize")
                return

            model_type = self.model_var.get()
            speed = self.speed_var.get()

            self.log_status(f"Generating speech with {model_type.upper()} model...")

            # Stop any playing audio and cleanup
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                time.sleep(0.1)  # Give pygame time to release the file

            # Cleanup previous temporary file
            temp_file = "temp_generated_audio.wav"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    # File might be locked, try a different name
                    import uuid
                    temp_file = f"temp_audio_{uuid.uuid4().hex[:8]}.wav"

            start_time = time.time()

            if model_type == "matcha":
                if not self.matcha_available:
                    self.log_status("⚠ Matcha-TTS model not available")
                    return

                self.load_matcha_model()
                audio = self.matcha_tts.generate(text, sid=0, speed=speed)

            elif model_type == "kokoro":
                if not self.kokoro_available:
                    self.log_status("⚠ Kokoro model not available")
                    return

                self.load_kokoro_model()
                speaker_id = int(self.speaker_var.get())
                audio = self.kokoro_tts.generate(text, sid=speaker_id, speed=speed)

            # Save audio to temporary file
            self.current_audio_file = temp_file
            sf.write(self.current_audio_file, audio.samples, audio.sample_rate)

            # Store audio data for enhanced playback controls
            # Convert to numpy array if it's a list
            if isinstance(audio.samples, list):
                self.audio_data = np.array(audio.samples, dtype=np.float32)
            else:
                self.audio_data = np.array(audio.samples, dtype=np.float32)
            self.sample_rate = audio.sample_rate
            self.audio_duration = len(self.audio_data) / audio.sample_rate
            self.pause_position = 0.0

            # Calculate statistics
            elapsed_time = time.time() - start_time
            duration = len(audio.samples) / audio.sample_rate
            rtf = elapsed_time / duration

            self.log_status(f"✓ Speech generated successfully!")
            self.log_status(f"  Duration: {duration:.2f} seconds")
            self.log_status(f"  Generation time: {elapsed_time:.2f} seconds")
            self.log_status(f"  RTF (Real-time factor): {rtf:.3f}")

            # Enable playback buttons and controls
            self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.seek_scale.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.update_time_display(0.0))

        except Exception as e:
            self.log_status(f"✗ Error generating speech: {str(e)}")

        finally:
            # Re-enable generate button and hide progress
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.progress.stop())

    def generate_speech(self):
        """Start speech generation"""
        self.generate_btn.config(state=tk.DISABLED)
        self.progress.start()

        # Run generation in separate thread
        thread = threading.Thread(target=self.generate_speech_thread)
        thread.daemon = True
        thread.start()

    def create_speed_adjusted_audio(self, speed_factor):
        """Create speed-adjusted audio data using time-stretching (preserves pitch)"""
        if self.audio_data is None:
            return None

        # Ensure audio_data is a numpy array
        if not isinstance(self.audio_data, np.ndarray):
            audio_array = np.array(self.audio_data, dtype=np.float32)
        else:
            audio_array = self.audio_data

        # No adjustment needed
        if speed_factor == 1.0:
            return audio_array

        # Use time-stretching algorithm that preserves pitch
        return self.time_stretch_audio(audio_array, speed_factor)

    def time_stretch_audio(self, audio, speed_factor):
        """Time-stretch audio using overlap-add method (preserves pitch)"""
        if speed_factor == 1.0:
            return audio

        # Parameters for overlap-add time stretching
        frame_size = 2048  # Size of each frame
        hop_size = frame_size // 4  # Overlap between frames

        # Calculate new hop size based on speed factor
        new_hop_size = int(hop_size * speed_factor)

        # Pad audio to ensure we have enough samples
        padded_audio = np.pad(audio, (0, frame_size), mode='constant')

        # Calculate output length
        num_frames = (len(padded_audio) - frame_size) // new_hop_size + 1
        output_length = num_frames * hop_size
        output = np.zeros(output_length, dtype=np.float32)

        # Create window function (Hann window)
        window = np.hanning(frame_size).astype(np.float32)

        # Process each frame
        for i in range(num_frames):
            # Input position (stretched)
            input_pos = i * new_hop_size
            # Output position (original spacing)
            output_pos = i * hop_size

            # Extract frame from input
            if input_pos + frame_size <= len(padded_audio):
                frame = padded_audio[input_pos:input_pos + frame_size] * window

                # Add to output with overlap
                if output_pos + frame_size <= len(output):
                    output[output_pos:output_pos + frame_size] += frame

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95

        return output

    def play_audio(self):
        """Play generated audio with enhanced controls"""
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            try:
                # Stop any currently playing audio
                if self.current_sound:
                    self.current_sound.stop()

                # Get playback speed
                speed_factor = self.playback_speed_var.get()

                # Create speed-adjusted audio if needed
                if speed_factor != 1.0:
                    adjusted_audio = self.create_speed_adjusted_audio(speed_factor)
                    if adjusted_audio is not None:
                        # Convert to pygame sound format
                        audio_to_use = adjusted_audio
                    else:
                        # Fallback to original audio
                        audio_to_use = self.audio_data
                else:
                    audio_to_use = self.audio_data

                # Convert audio to proper format for pygame
                if audio_to_use is not None:
                    # Ensure it's a numpy array
                    if not isinstance(audio_to_use, np.ndarray):
                        audio_to_use = np.array(audio_to_use, dtype=np.float32)

                    # Convert to stereo if mono
                    if len(audio_to_use.shape) == 1:
                        stereo_audio = np.column_stack((audio_to_use, audio_to_use))
                    else:
                        stereo_audio = audio_to_use

                    # Normalize and convert to 16-bit integers
                    # Clamp values to [-1, 1] range first
                    stereo_audio = np.clip(stereo_audio, -1.0, 1.0)
                    stereo_audio_int16 = (stereo_audio * 32767).astype(np.int16)

                    self.current_sound = pygame.sndarray.make_sound(stereo_audio_int16)
                else:
                    # Fallback to file loading
                    self.current_sound = pygame.mixer.Sound(self.current_audio_file)

                # Set volume
                volume = self.volume_var.get() / 100.0
                self.current_sound.set_volume(volume)

                # Calculate start position based on seek
                if self.pause_position > 0:
                    # For seeking, we need to create a subset of the audio
                    start_sample = int(self.pause_position * self.sample_rate)
                    if start_sample < len(self.audio_data):
                        remaining_audio = self.audio_data[start_sample:]
                        if speed_factor != 1.0:
                            remaining_audio = self.create_speed_adjusted_audio_from_data(remaining_audio, speed_factor)

                        # Ensure it's a numpy array
                        if not isinstance(remaining_audio, np.ndarray):
                            remaining_audio = np.array(remaining_audio, dtype=np.float32)

                        # Convert to stereo if mono
                        if len(remaining_audio.shape) == 1:
                            stereo_audio = np.column_stack((remaining_audio, remaining_audio))
                        else:
                            stereo_audio = remaining_audio

                        # Normalize and convert to 16-bit integers
                        stereo_audio = np.clip(stereo_audio, -1.0, 1.0)
                        stereo_audio_int16 = (stereo_audio * 32767).astype(np.int16)

                        self.current_sound = pygame.sndarray.make_sound(stereo_audio_int16)
                        self.current_sound.set_volume(volume)

                # Start playback
                self.current_sound.play()
                self.is_playing = True
                self.is_paused = False
                self.playback_start_time = time.time()

                self.log_status("▶ Playing audio...")
                self.play_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)

                # Monitor playback
                self.monitor_playback()

            except Exception as e:
                self.log_status(f"✗ Error playing audio: {str(e)}")

    def create_speed_adjusted_audio_from_data(self, audio_data, speed_factor):
        """Create speed-adjusted audio from given data using time-stretching"""
        # Ensure audio_data is a numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_array = np.array(audio_data, dtype=np.float32)
        else:
            audio_array = audio_data

        if speed_factor == 1.0:
            return audio_array

        # Use the same time-stretching algorithm
        return self.time_stretch_audio(audio_array, speed_factor)

    def monitor_playback(self):
        """Monitor audio playback status with enhanced controls"""
        if self.is_playing and self.current_sound:
            # Check if sound is still playing
            if pygame.mixer.get_busy():
                # Update time display and seek bar
                self.update_time_display()

                # Check if we've reached the end
                elapsed = time.time() - self.playback_start_time
                current_time = self.pause_position + elapsed * self.playback_speed_var.get()

                if current_time >= self.audio_duration:
                    self.playback_finished()
                else:
                    # Continue monitoring
                    self.root.after(50, self.monitor_playback)
            else:
                # Playback finished
                self.playback_finished()

    def playback_finished(self):
        """Handle playback completion"""
        self.is_playing = False
        self.is_paused = False
        self.pause_position = 0.0
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_time_display(0.0)
        self.log_status("⏹ Playback finished")

    def stop_audio(self):
        """Stop audio playback"""
        if self.current_sound:
            self.current_sound.stop()

        if self.is_playing:
            # Calculate current position for pause
            elapsed = time.time() - self.playback_start_time
            self.pause_position += elapsed * self.playback_speed_var.get()
            self.pause_position = min(self.pause_position, self.audio_duration)

        self.is_playing = False
        self.is_paused = True
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_time_display()
        self.log_status("⏸ Playback paused")

    def save_audio(self):
        """Save audio to file"""
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            file_path = filedialog.asksaveasfilename(
                title="Save Audio As",
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )

            if file_path:
                try:
                    # Copy temporary file to chosen location
                    import shutil
                    shutil.copy2(self.current_audio_file, file_path)
                    self.log_status(f"💾 Audio saved to: {file_path}")
                except Exception as e:
                    self.log_status(f"✗ Error saving audio: {str(e)}")

def main():
    """Main function"""
    # Check if required packages are available
    try:
        import sherpa_onnx
        import pygame
        import soundfile
    except ImportError as e:
        print(f"Required package missing: {e}")
        print("Please install required packages:")
        print("pip install sherpa-onnx pygame soundfile")
        return

    root = tk.Tk()
    app = TTSGui(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if hasattr(app, 'current_sound') and app.current_sound:
            try:
                app.current_sound.stop()
            except:
                pass
        pygame.mixer.quit()
        if hasattr(app, 'current_audio_file') and app.current_audio_file:
            try:
                os.remove(app.current_audio_file)
            except:
                pass

if __name__ == "__main__":
    main()
