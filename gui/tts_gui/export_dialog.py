#!/usr/bin/env python3

from tts_gui.common import tk, ttk, messagebox


class ExportOptionsDialog:

    def __init__(
        self, parent, audio_exporter, audio_data, sample_rate, colors, original_text=""
    ):
        self.parent = parent
        self.exporter = audio_exporter
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.colors = colors
        self.original_text = original_text
        self.result = None

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Advanced Export Options")
        self.dialog.geometry("650x850")
        self.dialog.configure(bg=colors["bg_primary"])
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 650) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 850) // 2
        self.dialog.geometry(f"+{x}+{y}")

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI"""
        # Main container with padding
        main_frame = ttk.Frame(self.dialog, style="Dark.TFrame", padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(
            main_frame,
            text="🎵 Advanced Audio Export",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_primary"],
        )
        title_label.pack(pady=(0, 15))

        # === Format Selection Frame ===
        format_frame = ttk.LabelFrame(
            main_frame, text="📀 Output Format", style="Dark.TLabelframe", padding="10"
        )
        format_frame.pack(fill=tk.X, pady=(0, 10))

        self.format_var = tk.StringVar(value="wav")
        available_formats = self.exporter.get_available_formats()

        for fmt in available_formats:
            fmt_config = self.exporter.FORMATS[fmt]
            rb = ttk.Radiobutton(
                format_frame,
                text=fmt_config["name"],
                variable=self.format_var,
                value=fmt,
                command=self._on_format_change,
                style="Dark.TRadiobutton",
            )
            rb.pack(anchor=tk.W, pady=2)

            # Description
            desc_label = tk.Label(
                format_frame,
                text=f"    {fmt_config['description']}",
                bg=self.colors["bg_secondary"],
                fg=self.colors["fg_muted"],
                font=("Segoe UI", 9),
            )
            desc_label.pack(anchor=tk.W)

        # Format availability note
        if "mp3" not in available_formats or "ogg" not in available_formats:
            note_label = tk.Label(
                format_frame,
                text="ℹ️ Install ffmpeg for MP3/OGG support",
                bg=self.colors["bg_secondary"],
                fg=self.colors["accent_orange"],
                font=("Segoe UI", 9),
            )
            note_label.pack(anchor=tk.W, pady=(10, 0))

        # === Quality Settings Frame ===
        quality_frame = ttk.LabelFrame(
            main_frame,
            text="⚙️ Quality Settings",
            style="Dark.TLabelframe",
            padding="10",
        )
        quality_frame.pack(fill=tk.X, pady=(0, 10))

        # Sample Rate
        sr_frame = ttk.Frame(quality_frame, style="Dark.TFrame")
        sr_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sr_frame, text="Sample Rate:", style="Dark.TLabel").pack(side=tk.LEFT)
        self.sample_rate_var = tk.StringVar(value=str(self.sample_rate))
        sr_combo = ttk.Combobox(
            sr_frame,
            textvariable=self.sample_rate_var,
            values=[str(sr) for sr in self.exporter.SAMPLE_RATES],
            width=10,
            state="readonly",
        )
        sr_combo.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(sr_frame, text="Hz", style="Dark.TLabel").pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Bitrate (for lossy formats)
        self.bitrate_frame = ttk.Frame(quality_frame, style="Dark.TFrame")
        self.bitrate_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.bitrate_frame, text="Bitrate:", style="Dark.TLabel").pack(
            side=tk.LEFT
        )
        self.bitrate_var = tk.StringVar(value="192")
        self.bitrate_combo = ttk.Combobox(
            self.bitrate_frame,
            textvariable=self.bitrate_var,
            values=["64", "96", "128", "160", "192", "224", "256", "320"],
            width=10,
            state="readonly",
        )
        self.bitrate_combo.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(self.bitrate_frame, text="kbps", style="Dark.TLabel").pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Initially hide bitrate for WAV
        self.bitrate_frame.pack_forget()

        # Normalize checkbox
        self.normalize_var = tk.BooleanVar(value=False)
        norm_cb = ttk.Checkbutton(
            quality_frame,
            text="Normalize audio (maximize volume without clipping)",
            variable=self.normalize_var,
            style="Dark.TRadiobutton",
        )
        norm_cb.pack(anchor=tk.W, pady=5)

        # === Split Options Frame ===
        split_frame = ttk.LabelFrame(
            main_frame, text="✂️ Split Options", style="Dark.TLabelframe", padding="10"
        )
        split_frame.pack(fill=tk.X, pady=(0, 10))

        self.split_mode_var = tk.StringVar(value="none")

        # No split
        ttk.Radiobutton(
            split_frame,
            text="Export as single file",
            variable=self.split_mode_var,
            value="none",
            command=self._on_split_mode_change,
            style="Dark.TRadiobutton",
        ).pack(anchor=tk.W, pady=2)

        # Split by silence
        ttk.Radiobutton(
            split_frame,
            text="Split by silence (automatic track detection)",
            variable=self.split_mode_var,
            value="silence",
            command=self._on_split_mode_change,
            style="Dark.TRadiobutton",
        ).pack(anchor=tk.W, pady=2)

        # Split by chapters
        ttk.Radiobutton(
            split_frame,
            text="Split by chapters/sections (from text markers)",
            variable=self.split_mode_var,
            value="chapters",
            command=self._on_split_mode_change,
            style="Dark.TRadiobutton",
        ).pack(anchor=tk.W, pady=2)

        # Silence detection settings
        self.silence_settings_frame = ttk.Frame(split_frame, style="Dark.TFrame")

        # Min silence length
        sl_frame = ttk.Frame(self.silence_settings_frame, style="Dark.TFrame")
        sl_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sl_frame, text="Min silence length:", style="Dark.TLabel").pack(
            side=tk.LEFT
        )
        self.min_silence_var = tk.StringVar(value="500")
        sl_spin = ttk.Spinbox(
            sl_frame,
            from_=100,
            to=5000,
            increment=100,
            textvariable=self.min_silence_var,
            width=8,
        )
        sl_spin.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(sl_frame, text="ms", style="Dark.TLabel").pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Silence threshold
        st_frame = ttk.Frame(self.silence_settings_frame, style="Dark.TFrame")
        st_frame.pack(fill=tk.X, pady=2)
        ttk.Label(st_frame, text="Silence threshold:", style="Dark.TLabel").pack(
            side=tk.LEFT
        )
        self.silence_thresh_var = tk.StringVar(value="-40")
        st_spin = ttk.Spinbox(
            st_frame,
            from_=-60,
            to=-20,
            increment=5,
            textvariable=self.silence_thresh_var,
            width=8,
        )
        st_spin.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(st_frame, text="dB", style="Dark.TLabel").pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Preview silence detection button
        preview_btn = ttk.Button(
            self.silence_settings_frame,
            text="🔍 Preview Splits",
            command=self._preview_silence_splits,
            style="Dark.TButton",
        )
        preview_btn.pack(anchor=tk.W, pady=(5, 0))

        # Detected chapters display
        self.chapters_frame = ttk.Frame(split_frame, style="Dark.TFrame")

        chapters_label = tk.Label(
            self.chapters_frame,
            text="Detected chapters from text:",
            bg=self.colors["bg_secondary"],
            fg=self.colors["fg_primary"],
            font=("Segoe UI", 10),
        )
        chapters_label.pack(anchor=tk.W)

        self.chapters_listbox = tk.Listbox(
            self.chapters_frame,
            height=4,
            bg=self.colors["bg_primary"],
            fg=self.colors["fg_primary"],
            selectbackground=self.colors["selection"],
        )
        self.chapters_listbox.pack(fill=tk.X, pady=5)

        # Detect chapters
        self._detect_chapters()

        # === Metadata Frame ===
        metadata_frame = ttk.LabelFrame(
            main_frame,
            text="🏷️ Metadata (Optional)",
            style="Dark.TLabelframe",
            padding="10",
        )
        metadata_frame.pack(fill=tk.X, pady=(0, 10))

        # Title
        title_row = ttk.Frame(metadata_frame, style="Dark.TFrame")
        title_row.pack(fill=tk.X, pady=2)
        ttk.Label(title_row, text="Title:", style="Dark.TLabel", width=10).pack(
            side=tk.LEFT
        )
        self.meta_title_var = tk.StringVar()
        ttk.Entry(title_row, textvariable=self.meta_title_var, width=40).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Artist
        artist_row = ttk.Frame(metadata_frame, style="Dark.TFrame")
        artist_row.pack(fill=tk.X, pady=2)
        ttk.Label(artist_row, text="Artist:", style="Dark.TLabel", width=10).pack(
            side=tk.LEFT
        )
        self.meta_artist_var = tk.StringVar(value="Sherpa-ONNX TTS")
        ttk.Entry(artist_row, textvariable=self.meta_artist_var, width=40).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Album
        album_row = ttk.Frame(metadata_frame, style="Dark.TFrame")
        album_row.pack(fill=tk.X, pady=2)
        ttk.Label(album_row, text="Album:", style="Dark.TLabel", width=10).pack(
            side=tk.LEFT
        )
        self.meta_album_var = tk.StringVar()
        ttk.Entry(album_row, textvariable=self.meta_album_var, width=40).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # === Info Display ===
        info_frame = ttk.Frame(main_frame, style="Card.TFrame", padding="8")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        duration = len(self.audio_data) / self.sample_rate
        self.info_label = tk.Label(
            info_frame,
            text=f"Duration: {duration:.1f}s | Sample Rate: {self.sample_rate} Hz | Samples: {len(self.audio_data):,}",
            bg=self.colors["bg_tertiary"],
            fg=self.colors["accent_cyan"],
            font=("Consolas", 10),
        )
        self.info_label.pack()

        # === Buttons ===
        btn_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame, text="Cancel", command=self._cancel, style="Dark.TButton"
        ).pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(
            btn_frame, text="💾 Export", command=self._export, style="Primary.TButton"
        ).pack(side=tk.RIGHT)

    def _on_format_change(self):
        """Handle format selection change"""
        fmt = self.format_var.get()
        fmt_config = self.exporter.FORMATS.get(fmt, {})

        if fmt_config.get("supports_bitrate", False):
            self.bitrate_frame.pack(
                fill=tk.X, pady=5, after=self.bitrate_frame.master.winfo_children()[0]
            )
            # Set default bitrate for format
            default_br = fmt_config.get("default_bitrate", 192)
            self.bitrate_var.set(str(default_br))
            # Update available bitrates
            if "bitrates" in fmt_config:
                self.bitrate_combo["values"] = [
                    str(br) for br in fmt_config["bitrates"]
                ]
        else:
            self.bitrate_frame.pack_forget()

    def _on_split_mode_change(self):
        """Handle split mode change"""
        mode = self.split_mode_var.get()

        # Hide all split settings first
        self.silence_settings_frame.pack_forget()
        self.chapters_frame.pack_forget()

        if mode == "silence":
            self.silence_settings_frame.pack(fill=tk.X, pady=(10, 0))
        elif mode == "chapters":
            self.chapters_frame.pack(fill=tk.X, pady=(10, 0))

    def _detect_chapters(self):
        """Detect chapters from original text"""
        if not self.original_text:
            return

        chapters = self.exporter.detect_chapters_from_text(self.original_text)

        self.chapters_listbox.delete(0, tk.END)
        self.detected_chapters = chapters

        if chapters:
            for i, ch in enumerate(chapters, 1):
                self.chapters_listbox.insert(tk.END, f"{i}. {ch['title']}")
        else:
            self.chapters_listbox.insert(tk.END, "(No chapters detected)")
            self.detected_chapters = []

    def _preview_silence_splits(self):
        """Preview silence detection results"""
        try:
            min_silence = int(self.min_silence_var.get())
            threshold = int(self.silence_thresh_var.get())

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

            messagebox.showinfo("Silence Detection Preview", msg, parent=self.dialog)

        except Exception as e:
            messagebox.showerror(
                "Error", f"Preview failed: {str(e)}", parent=self.dialog
            )

    def _cancel(self):
        """Cancel and close dialog"""
        self.result = None
        self.dialog.destroy()

    def _export(self):
        """Perform export with selected options"""
        # Build options dict
        self.result = {
            "format": self.format_var.get(),
            "target_sample_rate": int(self.sample_rate_var.get()),
            "normalize": self.normalize_var.get(),
            "split_mode": self.split_mode_var.get(),
            "metadata": {
                "title": self.meta_title_var.get(),
                "artist": self.meta_artist_var.get(),
                "album": self.meta_album_var.get(),
            },
        }

        # Add format-specific options
        fmt = self.format_var.get()
        if self.exporter.FORMATS.get(fmt, {}).get("supports_bitrate", False):
            self.result["bitrate"] = int(self.bitrate_var.get())

        # Add split options
        if self.result["split_mode"] == "silence":
            self.result["silence_settings"] = {
                "min_silence_len": int(self.min_silence_var.get()),
                "silence_thresh": int(self.silence_thresh_var.get()),
            }
        elif self.result["split_mode"] == "chapters":
            self.result["chapters"] = self.detected_chapters

        self.dialog.destroy()

    def show(self):
        """Show dialog and wait for result"""
        self.dialog.wait_window()
        return self.result
