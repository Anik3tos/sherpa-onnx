#!/usr/bin/env python3

from tts_gui import common as tgc

class TTSGuiVoiceMixin:
    def check_available_voices(self):
        """Check which voice models are available on the system"""
        self.available_voice_configs = {}

        for config_id, config in tgc.VOICE_CONFIGS.items():
            model_files = config["model_files"]
            available = True

            # Check if required model files exist
            if config["model_type"] == "kokoro":
                required_files = ["model", "voices", "tokens", "data_dir"]
            elif config["model_type"] == "matcha":
                required_files = ["acoustic_model", "vocoder", "tokens", "data_dir"]
            elif config["model_type"] == "vits":
                required_files = ["model", "tokens", "data_dir"]
            else:
                continue

            for file_key in required_files:
                if file_key in model_files:
                    file_path = model_files[file_key]
                    if file_key.endswith("_dir"):
                        # Check if directory exists
                        if not tgc.os.path.isdir(file_path):
                            available = False
                            break
                    else:
                        # Check if file exists
                        if not tgc.os.path.isfile(file_path):
                            available = False
                            break

            if available:
                self.available_voice_configs[config_id] = config
                self.log_status(f"✓ Found voice model: {config['name']}")
            else:
                self.log_status(f"⚠ Voice model not available: {config['name']}")

    def populate_voice_selections(self):
        """Populate the voice selection dropdowns"""
        if not self.available_voice_configs:
            self.log_status("⚠ No voice models found. Please download TTS models.")
            return

        # Populate model selection
        model_options = []
        for config_id, config in self.available_voice_configs.items():
            quality_indicator = "⭐" * (4 if config["quality"] == "excellent" else
                                     3 if config["quality"] == "very_high" else 2)
            model_options.append(f"{quality_indicator} {config['name']}")

        self.voice_model_combo['values'] = model_options

        # Select first available model
        if model_options:
            self.voice_model_combo.current(0)
            self.on_voice_model_changed(None)

    def on_voice_model_changed(self, event):
        """Handle voice model selection change"""
        if not self.voice_model_combo.get():
            return

        # Find the selected config
        selected_text = self.voice_model_combo.get()
        selected_config = None
        selected_config_id = None

        for config_id, config in self.available_voice_configs.items():
            quality_indicator = "⭐" * (4 if config["quality"] == "excellent" else
                                     3 if config["quality"] == "very_high" else 2)
            if f"{quality_indicator} {config['name']}" == selected_text:
                selected_config = config
                selected_config_id = config_id
                break

        if not selected_config:
            return

        self.selected_voice_config = (selected_config_id, selected_config)

        # Update speaker selection
        speaker_options = []
        for speaker_id, speaker_info in selected_config["speakers"].items():
            gender_icon = "👩" if speaker_info["gender"] == "female" else "👨"
            accent_text = f" ({speaker_info['accent']})" if speaker_info.get('accent') else ""
            speaker_options.append(f"{gender_icon} {speaker_info['name']}{accent_text} - {speaker_info['description']}")

        self.speaker_combo['values'] = speaker_options

        # Select first speaker
        if speaker_options:
            self.speaker_combo.current(0)
            self.on_speaker_changed(None)

        # Update model info
        info_text = f"{selected_config['description']} | Quality: {selected_config['quality'].replace('_', ' ').title()}"
        self.voice_info_label.config(text=info_text)

    def on_speaker_changed(self, event):
        """Handle speaker selection change"""
        if not self.speaker_combo.get() or not self.selected_voice_config:
            return

        # Update voice info with speaker details
        selected_speaker_text = self.speaker_combo.get()
        config_id, config = self.selected_voice_config

        # Find selected speaker
        for speaker_id, speaker_info in config["speakers"].items():
            gender_icon = "👩" if speaker_info["gender"] == "female" else "👨"
            accent_text = f" ({speaker_info['accent']})" if speaker_info.get('accent') else ""
            if f"{gender_icon} {speaker_info['name']}{accent_text} - {speaker_info['description']}" == selected_speaker_text:
                info_text = (f"{config['description']} | "
                           f"Speaker: {speaker_info['name']} ({speaker_info['gender']}) | "
                           f"Quality: {config['quality'].replace('_', ' ').title()}")
                self.voice_info_label.config(text=info_text)
                break

    def preview_voice(self):
        """Preview the selected voice with sample text"""
        if not self.selected_voice_config:
            tgc.messagebox.showwarning("No Voice Selected", "Please select a voice model and speaker first.")
            return

        # Use a short sample text for preview
        sample_text = "Hello! This is a preview of the selected voice. How does it sound?"

        # Temporarily store current text
        current_text = self.text_widget.get(1.0, tgc.tk.END).strip()

        # Set sample text
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, sample_text)

        # Generate speech
        self.generate_speech()

        # Restore original text after a short delay
        def restore_text():
            tgc.time.sleep(0.5)  # Wait for generation to start
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, current_text)

        tgc.threading.Thread(target=restore_text, daemon=True).start()

    def preload_models(self):
        """Preload available models in background for better performance"""
        if not self.available_voice_configs:
            self.log_status("⚠ No voice models available for preloading")
            return

        def preload_thread():
            try:
                self.model_loading_in_progress = True
                self.log_status("🚀 Preloading available voice models in background...")

                # Preload the first few available models for better performance
                preload_count = 0
                max_preload = 2  # Limit to avoid excessive memory usage

                for config_id, config in self.available_voice_configs.items():
                    if preload_count >= max_preload:
                        break

                    try:
                        self.load_voice_model(config_id, config)
                        preload_count += 1
                    except Exception as e:
                        self.log_status(f"⚠ Failed to preload {config['name']}: {str(e)}")

                if preload_count > 0:
                    self.log_status(f"✓ Preloaded {preload_count} voice models - ready for fast generation!")
                else:
                    self.log_status("⚠ No models could be preloaded")

            except Exception as e:
                self.log_status(f"⚠ Model preloading failed: {str(e)}")
            finally:
                self.model_loading_in_progress = False

        # Start preloading in background thread
        preload_thread = tgc.threading.Thread(target=preload_thread, daemon=True)
        preload_thread.start()

    def load_voice_model(self, config_id, config):
        """Load a voice model based on configuration with robust error handling"""
        if config_id in self.tts_models:
            return self.tts_models[config_id]

        self.log_status(f"Loading {config['name']}...")

        # Wrap everything in try-catch to prevent crashes
        try:
            model_files = config["model_files"]

            if config["model_type"] == "matcha":
                tts_config = tgc.sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                            acoustic_model=model_files["acoustic_model"],
                            vocoder=model_files["vocoder"],
                            lexicon=model_files.get("lexicon", ""),
                            tokens=model_files["tokens"],
                            data_dir=model_files["data_dir"],
                        ),
                        num_threads=2,
                        debug=False,
                        provider="cpu",
                    ),
                    max_num_sentences=1,
                )

            elif config["model_type"] == "kokoro":
                # Handle multi-lingual Kokoro models properly
                lexicon_path = model_files.get("lexicon", "")
                dict_dir_path = model_files.get("dict_dir", "")

                # For multi-lingual models, ensure lexicon and dict_dir are properly set
                if "multi-lang" in config_id or "enhanced" in config_id:
                    if not lexicon_path:
                        # Use default English lexicon for multi-lang models
                        lexicon_path = f"{model_files['model'].split('/')[0]}/lexicon-us-en.txt"
                    if not dict_dir_path:
                        dict_dir_path = f"{model_files['model'].split('/')[0]}/dict"

                try:
                    tts_config = tgc.sherpa_onnx.OfflineTtsConfig(
                        model=tgc.sherpa_onnx.OfflineTtsModelConfig(
                            kokoro=tgc.sherpa_onnx.OfflineTtsKokoroModelConfig(
                                model=model_files["model"],
                                voices=model_files["voices"],
                                tokens=model_files["tokens"],
                                lexicon=lexicon_path,
                                data_dir=model_files["data_dir"],
                                dict_dir=dict_dir_path,
                            ),
                            num_threads=2,
                            debug=False,
                            provider="cpu",
                        ),
                        max_num_sentences=1,
                    )
                except Exception as kokoro_error:
                    # If multi-lingual setup fails, try without dict_dir (fallback to single-lang mode)
                    self.log_status(f"⚠ Multi-lingual setup failed for {config['name']}, trying single-language mode...")
                    try:
                        tts_config = tgc.sherpa_onnx.OfflineTtsConfig(
                            model=tgc.sherpa_onnx.OfflineTtsModelConfig(
                                kokoro=tgc.sherpa_onnx.OfflineTtsKokoroModelConfig(
                                    model=model_files["model"],
                                    voices=model_files["voices"],
                                    tokens=model_files["tokens"],
                                    lexicon="",  # Empty for single-language
                                    data_dir=model_files["data_dir"],
                                    dict_dir="",  # Empty for single-language
                                ),
                                num_threads=2,
                                debug=False,
                                provider="cpu",
                            ),
                            max_num_sentences=1,
                        )
                        self.log_status(f"✓ {config['name']} loaded in single-language mode")
                    except Exception as fallback_error:
                        raise kokoro_error  # Re-raise original error if fallback also fails

            elif config["model_type"] == "vits":
                tts_config = tgc.sherpa_onnx.OfflineTtsConfig(
                    model=tgc.sherpa_onnx.OfflineTtsModelConfig(
                        vits=tgc.sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=model_files["model"],
                            lexicon=model_files.get("lexicon", ""),
                            tokens=model_files["tokens"],
                            data_dir=model_files["data_dir"],
                            dict_dir=model_files.get("dict_dir", ""),
                        ),
                        num_threads=2,
                        debug=False,
                        provider="cpu",
                    ),
                    max_num_sentences=1,
                )
            else:
                raise ValueError(f"Unsupported model type: {config['model_type']}")

            # Try to create the TTS model with comprehensive error handling
            try:
                # Wrap model creation in additional safety
                import subprocess
                import tempfile

                # All models should work now with proper fixes applied
                # But wrap in additional safety for any remaining issues

                tts_model = tgc.sherpa_onnx.OfflineTts(tts_config)
                self.tts_models[config_id] = tts_model
                self.log_status(f"✓ {config['name']} loaded successfully")
                return tts_model

            except SystemExit:
                # Catch system exit calls that cause crashes
                self.log_status(f"✗ {config['name']} caused system exit - model incompatible")
                self.log_status("💡 This model may require different parameters or files")
                return None
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                self.log_status(f"✗ {config['name']} loading interrupted by user")
                return None
            except Exception as model_error:
                # Handle model creation errors
                error_msg = str(model_error)
                if "multi-lingual" in error_msg or "lexicon" in error_msg or "dict-dir" in error_msg:
                    self.log_status(f"✗ {config['name']} requires multi-lingual setup - skipping for stability")
                    self.log_status("💡 This model needs specific lexicon and dictionary configuration")
                elif "version" in error_msg and "Kokoro" in error_msg:
                    self.log_status(f"✗ {config['name']} version mismatch - skipping for stability")
                    self.log_status("💡 This model version may not be compatible with current sherpa-onnx")
                else:
                    self.log_status(f"✗ {config['name']} model creation failed: {error_msg}")
                return None

        except Exception as e:
            error_msg = str(e)
            if "phontab" in error_msg or "espeak-ng-data" in error_msg:
                self.log_status(f"✗ Failed to load {config['name']}: Missing espeak-ng-data files")
                self.log_status("💡 Run fix_espeak_data.py to download missing language data")
            elif "dict" in error_msg and "utf8" in error_msg:
                self.log_status(f"✗ Failed to load {config['name']}: Missing dictionary files")
                self.log_status("💡 Run fix_kokoro_dict.py to download missing dictionary files")
            else:
                self.log_status(f"✗ Failed to load {config['name']}: {error_msg}")
            return None

    def get_current_speaker_id(self):
        """Get the currently selected speaker ID with proper mapping for multi-speaker models"""
        if not self.selected_voice_config or not self.speaker_combo.get():
            return 0

        config_id, config = self.selected_voice_config
        selected_speaker_text = self.speaker_combo.get()

        # Find the speaker ID from the display text
        for speaker_id, speaker_info in config["speakers"].items():
            gender_icon = "👩" if speaker_info["gender"] == "female" else "👨"
            accent_text = f" ({speaker_info['accent']})" if speaker_info.get('accent') else ""
            if f"{gender_icon} {speaker_info['name']}{accent_text} - {speaker_info['description']}" == selected_speaker_text:
                # All configurations now use sequential speaker IDs (0, 1, 2, etc.)
                # so we can use the speaker_id directly
                self.log_status(f"🎯 Selected speaker: {speaker_info['name']} ({speaker_info['gender']}) - ID {speaker_id}")
                return speaker_id

        return 0
