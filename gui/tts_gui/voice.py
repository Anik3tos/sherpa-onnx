#!/usr/bin/env python3
"""
Voice management mixin for the TTS GUI using PySide6 (Qt).
"""

from tts_gui.common import VOICE_CONFIGS, sherpa_onnx, os, threading, QMessageBox


class TTSGuiVoiceMixin:
    """Mixin class providing voice model management functionality."""

    def detect_available_providers(self):
        """Detect available ONNX Runtime providers."""
        try:
            import onnxruntime as ort

            return ort.get_available_providers()
        except Exception:
            return []

    def get_provider(self):
        """Resolve the provider string for sherpa-onnx based on settings."""
        providers = getattr(self, "available_onnx_providers", [])
        if getattr(self, "use_gpu", False):
            if (
                "CUDAExecutionProvider" in providers
                or "TensorrtExecutionProvider" in providers
            ):
                return "cuda"
            if "DmlExecutionProvider" in providers:
                return "directml"
            if "CoreMLExecutionProvider" in providers:
                return "coreml"
        return "cpu"

    def update_provider_ui(self):
        """Update provider status label and checkbox state."""
        if not hasattr(self, "provider_label"):
            return
        provider = self.get_provider()
        if getattr(self, "use_gpu", False) and provider == "cpu":
            self.provider_label.setText("Provider: CPU (GPU unavailable)")
        else:
            self.provider_label.setText(f"Provider: {provider.upper()}")

    def on_gpu_toggle(self, state):
        """Handle GPU acceleration toggle."""
        self.use_gpu = bool(state)
        self.update_provider_ui()

        # Clear loaded models so they reload with the new provider.
        if self.tts_models:
            self.tts_models.clear()
            self.log_status("🔄 GPU setting changed - reloading models on next use")
        if self.use_gpu and self.get_provider() == "cpu":
            self.log_status("⚠ GPU requested but not available - using CPU")

    def _find_matcha_vocoder(self, model_files):
        """Try to locate a matcha vocoder file if the configured path is missing."""
        acoustic_model = model_files.get("acoustic_model", "")
        if not acoustic_model:
            return None

        search_dir = os.path.dirname(acoustic_model)
        if not os.path.isdir(search_dir):
            return None

        try:
            candidates = [
                os.path.join(search_dir, name)
                for name in os.listdir(search_dir)
                if name.lower().endswith(".onnx") and "hifigan" in name.lower()
            ]
        except OSError:
            return None

        if candidates:
            candidates.sort()
            return candidates[0]
        return None

    def check_available_voices(self):
        """Check which voice models are available on the system."""
        self.available_voice_configs = {}

        for config_id, config in VOICE_CONFIGS.items():
            model_files = config["model_files"]
            available = True
            missing_files = []

            # Check if required model files exist
            if config["model_type"] == "kokoro":
                required_files = ["model", "voices", "tokens", "data_dir"]
            elif config["model_type"] == "matcha":
                if not os.path.isfile(model_files.get("vocoder", "")):
                    fallback_vocoder = self._find_matcha_vocoder(model_files)
                    if fallback_vocoder:
                        if not hasattr(self, "_vocoder_overrides"):
                            self._vocoder_overrides = {}
                        self._vocoder_overrides[config_id] = fallback_vocoder
                required_files = ["acoustic_model", "vocoder", "tokens", "data_dir"]
            elif config["model_type"] == "vits":
                required_files = ["model", "tokens", "data_dir"]
            else:
                continue

            for file_key in required_files:
                if file_key in model_files:
                    file_path = model_files[file_key]
                    if file_key.endswith("_dir"):
                        if not os.path.isdir(file_path):
                            missing_files.append(file_path)
                            available = False
                            break
                    else:
                        if not os.path.isfile(file_path):
                            missing_files.append(file_path)
                            available = False
                            break

            if available:
                self.available_voice_configs[config_id] = config
                self.log_status(f"✓ Found voice model: {config['name']}")
            else:
                if missing_files:
                    missing_preview = ", ".join(missing_files[:2])
                    more = ""
                    if len(missing_files) > 2:
                        more = f" (+{len(missing_files) - 2} more)"
                    self.log_status(
                        f"⚠ Voice model not available: {config['name']} (missing: {missing_preview}{more})"
                    )
                else:
                    self.log_status(f"⚠ Voice model not available: {config['name']}")

    def populate_voice_selections(self):
        """Populate the voice selection dropdowns."""
        if not self.available_voice_configs:
            self.log_status("⚠ No voice models found. Please download TTS models.")
            QMessageBox.warning(
                self.main_window,
                "No Voice Models Found",
                "No compatible TTS models were found.\n\n"
                "Check the status log for missing files and ensure the model folders are present in the repo root.",
            )
            return

        # Populate model selection
        self.voice_model_combo.clear()
        for config_id, config in self.available_voice_configs.items():
            quality_indicator = "⭐" * (
                4
                if config["quality"] == "excellent"
                else 3 if config["quality"] == "very_high" else 2
            )
            self.voice_model_combo.addItem(
                f"{quality_indicator} {config['name']}", config_id
            )

        # Select first available model
        if self.voice_model_combo.count() > 0:
            self.voice_model_combo.setCurrentIndex(0)
            self.on_voice_model_changed(0)

    def on_voice_model_changed(self, index):
        """Handle voice model selection change."""
        if index < 0 or self.voice_model_combo.count() == 0:
            return

        # Get the selected config ID from item data
        config_id = self.voice_model_combo.itemData(index)
        if config_id is None:
            return

        config = self.available_voice_configs.get(config_id)
        if not config:
            return

        self.selected_voice_config = (config_id, config)

        # Update speaker selection
        self.speaker_combo.clear()
        for speaker_id, speaker_info in config["speakers"].items():
            gender_icon = "👩" if speaker_info["gender"] == "female" else "👨"
            accent_text = (
                f" ({speaker_info['accent']})" if speaker_info.get("accent") else ""
            )
            display_text = f"{gender_icon} {speaker_info['name']}{accent_text} - {speaker_info['description']}"
            self.speaker_combo.addItem(display_text, speaker_id)

        # Select first speaker
        if self.speaker_combo.count() > 0:
            self.speaker_combo.setCurrentIndex(0)
            self.on_speaker_changed(0)

        # Update model info
        info_text = f"{config['description']} | Quality: {config['quality'].replace('_', ' ').title()}"
        self.voice_info_label.setText(info_text)

    def on_speaker_changed(self, index):
        """Handle speaker selection change."""
        if index < 0 or not self.selected_voice_config:
            return

        config_id, config = self.selected_voice_config
        speaker_id = self.speaker_combo.itemData(index)

        if speaker_id is None:
            return

        speaker_info = config["speakers"].get(speaker_id)
        if speaker_info:
            info_text = (
                f"{config['description']} | "
                f"Speaker: {speaker_info['name']} ({speaker_info['gender']}) | "
                f"Quality: {config['quality'].replace('_', ' ').title()}"
            )
            self.voice_info_label.setText(info_text)

    def preview_voice(self):
        """Preview the selected voice with sample text."""
        if not self.selected_voice_config:
            QMessageBox.warning(
                self.main_window,
                "No Voice Selected",
                "Please select a voice model and speaker first.",
            )
            return

        # Use a short sample text for preview
        sample_text = (
            "Hello! This is a preview of the selected voice. How does it sound?"
        )

        # Temporarily store current text
        current_text = self.text_widget.toPlainText()

        # Set sample text
        self.text_widget.setPlainText(sample_text)

        # Auto-play after generation for preview
        self.auto_play_after_generation = True

        # Generate speech
        self.generate_speech()

        # Restore original text after a short delay (UI thread)
        from PySide6.QtCore import QTimer

        QTimer.singleShot(500, lambda: self.text_widget.setPlainText(current_text))

    def preload_models(self):
        """Preload available models in background for better performance."""
        if not self.available_voice_configs:
            self.log_status("⚠ No voice models available for preloading")
            return

        def preload_thread():
            try:
                self.model_loading_in_progress = True
                self.log_status("🚀 Preloading available voice models in background...")

                preload_count = 0
                max_preload = 2

                for config_id, config in self.available_voice_configs.items():
                    if preload_count >= max_preload:
                        break

                    try:
                        self.load_voice_model(config_id, config)
                        preload_count += 1
                    except Exception as e:
                        self.log_status(
                            f"⚠ Failed to preload {config['name']}: {str(e)}"
                        )

                if preload_count > 0:
                    self.log_status(
                        f"✓ Preloaded {preload_count} voice models - ready for fast generation!"
                    )
                else:
                    self.log_status("⚠ No models could be preloaded")

            except Exception as e:
                self.log_status(f"⚠ Model preloading failed: {str(e)}")
            finally:
                self.model_loading_in_progress = False

        thread = threading.Thread(target=preload_thread, daemon=True)
        thread.start()

    def load_voice_model(self, config_id, config):
        """Load a voice model based on configuration with robust error handling."""
        if config_id in self.tts_models:
            return self.tts_models[config_id]

        self.log_status(f"Loading {config['name']}...")

        try:
            model_files = config["model_files"]

            if config["model_type"] == "matcha":
                if (
                    hasattr(self, "_vocoder_overrides")
                    and config_id in self._vocoder_overrides
                ):
                    model_files = dict(model_files)
                    model_files["vocoder"] = self._vocoder_overrides[config_id]
                tts_config = sherpa_onnx.OfflineTtsConfig(
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
                        provider=self.get_provider(),
                    ),
                    max_num_sentences=1,
                )

            elif config["model_type"] == "kokoro":
                lexicon_path = model_files.get("lexicon", "")
                dict_dir_path = model_files.get("dict_dir", "")

                if "multi-lang" in config_id or "enhanced" in config_id:
                    if not lexicon_path:
                        model_dir = os.path.dirname(model_files["model"])
                        lexicon_path = os.path.join(model_dir, "lexicon-us-en.txt")
                    if not dict_dir_path:
                        model_dir = os.path.dirname(model_files["model"])
                        dict_dir_path = os.path.join(model_dir, "dict")

                try:
                    tts_config = sherpa_onnx.OfflineTtsConfig(
                        model=sherpa_onnx.OfflineTtsModelConfig(
                            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                                model=model_files["model"],
                                voices=model_files["voices"],
                                tokens=model_files["tokens"],
                                lexicon=lexicon_path,
                                data_dir=model_files["data_dir"],
                                dict_dir=dict_dir_path,
                            ),
                            num_threads=2,
                            debug=False,
                            provider=self.get_provider(),
                        ),
                        max_num_sentences=1,
                    )
                except Exception as kokoro_error:
                    self.log_status(
                        f"⚠ Multi-lingual setup failed for {config['name']}, trying single-language mode..."
                    )
                    try:
                        tts_config = sherpa_onnx.OfflineTtsConfig(
                            model=sherpa_onnx.OfflineTtsModelConfig(
                                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                                    model=model_files["model"],
                                    voices=model_files["voices"],
                                    tokens=model_files["tokens"],
                                    lexicon="",
                                    data_dir=model_files["data_dir"],
                                    dict_dir="",
                                ),
                                num_threads=2,
                                debug=False,
                                provider=self.get_provider(),
                            ),
                            max_num_sentences=1,
                        )
                        self.log_status(
                            f"✓ {config['name']} loaded in single-language mode"
                        )
                    except Exception:
                        raise kokoro_error

            elif config["model_type"] == "vits":
                tts_config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=model_files["model"],
                            lexicon=model_files.get("lexicon", ""),
                            tokens=model_files["tokens"],
                            data_dir=model_files["data_dir"],
                            dict_dir=model_files.get("dict_dir", ""),
                        ),
                        num_threads=2,
                        debug=False,
                        provider=self.get_provider(),
                    ),
                    max_num_sentences=1,
                )
            else:
                raise ValueError(f"Unsupported model type: {config['model_type']}")

            try:
                tts_model = sherpa_onnx.OfflineTts(tts_config)
                self.tts_models[config_id] = tts_model
                self.log_status(f"✓ {config['name']} loaded successfully")
                return tts_model

            except SystemExit:
                self.log_status(
                    f"✗ {config['name']} caused system exit - model incompatible"
                )
                return None
            except KeyboardInterrupt:
                self.log_status(f"✗ {config['name']} loading interrupted by user")
                return None
            except Exception as model_error:
                error_msg = str(model_error)
                if "multi-lingual" in error_msg or "lexicon" in error_msg:
                    self.log_status(
                        f"✗ {config['name']} requires multi-lingual setup - skipping"
                    )
                else:
                    self.log_status(
                        f"✗ {config['name']} model creation failed: {error_msg}"
                    )
                return None

        except Exception as e:
            error_msg = str(e)
            if "phontab" in error_msg or "espeak-ng-data" in error_msg:
                self.log_status(
                    f"✗ Failed to load {config['name']}: Missing espeak-ng-data files"
                )
            elif "dict" in error_msg and "utf8" in error_msg:
                self.log_status(
                    f"✗ Failed to load {config['name']}: Missing dictionary files"
                )
            else:
                self.log_status(f"✗ Failed to load {config['name']}: {error_msg}")
            return None

    def get_current_speaker_id(self):
        """Get the currently selected speaker ID."""
        if not self.selected_voice_config or self.speaker_combo.currentIndex() < 0:
            return 0

        speaker_id = self.speaker_combo.itemData(self.speaker_combo.currentIndex())
        if speaker_id is None:
            return 0

        config_id, config = self.selected_voice_config
        speaker_info = config["speakers"].get(speaker_id)

        if speaker_info:
            self.log_status(
                f"🎯 Selected speaker: {speaker_info['name']} ({speaker_info['gender']}) - ID {speaker_id}"
            )

        return speaker_id
