#!/usr/bin/env python3
"""
Audio transcription mixin for the TTS GUI using PySide6 (Qt).
"""

import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path

import numpy as np

from tts_gui.common import QFileDialog, QMessageBox, QTimer, sf, sherpa_onnx


class TTSGuiTranscriptionMixin:
    """Mixin class providing upload-audio-to-text functionality."""

    ASR_MODELS = {
        "whisper_base_en": {
            "name": "Whisper Base.en (Recommended Accuracy)",
            "description": "Better accuracy than Tiny for English speech. Auto-download on first use.",
            "model_type": "whisper",
            "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2",
            "extract_dir": "sherpa-onnx-whisper-base.en",
            "size_hint": "199 MB",
            "language": "",
            "task": "transcribe",
            "tail_paddings": -1,
            "files": {
                "encoder": "base.en-encoder.int8.onnx",
                "decoder": "base.en-decoder.int8.onnx",
                "tokens": "base.en-tokens.txt",
            },
        },
        "whisper_small_en": {
            "name": "Whisper Small.en (Higher Accuracy, Slower)",
            "description": "Higher accuracy for English, especially tougher audio. Larger/slower model.",
            "model_type": "whisper",
            "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2",
            "extract_dir": "sherpa-onnx-whisper-small.en",
            "size_hint": "606 MB",
            "language": "",
            "task": "transcribe",
            "tail_paddings": -1,
            "files": {
                "encoder": "small.en-encoder.int8.onnx",
                "decoder": "small.en-decoder.int8.onnx",
                "tokens": "small.en-tokens.txt",
            },
        },
        "whisper_tiny_en": {
            "name": "Whisper Tiny.en (Fastest)",
            "description": "Fastest transcription, but noticeably lower accuracy than Base/Small.",
            "model_type": "whisper",
            "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2",
            "extract_dir": "sherpa-onnx-whisper-tiny.en",
            "size_hint": "113 MB",
            "language": "",
            "task": "transcribe",
            "tail_paddings": -1,
            "files": {
                "encoder": "tiny.en-encoder.int8.onnx",
                "decoder": "tiny.en-decoder.int8.onnx",
                "tokens": "tiny.en-tokens.txt",
            },
        }
    }

    def _ui_call_transcription(self, func):
        """Run a callable on the UI thread."""
        QTimer.singleShot(0, self.main_window, func)

    def populate_asr_models(self):
        """Populate ASR model selection dropdown."""
        if not hasattr(self, "asr_model_combo"):
            return

        self.asr_model_combo.clear()
        for model_id, config in self.ASR_MODELS.items():
            self.asr_model_combo.addItem(config["name"], model_id)

        preferred_model_id = getattr(self, "_preferred_asr_model_id", None)
        if preferred_model_id:
            preferred_index = -1
            for i in range(self.asr_model_combo.count()):
                if self.asr_model_combo.itemData(i) == preferred_model_id:
                    preferred_index = i
                    break
            if preferred_index >= 0:
                self.asr_model_combo.setCurrentIndex(preferred_index)

        if not preferred_model_id and self.asr_model_combo.count() > 0:
            default_model_id = "whisper_base_en"
            default_index = -1
            for i in range(self.asr_model_combo.count()):
                if self.asr_model_combo.itemData(i) == default_model_id:
                    default_index = i
                    break
            if default_index >= 0:
                self.asr_model_combo.setCurrentIndex(default_index)

        if self.asr_model_combo.count() > 0:
            self.on_asr_model_changed(self.asr_model_combo.currentIndex())

    def on_asr_model_changed(self, index):
        """Handle ASR model selection changes."""
        if index < 0 or not hasattr(self, "asr_model_combo"):
            return

        model_id = self.asr_model_combo.itemData(index)
        config = self.ASR_MODELS.get(model_id)
        if not config:
            return

        self.selected_asr_model_id = model_id

        if hasattr(self, "asr_model_info_label"):
            size_hint = config.get("size_hint", "")
            if size_hint:
                self.asr_model_info_label.setText(
                    f"{config['description']} Approx download size: {size_hint}."
                )
            else:
                self.asr_model_info_label.setText(config["description"])

        if hasattr(self, "schedule_config_save"):
            self.schedule_config_save()

    def _set_transcription_controls_enabled(self, enabled):
        """Enable/disable transcription controls."""
        if hasattr(self, "upload_audio_btn"):
            self.upload_audio_btn.setEnabled(enabled)
        if hasattr(self, "asr_model_combo"):
            self.asr_model_combo.setEnabled(enabled)
        if hasattr(self, "transcription_replace_cb"):
            self.transcription_replace_cb.setEnabled(enabled)
        if hasattr(self, "transcribe_btn"):
            has_audio = bool(getattr(self, "selected_audio_file", ""))
            self.transcribe_btn.setEnabled(enabled and has_audio)

    def upload_audio_file(self):
        """Select an audio file for transcription."""
        ffmpeg_available = bool(shutil.which("ffmpeg"))
        if ffmpeg_available:
            audio_filter = (
                "Audio files (*.wav *.flac *.ogg *.opus *.mp3 *.m4a *.aac *.mp4);;"
                "All files (*.*)"
            )
        else:
            audio_filter = "Audio files (*.wav *.flac *.ogg *.opus);;All files (*.*)"

        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Audio File",
            "",
            audio_filter,
        )

        if not file_path:
            return

        self.selected_audio_file = file_path
        file_name = os.path.basename(file_path)

        if hasattr(self, "audio_file_label"):
            self.audio_file_label.setText(f"Selected: {file_name}")
            self.audio_file_label.setToolTip(file_path)

        if hasattr(self, "transcribe_btn"):
            self.transcribe_btn.setEnabled(True)

        self.log_status(f"📂 Audio selected: {file_name}")

    def start_transcription(self):
        """Start audio transcription in a background thread."""
        if getattr(self, "transcription_in_progress", False):
            return

        if self.generation_thread and self.generation_thread.is_alive():
            QMessageBox.information(
                self.main_window,
                "Generation In Progress",
                "Please wait for speech generation to finish (or cancel it) before starting transcription.",
            )
            return

        audio_path = getattr(self, "selected_audio_file", "")
        if not audio_path:
            QMessageBox.warning(
                self.main_window,
                "No Audio Selected",
                "Please upload an audio file first.",
            )
            return

        if not os.path.isfile(audio_path):
            QMessageBox.warning(
                self.main_window,
                "Missing Audio File",
                "The selected audio file no longer exists. Please select it again.",
            )
            return

        model_id = getattr(self, "selected_asr_model_id", None)
        if not model_id:
            model_id = "whisper_base_en"

        self.transcription_in_progress = True
        self.transcription_cancelled = False
        self.generate_btn.setEnabled(False)
        self.cancel_btn.show()
        self.progress_frame.show()
        self.progress.setRange(0, 0)
        self.progress_label.setText("Preparing transcription...")
        self._set_transcription_controls_enabled(False)

        self.log_status(
            f"🎙️ Starting transcription: {os.path.basename(audio_path)}"
        )

        replace_text = (
            self.transcription_replace_cb.isChecked()
            if hasattr(self, "transcription_replace_cb")
            else True
        )

        self.transcription_thread = threading.Thread(
            target=self._transcription_thread,
            args=(audio_path, model_id, replace_text),
            daemon=True,
        )
        self.transcription_thread.start()

    def cancel_transcription(self):
        """Request cancellation of active transcription."""
        if not getattr(self, "transcription_in_progress", False):
            return

        self.transcription_cancelled = True
        self.progress.setRange(0, 0)
        self.progress_label.setText("Cancelling transcription...")
        self.log_status("🚫 Transcription cancellation requested")

    def _transcription_thread(self, audio_path, model_id, replace_text):
        """Background transcription thread function."""
        temp_wav = None
        try:
            self._ui_call_transcription(
                lambda: self.progress_label.setText("Checking ASR model...")
            )
            model_paths = self._ensure_asr_model_ready(model_id)
            if self.transcription_cancelled:
                return

            self._ui_call_transcription(
                lambda: self.progress_label.setText("Loading recognizer...")
            )
            recognizer = self.load_asr_model(model_id, model_paths)
            if self.transcription_cancelled:
                return

            self._ui_call_transcription(
                lambda: self.progress_label.setText("Loading audio...")
            )
            samples, sample_rate, temp_wav = self._read_audio_samples(audio_path)
            if self.transcription_cancelled:
                return

            duration = len(samples) / sample_rate if sample_rate > 0 else 0.0
            self.log_status(
                f"🎧 Audio ready: {duration:.1f}s at {sample_rate} Hz"
            )

            self._ui_call_transcription(
                lambda: self.progress_label.setText("Transcribing audio...")
            )
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)

            if self.transcription_cancelled:
                return

            result_text = (stream.result.text or "").strip()
            if not result_text:
                self.log_status("⚠ Transcription completed but produced no text")
                return

            self._apply_transcription_result(result_text, replace_text)
            self.log_status(
                f"✓ Transcription complete ({len(result_text)} characters)"
            )

        except Exception as e:
            err = str(e)
            if self.transcription_cancelled or "cancelled" in err.lower():
                self.log_status("🚫 Transcription cancelled")
            else:
                self.log_status(f"✗ Transcription failed: {err}")
                self._ui_call_transcription(
                    lambda m=err: QMessageBox.critical(
                        self.main_window,
                        "Transcription Error",
                        f"Failed to transcribe audio:\n{m}",
                    )
                )
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

            self.transcription_in_progress = False
            self.transcription_cancelled = False

            def _finalize_ui():
                self.generate_btn.setEnabled(True)
                self.cancel_btn.hide()
                self.progress.setRange(0, 100)
                self.progress_frame.hide()
                self.progress_label.setText("Idle")
                self._set_transcription_controls_enabled(True)

            self._ui_call_transcription(_finalize_ui)

    def _ensure_asr_model_ready(self, model_id):
        """Ensure the selected ASR model is present, downloading if needed."""
        config = self.ASR_MODELS.get(model_id)
        if not config:
            raise ValueError(f"Unknown ASR model: {model_id}")

        base_dir = Path("asr_models")
        model_dir = base_dir / config["extract_dir"]
        model_paths = self._resolve_model_files(model_dir, config)
        if model_paths:
            return model_paths

        base_dir.mkdir(parents=True, exist_ok=True)
        archive_path = base_dir / f"{config['extract_dir']}.tar.bz2"

        self.log_status(f"⬇ Downloading ASR model: {config['name']}")
        self._download_file(config["download_url"], archive_path)
        if self.transcription_cancelled:
            raise RuntimeError("Transcription cancelled")

        self._ui_call_transcription(
            lambda: self.progress_label.setText("Extracting ASR model...")
        )
        self._ui_call_transcription(lambda: self.progress.setRange(0, 0))

        self._extract_archive_safe(archive_path, base_dir)
        try:
            archive_path.unlink(missing_ok=True)
        except OSError:
            pass

        model_paths = self._resolve_model_files(model_dir, config)
        if not model_paths:
            raise RuntimeError(
                "ASR model download finished but required files are missing"
            )

        self.log_status("✓ ASR model downloaded and ready")
        return model_paths

    def _resolve_model_files(self, model_dir, config):
        """Resolve encoder/decoder/tokens files from a model directory."""
        expected = config.get("files", {})
        model_dir = Path(model_dir)

        explicit = {}
        if expected:
            explicit = {
                key: model_dir / filename for key, filename in expected.items()
            }
            if all(path.is_file() for path in explicit.values()):
                return {k: str(v) for k, v in explicit.items()}

        onnx_files = sorted(model_dir.glob("*.onnx"))
        token_files = sorted(model_dir.glob("*tokens*.txt"))

        if not onnx_files or not token_files:
            return None

        model_tag = config.get("extract_dir", "").replace("sherpa-onnx-whisper-", "")
        encoder = None
        decoder = None

        preferred_onnx = [
            f
            for f in onnx_files
            if model_tag in f.name and "int8" in f.name
        ]
        if not preferred_onnx:
            preferred_onnx = onnx_files

        for f in preferred_onnx:
            name = f.name.lower()
            if "encoder" in name and encoder is None:
                encoder = f
            if "decoder" in name and decoder is None:
                decoder = f

        if encoder is None or decoder is None:
            return None

        tokens = token_files[0]
        return {
            "encoder": str(encoder),
            "decoder": str(decoder),
            "tokens": str(tokens),
        }

    def _download_file(self, url, dest_path):
        """Download a file with progress updates."""
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=120) as response:
            total = int(response.headers.get("Content-Length", "0") or 0)

            if total > 0:
                self._ui_call_transcription(lambda: self.progress.setRange(0, 100))
                self._ui_call_transcription(lambda: self.progress.setValue(0))
            else:
                self._ui_call_transcription(lambda: self.progress.setRange(0, 0))

            downloaded = 0
            next_report = 0
            with open(dest_path, "wb") as f:
                while True:
                    if self.transcription_cancelled:
                        raise RuntimeError("Transcription cancelled")

                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        percent = int(downloaded * 100 / total)
                        if percent >= next_report:
                            self._ui_call_transcription(
                                lambda p=percent: self.progress.setValue(p)
                            )
                            self._ui_call_transcription(
                                lambda p=percent: self.progress_label.setText(
                                    f"Downloading ASR model... {p}%"
                                )
                            )
                            next_report = min(100, percent + 1)

    def _extract_archive_safe(self, archive_path, target_dir):
        """Extract tar archive with a basic path safety check."""
        target_dir = Path(target_dir).resolve()

        with tarfile.open(archive_path, "r:bz2") as tar:
            for member in tar.getmembers():
                if self.transcription_cancelled:
                    raise RuntimeError("Transcription cancelled")

                member_path = (target_dir / member.name).resolve()
                if not str(member_path).startswith(str(target_dir)):
                    raise RuntimeError("Blocked unsafe archive member path")

            tar.extractall(path=target_dir)

    def load_asr_model(self, model_id, model_paths):
        """Load an ASR recognizer and cache it for reuse."""
        cached = self.asr_models.get(model_id)
        if cached is not None:
            return cached

        config = self.ASR_MODELS.get(model_id)
        if not config:
            raise ValueError(f"Unknown ASR model: {model_id}")

        provider = self.get_provider() if hasattr(self, "get_provider") else "cpu"
        if config["model_type"] != "whisper":
            raise ValueError(f"Unsupported ASR model type: {config['model_type']}")

        self.log_status(f"🧠 Loading ASR recognizer ({provider.upper()})...")

        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=model_paths["encoder"],
                decoder=model_paths["decoder"],
                tokens=model_paths["tokens"],
                language=config.get("language", ""),
                task=config.get("task", "transcribe"),
                tail_paddings=int(config.get("tail_paddings", -1)),
                num_threads=2,
                provider=provider,
            )
        except Exception:
            if provider == "cpu":
                raise

            self.log_status(
                f"⚠ ASR provider {provider.upper()} unavailable - falling back to CPU"
            )
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=model_paths["encoder"],
                decoder=model_paths["decoder"],
                tokens=model_paths["tokens"],
                language=config.get("language", ""),
                task=config.get("task", "transcribe"),
                tail_paddings=int(config.get("tail_paddings", -1)),
                num_threads=2,
                provider="cpu",
            )

        self.asr_models[model_id] = recognizer
        self.log_status("✓ ASR recognizer loaded")
        return recognizer

    def _read_audio_samples(self, audio_path):
        """Read audio file and return mono float32 samples for ASR."""
        temp_wav = None
        try:
            data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        except Exception:
            temp_wav = self._convert_to_wav_with_ffmpeg(audio_path)
            data, sample_rate = sf.read(temp_wav, dtype="float32", always_2d=True)

        if data.size == 0:
            raise RuntimeError("Audio file is empty")

        if data.shape[1] == 1:
            samples = data[:, 0]
        else:
            samples = np.mean(data, axis=1)

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        samples = self._trim_silence(samples, sample_rate)
        if sample_rate != 16000:
            samples = self._resample_audio_linear(samples, sample_rate, 16000)
            sample_rate = 16000

        peak = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
        if peak > 1.0:
            samples = samples / peak

        return samples, int(sample_rate), temp_wav

    def _trim_silence(self, samples, sample_rate):
        """Trim long leading/trailing near-silence to improve recognition."""
        if samples.size == 0:
            return samples

        abs_samples = np.abs(samples)
        threshold = 0.003
        voiced = np.where(abs_samples > threshold)[0]
        if voiced.size == 0:
            return samples

        pad = int(sample_rate * 0.2)
        start = max(0, int(voiced[0]) - pad)
        end = min(samples.size, int(voiced[-1]) + pad)
        if end <= start:
            return samples
        return samples[start:end]

    def _resample_audio_linear(self, samples, src_rate, target_rate):
        """Simple linear resampling to target sample rate."""
        if src_rate == target_rate or samples.size == 0:
            return samples

        duration = samples.size / float(src_rate)
        target_size = int(duration * target_rate)
        if target_size <= 1:
            return samples

        x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
        resampled = np.interp(x_new, x_old, samples)
        return np.ascontiguousarray(resampled, dtype=np.float32)

    def _convert_to_wav_with_ffmpeg(self, audio_path):
        """Convert an audio file to temporary WAV using ffmpeg."""
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "Unsupported/undecodable audio format. Please install ffmpeg or use WAV/FLAC/OGG."
            )

        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="sherpa_asr_")
        os.close(fd)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            audio_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            out_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode != 0:
            try:
                os.remove(out_path)
            except OSError:
                pass
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                f"ffmpeg failed to convert audio. {stderr[:300] if stderr else ''}"
            )

        return out_path

    def _apply_transcription_result(self, result_text, replace_text):
        """Apply transcription result to the main text editor."""

        def _apply():
            existing = self.text_widget.toPlainText().strip()
            if replace_text or not existing:
                new_text = result_text
            else:
                joiner = "\n\n" if existing else ""
                new_text = f"{existing}{joiner}{result_text}"

            self.text_widget.setPlainText(new_text)
            self.on_text_change()

        self._ui_call_transcription(_apply)
