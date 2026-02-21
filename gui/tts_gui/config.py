#!/usr/bin/env python3
"""
Configuration persistence mixin for the TTS GUI.
"""

import json
import os
from pathlib import Path

from tts_gui.common import QTimer


class TTSGuiConfigMixin:
    """Mixin class providing configuration persistence."""

    def get_config_path(self):
        return Path.home() / ".sherpa-tts-gui.conf"

    def _get_config_save_timer(self):
        timer = getattr(self, "_config_save_timer", None)
        if timer is None:
            parent = self.main_window if hasattr(self, "main_window") else None
            timer = QTimer(parent)
            timer.setSingleShot(True)
            timer.timeout.connect(self.save_config)
            self._config_save_timer = timer
        return timer

    def schedule_config_save(self, delay_ms=500):
        """Debounce config writes so rapid UI changes don't thrash disk."""
        try:
            timer = self._get_config_save_timer()
            if timer.isActive():
                timer.stop()
            timer.start(int(delay_ms))
        except Exception:
            try:
                self.save_config()
            except Exception:
                pass

    def load_config(self):
        """Load configuration from disk and apply to runtime defaults."""
        path = self.get_config_path()
        if not path.exists():
            return

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            self.log_status(f"⚠ Failed to load config: {str(e)}")
            return

        settings = data.get("settings", data)

        text_options = settings.get("text_options")
        if isinstance(text_options, dict):
            self.text_options.update(text_options)

        self.ssml_enabled = bool(settings.get("ssml_enabled", self.ssml_enabled))
        self.ssml_auto_detect = bool(
            settings.get("ssml_auto_detect", self.ssml_auto_detect)
        )
        self.follow_along_enabled = bool(
            settings.get("follow_along_enabled", self.follow_along_enabled)
        )

        if "use_gpu" in settings:
            self.use_gpu = bool(settings.get("use_gpu"))

        self._preferred_voice_config_id = settings.get("voice_config_id")
        self._preferred_speaker_id = settings.get("speaker_id")
        self._preferred_asr_model_id = settings.get("asr_model_id")
        self.transcription_replace_text = bool(
            settings.get(
                "transcription_replace_text",
                getattr(self, "transcription_replace_text", True),
            )
        )

        if "speed_var" in settings:
            self.speed_var = float(settings.get("speed_var", self.speed_var))
        if "volume_var" in settings:
            self.volume_var = int(settings.get("volume_var", self.volume_var))
        if "playback_speed_var" in settings:
            self.playback_speed_var = float(
                settings.get("playback_speed_var", self.playback_speed_var)
            )

    def apply_config_to_ui(self):
        """Apply loaded configuration to UI widgets."""
        if hasattr(self, "ssml_enabled_cb"):
            self.ssml_enabled_cb.setChecked(self.ssml_enabled)
        if hasattr(self, "ssml_auto_detect_cb"):
            self.ssml_auto_detect_cb.setChecked(self.ssml_auto_detect)
        if hasattr(self, "follow_along_cb"):
            self.follow_along_cb.setChecked(self.follow_along_enabled)
        if hasattr(self, "gpu_checkbox"):
            self.gpu_checkbox.setChecked(self.use_gpu)
            self.update_provider_ui()

        if hasattr(self, "speed_slider"):
            self.speed_slider.setValue(int(self.speed_var * 100))
        if hasattr(self, "volume_slider"):
            self.volume_slider.setValue(int(self.volume_var))
        if hasattr(self, "playback_speed_slider"):
            self.playback_speed_slider.setValue(int(self.playback_speed_var * 100))

        preferred_voice_config_id = getattr(self, "_preferred_voice_config_id", None)
        preferred_speaker_id = getattr(self, "_preferred_speaker_id", None)

        if hasattr(self, "voice_model_combo") and preferred_voice_config_id:
            model_index = -1
            for i in range(self.voice_model_combo.count()):
                if self.voice_model_combo.itemData(i) == preferred_voice_config_id:
                    model_index = i
                    break
            if model_index >= 0:
                self.voice_model_combo.setCurrentIndex(model_index)

        if hasattr(self, "speaker_combo") and preferred_speaker_id is not None:
            speaker_index = -1
            for i in range(self.speaker_combo.count()):
                if self.speaker_combo.itemData(i) == preferred_speaker_id:
                    speaker_index = i
                    break
            if speaker_index >= 0:
                self.speaker_combo.setCurrentIndex(speaker_index)

        preferred_asr_model_id = getattr(self, "_preferred_asr_model_id", None)
        if hasattr(self, "asr_model_combo") and preferred_asr_model_id:
            asr_index = -1
            for i in range(self.asr_model_combo.count()):
                if self.asr_model_combo.itemData(i) == preferred_asr_model_id:
                    asr_index = i
                    break
            if asr_index >= 0:
                self.asr_model_combo.setCurrentIndex(asr_index)

        if hasattr(self, "transcription_replace_cb"):
            self.transcription_replace_cb.setChecked(
                bool(getattr(self, "transcription_replace_text", True))
            )

    def save_config(self):
        """Persist current settings to disk."""
        path = self.get_config_path()
        settings = {
            "text_options": dict(self.text_options),
            "ssml_enabled": bool(self.ssml_enabled),
            "ssml_auto_detect": bool(self.ssml_auto_detect),
            "follow_along_enabled": bool(self.follow_along_enabled),
            "use_gpu": bool(getattr(self, "use_gpu", False)),
            "speed_var": float(self.speed_var),
            "volume_var": int(self.volume_var),
            "playback_speed_var": float(self.playback_speed_var),
        }

        if self.selected_voice_config:
            settings["voice_config_id"] = self.selected_voice_config[0]
        elif getattr(self, "_preferred_voice_config_id", None):
            settings["voice_config_id"] = self._preferred_voice_config_id

        if hasattr(self, "speaker_combo") and self.speaker_combo.count() > 0:
            speaker_id = self.speaker_combo.itemData(
                self.speaker_combo.currentIndex()
            )
            settings["speaker_id"] = speaker_id
        elif getattr(self, "_preferred_speaker_id", None) is not None:
            settings["speaker_id"] = self._preferred_speaker_id

        if hasattr(self, "asr_model_combo") and self.asr_model_combo.count() > 0:
            asr_model_id = self.asr_model_combo.itemData(
                self.asr_model_combo.currentIndex()
            )
            settings["asr_model_id"] = asr_model_id
        elif getattr(self, "_preferred_asr_model_id", None):
            settings["asr_model_id"] = self._preferred_asr_model_id

        if hasattr(self, "transcription_replace_cb"):
            settings["transcription_replace_text"] = bool(
                self.transcription_replace_cb.isChecked()
            )
        else:
            settings["transcription_replace_text"] = bool(
                getattr(self, "transcription_replace_text", True)
            )

        payload = {
            "version": 1,
            "settings": settings,
        }

        try:
            os.makedirs(path.parent, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            self.log_status(f"⚠ Failed to save config: {str(e)}")
