#!/usr/bin/env python3
"""
Lifecycle management mixin for the TTS GUI using PySide6 (Qt).
"""

import os
import pygame


class TTSGuiLifecycleMixin:
    """Mixin class providing lifecycle management functionality."""

    def cleanup(self):
        """Cleanup resources on exit."""
        if getattr(self, "_cleanup_done", False):
            return
        self._cleanup_done = True
        try:
            if hasattr(self, "save_config"):
                self.save_config()
            # Save audio cache
            self.audio_cache.save_cache()

            # Stop any playing audio
            if self.current_sound:
                self.current_sound.stop()

            # Request cancellation for in-flight transcription/download
            if getattr(self, "transcription_in_progress", False):
                self.transcription_cancelled = True

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=False)

            # Clean up temporary files
            if hasattr(self, "current_audio_file") and self.current_audio_file:
                try:
                    if os.path.exists(self.current_audio_file):
                        os.remove(self.current_audio_file)
                except:
                    pass

            # Quit pygame
            pygame.mixer.quit()

        except Exception:
            pass  # Ignore cleanup errors
