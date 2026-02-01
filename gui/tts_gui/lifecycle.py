#!/usr/bin/env python3

import os
import pygame


class TTSGuiLifecycleMixin:
    def cleanup(self):
        """Cleanup resources on exit"""
        try:
            # Save audio cache
            self.audio_cache.save_cache()

            # Stop any playing audio
            if self.current_sound:
                self.current_sound.stop()

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
