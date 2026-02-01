#!/usr/bin/env python3

import os
import time
import tkinter as tk
import numpy as np
import pygame


class TTSGuiPlaybackMixin:
    def update_follow_along_highlight(self, current_time):
        """Update the word highlighting based on current playback time"""
        if not self.follow_along_enabled.get():
            return
        
        if not self.follow_along_manager.word_timings:
            return
        
        # Get current word info
        word_info = self.follow_along_manager.get_word_at_time(current_time)
        
        if word_info is None:
            return
        
        word_index, word, start_idx, end_idx = word_info
        
        # Update the current word label
        self.follow_along_word_label.config(text=word)
        
        # Update progress
        total_words = len(self.follow_along_manager.word_timings)
        self.follow_along_progress_label.config(text=f"{word_index + 1} / {total_words} words")
        
        # Only update text highlighting if the word index changed
        if word_index != self.follow_along_manager.current_word_index:
            self.follow_along_manager.current_word_index = word_index
            self._highlight_current_word(word_index, start_idx, end_idx)

    def _highlight_current_word(self, word_index, start_idx, end_idx):
        """Apply highlighting to the current word in the text widget"""
        try:
            # Remove all existing highlight tags
            self.text_widget.tag_remove('current_word', '1.0', tk.END)
            self.text_widget.tag_remove('spoken_word', '1.0', tk.END)
            
            # Convert character index to text widget position
            start_pos = f"1.0+{start_idx}c"
            end_pos = f"1.0+{end_idx}c"
            
            # Mark all words before current as spoken
            if word_index > 0 and self.follow_along_manager.word_timings:
                first_word = self.follow_along_manager.word_timings[0]
                prev_word = self.follow_along_manager.word_timings[word_index - 1]
                first_start = f"1.0+{first_word[3]}c"
                prev_end = f"1.0+{prev_word[4]}c"
                self.text_widget.tag_add('spoken_word', first_start, prev_end)
            
            # Highlight current word
            self.text_widget.tag_add('current_word', start_pos, end_pos)
            
            # Scroll to make the current word visible
            self.text_widget.see(start_pos)
            
        except tk.TclError:
            # Handle any text widget errors gracefully
            pass

    def clear_follow_along_highlight(self):
        """Clear all follow-along highlighting"""
        try:
            self.text_widget.tag_remove('current_word', '1.0', tk.END)
            self.text_widget.tag_remove('spoken_word', '1.0', tk.END)
            self.follow_along_word_label.config(text="---")
            self.follow_along_progress_label.config(text="0 / 0 words")
            self.follow_along_manager.reset()
        except tk.TclError:
            pass

    def setup_follow_along_for_audio(self, text, audio_duration, generation_speed=1.0):
        """Setup follow-along word timings for the generated audio"""
        if not self.follow_along_enabled.get():
            return
        
        # Store the text used for generation
        self.generated_text_for_follow_along = text
        
        # Calculate word timings, accounting for generation speed
        self.follow_along_manager.calculate_word_timings(text, audio_duration, generation_speed)
        
        # Update initial display
        total_words = len(self.follow_along_manager.word_timings)
        self.follow_along_progress_label.config(text=f"0 / {total_words} words")
        self.follow_along_word_label.config(text="Ready")
        
        speed_info = f" (speed: {generation_speed}x)" if generation_speed != 1.0 else ""
        self.log_status(f"📖 Follow-along ready: {total_words} words mapped{speed_info}")

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

                # Update follow-along word highlighting
                self.update_follow_along_highlight(current_time)

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
        self.clear_follow_along_highlight()
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
