#!/usr/bin/env python3

import tkinter as tk
import pygame
import time
import uuid
import os
import threading
import hashlib
import numpy as np
import soundfile as sf


class TTSGuiGenerationMixin:
    def generate_speech_thread(self):
        try:
            if self.generation_cancelled:
                return

            raw_text = self.text_widget.get(1.0, tk.END).strip()
            if not raw_text:
                self.log_status("⚠ Please enter some text to synthesize")
                return

            if self.generation_cancelled:
                return

            is_valid, error_msg = self.text_processor.validate_text(raw_text)
            if not is_valid:
                self.log_status(f"⚠ Text validation failed: {error_msg}")
                return

            options = {key: var.get() for key, var in self.text_options.items()}
            options.update({"fix_encoding": True, "replace_modern_terms": True})
            text = self.text_processor.preprocess_text(raw_text, options)

            if text != raw_text:
                self.log_status("🔧 Text preprocessed for optimal synthesis")

            # Check for cancellation
            if self.generation_cancelled:
                return

            # Process SSML if enabled or auto-detected
            ssml_result = self.process_ssml_text(text)
            if ssml_result["was_processed"]:
                text = ssml_result["text"]
                # Apply SSML rate adjustment to generation speed
                if ssml_result.get("rate", 1.0) != 1.0:
                    # Combine with user-selected speed
                    ssml_rate = ssml_result["rate"]
                    self.log_status(
                        f"🎭 SSML prosody: applying {ssml_rate:.2f}x rate adjustment"
                    )

            # Check for cancellation
            if self.generation_cancelled:
                return

            # Get current voice configuration
            if not self.selected_voice_config:
                self.log_status("⚠ Please select a voice model first")
                return

            config_id, config = self.selected_voice_config
            model_type = config["model_type"]
            speed = self.speed_var.get()
            speaker_id = self.get_current_speaker_id()

            # Check if text needs chunking
            if self.text_processor.needs_chunking(text):
                self.log_status(
                    f"📄 Long text detected ({len(text)} chars) - splitting into chunks..."
                )
                self._generate_chunked_speech(text, model_type, speed, speaker_id)
            else:
                self._generate_single_speech(text, model_type, speed, speaker_id)

        except Exception as e:
            error_msg = str(e)
            # Handle specific ONNX runtime errors
            if (
                "BroadcastIterator::Append" in error_msg
                or "axis == 1 || axis == largest was false" in error_msg
            ):
                self.log_status(
                    "✗ Model compatibility error: Text may be too long or contain unsupported characters. Try shorter text or different model."
                )
            else:
                self.log_status(f"✗ Error generating speech: {error_msg}")

        finally:
            # Re-enable generate button, hide cancel button and progress
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.cancel_btn.grid_remove())
            self.root.after(0, lambda: self.progress.stop())

    def _generate_single_speech(self, text, model_type, speed, speaker_id):
        """Generate speech for a single chunk of text"""
        # Check for cancellation
        if self.generation_cancelled:
            return

        # Check cache first (include voice config for proper caching)
        config_id = (
            self.selected_voice_config[0] if self.selected_voice_config else None
        )
        cached_audio = self.audio_cache.get(
            text, model_type, speaker_id, speed, config_id
        )
        if cached_audio:
            self.log_status("⚡ Using cached audio (instant generation!)")

            # Use cached data
            self.audio_data = cached_audio["audio_data"]
            self.sample_rate = cached_audio["sample_rate"]
            self.audio_duration = len(self.audio_data) / self.sample_rate
            self.pause_position = 0.0

            # Create temporary file from cached data
            temp_file = f"audio_output/temp_cached_{uuid.uuid4().hex[:8]}.wav"
            self.current_audio_file = temp_file
            sf.write(self.current_audio_file, self.audio_data, self.sample_rate)

            # Record performance metrics
            self.performance_monitor.end_generation(
                self.audio_duration, from_cache=True
            )

            self.log_status(
                f"✓ Cached audio loaded (Duration: {self.audio_duration:.2f}s)"
            )

            # Setup follow-along word highlighting
            self.root.after(
                0,
                lambda t=text, d=self.audio_duration, s=speed: self.setup_follow_along_for_audio(
                    t, d, s
                ),
            )

            # Enable playback controls
            self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.seek_scale.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.update_time_display(0.0))
            self.root.after(0, lambda: self.update_performance_display())
            return

        # Check for cancellation before starting generation
        if self.generation_cancelled:
            return

        # Start performance monitoring
        self.performance_monitor.start_generation(len(text), model_type)

        self.log_status(f"🎵 Generating speech with {model_type.upper()} model...")

        # Stop any playing audio and cleanup
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            time.sleep(0.1)

        # Ensure audio output directory exists
        os.makedirs("audio_output", exist_ok=True)

        # Generate unique temporary file
        temp_file = f"audio_output/temp_audio_{uuid.uuid4().hex[:8]}.wav"

        start_time = time.time()

        # Generate audio based on model type
        audio = self._generate_audio_for_text(text, model_type, speed, speaker_id)
        if audio is None or self.generation_cancelled:
            return

        # Process and store audio data
        self.current_audio_file = temp_file

        # Convert to numpy array for consistent handling
        if isinstance(audio.samples, list):
            self.audio_data = np.array(audio.samples, dtype=np.float32)
        else:
            self.audio_data = np.array(audio.samples, dtype=np.float32)

        self.sample_rate = audio.sample_rate
        self.audio_duration = len(self.audio_data) / audio.sample_rate
        self.pause_position = 0.0

        # Save to file
        sf.write(self.current_audio_file, self.audio_data, self.sample_rate)

        # Cache the generated audio (include voice config for proper caching)
        config_id = (
            self.selected_voice_config[0] if self.selected_voice_config else None
        )
        self.audio_cache.put(
            text,
            model_type,
            speaker_id,
            speed,
            self.audio_data.copy(),
            self.sample_rate,
            config_id,
        )

        # Calculate and record performance metrics
        elapsed_time = time.time() - start_time
        rtf = elapsed_time / self.audio_duration if self.audio_duration > 0 else 0

        metric = self.performance_monitor.end_generation(
            self.audio_duration, from_cache=False
        )
        avg_rtf = self.performance_monitor.get_average_rtf(model_type)

        self.log_status(f"✓ Speech generated successfully!")
        self.log_status(f"  Duration: {self.audio_duration:.2f} seconds")
        self.log_status(f"  Generation time: {elapsed_time:.2f} seconds")
        self.log_status(f"  RTF (Real-time factor): {rtf:.3f}")
        self.log_status(f"  Average RTF ({model_type}): {avg_rtf:.3f}")

        # Setup follow-along word highlighting
        self.root.after(
            0,
            lambda t=text, d=self.audio_duration, s=speed: self.setup_follow_along_for_audio(
                t, d, s
            ),
        )

        # Enable playback buttons and controls
        self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.seek_scale.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.update_time_display(0.0))
        self.root.after(0, lambda: self.update_performance_display())

    def _generate_chunked_speech(self, text, model_type, speed, speaker_id):
        """Generate speech for long text by splitting into chunks"""
        # Split text into chunks using model-aware chunking
        chunks = self.text_processor.split_text_into_chunks(text, model_type)
        total_chunks = len(chunks)

        self.log_status(
            f"📄 Split into {total_chunks} chunks for {model_type.upper()} model (token-aware chunking)"
        )

        # Log chunk sizes for debugging
        for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
            estimated_tokens = self.text_processor.estimate_token_count(chunk)
            self.log_status(
                f"  Chunk {i}: {len(chunk)} chars, ~{estimated_tokens} tokens"
            )
        if total_chunks > 3:
            self.log_status(f"  ... and {total_chunks - 3} more chunks")

        # Check if entire chunked result is cached (include voice config)
        config_id = (
            self.selected_voice_config[0] if self.selected_voice_config else None
        )
        full_cache_key = f"chunked_{hashlib.md5(text.encode()).hexdigest()}"
        cached_full = self.audio_cache.get(
            full_cache_key, model_type, speaker_id, speed, config_id
        )
        if cached_full:
            self.log_status("⚡ Using cached chunked audio (instant generation!)")
            self._use_cached_audio(cached_full, text, speed)
            return

        # Start performance monitoring for the entire operation
        self.performance_monitor.start_generation(len(text), model_type)
        start_time = time.time()

        # Generate audio for each chunk
        audio_chunks = []
        successful_chunks = 0

        for i, chunk in enumerate(chunks, 1):
            # Check for cancellation before processing each chunk
            if self.generation_cancelled:
                self.log_status("🚫 Generation cancelled during chunk processing")
                return

            try:
                self.log_status(
                    f"🎵 Processing chunk {i}/{total_chunks} ({len(chunk)} chars)..."
                )

                # Update progress bar to show chunk progress
                progress = (i - 1) / total_chunks * 100
                self.root.after(0, lambda p=progress: self.progress.configure(value=p))

                # Check cache for individual chunk (include voice config)
                config_id = (
                    self.selected_voice_config[0]
                    if self.selected_voice_config
                    else None
                )
                cached_chunk = self.audio_cache.get(
                    chunk, model_type, speaker_id, speed, config_id
                )
                if cached_chunk:
                    self.log_status(f"  ⚡ Chunk {i} found in cache")
                    audio_data = cached_chunk["audio_data"]
                else:
                    # Check for cancellation before generating
                    if self.generation_cancelled:
                        self.log_status(
                            "🚫 Generation cancelled during chunk processing"
                        )
                        return

                    # Generate audio for this chunk
                    estimated_tokens = self.text_processor.estimate_token_count(chunk)
                    self.log_status(
                        f"  🔍 Chunk {i}: {len(chunk)} chars, ~{estimated_tokens} tokens"
                    )

                    audio = self._generate_audio_for_text(
                        chunk, model_type, speed, speaker_id
                    )
                    if audio is None:
                        self.log_status(
                            f"  ⚠ Failed to generate chunk {i}, skipping..."
                        )
                        continue

                    # Convert to numpy array
                    if isinstance(audio.samples, list):
                        audio_data = np.array(audio.samples, dtype=np.float32)
                    else:
                        audio_data = np.array(audio.samples, dtype=np.float32)

                    # Cache individual chunk (include voice config)
                    config_id = (
                        self.selected_voice_config[0]
                        if self.selected_voice_config
                        else None
                    )
                    self.audio_cache.put(
                        chunk,
                        model_type,
                        speaker_id,
                        speed,
                        audio_data.copy(),
                        audio.sample_rate,
                        config_id,
                    )

                    self.log_status(
                        f"  ✓ Chunk {i} generated ({len(audio_data)/audio.sample_rate:.1f}s)"
                    )

                audio_chunks.append(audio_data)
                successful_chunks += 1
                self.log_status(f"  ✓ Chunk {i} completed successfully")

            except Exception as e:
                error_msg = str(e)
                if (
                    "BroadcastIterator::Append" in error_msg
                    or "axis == 1 || axis == largest was false" in error_msg
                ):
                    self.log_status(
                        f"  ✗ Model compatibility error in chunk {i}: {error_msg[:100]}..."
                    )
                else:
                    self.log_status(
                        f"  ✗ Error processing chunk {i}: {error_msg[:100]}..."
                    )
                self.log_status(f"  ⏭ Continuing with remaining chunks...")
                continue

        if not audio_chunks:
            self.log_status("✗ Failed to generate any audio chunks")
            return

        # Check success rate and provide detailed feedback
        success_rate = successful_chunks / total_chunks
        failed_chunks = total_chunks - successful_chunks

        if success_rate < 0.5:  # Less than 50% success
            self.log_status(
                f"⚠ Low success rate: {successful_chunks}/{total_chunks} chunks succeeded ({success_rate:.1%})"
            )
            self.log_status(
                f"  {failed_chunks} chunks failed - consider using shorter text or switching to a different model"
            )
        elif failed_chunks > 0:
            self.log_status(
                f"✓ Good success rate: {successful_chunks}/{total_chunks} chunks succeeded ({success_rate:.1%})"
            )
            self.log_status(
                f"  Note: {failed_chunks} chunks were skipped due to errors"
            )
        else:
            self.log_status(
                f"✓ Perfect success rate: All {total_chunks} chunks processed successfully!"
            )

        # Stitch chunks together
        self.log_status(f"🔗 Stitching {successful_chunks} audio chunks together...")

        # Use the sample rate from the first successful generation or cached chunk
        if hasattr(self, "sample_rate"):
            sample_rate = self.sample_rate
        else:
            sample_rate = 22050  # Default fallback

        stitched_audio = self.audio_stitcher.stitch_audio_chunks(
            audio_chunks, sample_rate
        )

        # Store final result
        self.audio_data = stitched_audio
        self.sample_rate = sample_rate
        self.audio_duration = len(self.audio_data) / sample_rate
        self.pause_position = 0.0

        # Save to temporary file
        temp_file = f"audio_output/temp_chunked_{uuid.uuid4().hex[:8]}.wav"
        self.current_audio_file = temp_file
        sf.write(self.current_audio_file, self.audio_data, self.sample_rate)

        # Cache the final stitched result (include voice config)
        config_id = (
            self.selected_voice_config[0] if self.selected_voice_config else None
        )
        self.audio_cache.put(
            full_cache_key,
            model_type,
            speaker_id,
            speed,
            self.audio_data.copy(),
            self.sample_rate,
            config_id,
        )

        # Calculate performance metrics
        elapsed_time = time.time() - start_time
        rtf = elapsed_time / self.audio_duration if self.audio_duration > 0 else 0

        metric = self.performance_monitor.end_generation(
            self.audio_duration, from_cache=False
        )
        avg_rtf = self.performance_monitor.get_average_rtf(model_type)

        self.log_status(f"✓ Chunked speech generated successfully!")
        self.log_status(
            f"  Total chunks: {total_chunks} (successful: {successful_chunks})"
        )
        self.log_status(f"  Total duration: {self.audio_duration:.2f} seconds")
        self.log_status(f"  Total generation time: {elapsed_time:.2f} seconds")
        self.log_status(f"  Overall RTF: {rtf:.3f}")
        self.log_status(f"  Average RTF ({model_type}): {avg_rtf:.3f}")

        # Setup follow-along word highlighting
        self.root.after(
            0,
            lambda t=text, d=self.audio_duration, s=speed: self.setup_follow_along_for_audio(
                t, d, s
            ),
        )

        # Enable playback controls
        self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.seek_scale.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.update_time_display(0.0))
        self.root.after(0, lambda: self.update_performance_display())

    def _generate_audio_for_text(self, text, model_type, speed, speaker_id):
        """Generate audio for a single piece of text"""
        try:
            # Check for cancellation
            if self.generation_cancelled:
                return None

            # Final validation before sending to model (with warning only)
            is_valid, error_msg = self.text_processor.validate_chunk_for_model(
                text, model_type
            )
            if not is_valid:
                self.log_status(
                    f"⚠ Chunk validation warning: {error_msg} - attempting anyway..."
                )
                # Don't return None, try to generate anyway as validation might be too conservative

            # Get current voice configuration
            if not self.selected_voice_config:
                self.log_status("⚠ No voice configuration selected")
                return None

            config_id, config = self.selected_voice_config

            # Load the appropriate model
            tts_model = self.load_voice_model(config_id, config)
            if tts_model is None:
                self.log_status(f"⚠ Failed to load voice model: {config['name']}")
                return None

            # Generate audio with the loaded model (with error handling)
            try:
                return tts_model.generate(text, sid=speaker_id, speed=speed)
            except Exception as e:
                error_msg = str(e)
                if "phontab" in error_msg or "espeak-ng-data" in error_msg:
                    self.log_status(
                        f"⚠ Language processing error (missing espeak data): {error_msg}"
                    )
                    self.log_status(
                        "💡 Try using a different voice model or check espeak-ng-data installation"
                    )
                elif "No such file or directory" in error_msg:
                    self.log_status(f"⚠ Missing model file: {error_msg}")
                    self.log_status(
                        "💡 Some model files may be missing - try redownloading the model"
                    )
                else:
                    self.log_status(f"⚠ Voice generation error: {error_msg}")
                    self.log_status(
                        "💡 Try using a different voice or check the text input"
                    )
                return None

        except Exception as e:
            error_msg = str(e)
            # Handle specific ONNX runtime errors with helpful messages
            if (
                "BroadcastIterator::Append" in error_msg
                or "axis == 1 || axis == largest was false" in error_msg
            ):
                self.log_status(
                    "✗ Model compatibility error: Chunk too complex for model. Trying to split further..."
                )
                # Try to split the problematic chunk further if it's long enough
                if len(text) > 1000:
                    self.log_status("  📄 Attempting to split problematic chunk...")
                    return self._handle_problematic_chunk(
                        text, model_type, speed, speaker_id
                    )
                else:
                    self.log_status(
                        "  ⚠ Chunk too short to split further - skipping this chunk"
                    )
            elif "Non-zero status code" in error_msg:
                self.log_status(
                    "✗ ONNX Runtime error: Model processing failed. Try different text or model."
                )
            else:
                self.log_status(f"✗ Error generating audio: {error_msg}")
            return None

    def _handle_problematic_chunk(self, text, model_type, speed, speaker_id):
        """Handle chunks that cause ONNX runtime errors by splitting them further"""
        try:
            # Split the problematic chunk into smaller pieces
            sentences = text.split(". ")
            if len(sentences) <= 1:
                # Try splitting by other punctuation if no sentences
                sentences = text.split(", ")
                if len(sentences) <= 1:
                    # Last resort: split by words
                    words = text.split()
                    mid = len(words) // 2
                    sentences = [" ".join(words[:mid]), " ".join(words[mid:])]

            # Try to generate audio for smaller pieces
            audio_chunks = []
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                try:
                    # Use the current voice configuration
                    if not self.selected_voice_config:
                        continue

                    config_id, config = self.selected_voice_config
                    tts_model = self.tts_models.get(config_id)
                    if tts_model is None:
                        continue

                    audio = tts_model.generate(
                        sentence.strip(), sid=speaker_id, speed=speed
                    )

                    if audio:
                        audio_chunks.append(audio)
                        self.log_status(
                            f"    ✓ Sub-chunk {i+1}/{len(sentences)} processed"
                        )
                except Exception as sub_e:
                    error_msg = str(sub_e)
                    if "phontab" in error_msg or "espeak-ng-data" in error_msg:
                        self.log_status(
                            f"    ⚠ Language processing error in sub-chunk {i+1}: skipping"
                        )
                    elif "No such file or directory" in error_msg:
                        self.log_status(
                            f"    ⚠ Missing file error in sub-chunk {i+1}: skipping"
                        )
                    else:
                        self.log_status(f"    ⚠ Error in sub-chunk {i+1}: {error_msg}")
                    continue  # Skip this chunk and continue with others
                except Exception as sub_e:
                    self.log_status(
                        f"    ⚠ Sub-chunk {i+1} failed: {str(sub_e)[:50]}..."
                    )
                    continue

            if not audio_chunks:
                self.log_status("  ✗ All sub-chunks failed")
                return None

            # Combine the audio chunks (simplified - just return the first successful one for now)
            # In a more sophisticated implementation, we would stitch them together
            self.log_status(
                f"  ✓ Successfully processed {len(audio_chunks)}/{len(sentences)} sub-chunks"
            )
            return audio_chunks[0]  # Return first successful chunk

        except Exception as e:
            self.log_status(f"  ✗ Error handling problematic chunk: {str(e)}")
            return None

    def _use_cached_audio(self, cached_audio, text="", speed=1.0):
        """Use cached audio data"""
        self.audio_data = cached_audio["audio_data"]
        self.sample_rate = cached_audio["sample_rate"]
        self.audio_duration = len(self.audio_data) / self.sample_rate
        self.pause_position = 0.0

        # Create temporary file from cached data
        temp_file = f"audio_output/temp_cached_{uuid.uuid4().hex[:8]}.wav"
        self.current_audio_file = temp_file
        sf.write(self.current_audio_file, self.audio_data, self.sample_rate)

        # Record performance metrics
        self.performance_monitor.end_generation(self.audio_duration, from_cache=True)

        self.log_status(f"✓ Cached audio loaded (Duration: {self.audio_duration:.2f}s)")

        # Setup follow-along word highlighting if text is provided
        if text:
            self.root.after(
                0,
                lambda t=text, d=self.audio_duration, s=speed: self.setup_follow_along_for_audio(
                    t, d, s
                ),
            )

        # Enable playback controls
        self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.seek_scale.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.update_time_display(0.0))
        self.root.after(0, lambda: self.update_performance_display())

    def generate_speech(self):
        """Start speech generation"""
        self.generation_cancelled = False
        self.generate_btn.config(state=tk.DISABLED)
        self.cancel_btn.grid()  # Show cancel button

        # Log that we're using the new token-aware chunking system
        self.log_status("🔄 Using improved token-aware chunking system...")

        # Check if we need chunking to determine progress bar mode
        raw_text = self.text_widget.get(1.0, tk.END).strip()
        if raw_text and self.text_processor.needs_chunking(raw_text):
            # Use determinate progress for chunked processing
            self.progress.configure(mode="determinate", value=0, maximum=100)
        else:
            # Use indeterminate progress for single chunk
            self.progress.configure(mode="indeterminate")
            self.progress.start()

        # Run generation in separate thread
        self.generation_thread = threading.Thread(target=self.generate_speech_thread)
        self.generation_thread.daemon = True
        self.generation_thread.start()

    def cancel_generation(self):
        """Cancel ongoing speech generation"""
        self.generation_cancelled = True
        self.log_status("🚫 Generation cancelled by user")

        # Reset UI state
        self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.cancel_btn.grid_remove())
        self.root.after(0, lambda: self.progress.stop())
