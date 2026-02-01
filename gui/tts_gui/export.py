#!/usr/bin/env python3

import os
import shutil
import re
import tkinter as tk
from tkinter import messagebox, filedialog

from tts_gui.export_dialog import ExportOptionsDialog


class TTSGuiExportMixin:
    def save_audio(self):
        """Save audio to file - opens advanced export options dialog"""
        if self.audio_data is None or len(self.audio_data) == 0:
            messagebox.showwarning(
                "No Audio", "No audio to save. Generate speech first."
            )
            return

        # Get original text for chapter detection
        original_text = self.text_widget.get(1.0, tk.END).strip()

        # Show advanced export dialog
        dialog = ExportOptionsDialog(
            self.root,
            self.audio_exporter,
            self.audio_data,
            self.sample_rate,
            self.colors,
            original_text,
        )

        export_options = dialog.show()

        if not export_options:
            return  # User cancelled

        # Determine save location based on split mode
        split_mode = export_options.get("split_mode", "none")
        fmt = export_options.get("format", "wav")
        fmt_config = self.audio_exporter.FORMATS.get(fmt, {})
        ext = fmt_config.get("extension", ".wav")

        if split_mode == "none":
            # Single file export
            file_path = filedialog.asksaveasfilename(
                title="Save Audio As",
                defaultextension=ext,
                initialdir="audio_output",
                filetypes=[(f"{fmt.upper()} files", f"*{ext}"), ("All files", "*.*")],
            )

            if file_path:
                self._export_single_file(file_path, export_options)
        else:
            # Multiple file export - choose directory
            output_dir = filedialog.askdirectory(
                title="Select Output Directory for Split Files",
                initialdir="audio_output",
            )

            if output_dir:
                self._export_split_files(output_dir, export_options)

    def _export_single_file(self, file_path, options):
        """Export audio as a single file"""
        try:
            fmt = options.get("format", "wav")

            export_options = {
                "target_sample_rate": options.get(
                    "target_sample_rate", self.sample_rate
                ),
                "normalize": options.get("normalize", False),
                "metadata": options.get("metadata", {}),
            }

            if "bitrate" in options:
                export_options["bitrate"] = options["bitrate"]

            self.log_status(f"💾 Exporting to {fmt.upper()}...")

            success, message, output_path = self.audio_exporter.export(
                self.audio_data, self.sample_rate, file_path, fmt, export_options
            )

            if success:
                file_size = os.path.getsize(output_path) / 1024  # KB
                self.log_status(f"✓ {message}")
                self.log_status(f"  File size: {file_size:.1f} KB")
                messagebox.showinfo(
                    "Export Complete", f"Audio exported successfully!\n\n{output_path}"
                )
            else:
                self.log_status(f"✗ Export failed: {message}")
                messagebox.showerror("Export Failed", message)

        except Exception as e:
            self.log_status(f"✗ Export error: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export audio:\n{str(e)}")

    def _export_split_files(self, output_dir, options):
        """Export audio as multiple split files"""
        try:
            split_mode = options.get("split_mode", "silence")
            fmt = options.get("format", "wav")

            # Prepare segments based on split mode
            if split_mode == "silence":
                silence_settings = options.get("silence_settings", {})
                min_silence = silence_settings.get("min_silence_len", 500)
                threshold = silence_settings.get("silence_thresh", -40)

                self.log_status(
                    f"🔍 Detecting silence regions (threshold: {threshold}dB, min: {min_silence}ms)..."
                )

                segments = self.audio_exporter.split_by_silence(
                    self.audio_data,
                    self.sample_rate,
                    min_silence_len=min_silence,
                    silence_thresh=threshold,
                )

                self.log_status(f"  Found {len(segments)} segment(s)")

                # Convert to (title, audio) format
                audio_segments = [
                    (f"Track {i:02d}", seg) for i, seg in enumerate(segments, 1)
                ]

            elif split_mode == "chapters":
                chapters = options.get("chapters", [])

                if not chapters:
                    self.log_status(
                        "⚠ No chapters detected, exporting as single file..."
                    )
                    self._export_single_file(
                        os.path.join(output_dir, f"audio.{fmt}"), options
                    )
                    return

                self.log_status(f"📚 Splitting by {len(chapters)} chapter(s)...")

                # For chapter splitting, we need to estimate timing
                # Calculate approximate timing based on text position
                audio_duration_ms = len(self.audio_data) * 1000 // self.sample_rate

                chapter_markers = []
                current_time = 0

                for i, ch in enumerate(chapters):
                    chapter_markers.append(
                        {
                            "start_ms": current_time,
                            "title": ch.get("title", f"Chapter {i+1}"),
                        }
                    )
                    # Estimate time for this chapter (proportional to text)
                    if i < len(chapters) - 1:
                        # Use roughly equal distribution for simplicity
                        current_time += audio_duration_ms // len(chapters)

                audio_segments = self.audio_exporter.split_by_chapters(
                    self.audio_data, self.sample_rate, chapter_markers
                )
            else:
                # Fallback - single segment
                audio_segments = [("Full Audio", self.audio_data)]

            # Export all segments
            export_options = {
                "sample_rate": self.sample_rate,
                "target_sample_rate": options.get(
                    "target_sample_rate", self.sample_rate
                ),
                "normalize": options.get("normalize", False),
                "metadata": options.get("metadata", {}),
            }

            if "bitrate" in options:
                export_options["bitrate"] = options["bitrate"]

            base_name = options.get("metadata", {}).get("title", "audio") or "audio"
            base_name = re.sub(r'[<>:"/\\|?*]', "", base_name)[:30]

            self.log_status(
                f"💾 Exporting {len(audio_segments)} file(s) to {fmt.upper()}..."
            )

            results = self.audio_exporter.export_multiple_tracks(
                audio_segments, output_dir, base_name, fmt, export_options
            )

            # Report results
            success_count = sum(1 for r in results if r[0])
            fail_count = len(results) - success_count

            for success, message, path in results:
                if success:
                    file_size = os.path.getsize(path) / 1024 if path else 0
                    self.log_status(
                        f"  ✓ {os.path.basename(path)} ({file_size:.1f} KB)"
                    )
                else:
                    self.log_status(f"  ✗ {message}")

            self.log_status(
                f"✓ Export complete: {success_count} succeeded, {fail_count} failed"
            )

            if success_count > 0:
                messagebox.showinfo(
                    "Export Complete",
                    f"Successfully exported {success_count} file(s) to:\n{output_dir}",
                )
            else:
                messagebox.showerror(
                    "Export Failed",
                    "All exports failed. Check the status log for details.",
                )

        except Exception as e:
            self.log_status(f"✗ Split export error: {str(e)}")
            messagebox.showerror(
                "Export Error", f"Failed to export split files:\n{str(e)}"
            )

    def save_audio_quick(self):
        """Quick save to WAV (original simple export behavior)"""
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            file_path = filedialog.asksaveasfilename(
                title="Quick Save as WAV",
                defaultextension=".wav",
                initialdir="audio_output",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            )

            if file_path:
                try:
                    shutil.copy2(self.current_audio_file, file_path)
                    self.log_status(f"💾 Audio saved to: {file_path}")
                except Exception as e:
                    self.log_status(f"✗ Error saving audio: {str(e)}")
