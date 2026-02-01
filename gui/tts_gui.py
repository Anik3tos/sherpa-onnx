#!/usr/bin/env python3

import atexit
import hashlib
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import time
import tkinter as tk
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree.ElementTree import ParseError

sys.path.insert(0, os.path.dirname(__file__))

try:
    import numpy as np
except Exception:

    class _NumpyStub:
        ndarray = (list,)

        def array(self, data, dtype=None):
            return list(data) if not isinstance(data, list) else data

        def linspace(self, start, stop, num):
            if num <= 1:
                return [start]
            return [start + (stop - start) * i / (num - 1) for i in range(num)]

        def interp(self, x_new, x_old, y):
            # Very simple nearest-neighbor like interpolation fallback
            res = []
            for xn in x_new:
                idx = min(range(len(x_old)), key=lambda i: abs(x_old[i] - xn))
                res.append(y[idx])
            return res

        def max(self, a):
            return max(a)

        def abs(self, a):
            if isinstance(a, list):
                return [abs(x) for x in a]
            return abs(a)

        def zeros(self, n, dtype=None):
            return [0] * n

        def concatenate(self, parts):
            res = []
            for p in parts:
                res.extend(list(p))
            return res

        float32 = float

    np = _NumpyStub()
import pygame
import sherpa_onnx
import soundfile as sf

from tts_gui.export import TTSGuiExportMixin
from tts_gui.generation import TTSGuiGenerationMixin
from tts_gui.lifecycle import TTSGuiLifecycleMixin
from tts_gui.playback import TTSGuiPlaybackMixin
from tts_gui.shortcuts import TTSGuiShortcutsMixin
from tts_gui.ssml import TTSGuiSSMLMixin
from tts_gui.text import TTSGuiTextMixin
from tts_gui.theme import TTSGuiThemeMixin
from tts_gui.ui import TTSGuiUiMixin
from tts_gui.voice import TTSGuiVoiceMixin


class AudioExporter:
    """
    Advanced Audio Export System

    Supports multiple audio formats with configurable quality settings,
    silence detection for automatic track splitting, and chapter/section markers.
    """

    # Supported audio formats with their configurations
    FORMATS = {
        "wav": {
            "name": "WAV (Lossless)",
            "extension": ".wav",
            "description": "Uncompressed audio, highest quality",
            "supports_bitrate": False,
            "default_sample_rate": 44100,
        },
        "flac": {
            "name": "FLAC (Lossless Compressed)",
            "extension": ".flac",
            "description": "Lossless compression, ~50% smaller than WAV",
            "supports_bitrate": False,
            "default_sample_rate": 44100,
            "compression_levels": list(range(9)),  # 0-8
        },
        "mp3": {
            "name": "MP3 (Lossy)",
            "extension": ".mp3",
            "description": "Universal compatibility, good compression",
            "supports_bitrate": True,
            "bitrates": [64, 96, 128, 160, 192, 224, 256, 320],
            "default_bitrate": 192,
            "default_sample_rate": 44100,
        },
        "ogg": {
            "name": "OGG Vorbis (Lossy)",
            "extension": ".ogg",
            "description": "Open format, excellent quality at lower bitrates",
            "supports_bitrate": True,
            "bitrates": [64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
            "default_bitrate": 160,
            "default_sample_rate": 44100,
        },
    }

    SAMPLE_RATES = [8000, 11025, 16000, 22050, 32000, 44100, 48000]

    def __init__(self):
        self.ffmpeg_available = self._check_ffmpeg()
        self.pydub_available = self._check_pydub()

    def _check_ffmpeg(self):
        """Check if ffmpeg is available in the system"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def _check_pydub(self):
        """Check if pydub is available"""
        try:
            from pydub import AudioSegment

            return True
        except ImportError:
            return False

    def get_available_formats(self):
        """Get list of formats that can be exported on this system"""
        available = ["wav"]  # WAV is always available via soundfile

        # FLAC is supported by soundfile
        available.append("flac")

        # MP3 and OGG require either ffmpeg or pydub
        if self.ffmpeg_available or self.pydub_available:
            available.extend(["mp3", "ogg"])

        return available

    def export(
        self, audio_data, sample_rate, output_path, format_type="wav", options=None
    ):
        """
        Export audio to specified format

        Args:
            audio_data: numpy array of audio samples
            sample_rate: source sample rate
            output_path: output file path
            format_type: one of 'wav', 'flac', 'mp3', 'ogg'
            options: dict with:
                - target_sample_rate: desired output sample rate
                - bitrate: for lossy formats (kbps)
                - compression_level: for FLAC (0-8)
                - normalize: whether to normalize audio
                - metadata: dict of metadata tags

        Returns:
            tuple (success: bool, message: str, output_path: str)
        """
        if options is None:
            options = {}

        format_config = self.FORMATS.get(format_type)
        if not format_config:
            return False, f"Unknown format: {format_type}", None

        # Ensure audio is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)

        # Normalize if requested
        if options.get("normalize", False):
            audio_data = self._normalize_audio(audio_data)

        # Resample if needed
        target_sr = options.get("target_sample_rate", sample_rate)
        if target_sr != sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, target_sr)
            sample_rate = target_sr

        # Ensure correct file extension
        if not output_path.lower().endswith(format_config["extension"]):
            output_path = output_path + format_config["extension"]

        try:
            if format_type == "wav":
                return self._export_wav(audio_data, sample_rate, output_path, options)
            elif format_type == "flac":
                return self._export_flac(audio_data, sample_rate, output_path, options)
            elif format_type in ["mp3", "ogg"]:
                return self._export_lossy(
                    audio_data, sample_rate, output_path, format_type, options
                )
            else:
                return False, f"Unsupported format: {format_type}", None
        except Exception as e:
            return False, f"Export failed: {str(e)}", None

    def _normalize_audio(self, audio_data, target_peak=0.95):
        """Normalize audio to target peak level"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data * (target_peak / max_val)
        return audio_data

    def _resample_audio(self, audio_data, src_rate, target_rate):
        """Resample audio to different sample rate"""
        if src_rate == target_rate:
            return audio_data

        # Simple linear interpolation resampling
        duration = len(audio_data) / src_rate
        target_samples = int(duration * target_rate)

        # Use numpy interpolation
        x_old = np.linspace(0, len(audio_data) - 1, len(audio_data))
        x_new = np.linspace(0, len(audio_data) - 1, target_samples)

        resampled = np.interp(x_new, x_old, audio_data)
        return resampled.astype(np.float32)

    def _export_wav(self, audio_data, sample_rate, output_path, options):
        """Export as WAV using soundfile"""
        try:
            # soundfile handles WAV export natively
            subtype = options.get("wav_subtype", "PCM_16")
            sf.write(output_path, audio_data, sample_rate, subtype=subtype)
            return True, f"Exported WAV: {output_path}", output_path
        except Exception as e:
            return False, f"WAV export failed: {str(e)}", None

    def _export_flac(self, audio_data, sample_rate, output_path, options):
        """Export as FLAC using soundfile"""
        try:
            # soundfile supports FLAC natively
            # options can be used for future compression level settings
            _ = options  # Reserved for future use (compression level, etc.)
            sf.write(output_path, audio_data, sample_rate, format="FLAC")
            return True, f"Exported FLAC: {output_path}", output_path
        except Exception as e:
            return False, f"FLAC export failed: {str(e)}", None

    def _export_lossy(self, audio_data, sample_rate, output_path, format_type, options):
        """Export as MP3 or OGG using ffmpeg or pydub"""
        format_config = self.FORMATS[format_type]
        bitrate = options.get("bitrate", format_config["default_bitrate"])

        # First, create a temporary WAV file
        temp_wav = output_path + ".temp.wav"

        try:
            # Write temporary WAV
            sf.write(temp_wav, audio_data, sample_rate, subtype="PCM_16")

            # Convert using ffmpeg or pydub
            if self.ffmpeg_available:
                success, msg = self._convert_with_ffmpeg(
                    temp_wav, output_path, format_type, bitrate, options
                )
            elif self.pydub_available:
                success, msg = self._convert_with_pydub(
                    temp_wav, output_path, format_type, bitrate, options
                )
            else:
                return (
                    False,
                    f"No encoder available for {format_type}. Install ffmpeg or pydub.",
                    None,
                )

            if success:
                return (
                    True,
                    f"Exported {format_type.upper()}: {output_path}",
                    output_path,
                )
            else:
                return False, msg, None

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    def _convert_with_ffmpeg(
        self, input_path, output_path, format_type, bitrate, options
    ):
        """Convert audio using ffmpeg"""
        try:
            cmd = ["ffmpeg", "-y", "-i", input_path]

            # Add format-specific options
            if format_type == "mp3":
                cmd.extend(["-codec:a", "libmp3lame", "-b:a", f"{bitrate}k"])
            elif format_type == "ogg":
                cmd.extend(["-codec:a", "libvorbis", "-b:a", f"{bitrate}k"])

            # Add metadata if provided
            metadata = options.get("metadata", {})
            for key, value in metadata.items():
                if value:
                    cmd.extend(["-metadata", f"{key}={value}"])

            cmd.append(output_path)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            if result.returncode == 0:
                return True, "Success"
            else:
                return False, f"ffmpeg error: {result.stderr}"

        except Exception as e:
            return False, f"ffmpeg conversion failed: {str(e)}"

    def _convert_with_pydub(
        self, input_path, output_path, format_type, bitrate, options
    ):
        """Convert audio using pydub"""
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_wav(input_path)

            # Add metadata if provided
            metadata = options.get("metadata", {})
            tags = {}
            for key, value in metadata.items():
                if value:
                    tags[key] = value

            if format_type == "mp3":
                audio.export(
                    output_path, format="mp3", bitrate=f"{bitrate}k", tags=tags
                )
            elif format_type == "ogg":
                audio.export(
                    output_path, format="ogg", bitrate=f"{bitrate}k", tags=tags
                )

            return True, "Success"

        except Exception as e:
            return False, f"pydub conversion failed: {str(e)}"

    def detect_silence(
        self,
        audio_data,
        sample_rate,
        min_silence_len=500,
        silence_thresh=-40,
        seek_step=10,
    ):
        """
        Detect silence regions in audio for automatic track splitting

        Args:
            audio_data: numpy array of audio samples
            sample_rate: sample rate
            min_silence_len: minimum length of silence in milliseconds
            silence_thresh: silence threshold in dB
            seek_step: step size in milliseconds for scanning

        Returns:
            list of tuples: [(start_ms, end_ms), ...] for each silence region
        """
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)

        # Convert parameters to samples
        min_silence_samples = int(min_silence_len * sample_rate / 1000)
        seek_samples = int(seek_step * sample_rate / 1000)

        # Convert threshold from dB to linear
        silence_thresh_linear = 10 ** (silence_thresh / 20)

        silence_regions = []
        in_silence = False
        silence_start = 0

        # Scan through audio
        for i in range(0, len(audio_data) - seek_samples, seek_samples):
            chunk = audio_data[i : i + seek_samples]
            chunk_level = np.max(np.abs(chunk))

            is_silent = chunk_level < silence_thresh_linear

            if is_silent and not in_silence:
                # Start of silence
                in_silence = True
                silence_start = i
            elif not is_silent and in_silence:
                # End of silence
                in_silence = False
                silence_end = i
                silence_duration = silence_end - silence_start

                if silence_duration >= min_silence_samples:
                    start_ms = int(silence_start * 1000 / sample_rate)
                    end_ms = int(silence_end * 1000 / sample_rate)
                    silence_regions.append((start_ms, end_ms))

        # Handle silence at the end
        if in_silence:
            silence_end = len(audio_data)
            silence_duration = silence_end - silence_start
            if silence_duration >= min_silence_samples:
                start_ms = int(silence_start * 1000 / sample_rate)
                end_ms = int(silence_end * 1000 / sample_rate)
                silence_regions.append((start_ms, end_ms))

        return silence_regions

    def split_by_silence(
        self,
        audio_data,
        sample_rate,
        min_silence_len=500,
        silence_thresh=-40,
        min_segment_len=1000,
        keep_silence=200,
    ):
        """
        Split audio at silence points

        Args:
            audio_data: numpy array of audio samples
            sample_rate: sample rate
            min_silence_len: minimum silence length for split point (ms)
            silence_thresh: silence threshold in dB
            min_segment_len: minimum segment length (ms)
            keep_silence: amount of silence to keep at edges (ms)

        Returns:
            list of numpy arrays (audio segments)
        """
        silence_regions = self.detect_silence(
            audio_data, sample_rate, min_silence_len, silence_thresh
        )

        if not silence_regions:
            return [audio_data]  # No silence found, return whole audio

        # Convert keep_silence to samples
        keep_samples = int(keep_silence * sample_rate / 1000)
        min_segment_samples = int(min_segment_len * sample_rate / 1000)

        segments = []
        prev_end = 0

        for start_ms, end_ms in silence_regions:
            # Calculate split point (middle of silence)
            split_point_samples = int((start_ms + end_ms) / 2 * sample_rate / 1000)

            # Create segment from previous end to current split point
            segment_start = max(0, prev_end - keep_samples)
            segment_end = min(len(audio_data), split_point_samples + keep_samples)

            segment = audio_data[segment_start:segment_end]

            # Only add if segment is long enough
            if len(segment) >= min_segment_samples:
                segments.append(segment)

            prev_end = segment_end

        # Add final segment
        if prev_end < len(audio_data):
            segment = audio_data[max(0, prev_end - keep_samples) :]
            if len(segment) >= min_segment_samples:
                segments.append(segment)

        return segments if segments else [audio_data]

    def split_by_chapters(self, audio_data, sample_rate, chapter_markers):
        """
        Split audio by chapter/section markers

        Args:
            audio_data: numpy array of audio samples
            sample_rate: sample rate
            chapter_markers: list of dicts with 'start_ms' and 'title' keys

        Returns:
            list of tuples: [(title, audio_segment), ...]
        """
        if not chapter_markers:
            return [("Full Audio", audio_data)]

        # Sort markers by start time
        markers = sorted(chapter_markers, key=lambda x: x.get("start_ms", 0))

        segments = []

        for i, marker in enumerate(markers):
            start_ms = marker.get("start_ms", 0)
            title = marker.get("title", f"Chapter {i+1}")

            # Determine end point
            if i + 1 < len(markers):
                end_ms = markers[i + 1]["start_ms"]
            else:
                end_ms = len(audio_data) * 1000 // sample_rate

            # Convert to samples
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)

            # Extract segment
            segment = audio_data[start_sample:end_sample]

            if len(segment) > 0:
                segments.append((title, segment))

        return segments if segments else [("Full Audio", audio_data)]

    def detect_chapters_from_text(self, text):
        """
        Detect potential chapter/section markers from text

        Args:
            text: input text to analyze

        Returns:
            list of chapter titles found
        """
        chapters = []

        # Common chapter patterns
        patterns = [
            r"^(?:Chapter|CHAPTER)\s+(\d+|[IVXLCDM]+)(?:\s*[:\-\.]\s*(.*))?$",
            r"^(?:Part|PART)\s+(\d+|[IVXLCDM]+)(?:\s*[:\-\.]\s*(.*))?$",
            r"^(?:Section|SECTION)\s+(\d+)(?:\s*[:\-\.]\s*(.*))?$",
            r"^#{1,3}\s+(.+)$",  # Markdown headers
            r"^\*\*\*+\s*$",  # Separator lines
            r"^[-=]{3,}\s*$",  # HR-style separators
        ]

        lines = text.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2 and groups[1]:
                        title = f"{groups[0]}: {groups[1]}"
                    elif groups[0]:
                        title = groups[0]
                    else:
                        title = line
                    chapters.append(
                        {
                            "title": title.strip(),
                            "line_number": i,
                            "original_line": line,
                        }
                    )
                    break

        return chapters

    def export_multiple_tracks(
        self, audio_segments, output_dir, base_name, format_type="wav", options=None
    ):
        """
        Export multiple audio segments as separate tracks

        Args:
            audio_segments: list of (title, audio_data) tuples or just audio_data arrays
            output_dir: output directory
            base_name: base file name
            format_type: audio format
            options: export options

        Returns:
            list of (success, message, output_path) tuples
        """
        if options is None:
            options = {}

        results = []
        os.makedirs(output_dir, exist_ok=True)

        for i, segment in enumerate(audio_segments, 1):
            if isinstance(segment, tuple):
                title, audio_data = segment
                # Clean title for filename
                safe_title = re.sub(r'[<>:"/\\|?*]', "", title)[:50]
                filename = f"{base_name}_{i:02d}_{safe_title}"
            else:
                audio_data = segment
                filename = f"{base_name}_{i:02d}"

            output_path = os.path.join(output_dir, filename)

            # Need sample rate for export
            sample_rate = options.get("sample_rate", 22050)

            result = self.export(
                audio_data, sample_rate, output_path, format_type, options
            )
            results.append(result)

        return results


class SSMLProcessor:
    def __init__(self):
        # Break time mappings (SSML strength to approximate pause text)
        self.break_mappings = {
            "none": "",
            "x-weak": ",",
            "weak": ", ",
            "medium": ". ",
            "strong": "... ",
            "x-strong": "...... ",
        }

        # Emphasis mappings (how to represent emphasis in plain text)
        self.emphasis_mappings = {
            "strong": ("*", "*"),  # Will be converted to natural emphasis cues
            "moderate": ("", ""),
            "reduced": ("", ""),
            "none": ("", ""),
        }

        # Say-as interpretation types
        self.say_as_types = [
            "cardinal",
            "ordinal",
            "digits",
            "fraction",
            "unit",
            "date",
            "time",
            "telephone",
            "address",
            "characters",
            "spell-out",
            "currency",
            "verbatim",
            "acronym",
            "expletive",
        ]

        # Track prosody state for nested elements
        self.prosody_stack = []
        self.current_rate = 1.0
        self.current_pitch = 1.0
        self.current_volume = 1.0

    def is_ssml(self, text):
        """Check if text appears to be SSML markup"""
        text = text.strip()
        # Check for common SSML patterns
        if text.startswith("<speak") or text.startswith("<?xml"):
            return True
        # Check for any SSML-like tags
        ssml_tags = [
            "<speak",
            "<break",
            "<emphasis",
            "<prosody",
            "<say-as",
            "<phoneme",
            "<sub",
            "<voice",
            "<p>",
            "<s>",
        ]
        return any(tag in text.lower() for tag in ssml_tags)

    def parse_ssml(self, ssml_text):
        result = {
            "text": "",
            "rate": 1.0,
            "segments": [],
            "errors": [],
            "has_prosody_changes": False,
        }

        # Reset prosody state
        self.prosody_stack = []
        self.current_rate = 1.0
        self.current_pitch = 1.0
        self.current_volume = 1.0

        # Ensure SSML has a root element
        ssml_text = ssml_text.strip()
        if not ssml_text.startswith("<speak"):
            # Wrap in speak tags if not present
            if not ssml_text.startswith("<?xml"):
                ssml_text = f"<speak>{ssml_text}</speak>"

        # Handle XML declaration if present
        if ssml_text.startswith("<?xml"):
            # Find the end of XML declaration and process the rest
            decl_end = ssml_text.find("?>")
            if decl_end != -1:
                ssml_text = ssml_text[decl_end + 2 :].strip()
                if not ssml_text.startswith("<speak"):
                    ssml_text = f"<speak>{ssml_text}</speak>"

        try:
            # Parse XML
            root = ET.fromstring(ssml_text)

            # Process the tree
            processed_text, segments = self._process_element(root)

            result["text"] = self._clean_text(processed_text)
            result["segments"] = segments

            # Calculate average rate from segments
            if segments:
                rates = [s["rate"] for s in segments if s.get("rate")]
                if rates:
                    result["rate"] = sum(rates) / len(rates)
                    if result["rate"] != 1.0:
                        result["has_prosody_changes"] = True

        except ParseError as e:
            result["errors"].append(f"XML parsing error: {str(e)}")
            # Fall back to stripping tags
            result["text"] = self._strip_tags(ssml_text)
        except Exception as e:
            result["errors"].append(f"SSML processing error: {str(e)}")
            result["text"] = self._strip_tags(ssml_text)

        return result

    def _process_element(self, element, depth=0):
        """Recursively process an SSML element and its children"""
        segments = []
        text_parts = []

        # Handle element's text content
        if element.text:
            text_parts.append(element.text)
            segments.append(
                {
                    "text": element.text,
                    "rate": self.current_rate,
                    "pitch": self.current_pitch,
                    "volume": self.current_volume,
                }
            )

        # Process child elements
        for child in element:
            child_text, child_segments = self._process_child_element(child, depth)
            text_parts.append(child_text)
            segments.extend(child_segments)

            # Handle tail text (text after the child element)
            if child.tail:
                text_parts.append(child.tail)
                segments.append(
                    {
                        "text": child.tail,
                        "rate": self.current_rate,
                        "pitch": self.current_pitch,
                        "volume": self.current_volume,
                    }
                )

        return "".join(text_parts), segments

    def _process_child_element(self, element, depth):
        """Process a specific SSML element based on its tag"""
        tag = element.tag.lower()

        # Remove namespace if present
        if "}" in tag:
            tag = tag.split("}")[1]

        # Handle different SSML elements
        if tag == "break":
            return self._handle_break(element)
        elif tag == "emphasis":
            return self._handle_emphasis(element, depth)
        elif tag == "prosody":
            return self._handle_prosody(element, depth)
        elif tag == "say-as":
            return self._handle_say_as(element, depth)
        elif tag == "phoneme":
            return self._handle_phoneme(element, depth)
        elif tag == "sub":
            return self._handle_sub(element)
        elif tag == "voice":
            return self._handle_voice(element, depth)
        elif tag in ["p", "paragraph"]:
            return self._handle_paragraph(element, depth)
        elif tag in ["s", "sentence"]:
            return self._handle_sentence(element, depth)
        elif tag == "audio":
            return self._handle_audio(element)
        elif tag == "speak":
            # Nested speak tags - just process content
            return self._process_element(element, depth + 1)
        else:
            # Unknown tag - process content anyway
            return self._process_element(element, depth + 1)

    def _handle_break(self, element):
        """Handle <break> element - insert pause"""
        time_attr = element.get("time", "")
        strength = element.get("strength", "medium")

        if time_attr:
            # Parse time value (e.g., "500ms", "1s", "1.5s")
            pause_text = self._time_to_pause(time_attr)
        else:
            # Use strength mapping
            pause_text = self.break_mappings.get(strength, ". ")

        return pause_text, [
            {
                "text": pause_text,
                "rate": self.current_rate,
                "is_break": True,
                "break_duration": time_attr or strength,
            }
        ]

    def _time_to_pause(self, time_str):
        """Convert time string to pause representation"""
        try:
            time_str = time_str.lower().strip()

            if time_str.endswith("ms"):
                ms = float(time_str[:-2])
                seconds = ms / 1000
            elif time_str.endswith("s"):
                seconds = float(time_str[:-1])
            else:
                seconds = float(time_str)

            # Convert to pause markers
            if seconds < 0.1:
                return ""
            elif seconds < 0.25:
                return ","
            elif seconds < 0.5:
                return ", "
            elif seconds < 1.0:
                return ". "
            elif seconds < 2.0:
                return "... "
            else:
                return "...... "
        except:
            return ". "

    def _handle_emphasis(self, element, depth):
        """Handle <emphasis> element"""
        level = element.get("level", "moderate")

        # Get content
        inner_text, segments = self._process_element(element, depth + 1)

        # Apply emphasis markers
        prefix, suffix = self.emphasis_mappings.get(level, ("", ""))

        # For strong emphasis, we can use natural speech patterns
        if level == "strong":
            # Add slight pause before and after for natural emphasis
            processed_text = f", {inner_text},"
        elif level == "reduced":
            # Keep text as-is for reduced emphasis
            processed_text = inner_text
        else:
            processed_text = inner_text

        # Update segments with emphasis info
        for seg in segments:
            seg["emphasis"] = level

        return processed_text, segments

    def _handle_prosody(self, element, depth):
        """Handle <prosody> element - pitch, rate, volume adjustments"""
        # Save current state
        old_rate = self.current_rate
        old_pitch = self.current_pitch
        old_volume = self.current_volume

        # Parse rate attribute
        rate = element.get("rate", "")
        if rate:
            self.current_rate = self._parse_prosody_value(rate, self.current_rate)

        # Parse pitch attribute
        pitch = element.get("pitch", "")
        if pitch:
            self.current_pitch = self._parse_prosody_value(pitch, self.current_pitch)

        # Parse volume attribute
        volume = element.get("volume", "")
        if volume:
            self.current_volume = self._parse_prosody_value(volume, self.current_volume)

        # Process content with new prosody settings
        inner_text, segments = self._process_element(element, depth + 1)

        # Restore previous state
        self.current_rate = old_rate
        self.current_pitch = old_pitch
        self.current_volume = old_volume

        return inner_text, segments

    def _parse_prosody_value(self, value, current):
        """Parse prosody attribute value (percentage, relative, or keyword)"""
        value = value.lower().strip()

        # Keyword values
        keywords = {
            "x-slow": 0.5,
            "slow": 0.75,
            "medium": 1.0,
            "fast": 1.25,
            "x-fast": 1.5,
            "x-low": 0.5,
            "low": 0.75,
            "high": 1.25,
            "x-high": 1.5,
            "silent": 0,
            "soft": 0.5,
            "loud": 1.5,
            "default": 1.0,
        }

        if value in keywords:
            return keywords[value]

        try:
            # Percentage (e.g., "150%", "+20%", "-10%")
            if value.endswith("%"):
                pct_str = value[:-1]
                if pct_str.startswith("+"):
                    return current * (1 + float(pct_str[1:]) / 100)
                elif pct_str.startswith("-"):
                    return current * (1 - float(pct_str[1:]) / 100)
                else:
                    return float(pct_str) / 100

            # Relative values (e.g., "+2st", "-3st" for semitones)
            if "st" in value:
                st_val = float(value.replace("st", "").replace("+", ""))
                # Convert semitones to multiplier (rough approximation)
                return current * (2 ** (st_val / 12))

            # Plain number
            return float(value)
        except:
            return current

    def _handle_say_as(self, element, depth):
        """Handle <say-as> element - pronunciation control"""
        interpret_as = element.get("interpret-as", "")
        format_attr = element.get("format", "")
        detail = element.get("detail", "")

        # Get the text content
        inner_text, segments = self._process_element(element, depth + 1)
        text = inner_text.strip()

        # Process based on interpret-as type
        if interpret_as == "characters" or interpret_as == "spell-out":
            # Spell out each character
            processed = " ".join(text)
        elif interpret_as == "digits":
            # Read each digit separately
            processed = " ".join(text)
        elif interpret_as == "ordinal":
            # Convert to ordinal (1 -> first, etc.)
            processed = self._number_to_ordinal(text)
        elif interpret_as == "cardinal":
            # Keep as number
            processed = text
        elif interpret_as == "telephone":
            # Format for telephone reading
            processed = self._format_telephone(text)
        elif interpret_as == "date":
            # Format date for reading
            processed = self._format_date(text, format_attr)
        elif interpret_as == "time":
            # Format time for reading
            processed = self._format_time(text, format_attr)
        elif interpret_as == "currency":
            # Format currency for reading
            processed = text  # Keep as-is, TTS usually handles this
        elif interpret_as == "verbatim":
            # Spell out exactly
            processed = " ".join(text)
        elif interpret_as == "acronym":
            # Spell out as acronym
            processed = " ".join(text.upper())
        elif interpret_as == "expletive":
            # Replace with beep or blank
            processed = "[expletive]"
        else:
            processed = text

        # Update segments
        for seg in segments:
            seg["interpret_as"] = interpret_as

        return processed, segments

    def _number_to_ordinal(self, text):
        """Convert number to ordinal text"""
        try:
            num = int(text)
            if 10 <= num % 100 <= 20:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(num % 10, "th")
            return f"{num}{suffix}"
        except:
            return text

    def _format_telephone(self, text):
        """Format telephone number for TTS reading"""
        # Extract digits only
        digits = "".join(c for c in text if c.isdigit())
        # Add spaces for natural reading
        return " ".join(digits)

    def _format_date(self, text, format_attr):
        """Format date for TTS reading"""
        # Basic date formatting - keep as-is mostly
        # Could expand this with format parsing
        return text

    def _format_time(self, text, format_attr):
        """Format time for TTS reading"""
        # Basic time formatting - keep as-is mostly
        return text

    def _handle_phoneme(self, element, depth):
        """Handle <phoneme> element - phonetic pronunciation"""
        alphabet = element.get("alphabet", "ipa")  # 'ipa' or 'x-sampa'
        ph = element.get("ph", "")

        # Get the text content (this is the fallback text)
        inner_text, segments = self._process_element(element, depth + 1)

        # For now, we use the original text since most TTS engines
        # don't support phoneme injection. The phoneme info is stored
        # in segments for engines that might support it.
        for seg in segments:
            seg["phoneme"] = ph
            seg["phoneme_alphabet"] = alphabet

        return inner_text, segments

    def _handle_sub(self, element):
        """Handle <sub> element - text substitution"""
        alias = element.get("alias", "")
        original = element.text or ""

        # Use the alias for TTS, original is what's displayed
        text_to_speak = alias if alias else original

        return text_to_speak, [
            {
                "text": text_to_speak,
                "original": original,
                "rate": self.current_rate,
                "is_substitution": True,
            }
        ]

    def _handle_voice(self, element, depth):
        """Handle <voice> element - voice selection hints"""
        # Voice attributes (informational - actual voice selection is in GUI)
        voice_name = element.get("name", "")
        gender = element.get("gender", "")
        age = element.get("age", "")
        variant = element.get("variant", "")

        # Process content
        inner_text, segments = self._process_element(element, depth + 1)

        # Add voice hints to segments
        for seg in segments:
            seg["voice_hint"] = {
                "name": voice_name,
                "gender": gender,
                "age": age,
                "variant": variant,
            }

        return inner_text, segments

    def _handle_paragraph(self, element, depth):
        """Handle <p> paragraph element"""
        inner_text, segments = self._process_element(element, depth + 1)

        # Add paragraph break after
        processed = inner_text.strip() + "\n\n"

        return processed, segments

    def _handle_sentence(self, element, depth):
        """Handle <s> sentence element"""
        inner_text, segments = self._process_element(element, depth + 1)

        # Ensure sentence ends properly
        text = inner_text.strip()
        if text and text[-1] not in ".!?":
            text += "."

        return text + " ", segments

    def _handle_audio(self, element):
        """Handle <audio> element - audio clips (informational only)"""
        src = element.get("src", "")
        # Audio elements are not supported - return description
        desc = element.text or f"[Audio: {src}]"
        return desc, [{"text": desc, "rate": self.current_rate, "is_audio_ref": True}]

    def _clean_text(self, text):
        """Clean up processed text"""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove multiple punctuation
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        # Clean up spaces around punctuation
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"([.,!?])\s*([.,!?])", r"\1", text)
        return text.strip()

    def _strip_tags(self, text):
        """Strip all XML/SSML tags from text as fallback"""
        # Remove XML declaration
        text = re.sub(r"<\?xml[^>]*\?>", "", text)
        # Remove all tags
        text = re.sub(r"<[^>]+>", "", text)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_ssml_template(self, template_name="basic"):
        """Get SSML template for user reference"""
        templates = {
            "basic": """<speak>
    Hello, this is a basic SSML example.
    <break time="500ms"/>
    With a pause in the middle.
</speak>""",
            "emphasis": """<speak>
    This is <emphasis level="strong">very important</emphasis> information.
    But this is <emphasis level="reduced">less important</emphasis>.
</speak>""",
            "prosody": """<speak>
    <prosody rate="slow">Speaking slowly for clarity.</prosody>
    <break time="300ms"/>
    <prosody rate="fast">Now speaking quickly!</prosody>
    <break time="300ms"/>
    <prosody pitch="high">With a higher pitch.</prosody>
    <prosody pitch="low">And a lower pitch.</prosody>
</speak>""",
            "say_as": """<speak>
    The number <say-as interpret-as="cardinal">42</say-as> is spelled 
    <say-as interpret-as="spell-out">42</say-as>.
    Call <say-as interpret-as="telephone">555-1234</say-as>.
    Today is <say-as interpret-as="date">2024-01-15</say-as>.
</speak>""",
            "full_example": """<speak>
    <p>Welcome to the SSML demonstration.</p>
    
    <s>This shows various SSML features.</s>
    
    <s><emphasis level="strong">Emphasis</emphasis> makes text stand out.</s>
    
    <s>Add pauses: short<break time="200ms"/>medium<break time="500ms"/>long<break time="1s"/>done.</s>
    
    <s><prosody rate="80%">Slower speech is clearer.</prosody></s>
    <s><prosody rate="120%">Faster speech saves time.</prosody></s>
    
    <s>Spell it out: <say-as interpret-as="characters">ABC</say-as></s>
    
    <s>Use <sub alias="Speech Synthesis Markup Language">SSML</sub> for control.</s>
</speak>""",
        }

        return templates.get(template_name, templates["basic"])

    def get_ssml_reference(self):
        """Get SSML reference documentation"""
        return """
SSML Quick Reference
====================

ROOT ELEMENT:
<speak>...</speak>
    Wraps all SSML content (optional - added automatically if missing)

PAUSES:
<break time="500ms"/>     - Pause for specific time (ms or s)
<break strength="medium"/> - Pause by strength: none, x-weak, weak, medium, strong, x-strong

EMPHASIS:
<emphasis level="strong">text</emphasis>
    Levels: strong, moderate (default), reduced, none

PROSODY (Speech Rate/Pitch/Volume):
<prosody rate="slow">text</prosody>
    Rate: x-slow, slow, medium, fast, x-fast, or percentage (80%, 120%)
<prosody pitch="high">text</prosody>
    Pitch: x-low, low, medium, high, x-high, or +/-Hz or +/-st (semitones)
<prosody volume="loud">text</prosody>
    Volume: silent, soft, medium, loud, default, or percentage

PRONUNCIATION:
<say-as interpret-as="type">content</say-as>
    Types: cardinal, ordinal, characters, spell-out, digits, telephone, date, time, currency

<phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme>
    Phonetic pronunciation hint (IPA or X-SAMPA alphabet)

<sub alias="World Wide Web">WWW</sub>
    Substitution - speak alias instead of text

STRUCTURE:
<p>Paragraph text</p>        - Paragraph (adds pause after)
<s>Sentence text.</s>        - Sentence marker

VOICE HINTS:
<voice name="en-US-female">text</voice>
    Voice selection hint (name, gender, age attributes)
"""


class TextProcessor:
    """Handles text preprocessing and validation"""

    def __init__(self):
        self.max_length = 100000  # Maximum total text length (doubled)
        self.min_length = 1  # Minimum text length
        self.chunk_size = 8000  # Target chunk size for long texts (characters)
        self.max_chunk_size = (
            9500  # Maximum chunk size before forced split (characters)
        )

        # Model-specific token limits (conservative but not overly restrictive)
        self.model_token_limits = {
            "matcha": 700,  # Conservative limit for Matcha-TTS (model max is ~1000)
            "kokoro": 1100,  # Conservative for Kokoro
        }

        # Approximate characters per token (varies by language/content)
        self.chars_per_token = 3.5  # More conservative estimate for English

    def validate_text(self, text):
        """Validate input text and return (is_valid, error_message)"""
        if not text or not text.strip():
            return False, "Text cannot be empty"

        if len(text) > self.max_length:
            return False, f"Text too long (max {self.max_length} characters)"

        if len(text.strip()) < self.min_length:
            return False, f"Text too short (min {self.min_length} characters)"

        return True, ""

    def estimate_token_count(self, text):
        """Estimate token count for text (conservative approximation)"""
        # More conservative estimation for better accuracy
        words = len(text.split())
        punctuation = sum(1 for c in text if c in ".,!?;:()[]{}\"-'")
        numbers = sum(1 for c in text if c.isdigit())
        special_chars = sum(
            1 for c in text if not c.isalnum() and c not in " .,!?;:()[]{}\"-'"
        )

        # Conservative token estimation with safety margins
        estimated_tokens = int(
            (words * 1.3)
            + (punctuation * 0.8)
            + (numbers * 0.3)
            + (special_chars * 0.5)
        )

        # Add extra safety margin for complex text
        if len(text) > 1000:
            estimated_tokens = int(
                estimated_tokens * 1.2
            )  # 20% extra buffer for long text

        return max(estimated_tokens, len(text) // 3)  # Minimum 1 token per 3 characters

    def get_model_safe_chunk_size(self, model_type):
        """Get safe chunk size for specific model based on token limits"""
        token_limit = self.model_token_limits.get(model_type, 600)
        # Convert token limit to character limit with aggressive safety margin
        safe_char_limit = int(
            token_limit * self.chars_per_token * 0.6
        )  # 40% safety margin
        return min(safe_char_limit, self.chunk_size)

    def validate_chunk_for_model(self, text, model_type):
        """Validate that a chunk is safe for the specified model"""
        token_count = self.estimate_token_count(text)
        token_limit = self.model_token_limits.get(model_type, 600)

        if token_count > token_limit:
            return (
                False,
                f"Chunk has ~{token_count} tokens, exceeds {model_type} limit of {token_limit}",
            )

        return True, ""

    def preprocess_text(self, text, options=None):
        """Preprocess text based on options with enhanced character and OOV handling"""
        if not text:
            return text

        if options is None:
            options = {}

        processed = text

        # Fix encoding issues and normalize unicode characters
        if options.get("fix_encoding", True):
            import unicodedata

            processed = unicodedata.normalize("NFKD", processed)

            # Fix common encoding corruption
            encoding_fixes = {
                "â€™": "'",  # Smart apostrophe
                "â€œ": '"',  # Smart quote open
                "â€": '"',  # Smart quote close
                'â€"': "-",  # Em dash
                'â€"': "-",  # En dash
                "â€¦": "...",  # Ellipsis
                "â?T": "'",  # Corrupted apostrophe
                'â?"': '"',  # Corrupted quote
                "â?~": '"',  # Another corrupted quote
                "â?¢": "•",  # Bullet point
            }

            for corrupt, fixed in encoding_fixes.items():
                processed = processed.replace(corrupt, fixed)

            # Remove any remaining problematic characters
            processed = re.sub(r'[^\w\s\.,!?;:\'"()-]', " ", processed)

        # Handle modern terms and brand names that might be OOV
        if options.get("replace_modern_terms", True):
            modern_replacements = {
                "Netflix": "streaming service",
                "YouTube": "video platform",
                "Google": "search engine",
                "Facebook": "social media",
                "Instagram": "photo sharing app",
                "Twitter": "social platform",
                "TikTok": "video app",
                "iPhone": "smartphone",
                "iPad": "tablet",
                "MacBook": "laptop",
                "PlayStation": "gaming console",
                "Xbox": "gaming console",
                "Tesla": "electric car",
                "Uber": "ride sharing",
                "Airbnb": "home sharing",
                "COVID": "coronavirus",
                "WiFi": "wireless internet",
                "Bluetooth": "wireless connection",
                "smartphone": "mobile phone",
                "app": "application",
                "blog": "web log",
                "email": "electronic mail",
                "website": "web site",
                "online": "on the internet",
                "offline": "not connected",
                "streaming": "live transmission",
                "podcast": "audio program",
                "hashtag": "topic tag",
                "selfie": "self portrait",
                "emoji": "emotion icon",
                "meme": "internet joke",
                "viral": "widely shared",
                "trending": "popular now",
            }

            for term, replacement in modern_replacements.items():
                processed = re.sub(
                    r"\b" + re.escape(term) + r"\b",
                    replacement,
                    processed,
                    flags=re.IGNORECASE,
                )

        # Normalize whitespace
        if options.get("normalize_whitespace", True):
            processed = re.sub(r"\s+", " ", processed)
            processed = processed.strip()

        # Normalize punctuation
        if options.get("normalize_punctuation", True):
            # Replace multiple punctuation marks
            processed = re.sub(r"[.]{2,}", "...", processed)
            processed = re.sub(r"[!]{2,}", "!", processed)
            processed = re.sub(r"[?]{2,}", "?", processed)

        # Remove URLs
        if options.get("remove_urls", False):
            processed = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                processed,
            )

        # Remove email addresses
        if options.get("remove_emails", False):
            processed = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", processed
            )

        # Remove duplicate consecutive lines
        if options.get("remove_duplicates", False):
            processed = self._remove_duplicate_lines(processed)

        # Convert numbers to words
        if options.get("numbers_to_words", False):
            processed = self._convert_numbers_to_words(processed)

        # Expand abbreviations
        if options.get("expand_abbreviations", False):
            processed = self._expand_abbreviations(processed)

        # Handle acronyms
        if options.get("handle_acronyms", False):
            processed = self._handle_acronyms(processed)

        # Add pause markers for natural timing
        if options.get("add_pauses", False):
            processed = self._add_pause_markers(processed)

        return processed

    def _remove_duplicate_lines(self, text):
        """Remove consecutive duplicate lines"""
        lines = text.split("\n")
        result = []
        prev_line = None

        for line in lines:
            stripped = line.strip()
            if stripped and stripped == prev_line:
                continue  # Skip duplicate
            result.append(line)
            prev_line = stripped

        return "\n".join(result)

    def _convert_numbers_to_words(self, text):
        """Convert numbers to words for better pronunciation"""
        import re

        def number_to_words(n):
            """Convert a number to words"""
            if n == 0:
                return "zero"

            ones = [
                "",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            ]
            tens = [
                "",
                "",
                "twenty",
                "thirty",
                "forty",
                "fifty",
                "sixty",
                "seventy",
                "eighty",
                "ninety",
            ]

            if n < 20:
                return ones[n]
            elif n < 100:
                return tens[n // 10] + (" " + ones[n % 10] if n % 10 != 0 else "")
            elif n < 1000:
                return (
                    ones[n // 100]
                    + " hundred"
                    + (" " + number_to_words(n % 100) if n % 100 != 0 else "")
                )
            elif n < 1000000:
                return (
                    number_to_words(n // 1000)
                    + " thousand"
                    + (" " + number_to_words(n % 1000) if n % 1000 != 0 else "")
                )
            elif n < 1000000000:
                return (
                    number_to_words(n // 1000000)
                    + " million"
                    + (" " + number_to_words(n % 1000000) if n % 1000000 != 0 else "")
                )
            else:
                return (
                    number_to_words(n // 1000000000)
                    + " billion"
                    + (
                        " " + number_to_words(n % 1000000000)
                        if n % 1000000000 != 0
                        else ""
                    )
                )

        def convert_currency(match):
            """Convert currency like $100 to words"""
            amount = match.group(1).replace(",", "")
            try:
                value = float(amount)
                dollars = int(value)
                cents = int(round((value - dollars) * 100))

                if cents == 0:
                    return number_to_words(dollars) + " dollars"
                elif cents == 1:
                    return (
                        number_to_words(dollars)
                        + " dollars and "
                        + number_to_words(cents)
                        + " cent"
                    )
                else:
                    return (
                        number_to_words(dollars)
                        + " dollars and "
                        + number_to_words(cents)
                        + " cents"
                    )
            except:
                return match.group(0)

        def convert_number(match):
            """Convert a standalone number to words"""
            num_str = match.group(0).replace(",", "")
            try:
                num = int(num_str)
                return number_to_words(num)
            except:
                return match.group(0)

        # Convert currency first ($100, $1.50, etc)
        text = re.sub(r"\$\s*([\d,]+(?:\.\d{2})?)", convert_currency, text)

        # Convert standalone numbers (but not years like 2024)
        text = re.sub(
            r"\b([1-9]\d{0,2}(?:,\d{3})*)(?!\s*(?:BC|AD|BCE|CE))\b",
            convert_number,
            text,
        )

        return text

    def _expand_abbreviations(self, text):
        """Expand common abbreviations for better pronunciation"""
        abbreviations = {
            # Titles
            r"\bMr\.?\b": "Mister",
            r"\bMrs\.?\b": "Missus",
            r"\bMs\.?\b": "Miss",
            r"\bDr\.?\b": "Doctor",
            r"\bProf\.?\b": "Professor",
            r"\bRev\.?\b": "Reverend",
            r"\bHon\.?\b": "Honorable",
            r"\bSgt\.?\b": "Sergeant",
            r"\bCapt\.?\b": "Captain",
            r"\bLt\.?\b": "Lieutenant",
            r"\bGen\.?\b": "General",
            r"\bCol\.?\b": "Colonel",
            r"\bRep\.?\b": "Representative",
            r"\bSen\.?\b": "Senator",
            r"\bGov\.?\b": "Governor",
            r"\bPres\.?\b": "President",
            r"\bVP\b": "Vice President",
            # Common abbreviations
            r"\betc\.?\b": "et cetera",
            r"\bie\.?\b": "that is",
            r"\beg\.?\b": "for example",
            r"\bvs\.?\b": "versus",
            r"\bapt\.?\b": "apartment",
            r"\bave\.?\b": "avenue",
            r"\bblvd\.?\b": "boulevard",
            r"\bdept\.?\b": "department",
            r"\bdiv\.?\b": "division",
            r"\binst\.?\b": "institute",
            r"\bprof\.?\b": "professor",
            r"\buni\.?\b": "university",
            r"\bassoc\.?\b": "associate",
            r"\bassn\.?\b": "association",
            r"\bave\.?\b": "avenue",
            r"\bblvd\.?\b": "boulevard",
            r"\bco\.?\b": "company",
            r"\bcorp\.?\b": "corporation",
            r"\binc\.?\b": "incorporated",
            r"\bltd\.?\b": "limited",
            r"\blb\.?\b": "pound",
            r"\boz\.?\b": "ounce",
            r"\bft\.?\b": "foot",
            r"\bhrs\.?\b": "hours",
            r"\bmin\.?\b": "minutes",
            r"\bsec\.?\b": "seconds",
            r"\bapprox\.?\b": "approximately",
            r"\bavg\.?\b": "average",
            r"\bmax\.?\b": "maximum",
            r"\bmin\.?\b": "minimum",
            r"\bnbr\.?\b": "number",
            r"\bno\.?\b": "number",
            r"\bpct\.?\b": "percent",
            r"\binfo\.?\b": "information",
            r"\bmsg\.?\b": "message",
            r"\badd\.?\b": "address",
            r"\bdept\.?\b": "department",
            r"\bdiag\.?\b": "diagnosis",
            r"\bdoc\.?\b": "document",
            r"\bex\.?\b": "example",
            r"\bext\.?\b": "extension",
            r"\bfig\.?\b": "figure",
            r"\bhr\.?\b": "hour",
            r"\bid\.?\b": "identification",
            r"\bmtg\.?\b": "meeting",
            r"\bmmbr\.?\b": "member",
            r"\breq\.?\b": "request",
            r"\bresp\.?\b": "responsible",
            r"\bst\.?\b": "street",
            r"\btemp\.?\b": "temporary",
            r"\btel\.?\b": "telephone",
            r"\btrans\.?\b": "transaction",
            r"\bvol\.?\b": "volume",
            # Tech abbreviations
            r"\bapp\.?\b": "application",
            r"\bconfig\.?\b": "configuration",
            r"\binfo\.?\b": "information",
            r"\bAPI\b": "A P I",
            r"\bURL\b": "U R L",
            r"\bSQL\b": "S Q L",
            r"\bGUI\b": "G U I",
            r"\bCPU\b": "C P U",
            r"\bGPU\b": "G P U",
            r"\bRAM\b": "R A M",
            r"\bSSD\b": "S S D",
            r"\bHDD\b": "H D D",
            r"\bOS\b": "Operating System",
            r"\bIoT\b": "I O T",
            r"\bPDF\b": "P D F",
            r"\bXML\b": "X M L",
            r"\bJSON\b": "J S O N",
            r"\bWiFi\b": "Wi Fi",
            # Time abbreviations
            r"\byr\.?\b": "year",
            r"\bmo\.?\b": "month",
            r"\bday\.?\b": "day",
            r"\bhr\.?\b": "hour",
            r"\bAM\b": "A M",
            r"\bPM\b": "P M",
            r"\bam\b": "a m",
            r"\bpm\b": "p m",
            r"\bEST\b": "Eastern Standard Time",
            r"\bPST\b": "Pacific Standard Time",
            r"\bGMT\b": "Greenwich Mean Time",
            r"\bUTC\b": "Universal Time",
            # Business/Legal
            r"\bCEO\b": "C E O",
            r"\bCFO\b": "C F O",
            r"\bCTO\b": "C T O",
            r"\bCOO\b": "C O O",
            r"\bVP\b": "Vice President",
            r"\bMBA\b": "M B A",
            r"\bPhD\.?\b": "P H D",
            r"\bMD\.?\b": "M D",
            r"\bDO\.?\b": "D O",
            r"\bRN\.?\b": "R N",
            r"\bLPN\.?\b": "L P N",
            r"\bJD\.?\b": "J D",
            r"\bLLC\b": "L L C",
            r"\bLTD\b": "Limited",
            r"\bPLC\b": "P L C",
        }

        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _handle_acronyms(self, text):
        """Handle acronyms - add spaces between letters for pronunciation"""
        import re

        # Common acronyms that should be pronounced as letters (not words)
        letter_acronyms = {
            # Government agencies
            "NASA": "N A S A",
            "FBI": "F B I",
            "CIA": "C I A",
            "FDA": "F D A",
            "SEC": "S E C",
            "FCC": "F C C",
            "FTC": "F T C",
            "EPA": "E P A",
            "DOT": "D O T",
            "DOJ": "D O J",
            "DHS": "D H S",
            "VA": "V A",
            "DMV": "D M V",
            "IRS": "I R S",
            "CDC": "C D C",
            "NIH": "N I H",
            # Sports
            "NFL": "N F L",
            "NBA": "N B A",
            "MLB": "M L B",
            "NCAA": "N C A A",
            # Politics
            "GOP": "G O P",
            "DNC": "D N C",
            # Media
            "CNN": "C N N",
            "MSNBC": "M S N B C",
            "BBC": "B B C",
            "CBS": "C B S",
            "NBC": "N B C",
            "ABC": "A B C",
            "HBO": "H B O",
            "PBS": "P B S",
            "NPR": "N P R",
            # Business titles
            "CEO": "C E O",
            "CFO": "C F O",
            "CTO": "C T O",
            "COO": "C O O",
            "CIO": "C I O",
            "CSO": "C S O",
            "CMO": "C M O",
            # General
            "IOU": "I O U",
            "RSVP": "R S V P",
            "AKA": "A K A",
            "VS": "V S",
            "ETC": "E T C",
            "FYI": "F Y I",
            "ASAP": "A S A P",
            "DIY": "D I Y",
            "TBA": "T B A",
            "TBD": "T B D",
            "RIP": "R I P",
            "VIP": "V I P",
            "ID": "I D",
            "GPS": "G P S",
            "USB": "U S B",
            "LED": "L E D",
            "LCD": "L C D",
            "DVD": "D V D",
            "CD": "C D",
            "PC": "P C",
            "IT": "I T",
            "HR": "H R",
            "PR": "P R",
            "QA": "Q A",
            "R&D": "R and D",
            "B2B": "B to B",
            "B2C": "B to C",
            # === TECH & CLOUD GENERAL ===
            "SaaS": "S A A S",
            "PaaS": "P A A S",
            "IaaS": "I A A S",
            "DevOps": "Dev Ops",
            "CI/CD": "C I C D",
            "VPC": "V P C",
            "CDN": "C D N",
            "DNS": "D N S",
            "SSH": "S S H",
            "SSL": "S S L",
            "TLS": "T L S",
            "FTP": "F T P",
            "SFTP": "S F T P",
            "HTTP": "H T T P",
            "HTTPS": "H T T P S",
            "REST": "R E S T",
            "SOAP": "S O A P",
            "GraphQL": "G R A P H Q L",
            "JSON": "J S O N",
            "XML": "X M L",
            "HTML": "H T M L",
            "CSS": "C S S",
            "JS": "J S",
            "TS": "T S",
            "SQL": "S Q L",
            "NoSQL": "N O S Q L",
            "BI": "B I",
            "ERP": "E R P",
            "CRM": "C R M",
            "CMS": "C M S",
            "LMS": "L M S",
            "POS": "P O S",
            "MFA": "M F A",
            "2FA": "two F A",
            "SSO": "S S O",
            "LDAP": "L D A P",
            "AD": "A D",
            "KPI": "K P I",
            "ROI": "R O I",
            "SLA": "S L A",
            "TOS": "T O S",
            "EULA": "E U L A",
            "GDPR": "G D P R",
            "CCPA": "C C P A",
            "HIPAA": "H I P A A",
            "SOC2": "S O C 2",
            "ISO": "I S O",
            "NIST": "N I S T",
            # === AZURE SPECIFIC ===
            "ARM": "A R M",
            "AzureAD": "Azure A D",
            "AAD": "A A D",
            "EntraID": "Entra I D",
            "MFA": "M F A",
            "AppService": "App Service",
            "AKS": "A K S",
            "ACR": "A C R",
            "ADF": "A D F",
            "ADLS": "A D L S",
            "AIP": "A I P",
            "APIM": "A P I M",
            "ASR": "A S R",
            "AVD": "A V D",
            "WVD": "W V D",
            "Bicep": "Bicep",
            "BGInfo": "B G Info",
            "CDN": "C D N",
            "CognitiveServices": "Cognitive Services",
            "CosmosDB": "Cosmos D B",
            "Databricks": "Data Bricks",
            "DataFactory": "Data Factory",
            "DataLake": "Data Lake",
            "Defender": "Defender",
            "Sentinel": "Sentinel",
            "DevOps": "Dev Ops",
            "ADO": "A D O",
            "AzureDevOps": "Azure Dev Ops",
            "ExpressRoute": "Express Route",
            "FrontDoor": "Front Door",
            "FunctionApp": "Function App",
            "GD": "G D",
            "GDI": "G D I",
            "HDI": "H D I",
            "HDInsight": "H D Insight",
            "IoTHub": "I O T Hub",
            "IoTEdge": "I O T Edge",
            "KeyVault": "Key Vault",
            "KV": "K V",
            "LogAnalytics": "Log Analytics",
            "LA": "L A",
            "Monitor": "Monitor",
            "NSG": "N S G",
            "ASG": "A S G",
            "PowerBI": "Power B I",
            "PBI": "P B I",
            "PowerAutomate": "Power Automate",
            "PowerApps": "Power Apps",
            "PrivateEndpoint": "Private Endpoint",
            "PLS": "P L S",
            "PublicIP": "Public I P",
            "PIP": "P I P",
            "RBAC": "R B A C",
            "RMS": "R M S",
            "ResourceGroup": "Resource Group",
            "RG": "R G",
            "RouteServer": "Route Server",
            "SAS": "S A S",
            "StorageAccount": "Storage Account",
            "SQLDW": "S Q L D W",
            "Synapse": "Synapse",
            "VM": "V M",
            "VMSS": "V M S S",
            "VNET": "V N E T",
            "VirtualNetwork": "Virtual Network",
            "VPN": "V P N",
            "VWAN": "V W A N",
            "WAF": "W A F",
            "WebApp": "Web App",
            # === AWS & GCP (for comparison docs) ===
            "EC2": "E C 2",
            "S3": "S 3",
            "RDS": "R D S",
            "VPC": "V P C",
            "IAM": "I A M",
            "Lambda": "Lambda",
            "GCP": "G C P",
            "GKE": "G K E",
            "CloudSQL": "Cloud S Q L",
            # === DEV & PROGRAMMING ===
            "IDE": "I D E",
            "CLI": "C L I",
            "GUI": "G U I",
            "API": "A P I",
            "SDK": "S D K",
            "DLL": "D L L",
            "EXE": "E X E",
            "UTF": "U T F",
            "ASCII": "A S C I I",
            "OOP": "O O P",
            "TDD": "T D D",
            "BDD": "B D D",
            "MVC": "M V C",
            "MVVM": "M V V M",
            "JWT": "J W T",
            "OAuth": "O Auth",
            "OIDC": "O I D C",
            "SAML": "S A M L",
            "WSFed": "W S Fed",
            "RBAC": "R B A C",
            "ABAC": "A B A C",
            "PBAC": "P B A C",
            "CI": "C I",
            "CD": "C D",
            "GitOps": "Git Ops",
            "Infra": "Infra",
            "IaC": "I a C",
            "PaaS": "P a a S",
            "FaaS": "F a a S",
            "CaaS": "C a a S",
            "XaaS": "X a a S",
            # === DEVOPS TOOLS ===
            "CI/CD": "C I C D",
            "VCS": "V C S",
            "SCM": "S C M",
            "JIRA": "J I R A",
            "Jenkins": "Jenkins",
            "GitHub": "Git Hub",
            "GitLab": "Git Lab",
            "Bitbucket": "Bit Bucket",
            "Docker": "Docker",
            "K8s": "K 8 s",
            "Kubernetes": "Kubernetes",
            "K8S": "K 8 S",
            "Helm": "Helm",
            "Istio": "Istio",
            "Prometheus": "Prometheus",
            "Grafana": "Grafana",
            "ELK": "E L K",
            "Splunk": "Splunk",
            "Datadog": "Data Dog",
            "NewRelic": "New Relic",
            "PagerDuty": "Pager Duty",
            "Terraform": "Terraform",
            "Ansible": "Ansible",
            "Chef": "Chef",
            "Puppet": "Puppet",
            "SaltStack": "Salt Stack",
            "Nagios": "Nagios",
            "Zabbix": "Zabbix",
            # === MONITORING & LOGGING ===
            "APM": "A P M",
            "NPM": "N P M",
            "RUM": "R U M",
            "SIEM": "S I E M",
            "SOAR": "S O A R",
            "XDR": "X D R",
            "EDR": "E D R",
            "MDR": "M D R",
            "DDoS": "D D o S",
            "DoS": "D o S",
            "MITRE": "M I T R E",
            "ATT&CK": "A T T C K",
            "CVE": "C V E",
            "CVSS": "C V S S",
            # === DATABASE ===
            "OLTP": "O L T P",
            "OLAP": "O L A P",
            "ETL": "E T L",
            "ELT": "E L T",
            "ACID": "A C I D",
            "BASE": "B A S E",
            "CAP": "C A P",
            "NoSQL": "N O S Q L",
            "CRUD": "C R U D",
            "ORM": "O R M",
            "ODBC": "O D B C",
            "JDBC": "J D B C",
            "DDL": "D D L",
            "DML": "D M L",
            "DCL": "D C L",
            "TCL": "T C L",
            # === NETWORKING ===
            "LAN": "L A N",
            "WAN": "W A N",
            "VLAN": "V L A N",
            "VPN": "V P N",
            "DNS": "D N S",
            "DHCP": "D H C P",
            "NAT": "N A T",
            "PAT": "P A T",
            "SNAT": "S N A T",
            "DNAT": "D N A T",
            "TCP": "T C P",
            "UDP": "U D P",
            "ICMP": "I C M P",
            "IP": "I P",
            "IPv4": "I P v 4",
            "IPv6": "I P v 6",
            "HTTP": "H T T P",
            "HTTPS": "H T T P S",
            "FTP": "F T P",
            "SFTP": "S F T P",
            "SSH": "S S H",
            "Telnet": "Telnet",
            "SMTP": "S M T P",
            "POP3": "P O P 3",
            "IMAP": "I M A P",
            "MQTT": "M Q T T",
            "CoAP": "C o A P",
            "AMQP": "A M Q P",
            "Kafka": "Kafka",
            "RabbitMQ": "Rabbit M Q",
            "Redis": "Redis",
            # === CONTAINERS & ORCHESTRATION ===
            "OCI": "O C I",
            "CRI": "C R I",
            "CNI": "C N I",
            "CSI": "C S I",
            # === CLOUD NATIVE ===
            "CNCF": "C N C F",
            "OSS": "O S S",
            "FOSS": "F O S S",
            "SaaS": "S a a S",
            "PaaS": "P a a S",
            "IaaS": "I a a S",
            "FaaS": "F a a S",
            "MSP": "M S P",
            "CSP": "C S P",
            # === AI/ML ===
            "AI": "A I",
            "ML": "M L",
            "DL": "D L",
            "NLP": "N L P",
            "CV": "C V",
            "LLM": "L L M",
            "GPT": "G P T",
            "BERT": "B E R T",
            "RAG": "R A G",
            "OCR": "O C R",
            "ASR": "A S R",
            "TTS": "T T S",
            "NLU": "N L U",
            "NLG": "N L G",
            # === SECURITY ===
            "PKI": "P K I",
            "CA": "C A",
            "CRL": "C R L",
            "OCSP": "O C S P",
            "HSM": "H S M",
            "TPM": "T P M",
            "YubiKey": "Yubi Key",
            "2FA": "2 F A",
            "MFA": "M F A",
            "TOTP": "T O T P",
            "HOTP": "H O T P",
            "SSO": "S S O",
            "IdP": "I d P",
            "SP": "S P",
            "RADIUS": "R A D I U S",
            "TACACS": "T A C A C S",
            "X.509": "X 5 0 9",
            "AES": "A E S",
            "RSA": "R S A",
            "ECC": "E C C",
            "PGP": "P G P",
            "GPG": "G P G",
            "TLS": "T L S",
            "SSL": "S S L",
            "SSH": "S S H",
            # === AGILE/PROJECT MGMT ===
            "Scrum": "Scrum",
            "Kanban": "Kanban",
            "MVP": "M V P",
            "PoC": "P o C",
            "MOKR": "M O K R",
            "OKR": "O K R",
            "KPI": "K P I",
            "SLA": "S L A",
            "SLO": "S L O",
            "SLI": "S L I",
            "MTTR": "M T T R",
            "MTTF": "M T T F",
            "MTBF": "M T B F",
            "RTO": "R T O",
            "RPO": "R P O",
        }

        for acronym, pronunciation in letter_acronyms.items():
            # Use word boundaries to avoid partial matches
            text = re.sub(r"\b" + acronym + r"\b", pronunciation, text)

        return text

    def _add_pause_markers(self, text):
        """Add pause markers for more natural TTS timing"""
        import re

        # Add short pauses after common abbreviations
        short_pause_after = [
            r"\b[A-Z]\.",
            r"\b[Mm]r\.",
            r"\b[Mm]s\.",
            r"\b[Dd]r\.",
            r"\b[Pp]rof\.",
            r"\b[Rr]ev\.",
            r"\b[Gg]en\.",
            r"\b[Ss]gt\.",
            r"\b[Cc]apt\.",
            r"\b[Lt]\.",
            r"\betc\.",
        ]
        for pattern in short_pause_after:
            text = re.sub(pattern, r"\g<0>,", text)

        # Add pauses before conjunctions in long sentences
        text = re.sub(r"\s+(and|or|but|yet|so|nor)\s+", r", \1 ", text)

        # Add pauses after introductory phrases
        intro_phrases = [
            r"However",
            r"Therefore",
            r"Furthermore",
            r"Moreover",
            r"Additionally",
            r"Consequently",
            r"Meanwhile",
            r"Nevertheless",
        ]
        for phrase in intro_phrases:
            text = re.sub(r"\b" + phrase + r",", r"\g<0>,", text)

        # Ensure pauses after colons and semicolons
        text = re.sub(r"[:;]", r"\g<0>,", text)

        return text

    def get_text_stats(self, text):
        """Get text statistics"""
        if not text:
            return {"chars": 0, "words": 0, "lines": 0, "sentences": 0}

        chars = len(text)
        words = len(text.split())
        lines = text.count("\n") + 1
        sentences = len(re.findall(r"[.!?]+", text))

        return {"chars": chars, "words": words, "lines": lines, "sentences": sentences}

    def needs_chunking(self, text):
        """Check if text needs to be split into chunks"""
        return len(text) > self.chunk_size

    def split_text_into_chunks(self, text, model_type="matcha"):
        """Split text into manageable chunks for TTS processing"""
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)

        if len(text) <= safe_chunk_size:
            # Double-check token count for single chunk
            if self.estimate_token_count(text) <= self.model_token_limits.get(
                model_type, 800
            ):
                return [text]

        chunks = []
        remaining_text = text

        while remaining_text:
            if len(remaining_text) <= safe_chunk_size:
                # Final chunk - check token count
                if self.estimate_token_count(
                    remaining_text
                ) <= self.model_token_limits.get(model_type, 800):
                    chunks.append(remaining_text.strip())
                    break
                else:
                    # Still too many tokens, need to split further
                    chunk = self._find_optimal_chunk(remaining_text, model_type)
                    chunks.append(chunk.strip())
                    remaining_text = remaining_text[len(chunk) :].strip()
            else:
                # Find the best split point
                chunk = self._find_optimal_chunk(remaining_text, model_type)
                chunks.append(chunk.strip())
                remaining_text = remaining_text[len(chunk) :].strip()

        # Filter out empty chunks and validate each chunk
        validated_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            is_valid, error_msg = self.validate_chunk_for_model(chunk, model_type)
            if not is_valid:
                # Chunk might be too large, but let's be less aggressive about splitting
                estimated_tokens = self.estimate_token_count(chunk)
                token_limit = self.model_token_limits.get(model_type, 700)

                # Only split if significantly over the limit (more than 20% over)
                if estimated_tokens > token_limit * 1.2:
                    print(
                        f"Warning: Chunk {i+1} significantly over limit ({estimated_tokens} > {token_limit * 1.2:.0f}): {error_msg}"
                    )
                    # Try to split this chunk into smaller pieces
                    sub_chunks = self._emergency_split_chunk(chunk, model_type)
                    validated_chunks.extend(sub_chunks)
                else:
                    # Close to limit but not too bad, let it through with warning
                    print(
                        f"Warning: Chunk {i+1} slightly over limit but allowing: {error_msg}"
                    )
                    validated_chunks.append(chunk)
            else:
                validated_chunks.append(chunk)

        return validated_chunks

    def _emergency_split_chunk(self, text, model_type):
        """Emergency splitting for chunks that are still too large"""
        token_limit = self.model_token_limits.get(model_type, 700)

        # Try splitting by sentences first (preserve original endings)
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            result_chunks = []
            current_chunk = ""

            for sentence in sentences:
                # Preserve original spacing
                separator = " " if current_chunk else ""
                test_chunk = current_chunk + separator + sentence
                if self.estimate_token_count(test_chunk) <= token_limit:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        result_chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                result_chunks.append(current_chunk.strip())

            return result_chunks

        # If no sentences, split by words
        words = text.split()
        result_chunks = []
        current_chunk = ""

        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            if self.estimate_token_count(test_chunk) <= token_limit:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    result_chunks.append(current_chunk)
                current_chunk = word

        if current_chunk:
            result_chunks.append(current_chunk)

        return result_chunks

    def _find_optimal_chunk(self, text, model_type="matcha"):
        """Find the optimal point to split text based on model constraints"""
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)

        if len(text) <= safe_chunk_size:
            # Check token count even for short text
            if self.estimate_token_count(text) <= token_limit:
                return text

        # Try to split at sentence boundaries first
        chunk = self._split_at_sentences(text, model_type)
        if chunk:
            return chunk

        # Try to split at clause boundaries
        chunk = self._split_at_clauses(text, model_type)
        if chunk:
            return chunk

        # Try to split at word boundaries
        chunk = self._split_at_words(text, model_type)
        if chunk:
            return chunk

        # Last resort: hard split at safe character limit
        return text[:safe_chunk_size]

    def _split_at_sentences(self, text, model_type="matcha"):
        """Try to split at sentence boundaries"""
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)

        best_pos = 0
        for i in range(min(len(text), safe_chunk_size), 0, -1):
            for ending in sentence_endings:
                if text[i - len(ending) : i] == ending:
                    # Found a sentence boundary - check token count
                    candidate = text[:i]
                    if self.estimate_token_count(candidate) <= token_limit:
                        return candidate
                elif (
                    i < len(text) - len(ending) and text[i : i + len(ending)] == ending
                ):
                    candidate_pos = i + len(ending)
                    candidate = text[:candidate_pos]
                    if self.estimate_token_count(candidate) <= token_limit:
                        best_pos = candidate_pos

        if best_pos > 0:
            return text[:best_pos]

        return None

    def _split_at_clauses(self, text, model_type="matcha"):
        """Try to split at clause boundaries"""
        clause_endings = [", ", "; ", ": ", ",\n", ";\n", ":\n"]
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)

        best_pos = 0
        for i in range(min(len(text), safe_chunk_size), 0, -1):
            for ending in clause_endings:
                if text[i - len(ending) : i] == ending:
                    candidate = text[:i]
                    if self.estimate_token_count(candidate) <= token_limit:
                        return candidate
                elif (
                    i < len(text) - len(ending) and text[i : i + len(ending)] == ending
                ):
                    candidate_pos = i + len(ending)
                    candidate = text[:candidate_pos]
                    if self.estimate_token_count(candidate) <= token_limit:
                        best_pos = candidate_pos

        if best_pos > 0:
            return text[:best_pos]

        return None

    def _split_at_words(self, text, model_type="matcha"):
        """Try to split at word boundaries"""
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)

        # Find the last space within safe chunk size
        for i in range(min(len(text), safe_chunk_size), 0, -1):
            if text[i - 1] == " ":
                candidate = text[: i - 1]
                if self.estimate_token_count(candidate) <= token_limit:
                    return candidate

        return None


def setup_crash_prevention():
    """Setup crash prevention and signal handling"""

    def signal_handler(signum, frame):
        print(f"\n[CRASH PREVENTION] Caught signal {signum}, exiting gracefully...")
        sys.exit(0)

    def exit_handler():
        print("[CRASH PREVENTION] Application exiting gracefully...")

    # Register signal handlers
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except:
        pass  # Some signals may not be available on all platforms

    # Register exit handler
    atexit.register(exit_handler)

    # Disable Windows error reporting dialog
    if os.name == "nt":  # Windows
        try:
            import ctypes

            # Disable Windows Error Reporting and "Press any key" prompts
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        except:
            pass


class AudioCache:
    """Manages caching of generated audio"""

    def __init__(self, max_size=50, cache_dir=None):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.cache_file = os.path.join(self.cache_dir, "tts_audio_cache.pkl")
        self.load_cache()

    def _generate_key(self, text, model_type, speaker_id, speed, voice_config_id=None):
        """Generate cache key from parameters including voice model"""
        key_data = (
            f"{text}|{model_type}|{speaker_id}|{speed}|{voice_config_id or 'default'}"
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, text, model_type, speaker_id, speed, voice_config_id=None):
        """Get cached audio if available"""
        key = self._generate_key(text, model_type, speaker_id, speed, voice_config_id)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(
        self,
        text,
        model_type,
        speaker_id,
        speed,
        audio_data,
        sample_rate,
        voice_config_id=None,
    ):
        """Cache audio data"""
        key = self._generate_key(text, model_type, speaker_id, speed, voice_config_id)

        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = {
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "timestamp": time.time(),
        }

        # Save cache periodically
        if len(self.cache) % 5 == 0:
            self.save_cache()

    def save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(dict(self.cache), f)
        except Exception:
            pass  # Ignore cache save errors

    def load_cache(self):
        """Load cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    self.cache = OrderedDict(cached_data)
        except Exception:
            pass  # Ignore cache load errors

    def clear(self):
        """Clear all cached data"""
        self.cache.clear()
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
        except Exception:
            pass


class PerformanceMonitor:
    """Monitors and tracks performance metrics"""

    def __init__(self):
        self.metrics = []
        self.current_generation = None

    def start_generation(self, text_length, model_type):
        """Start tracking a generation"""
        self.current_generation = {
            "start_time": time.time(),
            "text_length": text_length,
            "model_type": model_type,
        }

    def end_generation(self, audio_duration, from_cache=False):
        """End tracking and record metrics"""
        if not self.current_generation:
            return

        end_time = time.time()
        generation_time = end_time - self.current_generation["start_time"]

        metric = {
            "timestamp": end_time,
            "text_length": self.current_generation["text_length"],
            "model_type": self.current_generation["model_type"],
            "generation_time": generation_time,
            "audio_duration": audio_duration,
            "rtf": generation_time / audio_duration if audio_duration > 0 else 0,
            "from_cache": from_cache,
        }

        self.metrics.append(metric)

        # Keep only last 100 metrics
        if len(self.metrics) > 100:
            self.metrics = self.metrics[-100:]

        self.current_generation = None
        return metric

    def get_average_rtf(self, model_type=None, last_n=10):
        """Get average RTF for recent generations"""
        relevant_metrics = self.metrics
        if model_type:
            relevant_metrics = [
                m for m in self.metrics if m["model_type"] == model_type
            ]

        if not relevant_metrics:
            return 0

        recent_metrics = relevant_metrics[-last_n:]
        rtf_values = [m["rtf"] for m in recent_metrics if not m["from_cache"]]

        return sum(rtf_values) / len(rtf_values) if rtf_values else 0


class AudioStitcher:
    """Handles stitching multiple audio chunks together"""

    def __init__(self, silence_duration=0.2):
        self.silence_duration = silence_duration  # Seconds of silence between chunks

    def stitch_audio_chunks(self, audio_chunks, sample_rate):
        """Stitch multiple audio chunks together with optional silence"""
        if not audio_chunks:
            return np.array([], dtype=np.float32)

        if len(audio_chunks) == 1:
            return audio_chunks[0]

        # Calculate silence samples
        silence_samples = int(self.silence_duration * sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32)

        # Stitch chunks together
        stitched_audio = []
        for i, chunk in enumerate(audio_chunks):
            # Ensure chunk is numpy array
            if not isinstance(chunk, np.ndarray):
                chunk = np.array(chunk, dtype=np.float32)

            stitched_audio.append(chunk)

            # Add silence between chunks (but not after the last one)
            if i < len(audio_chunks) - 1 and silence_samples > 0:
                stitched_audio.append(silence)

        # Concatenate all parts
        result = np.concatenate(stitched_audio)
        return result.astype(np.float32)

    def estimate_total_duration(self, chunk_durations):
        """Estimate total duration including silence gaps"""
        if not chunk_durations:
            return 0.0

        total_audio_duration = sum(chunk_durations)
        silence_duration = (len(chunk_durations) - 1) * self.silence_duration
        return total_audio_duration + silence_duration


class FollowAlongManager:
    """
    Manages word-by-word highlighting during audio playback.

    Estimates word timings based on audio duration and text content,
    accounting for punctuation pauses and natural speech patterns.
    """

    def __init__(self):
        self.word_timings = (
            []
        )  # List of (start_time, end_time, word, start_idx, end_idx)
        self.current_word_index = -1
        self.is_active = False
        self.original_text = ""

        # Speech timing parameters
        self.punctuation_pause = {
            ".": 0.3,  # Period - longer pause
            "!": 0.3,  # Exclamation
            "?": 0.3,  # Question mark
            ",": 0.15,  # Comma - short pause
            ";": 0.2,  # Semicolon
            ":": 0.2,  # Colon
            "-": 0.1,  # Dash
            "—": 0.15,  # Em-dash
            "...": 0.4,  # Ellipsis - longer pause
            "\n": 0.25,  # Newline - paragraph pause
        }

    def calculate_word_timings(self, text, audio_duration, generation_speed=1.0):
        """
        Calculate estimated timing for each word in the text.

        Args:
            text: The original text being spoken
            audio_duration: Total duration of the audio in seconds
            generation_speed: The TTS generation speed (1.0 = normal)

        Returns:
            List of tuples: (start_time, end_time, word, text_start_idx, text_end_idx)
        """
        self.original_text = text
        self.word_timings = []
        self.current_word_index = -1

        if not text or audio_duration <= 0:
            return []

        # Parse text into words with their positions
        words_with_pos = self._extract_words_with_positions(text)

        if not words_with_pos:
            return []

        # Adjust pause weights based on generation speed
        # At higher speeds, TTS engines compress pauses more than speech
        # At lower speeds, pauses are relatively longer
        # Use inverse relationship: speed 2.0 -> pause_factor 0.5, speed 0.5 -> pause_factor 2.0
        pause_factor = 1.0 / generation_speed

        # Calculate total "weight" of all words (including punctuation pauses)
        total_weight = 0
        word_weights = []

        for i, (word, start_idx, end_idx) in enumerate(words_with_pos):
            # Base weight is proportional to word length (characters)
            weight = len(word)

            # Add weight for syllables (approximate: 1 syllable per 2.5 chars)
            syllable_bonus = max(1, len(word) / 2.5)
            weight += syllable_bonus

            # Check for punctuation after this word
            pause_weight = 0
            if end_idx < len(text):
                following_chars = text[end_idx : min(end_idx + 4, len(text))]
                for punct, pause in self.punctuation_pause.items():
                    if following_chars.startswith(punct):
                        # Scale pause weight by the pause factor
                        pause_weight = pause * 10 * pause_factor
                        break

            word_weights.append((weight, pause_weight))
            total_weight += weight + pause_weight

        # Distribute time across words based on weights
        if total_weight == 0:
            return []

        current_time = 0.0

        for i, (word, start_idx, end_idx) in enumerate(words_with_pos):
            word_weight, pause_weight = word_weights[i]

            # Calculate duration for this word
            word_duration = (word_weight / total_weight) * audio_duration
            pause_duration = (pause_weight / total_weight) * audio_duration

            # Record timing for this word
            start_time = current_time
            end_time = current_time + word_duration

            self.word_timings.append((start_time, end_time, word, start_idx, end_idx))

            # Move time forward (including any pause after this word)
            current_time = end_time + pause_duration

        return self.word_timings

    def _extract_words_with_positions(self, text):
        """
        Extract words from text along with their character positions.

        Returns:
            List of tuples: (word, start_index, end_index)
        """
        words_with_pos = []

        # Use regex to find word boundaries
        # Match sequences of alphanumeric characters and apostrophes (for contractions)
        pattern = r"[\w']+(?:-[\w']+)*"

        for match in re.finditer(pattern, text):
            word = match.group()
            start_idx = match.start()
            end_idx = match.end()
            words_with_pos.append((word, start_idx, end_idx))

        return words_with_pos

    def get_word_at_time(self, current_time):
        """
        Get the word being spoken at the given time.

        Args:
            current_time: Current playback time in seconds

        Returns:
            Tuple: (word_index, word, start_idx, end_idx) or None if no word found
        """
        if not self.word_timings:
            return None

        for i, (start_time, end_time, word, start_idx, end_idx) in enumerate(
            self.word_timings
        ):
            if start_time <= current_time < end_time:
                return (i, word, start_idx, end_idx)

        # If past the last word, return the last word
        if current_time >= self.word_timings[-1][1]:
            last = self.word_timings[-1]
            return (len(self.word_timings) - 1, last[2], last[3], last[4])

        return None

    def get_current_and_upcoming_words(self, current_time, lookahead=3):
        """
        Get the current word and a few upcoming words for context display.

        Args:
            current_time: Current playback time in seconds
            lookahead: Number of upcoming words to return

        Returns:
            List of word info tuples, with the current word first
        """
        current = self.get_word_at_time(current_time)
        if not current:
            return []

        result = [current]
        current_index = current[0]

        # Add upcoming words
        for i in range(1, lookahead + 1):
            next_index = current_index + i
            if next_index < len(self.word_timings):
                timing = self.word_timings[next_index]
                result.append((next_index, timing[2], timing[3], timing[4]))

        return result

    def reset(self):
        """Reset the follow-along state"""
        self.word_timings = []
        self.current_word_index = -1
        self.original_text = ""

    def get_progress_percentage(self, current_time, audio_duration):
        """Get the progress through the text as a percentage"""
        if not self.word_timings or audio_duration <= 0:
            return 0.0

        current = self.get_word_at_time(current_time)
        if current:
            return (current[0] / len(self.word_timings)) * 100
        return 0.0


class TTSGui(
    TTSGuiThemeMixin,
    TTSGuiUiMixin,
    TTSGuiShortcutsMixin,
    TTSGuiTextMixin,
    TTSGuiSSMLMixin,
    TTSGuiVoiceMixin,
    TTSGuiGenerationMixin,
    TTSGuiPlaybackMixin,
    TTSGuiExportMixin,
    TTSGuiLifecycleMixin,
):
    def __init__(self, root):
        # Setup crash prevention first
        setup_crash_prevention()

        self.root = root
        self.root.title("High-Quality English TTS - Sherpa-ONNX Enhanced")
        self.root.geometry("1300x1200")

        # Dracula theme color scheme
        self.colors = {
            "bg_primary": "#282a36",  # Main background (Dracula background)
            "bg_secondary": "#44475a",  # Secondary background (Dracula selection)
            "bg_tertiary": "#6272a4",  # Tertiary background (Dracula comment)
            "bg_accent": "#44475a",  # Accent background
            "fg_primary": "#f8f8f2",  # Primary text (Dracula foreground)
            "fg_secondary": "#bd93f9",  # Secondary text (Dracula purple)
            "fg_muted": "#6272a4",  # Muted text (Dracula comment)
            "accent_pink": "#ff79c6",  # Dracula pink (primary accent)
            "accent_pink_hover": "#ff92d0",  # Pink hover state
            "accent_cyan": "#8be9fd",  # Dracula cyan
            "accent_green": "#50fa7b",  # Dracula green (success)
            "accent_orange": "#ffb86c",  # Dracula orange (warning)
            "accent_red": "#ff5555",  # Dracula red (error/danger)
            "accent_purple": "#bd93f9",  # Dracula purple
            "selection": "#44475a",  # Selection background
            "border": "#6272a4",  # Border color
            "border_light": "#bd93f9",  # Light border (purple)
        }

        # Configure root window
        self.root.configure(bg=self.colors["bg_primary"])

        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

        # Initialize helper components
        self.text_processor = TextProcessor()
        self.audio_cache = AudioCache()
        self.performance_monitor = PerformanceMonitor()
        self.audio_stitcher = AudioStitcher(silence_duration=0.3)
        self.audio_exporter = AudioExporter()  # Advanced audio export system
        self.follow_along_manager = FollowAlongManager()  # Word-by-word highlighting
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # TTS model instances
        self.tts_models = {}  # Dictionary to store loaded models
        self.current_audio_file = None
        self.model_loading_in_progress = False

        # Create audio output directory if it doesn't exist
        self.audio_output_dir = Path("audio_output")
        self.audio_output_dir.mkdir(exist_ok=True)

        # Voice selection variables
        self.selected_voice_config = None
        self.available_voice_configs = {}

        # Audio playback control variables
        self.current_sound = None
        self.audio_duration = 0.0
        self.playback_start_time = 0.0
        self.is_playing = False
        self.is_paused = False
        self.pause_position = 0.0
        self.audio_data = None
        self.sample_rate = 22050

        # Generation control variables
        self.generation_cancelled = False
        self.generation_thread = None

        # Text processing options
        self.text_options = {
            "normalize_whitespace": tk.BooleanVar(value=True),
            "normalize_punctuation": tk.BooleanVar(value=True),
            "remove_urls": tk.BooleanVar(value=False),
            "remove_emails": tk.BooleanVar(value=False),
            "remove_duplicates": tk.BooleanVar(value=False),
            "numbers_to_words": tk.BooleanVar(value=False),
            "expand_abbreviations": tk.BooleanVar(value=False),
            "handle_acronyms": tk.BooleanVar(value=False),
            "add_pauses": tk.BooleanVar(value=False),
        }

        # SSML Support
        self.ssml_processor = SSMLProcessor()
        self.ssml_enabled = tk.BooleanVar(value=False)
        self.ssml_auto_detect = tk.BooleanVar(value=True)

        # Follow-along word highlighting
        self.follow_along_enabled = tk.BooleanVar(value=True)
        self.generated_text_for_follow_along = ""  # Store the text used for generation

        # Setup theme and UI
        self.setup_theme()
        self.setup_ui()

        # Setup keyboard shortcuts for power users
        self.setup_keyboard_shortcuts()

        # Check available voices and populate selections (after UI is ready)
        self.check_available_voices()
        self.populate_voice_selections()

        # Start model preloading in background
        self.preload_models()


def main():
    """Main function"""
    # Check if required packages are available
    try:
        import sherpa_onnx
        import pygame
        import soundfile
    except ImportError as e:
        print(f"Required package missing: {e}")
        print("Please install required packages:")
        print("pip install sherpa-onnx pygame soundfile")
        return

    root = tk.Tk()
    app = TTSGui(root)

    # Handle window close event
    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
