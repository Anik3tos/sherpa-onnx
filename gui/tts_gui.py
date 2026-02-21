#!/usr/bin/env python3
"""
High-Quality English TTS GUI using PySide6 (Qt).
Enhanced version with SSML support, follow-along highlighting, and advanced export options.
"""

import atexit
import hashlib
import os
import base64
import json
import re
import signal
import subprocess
import sys
import tempfile
import time
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

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import Qt

from tts_gui.export import TTSGuiExportMixin
from tts_gui.generation import TTSGuiGenerationMixin
from tts_gui.lifecycle import TTSGuiLifecycleMixin
from tts_gui.playback import TTSGuiPlaybackMixin
from tts_gui.shortcuts import TTSGuiShortcutsMixin
from tts_gui.ssml import TTSGuiSSMLMixin
from tts_gui.text import TTSGuiTextMixin
from tts_gui.transcription import TTSGuiTranscriptionMixin
from tts_gui.theme import TTSGuiThemeMixin
from tts_gui.ui import TTSGuiUiMixin
from tts_gui.voice import TTSGuiVoiceMixin
from tts_gui.config import TTSGuiConfigMixin


# ============================================================================
# Business Logic Classes (unchanged from Tkinter version)
# ============================================================================


class AudioExporter:
    """
    Advanced Audio Export System

    Supports multiple audio formats with configurable quality settings,
    silence detection for automatic track splitting, and chapter/section markers.
    """

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
            "compression_levels": list(range(9)),
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
        try:
            from pydub import AudioSegment

            return True
        except ImportError:
            return False

    def get_available_formats(self):
        available = ["wav", "flac"]
        if self.ffmpeg_available or self.pydub_available:
            available.extend(["mp3", "ogg"])
        return available

    def export(
        self, audio_data, sample_rate, output_path, format_type="wav", options=None
    ):
        if options is None:
            options = {}

        format_config = self.FORMATS.get(format_type)
        if not format_config:
            return False, f"Unknown format: {format_type}", None

        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)

        if options.get("normalize", False):
            audio_data = self._normalize_audio(audio_data)

        target_sr = options.get("target_sample_rate", sample_rate)
        if target_sr != sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, target_sr)
            sample_rate = target_sr

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
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data * (target_peak / max_val)
        return audio_data

    def _resample_audio(self, audio_data, src_rate, target_rate):
        if src_rate == target_rate:
            return audio_data
        duration = len(audio_data) / src_rate
        target_samples = int(duration * target_rate)
        x_old = np.linspace(0, len(audio_data) - 1, len(audio_data))
        x_new = np.linspace(0, len(audio_data) - 1, target_samples)
        resampled = np.interp(x_new, x_old, audio_data)
        return resampled.astype(np.float32)

    def _export_wav(self, audio_data, sample_rate, output_path, options):
        try:
            subtype = options.get("wav_subtype", "PCM_16")
            sf.write(output_path, audio_data, sample_rate, subtype=subtype)
            return True, f"Exported WAV: {output_path}", output_path
        except Exception as e:
            return False, f"WAV export failed: {str(e)}", None

    def _export_flac(self, audio_data, sample_rate, output_path, options):
        try:
            sf.write(output_path, audio_data, sample_rate, format="FLAC")
            return True, f"Exported FLAC: {output_path}", output_path
        except Exception as e:
            return False, f"FLAC export failed: {str(e)}", None

    def _export_lossy(self, audio_data, sample_rate, output_path, format_type, options):
        format_config = self.FORMATS[format_type]
        bitrate = options.get("bitrate", format_config["default_bitrate"])
        temp_wav = output_path + ".temp.wav"

        try:
            sf.write(temp_wav, audio_data, sample_rate, subtype="PCM_16")
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
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    def _convert_with_ffmpeg(
        self, input_path, output_path, format_type, bitrate, options
    ):
        try:
            cmd = ["ffmpeg", "-y", "-i", input_path]
            if format_type == "mp3":
                cmd.extend(["-codec:a", "libmp3lame", "-b:a", f"{bitrate}k"])
            elif format_type == "ogg":
                cmd.extend(["-codec:a", "libvorbis", "-b:a", f"{bitrate}k"])
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
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_wav(input_path)
            metadata = options.get("metadata", {})
            tags = {key: value for key, value in metadata.items() if value}
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
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        min_silence_samples = int(min_silence_len * sample_rate / 1000)
        seek_samples = int(seek_step * sample_rate / 1000)
        silence_thresh_linear = 10 ** (silence_thresh / 20)
        silence_regions = []
        in_silence = False
        silence_start = 0

        for i in range(0, len(audio_data) - seek_samples, seek_samples):
            chunk = audio_data[i : i + seek_samples]
            chunk_level = np.max(np.abs(chunk))
            is_silent = chunk_level < silence_thresh_linear

            if is_silent and not in_silence:
                in_silence = True
                silence_start = i
            elif not is_silent and in_silence:
                in_silence = False
                silence_end = i
                silence_duration = silence_end - silence_start
                if silence_duration >= min_silence_samples:
                    start_ms = int(silence_start * 1000 / sample_rate)
                    end_ms = int(silence_end * 1000 / sample_rate)
                    silence_regions.append((start_ms, end_ms))

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
        silence_regions = self.detect_silence(
            audio_data, sample_rate, min_silence_len, silence_thresh
        )
        if not silence_regions:
            return [audio_data]

        keep_samples = int(keep_silence * sample_rate / 1000)
        min_segment_samples = int(min_segment_len * sample_rate / 1000)
        segments = []
        prev_end = 0

        for start_ms, end_ms in silence_regions:
            split_point_samples = int((start_ms + end_ms) / 2 * sample_rate / 1000)
            segment_start = max(0, prev_end - keep_samples)
            segment_end = min(len(audio_data), split_point_samples + keep_samples)
            segment = audio_data[segment_start:segment_end]
            if len(segment) >= min_segment_samples:
                segments.append(segment)
            prev_end = segment_end

        if prev_end < len(audio_data):
            segment = audio_data[max(0, prev_end - keep_samples) :]
            if len(segment) >= min_segment_samples:
                segments.append(segment)

        return segments if segments else [audio_data]

    def split_by_chapters(self, audio_data, sample_rate, chapter_markers):
        if not chapter_markers:
            return [("Full Audio", audio_data)]
        markers = sorted(chapter_markers, key=lambda x: x.get("start_ms", 0))
        segments = []

        for i, marker in enumerate(markers):
            start_ms = marker.get("start_ms", 0)
            title = marker.get("title", f"Chapter {i+1}")
            if i + 1 < len(markers):
                end_ms = markers[i + 1]["start_ms"]
            else:
                end_ms = len(audio_data) * 1000 // sample_rate
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)
            segment = audio_data[start_sample:end_sample]
            if len(segment) > 0:
                segments.append((title, segment))

        return segments if segments else [("Full Audio", audio_data)]

    def detect_chapters_from_text(self, text):
        chapters = []
        patterns = [
            r"^(?:Chapter|CHAPTER)\s+(\d+|[IVXLCDM]+)(?:\s*[:\-\.]\s*(.*))?$",
            r"^(?:Part|PART)\s+(\d+|[IVXLCDM]+)(?:\s*[:\-\.]\s*(.*))?$",
            r"^(?:Section|SECTION)\s+(\d+)(?:\s*[:\-\.]\s*(.*))?$",
            r"^#{1,3}\s+(.+)$",
            r"^\*\*\*+\s*$",
            r"^[-=]{3,}\s*$",
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
        if options is None:
            options = {}
        results = []
        os.makedirs(output_dir, exist_ok=True)

        for i, segment in enumerate(audio_segments, 1):
            if isinstance(segment, tuple):
                title, audio_data = segment
                safe_title = re.sub(r'[<>:"/\\|?*]', "", title)[:50]
                filename = f"{base_name}_{i:02d}_{safe_title}"
            else:
                audio_data = segment
                filename = f"{base_name}_{i:02d}"
            output_path = os.path.join(output_dir, filename)
            sample_rate = options.get("sample_rate", 22050)
            result = self.export(
                audio_data, sample_rate, output_path, format_type, options
            )
            results.append(result)
        return results


class SSMLProcessor:
    """SSML (Speech Synthesis Markup Language) processor."""

    def __init__(self):
        self.break_mappings = {
            "none": "",
            "x-weak": ",",
            "weak": ", ",
            "medium": ". ",
            "strong": "... ",
            "x-strong": "...... ",
        }
        self.emphasis_mappings = {
            "strong": ("*", "*"),
            "moderate": ("", ""),
            "reduced": ("", ""),
            "none": ("", ""),
        }
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
        self.prosody_stack = []
        self.current_rate = 1.0
        self.current_pitch = 1.0
        self.current_volume = 1.0

    def is_ssml(self, text):
        text = text.strip()
        if text.startswith("<speak") or text.startswith("<?xml"):
            return True
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
        self.prosody_stack = []
        self.current_rate = 1.0
        self.current_pitch = 1.0
        self.current_volume = 1.0

        ssml_text = ssml_text.strip()
        if not ssml_text.startswith("<speak"):
            if not ssml_text.startswith("<?xml"):
                ssml_text = f"<speak>{ssml_text}</speak>"

        if ssml_text.startswith("<?xml"):
            decl_end = ssml_text.find("?>")
            if decl_end != -1:
                ssml_text = ssml_text[decl_end + 2 :].strip()
                if not ssml_text.startswith("<speak"):
                    ssml_text = f"<speak>{ssml_text}</speak>"

        try:
            root = ET.fromstring(ssml_text)
            processed_text, segments = self._process_element(root)
            result["text"] = self._clean_text(processed_text)
            result["segments"] = segments
            if segments:
                rates = [s["rate"] for s in segments if s.get("rate")]
                if rates:
                    result["rate"] = sum(rates) / len(rates)
                    if result["rate"] != 1.0:
                        result["has_prosody_changes"] = True
        except ParseError as e:
            result["errors"].append(f"XML parsing error: {str(e)}")
            result["text"] = self._strip_tags(ssml_text)
        except Exception as e:
            result["errors"].append(f"SSML processing error: {str(e)}")
            result["text"] = self._strip_tags(ssml_text)

        return result

    def _process_element(self, element, depth=0):
        segments = []
        text_parts = []
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

        for child in element:
            child_text, child_segments = self._process_child_element(child, depth)
            text_parts.append(child_text)
            segments.extend(child_segments)
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
        tag = element.tag.lower()
        if "}" in tag:
            tag = tag.split("}")[1]

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
            return self._process_element(element, depth + 1)
        else:
            return self._process_element(element, depth + 1)

    def _handle_break(self, element):
        time_attr = element.get("time", "")
        strength = element.get("strength", "medium")
        if time_attr:
            pause_text = self._time_to_pause(time_attr)
        else:
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
        try:
            time_str = time_str.lower().strip()
            if time_str.endswith("ms"):
                ms = float(time_str[:-2])
                seconds = ms / 1000
            elif time_str.endswith("s"):
                seconds = float(time_str[:-1])
            else:
                seconds = float(time_str)
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
        level = element.get("level", "moderate")
        inner_text, segments = self._process_element(element, depth + 1)
        if level == "strong":
            processed_text = f", {inner_text},"
        else:
            processed_text = inner_text
        for seg in segments:
            seg["emphasis"] = level
        return processed_text, segments

    def _handle_prosody(self, element, depth):
        old_rate = self.current_rate
        old_pitch = self.current_pitch
        old_volume = self.current_volume

        rate = element.get("rate", "")
        if rate:
            self.current_rate = self._parse_prosody_value(rate, self.current_rate)
        pitch = element.get("pitch", "")
        if pitch:
            self.current_pitch = self._parse_prosody_value(pitch, self.current_pitch)
        volume = element.get("volume", "")
        if volume:
            self.current_volume = self._parse_prosody_value(volume, self.current_volume)

        inner_text, segments = self._process_element(element, depth + 1)

        self.current_rate = old_rate
        self.current_pitch = old_pitch
        self.current_volume = old_volume

        return inner_text, segments

    def _parse_prosody_value(self, value, current):
        value = value.lower().strip()
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
            if value.endswith("%"):
                pct_str = value[:-1]
                if pct_str.startswith("+"):
                    return current * (1 + float(pct_str[1:]) / 100)
                elif pct_str.startswith("-"):
                    return current * (1 - float(pct_str[1:]) / 100)
                else:
                    return float(pct_str) / 100
            if "st" in value:
                st_val = float(value.replace("st", "").replace("+", ""))
                return current * (2 ** (st_val / 12))
            return float(value)
        except:
            return current

    def _handle_say_as(self, element, depth):
        interpret_as = element.get("interpret-as", "")
        inner_text, segments = self._process_element(element, depth + 1)
        text = inner_text.strip()

        if interpret_as in ["characters", "spell-out"]:
            processed = " ".join(text)
        elif interpret_as == "digits":
            processed = " ".join(text)
        elif interpret_as == "ordinal":
            processed = self._number_to_ordinal(text)
        elif interpret_as == "telephone":
            processed = " ".join(c for c in text if c.isdigit())
        elif interpret_as == "verbatim":
            processed = " ".join(text)
        elif interpret_as == "acronym":
            processed = " ".join(text.upper())
        elif interpret_as == "expletive":
            processed = "[expletive]"
        else:
            processed = text

        for seg in segments:
            seg["interpret_as"] = interpret_as
        return processed, segments

    def _number_to_ordinal(self, text):
        try:
            num = int(text)
            if 10 <= num % 100 <= 20:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(num % 10, "th")
            return f"{num}{suffix}"
        except:
            return text

    def _handle_phoneme(self, element, depth):
        alphabet = element.get("alphabet", "ipa")
        ph = element.get("ph", "")
        inner_text, segments = self._process_element(element, depth + 1)
        for seg in segments:
            seg["phoneme"] = ph
            seg["phoneme_alphabet"] = alphabet
        return inner_text, segments

    def _handle_sub(self, element):
        alias = element.get("alias", "")
        original = element.text or ""
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
        voice_name = element.get("name", "")
        gender = element.get("gender", "")
        age = element.get("age", "")
        variant = element.get("variant", "")
        inner_text, segments = self._process_element(element, depth + 1)
        for seg in segments:
            seg["voice_hint"] = {
                "name": voice_name,
                "gender": gender,
                "age": age,
                "variant": variant,
            }
        return inner_text, segments

    def _handle_paragraph(self, element, depth):
        inner_text, segments = self._process_element(element, depth + 1)
        return inner_text.strip() + "\n\n", segments

    def _handle_sentence(self, element, depth):
        inner_text, segments = self._process_element(element, depth + 1)
        text = inner_text.strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text + " ", segments

    def _handle_audio(self, element):
        src = element.get("src", "")
        desc = element.text or f"[Audio: {src}]"
        return desc, [{"text": desc, "rate": self.current_rate, "is_audio_ref": True}]

    def _clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"([.,!?])\s*([.,!?])", r"\1", text)
        return text.strip()

    def _strip_tags(self, text):
        text = re.sub(r"<\?xml[^>]*\?>", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_ssml_template(self, template_name="basic"):
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
    """Handles text preprocessing and validation."""

    def __init__(self):
        self.max_length = 100000
        self.min_length = 1
        self.chunk_size = 8000
        self.max_chunk_size = 9500
        self.model_token_limits = {"matcha": 700, "kokoro": 1100}
        self.chars_per_token = 3.5
        self._abbreviation_map = {
            "mr": "mister",
            "mrs": "missus",
            "ms": "miss",
            "dr": "doctor",
            "prof": "professor",
            "sr": "senior",
            "jr": "junior",
            "dept": "department",
            "univ": "university",
            "ave": "avenue",
            "rd": "road",
            "blvd": "boulevard",
            "st": "street",
            "mt": "mount",
            "no": "number",
            "vs": "versus",
            "etc": "et cetera",
        }
        self._small_numbers = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
            19: "nineteen",
        }
        self._tens_numbers = {
            20: "twenty",
            30: "thirty",
            40: "forty",
            50: "fifty",
            60: "sixty",
            70: "seventy",
            80: "eighty",
            90: "ninety",
        }
        self._digit_words = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }

    def validate_text(self, text):
        if not text or not text.strip():
            return False, "Text cannot be empty"
        if len(text) > self.max_length:
            return False, f"Text too long (max {self.max_length} characters)"
        if len(text.strip()) < self.min_length:
            return False, f"Text too short (min {self.min_length} characters)"
        return True, ""

    def estimate_token_count(self, text):
        words = len(text.split())
        punctuation = sum(1 for c in text if c in ".,!?;:()[]{}\"-'")
        numbers = sum(1 for c in text if c.isdigit())
        special_chars = sum(
            1 for c in text if not c.isalnum() and c not in " .,!?;:()[]{}\"-'"
        )
        estimated_tokens = int(
            (words * 1.3)
            + (punctuation * 0.8)
            + (numbers * 0.3)
            + (special_chars * 0.5)
        )
        if len(text) > 1000:
            estimated_tokens = int(estimated_tokens * 1.2)
        return max(estimated_tokens, len(text) // 3)

    def get_model_safe_chunk_size(self, model_type):
        token_limit = self.model_token_limits.get(model_type, 600)
        safe_char_limit = int(token_limit * self.chars_per_token * 0.6)
        return min(safe_char_limit, self.chunk_size)

    def validate_chunk_for_model(self, text, model_type):
        token_count = self.estimate_token_count(text)
        token_limit = self.model_token_limits.get(model_type, 600)
        if token_count > token_limit:
            return (
                False,
                f"Chunk has ~{token_count} tokens, exceeds {model_type} limit of {token_limit}",
            )
        return True, ""

    def preprocess_text(self, text, options=None):
        if not text:
            return text
        if options is None:
            options = {}
        processed = text

        if options.get("fix_encoding", True):
            import unicodedata

            processed = unicodedata.normalize("NFKD", processed)
            encoding_fixes = {
                "â€™": "'",
                "â€œ": '"',
                "â€": '"',
                'â€"': "-",
                'â€"': "-",
                "â€¦": "...",
                "â?T": "'",
                'â?"': '"',
                "â?~": '"',
                "â?¢": "•",
            }
            for corrupt, fixed in encoding_fixes.items():
                processed = processed.replace(corrupt, fixed)
            processed = re.sub(r'[^\w\s\.,!?;:\'"()-]', " ", processed)

        if options.get("remove_urls", False):
            processed = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                processed,
            )

        if options.get("remove_duplicates", False):
            processed = self._remove_duplicate_lines(processed)

        if options.get("expand_abbreviations", False):
            processed = self._expand_abbreviations(processed)

        if options.get("numbers_to_words", False):
            processed = self._expand_numbers(processed)

        if options.get("remove_word_dashes", False):
            processed = re.sub(r"\b([A-Za-z]+)-([A-Za-z]+)\b", r"\1 \2", processed)

        if options.get("handle_acronyms", False):
            # Expand acronyms such as NASA or U.S.A. into space-separated letters.
            processed = re.sub(
                r"\b(?:[A-Z]\.){2,}",
                lambda m: " ".join(ch for ch in m.group(0) if ch.isalpha()),
                processed,
            )
            processed = re.sub(
                r"\b[A-Z]{2,}\b",
                lambda m: " ".join(m.group(0)),
                processed,
            )

        if options.get("add_pauses", False):
            processed = self._add_natural_pauses(processed)

        if options.get("normalize_punctuation", True):
            processed = re.sub(r"[.]{2,}", "...", processed)
            processed = re.sub(r"[!]{2,}", "!", processed)
            processed = re.sub(r"[?]{2,}", "?", processed)

        if options.get("normalize_whitespace", True):
            processed = re.sub(r"\s+", " ", processed)
            processed = processed.strip()

        return processed

    def _remove_duplicate_lines(self, text):
        lines = text.splitlines()
        if len(lines) <= 1:
            return text

        deduplicated = []
        previous_non_empty = None
        for line in lines:
            stripped = line.strip()
            if stripped and stripped == previous_non_empty:
                continue
            deduplicated.append(line)
            previous_non_empty = stripped

        return "\n".join(deduplicated)

    def _expand_abbreviations(self, text):
        text = re.sub(r"\be\.g\.(?=\s|$)", "for example", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi\.e\.(?=\s|$)", "that is", text, flags=re.IGNORECASE)

        pattern = re.compile(
            r"\b("
            + "|".join(sorted(self._abbreviation_map.keys(), key=len, reverse=True))
            + r")\.(?=\s|$)",
            flags=re.IGNORECASE,
        )

        def replace(match):
            token = match.group(1)
            replacement = self._abbreviation_map.get(token.lower(), token)
            return self._match_word_case(token, replacement)

        return pattern.sub(replace, text)

    def _match_word_case(self, source, replacement):
        if not source:
            return replacement
        if source.isupper():
            return replacement.upper()
        if source.islower():
            return replacement
        if source[0].isupper():
            return replacement.capitalize()
        return replacement

    def _expand_numbers(self, text):
        number_pattern = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
        return number_pattern.sub(self._replace_number_token, text)

    def _replace_number_token(self, match):
        token = match.group(0)
        normalized = token.replace(",", "")
        try:
            if "." in normalized:
                whole, fractional = normalized.split(".", 1)
                whole_part = int(whole) if whole else 0
                whole_words = self._int_to_words(whole_part)
                if not fractional:
                    return whole_words
                fractional_words = " ".join(
                    self._digit_words.get(d, d) for d in fractional
                )
                return f"{whole_words} point {fractional_words}".strip()

            if len(normalized) > 1 and normalized.startswith("0"):
                return " ".join(self._digit_words.get(d, d) for d in normalized)

            return self._int_to_words(int(normalized))
        except Exception:
            return token

    def _int_to_words(self, number):
        if number < 0:
            return "minus " + self._int_to_words(abs(number))
        if number < 20:
            return self._small_numbers[number]
        if number < 100:
            tens = (number // 10) * 10
            remainder = number % 10
            if remainder == 0:
                return self._tens_numbers[tens]
            return f"{self._tens_numbers[tens]} {self._small_numbers[remainder]}"
        if number < 1000:
            hundreds = number // 100
            remainder = number % 100
            if remainder == 0:
                return f"{self._small_numbers[hundreds]} hundred"
            return (
                f"{self._small_numbers[hundreds]} hundred {self._int_to_words(remainder)}"
            )

        scales = [
            (1_000_000_000_000, "trillion"),
            (1_000_000_000, "billion"),
            (1_000_000, "million"),
            (1_000, "thousand"),
        ]
        for scale_value, scale_name in scales:
            if number >= scale_value:
                major = number // scale_value
                remainder = number % scale_value
                major_words = self._int_to_words(major)
                if remainder == 0:
                    return f"{major_words} {scale_name}"
                return f"{major_words} {scale_name} {self._int_to_words(remainder)}"
        return str(number)

    def _add_natural_pauses(self, text):
        # Turn line breaks into sentence boundaries unless punctuation already exists.
        text = re.sub(r"([^\s.!?])\s*\n+\s*", r"\1. ", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s*--+\s*", ", ", text)
        text = re.sub(r"\s*[-–—]\s*", ", ", text)
        text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
        return text

    def get_text_stats(self, text):
        if not text:
            return {"chars": 0, "words": 0, "lines": 0, "sentences": 0}
        chars = len(text)
        words = len(text.split())
        lines = text.count("\n") + 1
        sentences = len(re.findall(r"[.!?]+", text))
        return {"chars": chars, "words": words, "lines": lines, "sentences": sentences}

    def needs_chunking(self, text):
        return len(text) > self.chunk_size

    def split_text_into_chunks(self, text, model_type="matcha"):
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        if len(text) <= safe_chunk_size:
            if self.estimate_token_count(text) <= self.model_token_limits.get(
                model_type, 800
            ):
                return [text]

        chunks = []
        remaining_text = text

        while remaining_text:
            if len(remaining_text) <= safe_chunk_size:
                if self.estimate_token_count(
                    remaining_text
                ) <= self.model_token_limits.get(model_type, 800):
                    chunks.append(remaining_text.strip())
                    break
                else:
                    chunk = self._find_optimal_chunk(remaining_text, model_type)
                    chunks.append(chunk.strip())
                    remaining_text = remaining_text[len(chunk) :].strip()
            else:
                chunk = self._find_optimal_chunk(remaining_text, model_type)
                chunks.append(chunk.strip())
                remaining_text = remaining_text[len(chunk) :].strip()

        validated_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            is_valid, error_msg = self.validate_chunk_for_model(chunk, model_type)
            if not is_valid:
                estimated_tokens = self.estimate_token_count(chunk)
                token_limit = self.model_token_limits.get(model_type, 700)
                if estimated_tokens > token_limit * 1.2:
                    sub_chunks = self._emergency_split_chunk(chunk, model_type)
                    validated_chunks.extend(sub_chunks)
                else:
                    validated_chunks.append(chunk)
            else:
                validated_chunks.append(chunk)

        return validated_chunks

    def _emergency_split_chunk(self, text, model_type):
        token_limit = self.model_token_limits.get(model_type, 700)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            result_chunks = []
            current_chunk = ""
            for sentence in sentences:
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
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)

        if len(text) <= safe_chunk_size:
            if self.estimate_token_count(text) <= token_limit:
                return text

        chunk = self._split_at_sentences(text, model_type)
        if chunk:
            return chunk
        chunk = self._split_at_clauses(text, model_type)
        if chunk:
            return chunk
        chunk = self._split_at_words(text, model_type)
        if chunk:
            return chunk
        return text[:safe_chunk_size]

    def _split_at_sentences(self, text, model_type="matcha"):
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)
        best_pos = 0

        for i in range(min(len(text), safe_chunk_size), 0, -1):
            for ending in sentence_endings:
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

    def _split_at_clauses(self, text, model_type="matcha"):
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
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)

        for i in range(min(len(text), safe_chunk_size), 0, -1):
            if text[i - 1] == " ":
                candidate = text[: i - 1]
                if self.estimate_token_count(candidate) <= token_limit:
                    return candidate
        return None


def setup_crash_prevention():
    """Setup crash prevention and signal handling."""

    def signal_handler(signum, frame):
        print(f"\n[CRASH PREVENTION] Caught signal {signum}, exiting gracefully...")
        sys.exit(0)

    def exit_handler():
        print("[CRASH PREVENTION] Application exiting gracefully...")

    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except:
        pass

    atexit.register(exit_handler)

    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        except:
            pass


class AudioCache:
    """Manages caching of generated audio."""

    def __init__(self, max_size=50, cache_dir=None):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.cache_file = os.path.join(self.cache_dir, "tts_audio_cache.json")
        self.load_cache()

    def _generate_key(self, text, model_type, speaker_id, speed, voice_config_id=None):
        key_data = (
            f"{text}|{model_type}|{speaker_id}|{speed}|{voice_config_id or 'default'}"
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, text, model_type, speaker_id, speed, voice_config_id=None):
        key = self._generate_key(text, model_type, speaker_id, speed, voice_config_id)
        if key in self.cache:
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
        key = self._generate_key(text, model_type, speaker_id, speed, voice_config_id)
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = {
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "timestamp": time.time(),
        }
        if len(self.cache) % 5 == 0:
            self.save_cache()

    def save_cache(self):
        try:
            serializable = {}
            for key, value in self.cache.items():
                audio_data = value.get("audio_data")
                if audio_data is None:
                    continue
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data, dtype=np.float32)
                serializable[key] = {
                    "audio_b64": base64.b64encode(audio_data.tobytes()).decode("ascii"),
                    "dtype": str(audio_data.dtype),
                    "length": int(audio_data.size),
                    "sample_rate": value.get("sample_rate", 22050),
                    "timestamp": value.get("timestamp", time.time()),
                }
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f)
        except Exception:
            pass

    def load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                loaded = OrderedDict()
                for key, value in cached_data.items():
                    try:
                        audio_b64 = value.get("audio_b64")
                        dtype = value.get("dtype", "float32")
                        length = int(value.get("length", 0))
                        if not audio_b64 or length <= 0:
                            continue
                        raw = base64.b64decode(audio_b64.encode("ascii"))
                        audio = np.frombuffer(raw, dtype=dtype)
                        if audio.size != length:
                            continue
                        loaded[key] = {
                            "audio_data": audio,
                            "sample_rate": value.get("sample_rate", 22050),
                            "timestamp": value.get("timestamp", time.time()),
                        }
                    except Exception:
                        continue
                self.cache = loaded
        except Exception:
            pass

    def clear(self):
        self.cache.clear()
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
        except Exception:
            pass


class PerformanceMonitor:
    """Monitors and tracks performance metrics."""

    def __init__(self):
        self.metrics = []
        self.current_generation = None

    def start_generation(self, text_length, model_type):
        self.current_generation = {
            "start_time": time.time(),
            "text_length": text_length,
            "model_type": model_type,
        }

    def end_generation(self, audio_duration, from_cache=False):
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
        if len(self.metrics) > 100:
            self.metrics = self.metrics[-100:]
        self.current_generation = None
        return metric

    def get_average_rtf(self, model_type=None, last_n=10):
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
    """Handles stitching multiple audio chunks together."""

    def __init__(self, silence_duration=0.2):
        self.silence_duration = silence_duration

    def stitch_audio_chunks(self, audio_chunks, sample_rate):
        if not audio_chunks:
            return np.array([], dtype=np.float32)
        if len(audio_chunks) == 1:
            return audio_chunks[0]

        silence_samples = int(self.silence_duration * sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32)
        stitched_audio = []

        for i, chunk in enumerate(audio_chunks):
            if not isinstance(chunk, np.ndarray):
                chunk = np.array(chunk, dtype=np.float32)
            stitched_audio.append(chunk)
            if i < len(audio_chunks) - 1 and silence_samples > 0:
                stitched_audio.append(silence)

        result = np.concatenate(stitched_audio)
        return result.astype(np.float32)


class FollowAlongManager:
    """Manages word-by-word highlighting during audio playback."""

    def __init__(self):
        self.word_timings = []
        self.current_word_index = -1
        self.is_active = False
        self.original_text = ""
        self.punctuation_pause = {
            ".": 0.3,
            "!": 0.3,
            "?": 0.3,
            ",": 0.15,
            ";": 0.2,
            ":": 0.2,
            "-": 0.1,
            "—": 0.15,
            "...": 0.4,
            "\n": 0.25,
        }

    def calculate_word_timings(self, text, audio_duration, generation_speed=1.0):
        self.original_text = text
        self.word_timings = []
        self.current_word_index = -1

        if not text or audio_duration <= 0:
            return []

        words_with_pos = self._extract_words_with_positions(text)
        if not words_with_pos:
            return []

        pause_factor = 1.0 / generation_speed
        total_weight = 0
        word_weights = []

        for i, (word, start_idx, end_idx) in enumerate(words_with_pos):
            weight = len(word)
            syllable_bonus = max(1, len(word) / 2.5)
            weight += syllable_bonus
            pause_weight = 0
            if end_idx < len(text):
                following_chars = text[end_idx : min(end_idx + 4, len(text))]
                for punct, pause in self.punctuation_pause.items():
                    if following_chars.startswith(punct):
                        pause_weight = pause * 10 * pause_factor
                        break
            word_weights.append((weight, pause_weight))
            total_weight += weight + pause_weight

        if total_weight == 0:
            return []

        current_time = 0.0
        for i, (word, start_idx, end_idx) in enumerate(words_with_pos):
            word_weight, pause_weight = word_weights[i]
            word_duration = (word_weight / total_weight) * audio_duration
            pause_duration = (pause_weight / total_weight) * audio_duration
            start_time = current_time
            end_time = current_time + word_duration
            self.word_timings.append((start_time, end_time, word, start_idx, end_idx))
            current_time = end_time + pause_duration

        return self.word_timings

    def _extract_words_with_positions(self, text):
        words_with_pos = []
        pattern = r"[\w']+(?:-[\w']+)*"
        for match in re.finditer(pattern, text):
            word = match.group()
            start_idx = match.start()
            end_idx = match.end()
            words_with_pos.append((word, start_idx, end_idx))
        return words_with_pos

    def get_word_at_time(self, current_time):
        if not self.word_timings:
            return None
        for i, (start_time, end_time, word, start_idx, end_idx) in enumerate(
            self.word_timings
        ):
            if start_time <= current_time < end_time:
                return (i, word, start_idx, end_idx)
        if current_time >= self.word_timings[-1][1]:
            last = self.word_timings[-1]
            return (len(self.word_timings) - 1, last[2], last[3], last[4])
        return None

    def reset(self):
        self.word_timings = []
        self.current_word_index = -1
        self.original_text = ""


# ============================================================================
# Main TTSGui Class (PySide6 Version)
# ============================================================================


class TTSGui(
    QMainWindow,
    TTSGuiThemeMixin,
    TTSGuiUiMixin,
    TTSGuiShortcutsMixin,
    TTSGuiTextMixin,
    TTSGuiTranscriptionMixin,
    TTSGuiSSMLMixin,
    TTSGuiConfigMixin,
    TTSGuiVoiceMixin,
    TTSGuiGenerationMixin,
    TTSGuiPlaybackMixin,
    TTSGuiExportMixin,
    TTSGuiLifecycleMixin,
):
    """Main TTS GUI Application using PySide6."""

    def __init__(self):
        # Initialize QMainWindow first
        QMainWindow.__init__(self)

        # Setup crash prevention
        setup_crash_prevention()

        # Store reference to main window for mixins
        self.main_window = self
        self._cleanup_done = False

        self.setWindowTitle("High-Quality English TTS - Sherpa-ONNX Enhanced")
        self.resize(1300, 1200)

        # Dracula theme color scheme
        self.colors = {
            "bg_primary": "#282a36",
            "bg_secondary": "#44475a",
            "bg_tertiary": "#6272a4",
            "bg_accent": "#44475a",
            "fg_primary": "#f8f8f2",
            "fg_secondary": "#bd93f9",
            "fg_muted": "#6272a4",
            "accent_pink": "#ff79c6",
            "accent_pink_hover": "#ff92d0",
            "accent_cyan": "#8be9fd",
            "accent_green": "#50fa7b",
            "accent_orange": "#ffb86c",
            "accent_red": "#ff5555",
            "accent_purple": "#bd93f9",
            "accent_yellow": "#f1fa8c",
            "selection": "#44475a",
            "border": "#6272a4",
            "border_light": "#bd93f9",
        }

        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

        # Initialize helper components
        self.text_processor = TextProcessor()
        self.audio_cache = AudioCache()
        self.performance_monitor = PerformanceMonitor()
        self.audio_stitcher = AudioStitcher(silence_duration=0.3)
        self.audio_exporter = AudioExporter()
        self.follow_along_manager = FollowAlongManager()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # TTS model instances
        self.tts_models = {}
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
        self.auto_play_after_generation = False

        # Text processing options (using simple dict instead of tk.BooleanVar)
        self.text_options = {
            "normalize_whitespace": True,
            "normalize_punctuation": True,
            "remove_urls": False,
            "remove_duplicates": True,
            "remove_word_dashes": True,
            "numbers_to_words": False,
            "expand_abbreviations": True,
            "handle_acronyms": False,
            "add_pauses": False,
        }

        # SSML Support
        self.ssml_processor = SSMLProcessor()
        self.ssml_enabled = False
        self.ssml_auto_detect = True

        # Follow-along word highlighting
        self.follow_along_enabled = True
        self.generated_text_for_follow_along = ""

        # Audio transcription state
        self.asr_models = {}
        self.selected_asr_model_id = None
        self.selected_audio_file = ""
        self.transcription_replace_text = True
        self.transcription_in_progress = False
        self.transcription_cancelled = False
        self.transcription_thread = None

        # Provider selection (GPU auto-detect)
        self.available_onnx_providers = self.detect_available_providers()
        self.use_gpu = bool(
            {
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
            }
            & set(self.available_onnx_providers)
        )

        # Control variable values (accessed via properties in ui.py)
        self.speed_var = 1.0
        self.volume_var = 80
        self.playback_speed_var = 1.0

        # Load persisted settings after defaults are established
        self.load_config()

        # Setup theme and UI
        self.setup_theme()
        self.setup_ui()
        self.update_provider_ui()

        # Setup keyboard shortcuts for power users
        self.setup_keyboard_shortcuts()

        # Check available voices and populate selections (after UI is ready)
        self.check_available_voices()
        self.populate_voice_selections()
        self.apply_config_to_ui()

        # Start model preloading in background
        self.preload_models()

    def closeEvent(self, event):
        """Handle window close event."""
        self.cleanup()
        super().closeEvent(event)
        event.accept()


def main():
    """Main function."""
    # Check if required packages are available
    try:
        import sherpa_onnx
        import pygame
        import soundfile
    except ImportError as e:
        print(f"Required package missing: {e}")
        print("Please install required packages:")
        print("pip install sherpa-onnx pygame soundfile PySide6")
        return

    app = QApplication(sys.argv)

    # Set application-wide properties
    app.setApplicationName("Sherpa-ONNX TTS")
    app.setOrganizationName("Sherpa-ONNX")

    window = TTSGui()
    window.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        pass
    finally:
        window.cleanup()


if __name__ == "__main__":
    main()
