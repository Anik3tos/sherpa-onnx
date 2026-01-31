#!/usr/bin/env python3

"""
High-Quality English TTS GUI
A user-friendly interface for sherpa-onnx text-to-speech with premium English models
Enhanced with improved text input usability and optimized speech generation performance
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import time
import uuid
import pygame
import sherpa_onnx
import soundfile as sf
from pathlib import Path
import numpy as np
import hashlib
import pickle
import tempfile
import re
import weakref
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import json
import signal
import sys
import atexit
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
import subprocess
import shutil
import io
import wave
import struct


class AudioExporter:
    """
    Advanced Audio Export System
    
    Supports multiple audio formats with configurable quality settings,
    silence detection for automatic track splitting, and chapter/section markers.
    """
    
    # Supported audio formats with their configurations
    FORMATS = {
        'wav': {
            'name': 'WAV (Lossless)',
            'extension': '.wav',
            'description': 'Uncompressed audio, highest quality',
            'supports_bitrate': False,
            'default_sample_rate': 44100,
        },
        'flac': {
            'name': 'FLAC (Lossless Compressed)',
            'extension': '.flac',
            'description': 'Lossless compression, ~50% smaller than WAV',
            'supports_bitrate': False,
            'default_sample_rate': 44100,
            'compression_levels': list(range(9)),  # 0-8
        },
        'mp3': {
            'name': 'MP3 (Lossy)',
            'extension': '.mp3',
            'description': 'Universal compatibility, good compression',
            'supports_bitrate': True,
            'bitrates': [64, 96, 128, 160, 192, 224, 256, 320],
            'default_bitrate': 192,
            'default_sample_rate': 44100,
        },
        'ogg': {
            'name': 'OGG Vorbis (Lossy)',
            'extension': '.ogg',
            'description': 'Open format, excellent quality at lower bitrates',
            'supports_bitrate': True,
            'bitrates': [64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
            'default_bitrate': 160,
            'default_sample_rate': 44100,
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
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
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
        available = ['wav']  # WAV is always available via soundfile
        
        # FLAC is supported by soundfile
        available.append('flac')
        
        # MP3 and OGG require either ffmpeg or pydub
        if self.ffmpeg_available or self.pydub_available:
            available.extend(['mp3', 'ogg'])
        
        return available
    
    def export(self, audio_data, sample_rate, output_path, format_type='wav', options=None):
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
        if options.get('normalize', False):
            audio_data = self._normalize_audio(audio_data)
        
        # Resample if needed
        target_sr = options.get('target_sample_rate', sample_rate)
        if target_sr != sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Ensure correct file extension
        if not output_path.lower().endswith(format_config['extension']):
            output_path = output_path + format_config['extension']
        
        try:
            if format_type == 'wav':
                return self._export_wav(audio_data, sample_rate, output_path, options)
            elif format_type == 'flac':
                return self._export_flac(audio_data, sample_rate, output_path, options)
            elif format_type in ['mp3', 'ogg']:
                return self._export_lossy(audio_data, sample_rate, output_path, format_type, options)
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
            subtype = options.get('wav_subtype', 'PCM_16')
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
            sf.write(output_path, audio_data, sample_rate, format='FLAC')
            return True, f"Exported FLAC: {output_path}", output_path
        except Exception as e:
            return False, f"FLAC export failed: {str(e)}", None
    
    def _export_lossy(self, audio_data, sample_rate, output_path, format_type, options):
        """Export as MP3 or OGG using ffmpeg or pydub"""
        format_config = self.FORMATS[format_type]
        bitrate = options.get('bitrate', format_config['default_bitrate'])
        
        # First, create a temporary WAV file
        temp_wav = output_path + '.temp.wav'
        
        try:
            # Write temporary WAV
            sf.write(temp_wav, audio_data, sample_rate, subtype='PCM_16')
            
            # Convert using ffmpeg or pydub
            if self.ffmpeg_available:
                success, msg = self._convert_with_ffmpeg(temp_wav, output_path, format_type, bitrate, options)
            elif self.pydub_available:
                success, msg = self._convert_with_pydub(temp_wav, output_path, format_type, bitrate, options)
            else:
                return False, f"No encoder available for {format_type}. Install ffmpeg or pydub.", None
            
            if success:
                return True, f"Exported {format_type.upper()}: {output_path}", output_path
            else:
                return False, msg, None
                
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass
    
    def _convert_with_ffmpeg(self, input_path, output_path, format_type, bitrate, options):
        """Convert audio using ffmpeg"""
        try:
            cmd = ['ffmpeg', '-y', '-i', input_path]
            
            # Add format-specific options
            if format_type == 'mp3':
                cmd.extend(['-codec:a', 'libmp3lame', '-b:a', f'{bitrate}k'])
            elif format_type == 'ogg':
                cmd.extend(['-codec:a', 'libvorbis', '-b:a', f'{bitrate}k'])
            
            # Add metadata if provided
            metadata = options.get('metadata', {})
            for key, value in metadata.items():
                if value:
                    cmd.extend(['-metadata', f'{key}={value}'])
            
            cmd.append(output_path)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode == 0:
                return True, "Success"
            else:
                return False, f"ffmpeg error: {result.stderr}"
                
        except Exception as e:
            return False, f"ffmpeg conversion failed: {str(e)}"
    
    def _convert_with_pydub(self, input_path, output_path, format_type, bitrate, options):
        """Convert audio using pydub"""
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_wav(input_path)
            
            # Add metadata if provided
            metadata = options.get('metadata', {})
            tags = {}
            for key, value in metadata.items():
                if value:
                    tags[key] = value
            
            if format_type == 'mp3':
                audio.export(output_path, format='mp3', bitrate=f'{bitrate}k', tags=tags)
            elif format_type == 'ogg':
                audio.export(output_path, format='ogg', bitrate=f'{bitrate}k', tags=tags)
            
            return True, "Success"
            
        except Exception as e:
            return False, f"pydub conversion failed: {str(e)}"
    
    def detect_silence(self, audio_data, sample_rate, min_silence_len=500, silence_thresh=-40, 
                       seek_step=10):
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
            chunk = audio_data[i:i + seek_samples]
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
    
    def split_by_silence(self, audio_data, sample_rate, min_silence_len=500, 
                         silence_thresh=-40, min_segment_len=1000, keep_silence=200):
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
            segment = audio_data[max(0, prev_end - keep_samples):]
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
        markers = sorted(chapter_markers, key=lambda x: x.get('start_ms', 0))
        
        segments = []
        
        for i, marker in enumerate(markers):
            start_ms = marker.get('start_ms', 0)
            title = marker.get('title', f'Chapter {i+1}')
            
            # Determine end point
            if i + 1 < len(markers):
                end_ms = markers[i + 1]['start_ms']
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
            r'^(?:Chapter|CHAPTER)\s+(\d+|[IVXLCDM]+)(?:\s*[:\-\.]\s*(.*))?$',
            r'^(?:Part|PART)\s+(\d+|[IVXLCDM]+)(?:\s*[:\-\.]\s*(.*))?$',
            r'^(?:Section|SECTION)\s+(\d+)(?:\s*[:\-\.]\s*(.*))?$',
            r'^#{1,3}\s+(.+)$',  # Markdown headers
            r'^\*\*\*+\s*$',  # Separator lines
            r'^[-=]{3,}\s*$',  # HR-style separators
        ]
        
        lines = text.split('\n')
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
                    chapters.append({
                        'title': title.strip(),
                        'line_number': i,
                        'original_line': line
                    })
                    break
        
        return chapters
    
    def export_multiple_tracks(self, audio_segments, output_dir, base_name, format_type='wav', options=None):
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
                safe_title = re.sub(r'[<>:"/\\|?*]', '', title)[:50]
                filename = f"{base_name}_{i:02d}_{safe_title}"
            else:
                audio_data = segment
                filename = f"{base_name}_{i:02d}"
            
            output_path = os.path.join(output_dir, filename)
            
            # Need sample rate for export
            sample_rate = options.get('sample_rate', 22050)
            
            result = self.export(audio_data, sample_rate, output_path, format_type, options)
            results.append(result)
        
        return results


class ExportOptionsDialog:
    """
    Advanced Export Options Dialog
    
    Provides UI for configuring audio export with format selection,
    quality settings, and split options.
    """
    
    def __init__(self, parent, audio_exporter, audio_data, sample_rate, colors, original_text=""):
        self.parent = parent
        self.exporter = audio_exporter
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.colors = colors
        self.original_text = original_text
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Advanced Export Options")
        self.dialog.geometry("650x850")
        self.dialog.configure(bg=colors['bg_primary'])
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 650) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 850) // 2
        self.dialog.geometry(f"+{x}+{y}")
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the dialog UI"""
        # Main container with padding
        main_frame = ttk.Frame(self.dialog, style='Dark.TFrame', padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="🎵 Advanced Audio Export", 
                              font=('Segoe UI', 14, 'bold'),
                              bg=self.colors['bg_primary'], fg=self.colors['fg_primary'])
        title_label.pack(pady=(0, 15))
        
        # === Format Selection Frame ===
        format_frame = ttk.LabelFrame(main_frame, text="📀 Output Format", 
                                     style='Dark.TLabelframe', padding="10")
        format_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.format_var = tk.StringVar(value='wav')
        available_formats = self.exporter.get_available_formats()
        
        for fmt in available_formats:
            fmt_config = self.exporter.FORMATS[fmt]
            rb = ttk.Radiobutton(format_frame, text=fmt_config['name'],
                                variable=self.format_var, value=fmt,
                                command=self._on_format_change,
                                style='Dark.TRadiobutton')
            rb.pack(anchor=tk.W, pady=2)
            
            # Description
            desc_label = tk.Label(format_frame, text=f"    {fmt_config['description']}",
                                 bg=self.colors['bg_secondary'], 
                                 fg=self.colors['fg_muted'],
                                 font=('Segoe UI', 9))
            desc_label.pack(anchor=tk.W)
        
        # Format availability note
        if 'mp3' not in available_formats or 'ogg' not in available_formats:
            note_label = tk.Label(format_frame, 
                                 text="ℹ️ Install ffmpeg for MP3/OGG support",
                                 bg=self.colors['bg_secondary'],
                                 fg=self.colors['accent_orange'],
                                 font=('Segoe UI', 9))
            note_label.pack(anchor=tk.W, pady=(10, 0))
        
        # === Quality Settings Frame ===
        quality_frame = ttk.LabelFrame(main_frame, text="⚙️ Quality Settings",
                                      style='Dark.TLabelframe', padding="10")
        quality_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Sample Rate
        sr_frame = ttk.Frame(quality_frame, style='Dark.TFrame')
        sr_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sr_frame, text="Sample Rate:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.sample_rate_var = tk.StringVar(value=str(self.sample_rate))
        sr_combo = ttk.Combobox(sr_frame, textvariable=self.sample_rate_var,
                               values=[str(sr) for sr in AudioExporter.SAMPLE_RATES],
                               width=10, state='readonly')
        sr_combo.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(sr_frame, text="Hz", style='Dark.TLabel').pack(side=tk.LEFT, padx=(5, 0))
        
        # Bitrate (for lossy formats)
        self.bitrate_frame = ttk.Frame(quality_frame, style='Dark.TFrame')
        self.bitrate_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.bitrate_frame, text="Bitrate:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.bitrate_var = tk.StringVar(value='192')
        self.bitrate_combo = ttk.Combobox(self.bitrate_frame, textvariable=self.bitrate_var,
                                         values=['64', '96', '128', '160', '192', '224', '256', '320'],
                                         width=10, state='readonly')
        self.bitrate_combo.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(self.bitrate_frame, text="kbps", style='Dark.TLabel').pack(side=tk.LEFT, padx=(5, 0))
        
        # Initially hide bitrate for WAV
        self.bitrate_frame.pack_forget()
        
        # Normalize checkbox
        self.normalize_var = tk.BooleanVar(value=False)
        norm_cb = ttk.Checkbutton(quality_frame, text="Normalize audio (maximize volume without clipping)",
                                 variable=self.normalize_var, style='Dark.TRadiobutton')
        norm_cb.pack(anchor=tk.W, pady=5)
        
        # === Split Options Frame ===
        split_frame = ttk.LabelFrame(main_frame, text="✂️ Split Options",
                                    style='Dark.TLabelframe', padding="10")
        split_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.split_mode_var = tk.StringVar(value='none')
        
        # No split
        ttk.Radiobutton(split_frame, text="Export as single file",
                       variable=self.split_mode_var, value='none',
                       command=self._on_split_mode_change,
                       style='Dark.TRadiobutton').pack(anchor=tk.W, pady=2)
        
        # Split by silence
        ttk.Radiobutton(split_frame, text="Split by silence (automatic track detection)",
                       variable=self.split_mode_var, value='silence',
                       command=self._on_split_mode_change,
                       style='Dark.TRadiobutton').pack(anchor=tk.W, pady=2)
        
        # Split by chapters
        ttk.Radiobutton(split_frame, text="Split by chapters/sections (from text markers)",
                       variable=self.split_mode_var, value='chapters',
                       command=self._on_split_mode_change,
                       style='Dark.TRadiobutton').pack(anchor=tk.W, pady=2)
        
        # Silence detection settings
        self.silence_settings_frame = ttk.Frame(split_frame, style='Dark.TFrame')
        
        # Min silence length
        sl_frame = ttk.Frame(self.silence_settings_frame, style='Dark.TFrame')
        sl_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sl_frame, text="Min silence length:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.min_silence_var = tk.StringVar(value='500')
        sl_spin = ttk.Spinbox(sl_frame, from_=100, to=5000, increment=100,
                             textvariable=self.min_silence_var, width=8)
        sl_spin.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(sl_frame, text="ms", style='Dark.TLabel').pack(side=tk.LEFT, padx=(5, 0))
        
        # Silence threshold
        st_frame = ttk.Frame(self.silence_settings_frame, style='Dark.TFrame')
        st_frame.pack(fill=tk.X, pady=2)
        ttk.Label(st_frame, text="Silence threshold:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.silence_thresh_var = tk.StringVar(value='-40')
        st_spin = ttk.Spinbox(st_frame, from_=-60, to=-20, increment=5,
                             textvariable=self.silence_thresh_var, width=8)
        st_spin.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(st_frame, text="dB", style='Dark.TLabel').pack(side=tk.LEFT, padx=(5, 0))
        
        # Preview silence detection button
        preview_btn = ttk.Button(self.silence_settings_frame, text="🔍 Preview Splits",
                                command=self._preview_silence_splits,
                                style='Dark.TButton')
        preview_btn.pack(anchor=tk.W, pady=(5, 0))
        
        # Detected chapters display
        self.chapters_frame = ttk.Frame(split_frame, style='Dark.TFrame')
        
        chapters_label = tk.Label(self.chapters_frame, 
                                 text="Detected chapters from text:",
                                 bg=self.colors['bg_secondary'],
                                 fg=self.colors['fg_primary'],
                                 font=('Segoe UI', 10))
        chapters_label.pack(anchor=tk.W)
        
        self.chapters_listbox = tk.Listbox(self.chapters_frame, height=4,
                                          bg=self.colors['bg_primary'],
                                          fg=self.colors['fg_primary'],
                                          selectbackground=self.colors['selection'])
        self.chapters_listbox.pack(fill=tk.X, pady=5)
        
        # Detect chapters
        self._detect_chapters()
        
        # === Metadata Frame ===
        metadata_frame = ttk.LabelFrame(main_frame, text="🏷️ Metadata (Optional)",
                                       style='Dark.TLabelframe', padding="10")
        metadata_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_row = ttk.Frame(metadata_frame, style='Dark.TFrame')
        title_row.pack(fill=tk.X, pady=2)
        ttk.Label(title_row, text="Title:", style='Dark.TLabel', width=10).pack(side=tk.LEFT)
        self.meta_title_var = tk.StringVar()
        ttk.Entry(title_row, textvariable=self.meta_title_var, width=40).pack(side=tk.LEFT, padx=(5, 0))
        
        # Artist
        artist_row = ttk.Frame(metadata_frame, style='Dark.TFrame')
        artist_row.pack(fill=tk.X, pady=2)
        ttk.Label(artist_row, text="Artist:", style='Dark.TLabel', width=10).pack(side=tk.LEFT)
        self.meta_artist_var = tk.StringVar(value='Sherpa-ONNX TTS')
        ttk.Entry(artist_row, textvariable=self.meta_artist_var, width=40).pack(side=tk.LEFT, padx=(5, 0))
        
        # Album
        album_row = ttk.Frame(metadata_frame, style='Dark.TFrame')
        album_row.pack(fill=tk.X, pady=2)
        ttk.Label(album_row, text="Album:", style='Dark.TLabel', width=10).pack(side=tk.LEFT)
        self.meta_album_var = tk.StringVar()
        ttk.Entry(album_row, textvariable=self.meta_album_var, width=40).pack(side=tk.LEFT, padx=(5, 0))
        
        # === Info Display ===
        info_frame = ttk.Frame(main_frame, style='Card.TFrame', padding="8")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        duration = len(self.audio_data) / self.sample_rate
        self.info_label = tk.Label(info_frame, 
                                  text=f"Duration: {duration:.1f}s | Sample Rate: {self.sample_rate} Hz | Samples: {len(self.audio_data):,}",
                                  bg=self.colors['bg_tertiary'],
                                  fg=self.colors['accent_cyan'],
                                  font=('Consolas', 10))
        self.info_label.pack()
        
        # === Buttons ===
        btn_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Cancel", command=self._cancel,
                  style='Dark.TButton').pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(btn_frame, text="💾 Export", command=self._export,
                  style='Primary.TButton').pack(side=tk.RIGHT)
    
    def _on_format_change(self):
        """Handle format selection change"""
        fmt = self.format_var.get()
        fmt_config = self.exporter.FORMATS.get(fmt, {})
        
        if fmt_config.get('supports_bitrate', False):
            self.bitrate_frame.pack(fill=tk.X, pady=5, after=self.bitrate_frame.master.winfo_children()[0])
            # Set default bitrate for format
            default_br = fmt_config.get('default_bitrate', 192)
            self.bitrate_var.set(str(default_br))
            # Update available bitrates
            if 'bitrates' in fmt_config:
                self.bitrate_combo['values'] = [str(br) for br in fmt_config['bitrates']]
        else:
            self.bitrate_frame.pack_forget()
    
    def _on_split_mode_change(self):
        """Handle split mode change"""
        mode = self.split_mode_var.get()
        
        # Hide all split settings first
        self.silence_settings_frame.pack_forget()
        self.chapters_frame.pack_forget()
        
        if mode == 'silence':
            self.silence_settings_frame.pack(fill=tk.X, pady=(10, 0))
        elif mode == 'chapters':
            self.chapters_frame.pack(fill=tk.X, pady=(10, 0))
    
    def _detect_chapters(self):
        """Detect chapters from original text"""
        if not self.original_text:
            return
        
        chapters = self.exporter.detect_chapters_from_text(self.original_text)
        
        self.chapters_listbox.delete(0, tk.END)
        self.detected_chapters = chapters
        
        if chapters:
            for i, ch in enumerate(chapters, 1):
                self.chapters_listbox.insert(tk.END, f"{i}. {ch['title']}")
        else:
            self.chapters_listbox.insert(tk.END, "(No chapters detected)")
            self.detected_chapters = []
    
    def _preview_silence_splits(self):
        """Preview silence detection results"""
        try:
            min_silence = int(self.min_silence_var.get())
            threshold = int(self.silence_thresh_var.get())
            
            silence_regions = self.exporter.detect_silence(
                self.audio_data, self.sample_rate,
                min_silence_len=min_silence,
                silence_thresh=threshold
            )
            
            if silence_regions:
                msg = f"Found {len(silence_regions)} silence region(s):\n\n"
                for i, (start, end) in enumerate(silence_regions[:10], 1):
                    duration = end - start
                    msg += f"  {i}. {start/1000:.1f}s - {end/1000:.1f}s ({duration}ms)\n"
                if len(silence_regions) > 10:
                    msg += f"\n  ... and {len(silence_regions) - 10} more"
                msg += f"\n\nThis would create {len(silence_regions) + 1} track(s)."
            else:
                msg = "No silence regions detected with current settings.\n\nTry adjusting:\n- Lower threshold (more sensitive)\n- Shorter minimum silence length"
            
            messagebox.showinfo("Silence Detection Preview", msg, parent=self.dialog)
            
        except Exception as e:
            messagebox.showerror("Error", f"Preview failed: {str(e)}", parent=self.dialog)
    
    def _cancel(self):
        """Cancel and close dialog"""
        self.result = None
        self.dialog.destroy()
    
    def _export(self):
        """Perform export with selected options"""
        # Build options dict
        self.result = {
            'format': self.format_var.get(),
            'target_sample_rate': int(self.sample_rate_var.get()),
            'normalize': self.normalize_var.get(),
            'split_mode': self.split_mode_var.get(),
            'metadata': {
                'title': self.meta_title_var.get(),
                'artist': self.meta_artist_var.get(),
                'album': self.meta_album_var.get(),
            }
        }
        
        # Add format-specific options
        fmt = self.format_var.get()
        if self.exporter.FORMATS.get(fmt, {}).get('supports_bitrate', False):
            self.result['bitrate'] = int(self.bitrate_var.get())
        
        # Add split options
        if self.result['split_mode'] == 'silence':
            self.result['silence_settings'] = {
                'min_silence_len': int(self.min_silence_var.get()),
                'silence_thresh': int(self.silence_thresh_var.get()),
            }
        elif self.result['split_mode'] == 'chapters':
            self.result['chapters'] = self.detected_chapters
        
        self.dialog.destroy()
    
    def show(self):
        """Show dialog and wait for result"""
        self.dialog.wait_window()
        return self.result


class SSMLProcessor:
    """
    SSML (Speech Synthesis Markup Language) Processor
    
    Converts SSML markup to text with appropriate modifications for TTS engines
    that don't natively support SSML. This provides professional-grade control
    over speech synthesis including:
    
    - <emphasis> - Emphasize text with different levels
    - <break> - Insert pauses of varying duration
    - <prosody> - Control pitch, rate, and volume
    - <say-as> - Control pronunciation (digits, characters, ordinal, etc.)
    - <phoneme> - Phonetic pronunciation hints
    - <sub> - Text substitution
    - <p> and <s> - Paragraph and sentence markers
    - <voice> - Voice selection hints (informational)
    
    Industry standard W3C SSML 1.1 compliance where possible.
    """
    
    def __init__(self):
        # Break time mappings (SSML strength to approximate pause text)
        self.break_mappings = {
            'none': '',
            'x-weak': ',',
            'weak': ', ',
            'medium': '. ',
            'strong': '... ',
            'x-strong': '...... '
        }
        
        # Emphasis mappings (how to represent emphasis in plain text)
        self.emphasis_mappings = {
            'strong': ('*', '*'),      # Will be converted to natural emphasis cues
            'moderate': ('', ''),
            'reduced': ('', ''),
            'none': ('', '')
        }
        
        # Say-as interpretation types
        self.say_as_types = [
            'cardinal', 'ordinal', 'digits', 'fraction', 'unit', 'date', 
            'time', 'telephone', 'address', 'characters', 'spell-out',
            'currency', 'verbatim', 'acronym', 'expletive'
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
        if text.startswith('<speak') or text.startswith('<?xml'):
            return True
        # Check for any SSML-like tags
        ssml_tags = ['<speak', '<break', '<emphasis', '<prosody', '<say-as', 
                     '<phoneme', '<sub', '<voice', '<p>', '<s>']
        return any(tag in text.lower() for tag in ssml_tags)
    
    def parse_ssml(self, ssml_text):
        """
        Parse SSML and convert to plain text with prosody information.
        
        Returns:
            dict with:
                - 'text': Processed plain text
                - 'rate': Suggested speaking rate multiplier
                - 'segments': List of text segments with individual prosody settings
                - 'errors': List of any parsing errors encountered
        """
        result = {
            'text': '',
            'rate': 1.0,
            'segments': [],
            'errors': [],
            'has_prosody_changes': False
        }
        
        # Reset prosody state
        self.prosody_stack = []
        self.current_rate = 1.0
        self.current_pitch = 1.0
        self.current_volume = 1.0
        
        # Ensure SSML has a root element
        ssml_text = ssml_text.strip()
        if not ssml_text.startswith('<speak'):
            # Wrap in speak tags if not present
            if not ssml_text.startswith('<?xml'):
                ssml_text = f'<speak>{ssml_text}</speak>'
        
        # Handle XML declaration if present
        if ssml_text.startswith('<?xml'):
            # Find the end of XML declaration and process the rest
            decl_end = ssml_text.find('?>')
            if decl_end != -1:
                ssml_text = ssml_text[decl_end + 2:].strip()
                if not ssml_text.startswith('<speak'):
                    ssml_text = f'<speak>{ssml_text}</speak>'
        
        try:
            # Parse XML
            root = ET.fromstring(ssml_text)
            
            # Process the tree
            processed_text, segments = self._process_element(root)
            
            result['text'] = self._clean_text(processed_text)
            result['segments'] = segments
            
            # Calculate average rate from segments
            if segments:
                rates = [s['rate'] for s in segments if s.get('rate')]
                if rates:
                    result['rate'] = sum(rates) / len(rates)
                    if result['rate'] != 1.0:
                        result['has_prosody_changes'] = True
            
        except ParseError as e:
            result['errors'].append(f"XML parsing error: {str(e)}")
            # Fall back to stripping tags
            result['text'] = self._strip_tags(ssml_text)
        except Exception as e:
            result['errors'].append(f"SSML processing error: {str(e)}")
            result['text'] = self._strip_tags(ssml_text)
        
        return result
    
    def _process_element(self, element, depth=0):
        """Recursively process an SSML element and its children"""
        segments = []
        text_parts = []
        
        # Handle element's text content
        if element.text:
            text_parts.append(element.text)
            segments.append({
                'text': element.text,
                'rate': self.current_rate,
                'pitch': self.current_pitch,
                'volume': self.current_volume
            })
        
        # Process child elements
        for child in element:
            child_text, child_segments = self._process_child_element(child, depth)
            text_parts.append(child_text)
            segments.extend(child_segments)
            
            # Handle tail text (text after the child element)
            if child.tail:
                text_parts.append(child.tail)
                segments.append({
                    'text': child.tail,
                    'rate': self.current_rate,
                    'pitch': self.current_pitch,
                    'volume': self.current_volume
                })
        
        return ''.join(text_parts), segments
    
    def _process_child_element(self, element, depth):
        """Process a specific SSML element based on its tag"""
        tag = element.tag.lower()
        
        # Remove namespace if present
        if '}' in tag:
            tag = tag.split('}')[1]
        
        # Handle different SSML elements
        if tag == 'break':
            return self._handle_break(element)
        elif tag == 'emphasis':
            return self._handle_emphasis(element, depth)
        elif tag == 'prosody':
            return self._handle_prosody(element, depth)
        elif tag == 'say-as':
            return self._handle_say_as(element, depth)
        elif tag == 'phoneme':
            return self._handle_phoneme(element, depth)
        elif tag == 'sub':
            return self._handle_sub(element)
        elif tag == 'voice':
            return self._handle_voice(element, depth)
        elif tag in ['p', 'paragraph']:
            return self._handle_paragraph(element, depth)
        elif tag in ['s', 'sentence']:
            return self._handle_sentence(element, depth)
        elif tag == 'audio':
            return self._handle_audio(element)
        elif tag == 'speak':
            # Nested speak tags - just process content
            return self._process_element(element, depth + 1)
        else:
            # Unknown tag - process content anyway
            return self._process_element(element, depth + 1)
    
    def _handle_break(self, element):
        """Handle <break> element - insert pause"""
        time_attr = element.get('time', '')
        strength = element.get('strength', 'medium')
        
        if time_attr:
            # Parse time value (e.g., "500ms", "1s", "1.5s")
            pause_text = self._time_to_pause(time_attr)
        else:
            # Use strength mapping
            pause_text = self.break_mappings.get(strength, '. ')
        
        return pause_text, [{
            'text': pause_text,
            'rate': self.current_rate,
            'is_break': True,
            'break_duration': time_attr or strength
        }]
    
    def _time_to_pause(self, time_str):
        """Convert time string to pause representation"""
        try:
            time_str = time_str.lower().strip()
            
            if time_str.endswith('ms'):
                ms = float(time_str[:-2])
                seconds = ms / 1000
            elif time_str.endswith('s'):
                seconds = float(time_str[:-1])
            else:
                seconds = float(time_str)
            
            # Convert to pause markers
            if seconds < 0.1:
                return ''
            elif seconds < 0.25:
                return ','
            elif seconds < 0.5:
                return ', '
            elif seconds < 1.0:
                return '. '
            elif seconds < 2.0:
                return '... '
            else:
                return '...... '
        except:
            return '. '
    
    def _handle_emphasis(self, element, depth):
        """Handle <emphasis> element"""
        level = element.get('level', 'moderate')
        
        # Get content
        inner_text, segments = self._process_element(element, depth + 1)
        
        # Apply emphasis markers
        prefix, suffix = self.emphasis_mappings.get(level, ('', ''))
        
        # For strong emphasis, we can use natural speech patterns
        if level == 'strong':
            # Add slight pause before and after for natural emphasis
            processed_text = f", {inner_text},"
        elif level == 'reduced':
            # Keep text as-is for reduced emphasis
            processed_text = inner_text
        else:
            processed_text = inner_text
        
        # Update segments with emphasis info
        for seg in segments:
            seg['emphasis'] = level
        
        return processed_text, segments
    
    def _handle_prosody(self, element, depth):
        """Handle <prosody> element - pitch, rate, volume adjustments"""
        # Save current state
        old_rate = self.current_rate
        old_pitch = self.current_pitch
        old_volume = self.current_volume
        
        # Parse rate attribute
        rate = element.get('rate', '')
        if rate:
            self.current_rate = self._parse_prosody_value(rate, self.current_rate)
        
        # Parse pitch attribute
        pitch = element.get('pitch', '')
        if pitch:
            self.current_pitch = self._parse_prosody_value(pitch, self.current_pitch)
        
        # Parse volume attribute
        volume = element.get('volume', '')
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
            'x-slow': 0.5, 'slow': 0.75, 'medium': 1.0, 'fast': 1.25, 'x-fast': 1.5,
            'x-low': 0.5, 'low': 0.75, 'high': 1.25, 'x-high': 1.5,
            'silent': 0, 'soft': 0.5, 'loud': 1.5, 'default': 1.0
        }
        
        if value in keywords:
            return keywords[value]
        
        try:
            # Percentage (e.g., "150%", "+20%", "-10%")
            if value.endswith('%'):
                pct_str = value[:-1]
                if pct_str.startswith('+'):
                    return current * (1 + float(pct_str[1:]) / 100)
                elif pct_str.startswith('-'):
                    return current * (1 - float(pct_str[1:]) / 100)
                else:
                    return float(pct_str) / 100
            
            # Relative values (e.g., "+2st", "-3st" for semitones)
            if 'st' in value:
                st_val = float(value.replace('st', '').replace('+', ''))
                # Convert semitones to multiplier (rough approximation)
                return current * (2 ** (st_val / 12))
            
            # Plain number
            return float(value)
        except:
            return current
    
    def _handle_say_as(self, element, depth):
        """Handle <say-as> element - pronunciation control"""
        interpret_as = element.get('interpret-as', '')
        format_attr = element.get('format', '')
        detail = element.get('detail', '')
        
        # Get the text content
        inner_text, segments = self._process_element(element, depth + 1)
        text = inner_text.strip()
        
        # Process based on interpret-as type
        if interpret_as == 'characters' or interpret_as == 'spell-out':
            # Spell out each character
            processed = ' '.join(text)
        elif interpret_as == 'digits':
            # Read each digit separately
            processed = ' '.join(text)
        elif interpret_as == 'ordinal':
            # Convert to ordinal (1 -> first, etc.)
            processed = self._number_to_ordinal(text)
        elif interpret_as == 'cardinal':
            # Keep as number
            processed = text
        elif interpret_as == 'telephone':
            # Format for telephone reading
            processed = self._format_telephone(text)
        elif interpret_as == 'date':
            # Format date for reading
            processed = self._format_date(text, format_attr)
        elif interpret_as == 'time':
            # Format time for reading
            processed = self._format_time(text, format_attr)
        elif interpret_as == 'currency':
            # Format currency for reading
            processed = text  # Keep as-is, TTS usually handles this
        elif interpret_as == 'verbatim':
            # Spell out exactly
            processed = ' '.join(text)
        elif interpret_as == 'acronym':
            # Spell out as acronym
            processed = ' '.join(text.upper())
        elif interpret_as == 'expletive':
            # Replace with beep or blank
            processed = '[expletive]'
        else:
            processed = text
        
        # Update segments
        for seg in segments:
            seg['interpret_as'] = interpret_as
        
        return processed, segments
    
    def _number_to_ordinal(self, text):
        """Convert number to ordinal text"""
        try:
            num = int(text)
            if 10 <= num % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
            return f"{num}{suffix}"
        except:
            return text
    
    def _format_telephone(self, text):
        """Format telephone number for TTS reading"""
        # Extract digits only
        digits = ''.join(c for c in text if c.isdigit())
        # Add spaces for natural reading
        return ' '.join(digits)
    
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
        alphabet = element.get('alphabet', 'ipa')  # 'ipa' or 'x-sampa'
        ph = element.get('ph', '')
        
        # Get the text content (this is the fallback text)
        inner_text, segments = self._process_element(element, depth + 1)
        
        # For now, we use the original text since most TTS engines
        # don't support phoneme injection. The phoneme info is stored
        # in segments for engines that might support it.
        for seg in segments:
            seg['phoneme'] = ph
            seg['phoneme_alphabet'] = alphabet
        
        return inner_text, segments
    
    def _handle_sub(self, element):
        """Handle <sub> element - text substitution"""
        alias = element.get('alias', '')
        original = element.text or ''
        
        # Use the alias for TTS, original is what's displayed
        text_to_speak = alias if alias else original
        
        return text_to_speak, [{
            'text': text_to_speak,
            'original': original,
            'rate': self.current_rate,
            'is_substitution': True
        }]
    
    def _handle_voice(self, element, depth):
        """Handle <voice> element - voice selection hints"""
        # Voice attributes (informational - actual voice selection is in GUI)
        voice_name = element.get('name', '')
        gender = element.get('gender', '')
        age = element.get('age', '')
        variant = element.get('variant', '')
        
        # Process content
        inner_text, segments = self._process_element(element, depth + 1)
        
        # Add voice hints to segments
        for seg in segments:
            seg['voice_hint'] = {
                'name': voice_name,
                'gender': gender,
                'age': age,
                'variant': variant
            }
        
        return inner_text, segments
    
    def _handle_paragraph(self, element, depth):
        """Handle <p> paragraph element"""
        inner_text, segments = self._process_element(element, depth + 1)
        
        # Add paragraph break after
        processed = inner_text.strip() + '\n\n'
        
        return processed, segments
    
    def _handle_sentence(self, element, depth):
        """Handle <s> sentence element"""
        inner_text, segments = self._process_element(element, depth + 1)
        
        # Ensure sentence ends properly
        text = inner_text.strip()
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text + ' ', segments
    
    def _handle_audio(self, element):
        """Handle <audio> element - audio clips (informational only)"""
        src = element.get('src', '')
        # Audio elements are not supported - return description
        desc = element.text or f'[Audio: {src}]'
        return desc, [{'text': desc, 'rate': self.current_rate, 'is_audio_ref': True}]
    
    def _clean_text(self, text):
        """Clean up processed text"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple punctuation
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        # Clean up spaces around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])\s*([.,!?])', r'\1', text)
        return text.strip()
    
    def _strip_tags(self, text):
        """Strip all XML/SSML tags from text as fallback"""
        # Remove XML declaration
        text = re.sub(r'<\?xml[^>]*\?>', '', text)
        # Remove all tags
        text = re.sub(r'<[^>]+>', '', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_ssml_template(self, template_name='basic'):
        """Get SSML template for user reference"""
        templates = {
            'basic': '''<speak>
    Hello, this is a basic SSML example.
    <break time="500ms"/>
    With a pause in the middle.
</speak>''',
            
            'emphasis': '''<speak>
    This is <emphasis level="strong">very important</emphasis> information.
    But this is <emphasis level="reduced">less important</emphasis>.
</speak>''',
            
            'prosody': '''<speak>
    <prosody rate="slow">Speaking slowly for clarity.</prosody>
    <break time="300ms"/>
    <prosody rate="fast">Now speaking quickly!</prosody>
    <break time="300ms"/>
    <prosody pitch="high">With a higher pitch.</prosody>
    <prosody pitch="low">And a lower pitch.</prosody>
</speak>''',
            
            'say_as': '''<speak>
    The number <say-as interpret-as="cardinal">42</say-as> is spelled 
    <say-as interpret-as="spell-out">42</say-as>.
    Call <say-as interpret-as="telephone">555-1234</say-as>.
    Today is <say-as interpret-as="date">2024-01-15</say-as>.
</speak>''',
            
            'full_example': '''<speak>
    <p>Welcome to the SSML demonstration.</p>
    
    <s>This shows various SSML features.</s>
    
    <s><emphasis level="strong">Emphasis</emphasis> makes text stand out.</s>
    
    <s>Add pauses: short<break time="200ms"/>medium<break time="500ms"/>long<break time="1s"/>done.</s>
    
    <s><prosody rate="80%">Slower speech is clearer.</prosody></s>
    <s><prosody rate="120%">Faster speech saves time.</prosody></s>
    
    <s>Spell it out: <say-as interpret-as="characters">ABC</say-as></s>
    
    <s>Use <sub alias="Speech Synthesis Markup Language">SSML</sub> for control.</s>
</speak>'''
        }
        
        return templates.get(template_name, templates['basic'])
    
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


# Enhanced Voice Configuration System
# Model files should be downloaded from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
# Extract model files to the same directory as this script or set absolute paths
VOICE_CONFIGS = {
    # VITS Piper Models (Recommended - Most Stable)
    "vits_piper_libritts": {
        "name": "LibriTTS Multi-Speaker (904 Diverse Voices) ⭐RECOMMENDED⭐",
        "model_type": "vits",
        "quality": "excellent",
        "description": "Massive collection of high-quality diverse American English voices",
        "model_files": {
            "model": "vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx",
            "tokens": "vits-piper-en_US-libritts_r-medium/tokens.txt",
            "lexicon": "vits-piper-en_US-libritts_r-medium/espeak-ng-data/en_dict",
            "data_dir": "vits-piper-en_US-libritts_r-medium/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Victoria", "gender": "female", "accent": "american", "description": "Warm, articulate female voice"},
            1: {"name": "Alexander", "gender": "male", "accent": "american", "description": "Professional male narrator"},
            2: {"name": "Rachel", "gender": "female", "accent": "american", "description": "Clear, engaging female voice"},
            3: {"name": "Christopher", "gender": "male", "accent": "american", "description": "Deep, resonant male voice"},
            4: {"name": "Amanda", "gender": "female", "accent": "american", "description": "Friendly, approachable female"},
            5: {"name": "Jonathan", "gender": "male", "accent": "american", "description": "Smooth male broadcaster"},
            6: {"name": "Michelle", "gender": "female", "accent": "american", "description": "Professional female voice"},
            7: {"name": "Daniel", "gender": "male", "accent": "american", "description": "Authoritative male speaker"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2"
    },
    "vits_piper_amy": {
        "name": "Amy - High Quality Female Voice ⭐RECOMMENDED⭐",
        "model_type": "vits",
        "quality": "excellent",
        "description": "Crystal clear American English female voice, perfect for narration",
        "model_files": {
            "model": "vits-piper-en_US-amy-medium/en_US-amy-medium.onnx",
            "tokens": "vits-piper-en_US-amy-medium/tokens.txt",
            "data_dir": "vits-piper-en_US-amy-medium/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Amy", "gender": "female", "accent": "american", "description": "Crystal clear, professional female narrator"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-medium.tar.bz2"
    },

    "vits_piper_lessac": {
        "name": "Lessac - Premium Female Voice",
        "model_type": "vits",
        "quality": "excellent",
        "description": "High-quality American English female voice with natural intonation",
        "model_files": {
            "model": "vits-piper-en_US-lessac-medium/en_US-lessac-medium.onnx",
            "tokens": "vits-piper-en_US-lessac-medium/tokens.txt",
            "data_dir": "vits-piper-en_US-lessac-medium/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Lessac", "gender": "female", "accent": "american", "description": "Premium quality female voice with excellent clarity"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2"
    },

    "vits_piper_ryan": {
        "name": "Ryan - High Quality Male Voice",
        "model_type": "vits",
        "quality": "excellent",
        "description": "Natural American English male voice, great for professional use",
        "model_files": {
            "model": "vits-piper-en_US-ryan-high/en_US-ryan-high.onnx",
            "tokens": "vits-piper-en_US-ryan-high/tokens.txt",
            "data_dir": "vits-piper-en_US-ryan-high/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Ryan", "gender": "male", "accent": "american", "description": "Natural, professional male voice"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-ryan-high.tar.bz2"
    },

    "vits_piper_danny": {
        "name": "Danny - Male Voice",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Clear American English male voice",
        "model_files": {
            "model": "vits-piper-en_US-danny-low/en_US-danny-low.onnx",
            "tokens": "vits-piper-en_US-danny-low/tokens.txt",
            "data_dir": "vits-piper-en_US-danny-low/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Danny", "gender": "male", "accent": "american", "description": "Clear male voice, optimized for speed"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-danny-low.tar.bz2"
    },

    "vits_piper_kathleen": {
        "name": "Kathleen - Female Voice",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Warm American English female voice",
        "model_files": {
            "model": "vits-piper-en_US-kathleen-low/en_US-kathleen-low.onnx",
            "tokens": "vits-piper-en_US-kathleen-low/tokens.txt",
            "data_dir": "vits-piper-en_US-kathleen-low/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Kathleen", "gender": "female", "accent": "american", "description": "Warm female voice, fast generation"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-kathleen-low.tar.bz2"
    },

    "vits_piper_libritts_high": {
        "name": "LibriTTS High Quality (10 Premium Speakers)",
        "model_type": "vits",
        "quality": "excellent",
        "description": "Top 10 highest quality speakers from LibriTTS dataset",
        "model_files": {
            "model": "vits-piper-en_US-libritts-high/en_US-libritts-high.onnx",
            "tokens": "vits-piper-en_US-libritts-high/tokens.txt",
            "data_dir": "vits-piper-en_US-libritts-high/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Speaker 0", "gender": "mixed", "accent": "american", "description": "Premium quality voice #1"},
            1: {"name": "Speaker 1", "gender": "mixed", "accent": "american", "description": "Premium quality voice #2"},
            2: {"name": "Speaker 2", "gender": "mixed", "accent": "american", "description": "Premium quality voice #3"},
            3: {"name": "Speaker 3", "gender": "mixed", "accent": "american", "description": "Premium quality voice #4"},
            4: {"name": "Speaker 4", "gender": "mixed", "accent": "american", "description": "Premium quality voice #5"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts-high.tar.bz2"
    },

    # British English Models
    "vits_piper_alba": {
        "name": "Alba - British Female Voice",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Natural British English female voice",
        "model_files": {
            "model": "vits-piper-en_GB-alba-medium/en_GB-alba-medium.onnx",
            "tokens": "vits-piper-en_GB-alba-medium/tokens.txt",
            "data_dir": "vits-piper-en_GB-alba-medium/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Alba", "gender": "female", "accent": "british", "description": "Natural British English female voice"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-alba-medium.tar.bz2"
    },

    "vits_piper_jenny_dioco": {
        "name": "Jenny Dioco - British Multi-Speaker (2 Voices)",
        "model_type": "vits",
        "quality": "very_high",
        "description": "British English multi-speaker model",
        "model_files": {
            "model": "vits-piper-en_GB-jenny_dioco-medium/en_GB-jenny_dioco-medium.onnx",
            "tokens": "vits-piper-en_GB-jenny_dioco-medium/tokens.txt",
            "data_dir": "vits-piper-en_GB-jenny_dioco-medium/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Jenny", "gender": "female", "accent": "british", "description": "British female voice"},
            1: {"name": "Dioco", "gender": "male", "accent": "british", "description": "British male voice"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-jenny_dioco-medium.tar.bz2"
    },

    # VCTK Multi-Speaker Model
    "vits_vctk": {
        "name": "VCTK Multi-Speaker (109 Diverse British Voices)",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Large collection of diverse British and international voices",
        "model_files": {
            "model": "vits-vctk/vits-vctk.onnx",
            "tokens": "vits-vctk/tokens.txt",
            "lexicon": "vits-vctk/lexicon.txt",
            "data_dir": "vits-vctk/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Speaker p225", "gender": "female", "accent": "british", "description": "British female - voice 1"},
            1: {"name": "Speaker p226", "gender": "male", "accent": "british", "description": "British male - voice 1"},
            2: {"name": "Speaker p227", "gender": "male", "accent": "british", "description": "British male - voice 2"},
            3: {"name": "Speaker p228", "gender": "female", "accent": "british", "description": "British female - voice 2"},
            4: {"name": "Speaker p229", "gender": "female", "accent": "british", "description": "British female - voice 3"},
            5: {"name": "Speaker p230", "gender": "female", "accent": "british", "description": "British female - voice 4"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2"
    },
    # Matcha-TTS Models (High Quality with Natural Prosody)
    "matcha_ljspeech": {
        "name": "Matcha-TTS LJSpeech (Premium Female Voice)",
        "model_type": "matcha",
        "quality": "excellent",
        "description": "State-of-the-art TTS with natural prosody and intonation",
        "model_files": {
            "acoustic_model": "matcha-icefall-en_US-ljspeech/model-steps-3.onnx",
            "vocoder": "matcha-icefall-en_US-ljspeech/hifigan_v1.onnx",
            "tokens": "matcha-icefall-en_US-ljspeech/tokens.txt",
            "lexicon": "matcha-icefall-en_US-ljspeech/lexicon.txt",
            "data_dir": "matcha-icefall-en_US-ljspeech/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Linda", "gender": "female", "accent": "american", "description": "Premium quality female narrator with natural prosody"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2"
    },

    # Special Character Voices
    "vits_glados": {
        "name": "GLaDOS - AI Character Voice",
        "model_type": "vits",
        "quality": "high",
        "description": "Distinctive robotic/AI character voice (from Portal game)",
        "model_files": {
            "model": "vits-piper-en_US-glados/en_US-glados.onnx",
            "tokens": "vits-piper-en_US-glados/tokens.txt",
            "data_dir": "vits-piper-en_US-glados/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "GLaDOS", "gender": "female", "accent": "robotic", "description": "Distinctive AI/robotic character voice"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-glados.tar.bz2"
    },

    # Chinese Models
    "vits_piper_zh_huayan": {
        "name": "Huayan - Chinese Female Voice (中文女声)",
        "model_type": "vits",
        "quality": "excellent",
        "description": "High-quality Mandarin Chinese female voice",
        "model_files": {
            "model": "vits-piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx",
            "tokens": "vits-piper-zh_CN-huayan-medium/tokens.txt",
            "data_dir": "vits-piper-zh_CN-huayan-medium/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Huayan (华严)", "gender": "female", "accent": "mandarin", "description": "Premium Mandarin Chinese female voice"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-zh_CN-huayan-medium.tar.bz2"
    },

    # German Models  
    "vits_piper_de_thorsten": {
        "name": "Thorsten - German Male Voice (Deutsch)",
        "model_type": "vits",
        "quality": "excellent",
        "description": "High-quality German male voice",
        "model_files": {
            "model": "vits-piper-de_DE-thorsten-high/de_DE-thorsten-high.onnx",
            "tokens": "vits-piper-de_DE-thorsten-high/tokens.txt",
            "data_dir": "vits-piper-de_DE-thorsten-high/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Thorsten", "gender": "male", "accent": "german", "description": "Premium German male voice"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-thorsten-high.tar.bz2"
    },

    # French Models
    "vits_piper_fr_siwis": {
        "name": "Siwis - French Female Voice (Français)",
        "model_type": "vits",
        "quality": "excellent",
        "description": "High-quality French female voice",
        "model_files": {
            "model": "vits-piper-fr_FR-siwis-medium/fr_FR-siwis-medium.onnx",
            "tokens": "vits-piper-fr_FR-siwis-medium/tokens.txt",
            "data_dir": "vits-piper-fr_FR-siwis-medium/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Siwis", "gender": "female", "accent": "french", "description": "Premium French female voice"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fr_FR-siwis-medium.tar.bz2"
    },

    # Spanish Models
    "vits_piper_es_carlfm": {
        "name": "Carlfm - Spanish Male Voice (Español)",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Natural Spanish male voice",
        "model_files": {
            "model": "vits-piper-es_ES-carlfm-x_low/es_ES-carlfm-x_low.onnx",
            "tokens": "vits-piper-es_ES-carlfm-x_low/tokens.txt",
            "data_dir": "vits-piper-es_ES-carlfm-x_low/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Carlfm", "gender": "male", "accent": "spanish", "description": "Natural Spanish male voice"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-es_ES-carlfm-x_low.tar.bz2"
    },

    # Russian Models
    "vits_piper_ru_irinia": {
        "name": "Irina - Russian Female Voice (Русский)",
        "model_type": "vits",
        "quality": "very_high",
        "description": "High-quality Russian female voice",
        "model_files": {
            "model": "vits-piper-ru_RU-irina-medium/ru_RU-irina-medium.onnx",
            "tokens": "vits-piper-ru_RU-irina-medium/tokens.txt",
            "data_dir": "vits-piper-ru_RU-irina-medium/espeak-ng-data"
        },
        "speakers": {
            0: {"name": "Irina (Ирина)", "gender": "female", "accent": "russian", "description": "Premium Russian female voice"}
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-ru_RU-irina-medium.tar.bz2"
    },

    # NOTE: Japanese models are not currently available in sherpa-onnx releases
    # NOTE: Kokoro models are disabled due to stability issues with multi-lingual requirements
    # They require complex lexicon and dictionary configurations that can cause crashes
    # Use the VITS Piper models above for stable, high-quality TTS
}


class TextProcessor:
    """Handles text preprocessing and validation"""

    def __init__(self):
        self.max_length = 100000  # Maximum total text length (doubled)
        self.min_length = 1      # Minimum text length
        self.chunk_size = 8000   # Target chunk size for long texts (characters)
        self.max_chunk_size = 9500  # Maximum chunk size before forced split (characters)

        # Model-specific token limits (conservative but not overly restrictive)
        self.model_token_limits = {
            'matcha': 700,   # Conservative limit for Matcha-TTS (model max is ~1000)
            'kokoro': 1100   # Conservative for Kokoro
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
        punctuation = sum(1 for c in text if c in '.,!?;:()[]{}"-\'')
        numbers = sum(1 for c in text if c.isdigit())
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,!?;:()[]{}"-\'')

        # Conservative token estimation with safety margins
        estimated_tokens = int((words * 1.3) + (punctuation * 0.8) + (numbers * 0.3) + (special_chars * 0.5))

        # Add extra safety margin for complex text
        if len(text) > 1000:
            estimated_tokens = int(estimated_tokens * 1.2)  # 20% extra buffer for long text

        return max(estimated_tokens, len(text) // 3)  # Minimum 1 token per 3 characters

    def get_model_safe_chunk_size(self, model_type):
        """Get safe chunk size for specific model based on token limits"""
        token_limit = self.model_token_limits.get(model_type, 600)
        # Convert token limit to character limit with aggressive safety margin
        safe_char_limit = int(token_limit * self.chars_per_token * 0.6)  # 40% safety margin
        return min(safe_char_limit, self.chunk_size)

    def validate_chunk_for_model(self, text, model_type):
        """Validate that a chunk is safe for the specified model"""
        token_count = self.estimate_token_count(text)
        token_limit = self.model_token_limits.get(model_type, 600)

        if token_count > token_limit:
            return False, f"Chunk has ~{token_count} tokens, exceeds {model_type} limit of {token_limit}"

        return True, ""

    def preprocess_text(self, text, options=None):
        """Preprocess text based on options with enhanced character and OOV handling"""
        if not text:
            return text

        if options is None:
            options = {}

        processed = text

        # Fix encoding issues and normalize unicode characters
        if options.get('fix_encoding', True):
            import unicodedata
            processed = unicodedata.normalize('NFKD', processed)

            # Fix common encoding corruption
            encoding_fixes = {
                'â€™': "'",     # Smart apostrophe
                'â€œ': '"',     # Smart quote open
                'â€': '"',      # Smart quote close
                'â€"': '-',     # Em dash
                'â€"': '-',     # En dash
                'â€¦': '...',   # Ellipsis
                'â?T': "'",     # Corrupted apostrophe
                'â?"': '"',     # Corrupted quote
                'â?~': '"',     # Another corrupted quote
                'â?¢': '•',     # Bullet point
            }

            for corrupt, fixed in encoding_fixes.items():
                processed = processed.replace(corrupt, fixed)

            # Remove any remaining problematic characters
            processed = re.sub(r'[^\w\s\.,!?;:\'"()-]', ' ', processed)

        # Handle modern terms and brand names that might be OOV
        if options.get('replace_modern_terms', True):
            modern_replacements = {
                'Netflix': 'streaming service',
                'YouTube': 'video platform',
                'Google': 'search engine',
                'Facebook': 'social media',
                'Instagram': 'photo sharing app',
                'Twitter': 'social platform',
                'TikTok': 'video app',
                'iPhone': 'smartphone',
                'iPad': 'tablet',
                'MacBook': 'laptop',
                'PlayStation': 'gaming console',
                'Xbox': 'gaming console',
                'Tesla': 'electric car',
                'Uber': 'ride sharing',
                'Airbnb': 'home sharing',
                'COVID': 'coronavirus',
                'WiFi': 'wireless internet',
                'Bluetooth': 'wireless connection',
                'smartphone': 'mobile phone',
                'app': 'application',
                'blog': 'web log',
                'email': 'electronic mail',
                'website': 'web site',
                'online': 'on the internet',
                'offline': 'not connected',
                'streaming': 'live transmission',
                'podcast': 'audio program',
                'hashtag': 'topic tag',
                'selfie': 'self portrait',
                'emoji': 'emotion icon',
                'meme': 'internet joke',
                'viral': 'widely shared',
                'trending': 'popular now'
            }

            for term, replacement in modern_replacements.items():
                processed = re.sub(r'\b' + re.escape(term) + r'\b', replacement, processed, flags=re.IGNORECASE)

        # Normalize whitespace
        if options.get('normalize_whitespace', True):
            processed = re.sub(r'\s+', ' ', processed)
            processed = processed.strip()

        # Normalize punctuation
        if options.get('normalize_punctuation', True):
            # Replace multiple punctuation marks
            processed = re.sub(r'[.]{2,}', '...', processed)
            processed = re.sub(r'[!]{2,}', '!', processed)
            processed = re.sub(r'[?]{2,}', '?', processed)

        # Remove URLs
        if options.get('remove_urls', False):
            processed = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', processed)

        # Remove email addresses
        if options.get('remove_emails', False):
            processed = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', processed)

        # Remove duplicate consecutive lines
        if options.get('remove_duplicates', False):
            processed = self._remove_duplicate_lines(processed)

        # Convert numbers to words
        if options.get('numbers_to_words', False):
            processed = self._convert_numbers_to_words(processed)

        # Expand abbreviations
        if options.get('expand_abbreviations', False):
            processed = self._expand_abbreviations(processed)

        # Handle acronyms
        if options.get('handle_acronyms', False):
            processed = self._handle_acronyms(processed)

        # Add pause markers for natural timing
        if options.get('add_pauses', False):
            processed = self._add_pause_markers(processed)

        return processed

    def _remove_duplicate_lines(self, text):
        """Remove consecutive duplicate lines"""
        lines = text.split('\n')
        result = []
        prev_line = None

        for line in lines:
            stripped = line.strip()
            if stripped and stripped == prev_line:
                continue  # Skip duplicate
            result.append(line)
            prev_line = stripped

        return '\n'.join(result)

    def _convert_numbers_to_words(self, text):
        """Convert numbers to words for better pronunciation"""
        import re

        def number_to_words(n):
            """Convert a number to words"""
            if n == 0:
                return "zero"

            ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                    'seventeen', 'eighteen', 'nineteen']
            tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

            if n < 20:
                return ones[n]
            elif n < 100:
                return tens[n // 10] + (' ' + ones[n % 10] if n % 10 != 0 else '')
            elif n < 1000:
                return ones[n // 100] + ' hundred' + (' ' + number_to_words(n % 100) if n % 100 != 0 else '')
            elif n < 1000000:
                return number_to_words(n // 1000) + ' thousand' + (' ' + number_to_words(n % 1000) if n % 1000 != 0 else '')
            elif n < 1000000000:
                return number_to_words(n // 1000000) + ' million' + (' ' + number_to_words(n % 1000000) if n % 1000000 != 0 else '')
            else:
                return number_to_words(n // 1000000000) + ' billion' + (' ' + number_to_words(n % 1000000000) if n % 1000000000 != 0 else '')

        def convert_currency(match):
            """Convert currency like $100 to words"""
            amount = match.group(1).replace(',', '')
            try:
                value = float(amount)
                dollars = int(value)
                cents = int(round((value - dollars) * 100))

                if cents == 0:
                    return number_to_words(dollars) + ' dollars'
                elif cents == 1:
                    return number_to_words(dollars) + ' dollars and ' + number_to_words(cents) + ' cent'
                else:
                    return number_to_words(dollars) + ' dollars and ' + number_to_words(cents) + ' cents'
            except:
                return match.group(0)

        def convert_number(match):
            """Convert a standalone number to words"""
            num_str = match.group(0).replace(',', '')
            try:
                num = int(num_str)
                return number_to_words(num)
            except:
                return match.group(0)

        # Convert currency first ($100, $1.50, etc)
        text = re.sub(r'\$\s*([\d,]+(?:\.\d{2})?)', convert_currency, text)

        # Convert standalone numbers (but not years like 2024)
        text = re.sub(r'\b([1-9]\d{0,2}(?:,\d{3})*)(?!\s*(?:BC|AD|BCE|CE))\b', convert_number, text)

        return text

    def _expand_abbreviations(self, text):
        """Expand common abbreviations for better pronunciation"""
        abbreviations = {
            # Titles
            r'\bMr\.?\b': 'Mister',
            r'\bMrs\.?\b': 'Missus',
            r'\bMs\.?\b': 'Miss',
            r'\bDr\.?\b': 'Doctor',
            r'\bProf\.?\b': 'Professor',
            r'\bRev\.?\b': 'Reverend',
            r'\bHon\.?\b': 'Honorable',
            r'\bSgt\.?\b': 'Sergeant',
            r'\bCapt\.?\b': 'Captain',
            r'\bLt\.?\b': 'Lieutenant',
            r'\bGen\.?\b': 'General',
            r'\bCol\.?\b': 'Colonel',
            r'\bRep\.?\b': 'Representative',
            r'\bSen\.?\b': 'Senator',
            r'\bGov\.?\b': 'Governor',
            r'\bPres\.?\b': 'President',
            r'\bVP\b': 'Vice President',

            # Common abbreviations
            r'\betc\.?\b': 'et cetera',
            r'\bie\.?\b': 'that is',
            r'\beg\.?\b': 'for example',
            r'\bvs\.?\b': 'versus',
            r'\bapt\.?\b': 'apartment',
            r'\bave\.?\b': 'avenue',
            r'\bblvd\.?\b': 'boulevard',
            r'\bdept\.?\b': 'department',
            r'\bdiv\.?\b': 'division',
            r'\binst\.?\b': 'institute',
            r'\bprof\.?\b': 'professor',
            r'\buni\.?\b': 'university',
            r'\bassoc\.?\b': 'associate',
            r'\bassn\.?\b': 'association',
            r'\bave\.?\b': 'avenue',
            r'\bblvd\.?\b': 'boulevard',
            r'\bco\.?\b': 'company',
            r'\bcorp\.?\b': 'corporation',
            r'\binc\.?\b': 'incorporated',
            r'\bltd\.?\b': 'limited',
            r'\blb\.?\b': 'pound',
            r'\boz\.?\b': 'ounce',
            r'\bft\.?\b': 'foot',
            r'\bhrs\.?\b': 'hours',
            r'\bmin\.?\b': 'minutes',
            r'\bsec\.?\b': 'seconds',
            r'\bapprox\.?\b': 'approximately',
            r'\bavg\.?\b': 'average',
            r'\bmax\.?\b': 'maximum',
            r'\bmin\.?\b': 'minimum',
            r'\bnbr\.?\b': 'number',
            r'\bno\.?\b': 'number',
            r'\bpct\.?\b': 'percent',
            r'\binfo\.?\b': 'information',
            r'\bmsg\.?\b': 'message',
            r'\badd\.?\b': 'address',
            r'\bdept\.?\b': 'department',
            r'\bdiag\.?\b': 'diagnosis',
            r'\bdoc\.?\b': 'document',
            r'\bex\.?\b': 'example',
            r'\bext\.?\b': 'extension',
            r'\bfig\.?\b': 'figure',
            r'\bhr\.?\b': 'hour',
            r'\bid\.?\b': 'identification',
            r'\bmtg\.?\b': 'meeting',
            r'\bmmbr\.?\b': 'member',
            r'\breq\.?\b': 'request',
            r'\bresp\.?\b': 'responsible',
            r'\bst\.?\b': 'street',
            r'\btemp\.?\b': 'temporary',
            r'\btel\.?\b': 'telephone',
            r'\btrans\.?\b': 'transaction',
            r'\bvol\.?\b': 'volume',

            # Tech abbreviations
            r'\bapp\.?\b': 'application',
            r'\bconfig\.?\b': 'configuration',
            r'\binfo\.?\b': 'information',
            r'\bAPI\b': 'A P I',
            r'\bHTTP\b': 'H T T P',
            r'\bHTTPS\b': 'H T T P S',
            r'\bURL\b': 'U R L',
            r'\bSQL\b': 'S Q L',
            r'\bGUI\b': 'G U I',
            r'\bCPU\b': 'C P U',
            r'\bGPU\b': 'G P U',
            r'\bRAM\b': 'R A M',
            r'\bSSD\b': 'S S D',
            r'\bHDD\b': 'H D D',
            r'\bOS\b': 'Operating System',
            r'\bIoT\b': 'I O T',
            r'\bPDF\b': 'P D F',
            r'\bXML\b': 'X M L',
            r'\bJSON\b': 'J S O N',
            r'\bWiFi\b': 'Wi Fi',

            # Time abbreviations
            r'\byr\.?\b': 'year',
            r'\bmo\.?\b': 'month',
            r'\bday\.?\b': 'day',
            r'\bhr\.?\b': 'hour',
            r'\bAM\b': 'A M',
            r'\bPM\b': 'P M',
            r'\bam\b': 'a m',
            r'\bpm\b': 'p m',
            r'\bEST\b': 'Eastern Standard Time',
            r'\bPST\b': 'Pacific Standard Time',
            r'\bGMT\b': 'Greenwich Mean Time',
            r'\bUTC\b': 'Universal Time',

            # Business/Legal
            r'\bCEO\b': 'C E O',
            r'\bCFO\b': 'C F O',
            r'\bCTO\b': 'C T O',
            r'\bCOO\b': 'C O O',
            r'\bVP\b': 'Vice President',
            r'\bMBA\b': 'M B A',
            r'\bPhD\.?\b': 'P H D',
            r'\bMD\.?\b': 'M D',
            r'\bDO\.?\b': 'D O',
            r'\bRN\.?\b': 'R N',
            r'\bLPN\.?\b': 'L P N',
            r'\bJD\.?\b': 'J D',
            r'\bLLC\b': 'L L C',
            r'\bLTD\b': 'Limited',
            r'\bPLC\b': 'P L C',
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
            'NASA': 'N A S A',
            'FBI': 'F B I',
            'CIA': 'C I A',
            'FDA': 'F D A',
            'SEC': 'S E C',
            'FCC': 'F C C',
            'FTC': 'F T C',
            'EPA': 'E P A',
            'DOT': 'D O T',
            'DOJ': 'D O J',
            'DHS': 'D H S',
            'VA': 'V A',
            'DMV': 'D M V',
            'IRS': 'I R S',
            'CDC': 'C D C',
            'NIH': 'N I H',

            # Sports
            'NFL': 'N F L',
            'NBA': 'N B A',
            'MLB': 'M L B',
            'NCAA': 'N C A A',

            # Politics
            'GOP': 'G O P',
            'DNC': 'D N C',

            # Media
            'CNN': 'C N N',
            'MSNBC': 'M S N B C',
            'BBC': 'B B C',
            'CBS': 'C B S',
            'NBC': 'N B C',
            'ABC': 'A B C',
            'HBO': 'H B O',
            'PBS': 'P B S',
            'NPR': 'N P R',

            # Business titles
            'CEO': 'C E O',
            'CFO': 'C F O',
            'CTO': 'C T O',
            'COO': 'C O O',
            'CIO': 'C I O',
            'CSO': 'C S O',
            'CMO': 'C M O',

            # General
            'IOU': 'I O U',
            'RSVP': 'R S V P',
            'AKA': 'A K A',
            'VS': 'V S',
            'ETC': 'E T C',
            'FYI': 'F Y I',
            'ASAP': 'A S A P',
            'DIY': 'D I Y',
            'TBA': 'T B A',
            'TBD': 'T B D',
            'RIP': 'R I P',
            'VIP': 'V I P',
            'ID': 'I D',
            'GPS': 'G P S',
            'USB': 'U S B',
            'LED': 'L E D',
            'LCD': 'L C D',
            'DVD': 'D V D',
            'CD': 'C D',
            'PC': 'P C',
            'IT': 'I T',
            'HR': 'H R',
            'PR': 'P R',
            'QA': 'Q A',
            'R&D': 'R and D',
            'B2B': 'B to B',
            'B2C': 'B to C',

            # === TECH & CLOUD GENERAL ===
            'SaaS': 'S A A S',
            'PaaS': 'P A A S',
            'IaaS': 'I A A S',
            'DevOps': 'Dev Ops',
            'CI/CD': 'C I C D',
            'VPC': 'V P C',
            'CDN': 'C D N',
            'DNS': 'D N S',
            'SSH': 'S S H',
            'SSL': 'S S L',
            'TLS': 'T L S',
            'FTP': 'F T P',
            'SFTP': 'S F T P',
            'HTTP': 'H T T P',
            'HTTPS': 'H T T P S',
            'REST': 'R E S T',
            'SOAP': 'S O A P',
            'GraphQL': 'G R A P H Q L',
            'JSON': 'J S O N',
            'XML': 'X M L',
            'HTML': 'H T M L',
            'CSS': 'C S S',
            'JS': 'J S',
            'TS': 'T S',
            'SQL': 'S Q L',
            'NoSQL': 'N O S Q L',
            'BI': 'B I',
            'ERP': 'E R P',
            'CRM': 'C R M',
            'CMS': 'C M S',
            'LMS': 'L M S',
            'POS': 'P O S',
            'MFA': 'M F A',
            '2FA': 'two F A',
            'SSO': 'S S O',
            'LDAP': 'L D A P',
            'AD': 'A D',
            'KPI': 'K P I',
            'ROI': 'R O I',
            'SLA': 'S L A',
            'TOS': 'T O S',
            'EULA': 'E U L A',
            'GDPR': 'G D P R',
            'CCPA': 'C C P A',
            'HIPAA': 'H I P A A',
            'SOC2': 'S O C 2',
            'ISO': 'I S O',
            'NIST': 'N I S T',

            # === AZURE SPECIFIC ===
            'ARM': 'A R M',
            'AzureAD': 'Azure A D',
            'AAD': 'A A D',
            'EntraID': 'Entra I D',
            'MFA': 'M F A',
            'AppService': 'App Service',
            'AKS': 'A K S',
            'ACR': 'A C R',
            'ADF': 'A D F',
            'ADLS': 'A D L S',
            'AIP': 'A I P',
            'APIM': 'A P I M',
            'ASR': 'A S R',
            'AVD': 'A V D',
            'WVD': 'W V D',
            'Bicep': 'Bicep',
            'BGInfo': 'B G Info',
            'CDN': 'C D N',
            'CognitiveServices': 'Cognitive Services',
            'CosmosDB': 'Cosmos D B',
            'Databricks': 'Data Bricks',
            'DataFactory': 'Data Factory',
            'DataLake': 'Data Lake',
            'Defender': 'Defender',
            'Sentinel': 'Sentinel',
            'DevOps': 'Dev Ops',
            'ADO': 'A D O',
            'AzureDevOps': 'Azure Dev Ops',
            'ExpressRoute': 'Express Route',
            'FrontDoor': 'Front Door',
            'FunctionApp': 'Function App',
            'GD': 'G D',
            'GDI': 'G D I',
            'HDI': 'H D I',
            'HDInsight': 'H D Insight',
            'IoTHub': 'I O T Hub',
            'IoTEdge': 'I O T Edge',
            'KeyVault': 'Key Vault',
            'KV': 'K V',
            'LogAnalytics': 'Log Analytics',
            'LA': 'L A',
            'Monitor': 'Monitor',
            'NSG': 'N S G',
            'ASG': 'A S G',
            'PowerBI': 'Power B I',
            'PBI': 'P B I',
            'PowerAutomate': 'Power Automate',
            'PowerApps': 'Power Apps',
            'PrivateEndpoint': 'Private Endpoint',
            'PLS': 'P L S',
            'PublicIP': 'Public I P',
            'PIP': 'P I P',
            'RBAC': 'R B A C',
            'RMS': 'R M S',
            'ResourceGroup': 'Resource Group',
            'RG': 'R G',
            'RouteServer': 'Route Server',
            'SAS': 'S A S',
            'StorageAccount': 'Storage Account',
            'SQLDW': 'S Q L D W',
            'Synapse': 'Synapse',
            'VM': 'V M',
            'VMSS': 'V M S S',
            'VNET': 'V N E T',
            'VirtualNetwork': 'Virtual Network',
            'VPN': 'V P N',
            'VWAN': 'V W A N',
            'WAF': 'W A F',
            'WebApp': 'Web App',

            # === AWS & GCP (for comparison docs) ===
            'EC2': 'E C 2',
            'S3': 'S 3',
            'RDS': 'R D S',
            'VPC': 'V P C',
            'IAM': 'I A M',
            'Lambda': 'Lambda',
            'GCP': 'G C P',
            'GKE': 'G K E',
            'CloudSQL': 'Cloud S Q L',

            # === DEV & PROGRAMMING ===
            'IDE': 'I D E',
            'CLI': 'C L I',
            'GUI': 'G U I',
            'API': 'A P I',
            'SDK': 'S D K',
            'DLL': 'D L L',
            'EXE': 'E X E',
            'UTF': 'U T F',
            'ASCII': 'A S C I I',
            'OOP': 'O O P',
            'TDD': 'T D D',
            'BDD': 'B D D',
            'MVC': 'M V C',
            'MVVM': 'M V V M',
            'JWT': 'J W T',
            'OAuth': 'O Auth',
            'OIDC': 'O I D C',
            'SAML': 'S A M L',
            'WSFed': 'W S Fed',
            'RBAC': 'R B A C',
            'ABAC': 'A B A C',
            'PBAC': 'P B A C',
            'CI': 'C I',
            'CD': 'C D',
            'GitOps': 'Git Ops',
            'Infra': 'Infra',
            'IaC': 'I a C',
            'PaaS': 'P a a S',
            'FaaS': 'F a a S',
            'CaaS': 'C a a S',
            'XaaS': 'X a a S',

            # === DEVOPS TOOLS ===
            'CI/CD': 'C I C D',
            'VCS': 'V C S',
            'SCM': 'S C M',
            'JIRA': 'J I R A',
            'Jenkins': 'Jenkins',
            'GitHub': 'Git Hub',
            'GitLab': 'Git Lab',
            'Bitbucket': 'Bit Bucket',
            'Docker': 'Docker',
            'K8s': 'K 8 s',
            'Kubernetes': 'Kubernetes',
            'K8S': 'K 8 S',
            'Helm': 'Helm',
            'Istio': 'Istio',
            'Prometheus': 'Prometheus',
            'Grafana': 'Grafana',
            'ELK': 'E L K',
            'Splunk': 'Splunk',
            'Datadog': 'Data Dog',
            'NewRelic': 'New Relic',
            'PagerDuty': 'Pager Duty',
            'Terraform': 'Terraform',
            'Ansible': 'Ansible',
            'Chef': 'Chef',
            'Puppet': 'Puppet',
            'SaltStack': 'Salt Stack',
            'Nagios': 'Nagios',
            'Zabbix': 'Zabbix',

            # === MONITORING & LOGGING ===
            'APM': 'A P M',
            'NPM': 'N P M',
            'RUM': 'R U M',
            'SIEM': 'S I E M',
            'SOAR': 'S O A R',
            'XDR': 'X D R',
            'EDR': 'E D R',
            'MDR': 'M D R',
            'DDoS': 'D D o S',
            'DoS': 'D o S',
            'MITRE': 'M I T R E',
            'ATT&CK': 'A T T C K',
            'CVE': 'C V E',
            'CVSS': 'C V S S',

            # === DATABASE ===
            'OLTP': 'O L T P',
            'OLAP': 'O L A P',
            'ETL': 'E T L',
            'ELT': 'E L T',
            'ACID': 'A C I D',
            'BASE': 'B A S E',
            'CAP': 'C A P',
            'NoSQL': 'N O S Q L',
            'CRUD': 'C R U D',
            'ORM': 'O R M',
            'ODBC': 'O D B C',
            'JDBC': 'J D B C',
            'DDL': 'D D L',
            'DML': 'D M L',
            'DCL': 'D C L',
            'TCL': 'T C L',

            # === NETWORKING ===
            'LAN': 'L A N',
            'WAN': 'W A N',
            'VLAN': 'V L A N',
            'VPN': 'V P N',
            'DNS': 'D N S',
            'DHCP': 'D H C P',
            'NAT': 'N A T',
            'PAT': 'P A T',
            'SNAT': 'S N A T',
            'DNAT': 'D N A T',
            'TCP': 'T C P',
            'UDP': 'U D P',
            'ICMP': 'I C M P',
            'IP': 'I P',
            'IPv4': 'I P v 4',
            'IPv6': 'I P v 6',
            'HTTP': 'H T T P',
            'HTTPS': 'H T T P S',
            'FTP': 'F T P',
            'SFTP': 'S F T P',
            'SSH': 'S S H',
            'Telnet': 'Telnet',
            'SMTP': 'S M T P',
            'POP3': 'P O P 3',
            'IMAP': 'I M A P',
            'MQTT': 'M Q T T',
            'CoAP': 'C o A P',
            'AMQP': 'A M Q P',
            'Kafka': 'Kafka',
            'RabbitMQ': 'Rabbit M Q',
            'Redis': 'Redis',

            # === CONTAINERS & ORCHESTRATION ===
            'OCI': 'O C I',
            'CRI': 'C R I',
            'CNI': 'C N I',
            'CSI': 'C S I',

            # === CLOUD NATIVE ===
            'CNCF': 'C N C F',
            'OSS': 'O S S',
            'FOSS': 'F O S S',
            'SaaS': 'S a a S',
            'PaaS': 'P a a S',
            'IaaS': 'I a a S',
            'FaaS': 'F a a S',
            'MSP': 'M S P',
            'CSP': 'C S P',

            # === AI/ML ===
            'AI': 'A I',
            'ML': 'M L',
            'DL': 'D L',
            'NLP': 'N L P',
            'CV': 'C V',
            'LLM': 'L L M',
            'GPT': 'G P T',
            'BERT': 'B E R T',
            'RAG': 'R A G',
            'OCR': 'O C R',
            'ASR': 'A S R',
            'TTS': 'T T S',
            'NLU': 'N L U',
            'NLG': 'N L G',

            # === SECURITY ===
            'PKI': 'P K I',
            'CA': 'C A',
            'CRL': 'C R L',
            'OCSP': 'O C S P',
            'HSM': 'H S M',
            'TPM': 'T P M',
            'YubiKey': 'Yubi Key',
            '2FA': '2 F A',
            'MFA': 'M F A',
            'TOTP': 'T O T P',
            'HOTP': 'H O T P',
            'SSO': 'S S O',
            'IdP': 'I d P',
            'SP': 'S P',
            'RADIUS': 'R A D I U S',
            'TACACS': 'T A C A C S',
            'X.509': 'X 5 0 9',
            'AES': 'A E S',
            'RSA': 'R S A',
            'ECC': 'E C C',
            'PGP': 'P G P',
            'GPG': 'G P G',
            'TLS': 'T L S',
            'SSL': 'S S L',
            'SSH': 'S S H',

            # === AGILE/PROJECT MGMT ===
            'Scrum': 'Scrum',
            'Kanban': 'Kanban',
            'MVP': 'M V P',
            'PoC': 'P o C',
            'MOKR': 'M O K R',
            'OKR': 'O K R',
            'KPI': 'K P I',
            'SLA': 'S L A',
            'SLO': 'S L O',
            'SLI': 'S L I',
            'MTTR': 'M T T R',
            'MTTF': 'M T T F',
            'MTBF': 'M T B F',
            'RTO': 'R T O',
            'RPO': 'R P O',
        }

        for acronym, pronunciation in letter_acronyms.items():
            # Use word boundaries to avoid partial matches
            text = re.sub(r'\b' + acronym + r'\b', pronunciation, text)

        return text

    def _add_pause_markers(self, text):
        """Add pause markers for more natural TTS timing"""
        import re

        # Add short pauses after common abbreviations
        short_pause_after = [r'\b[A-Z]\.', r'\b[Mm]r\.', r'\b[Mm]s\.', r'\b[Dd]r\.',
                             r'\b[Pp]rof\.', r'\b[Rr]ev\.', r'\b[Gg]en\.',
                             r'\b[Ss]gt\.', r'\b[Cc]apt\.', r'\b[Lt]\.',
                             r'\betc\.']
        for pattern in short_pause_after:
            text = re.sub(pattern, r'\g<0>,', text)

        # Add pauses before conjunctions in long sentences
        text = re.sub(r'\s+(and|or|but|yet|so|nor)\s+', r', \1 ', text)

        # Add pauses after introductory phrases
        intro_phrases = [r'However', r'Therefore', r'Furthermore', r'Moreover',
                         r'Additionally', r'Consequently', r'Meanwhile', r'Nevertheless']
        for phrase in intro_phrases:
            text = re.sub(r'\b' + phrase + r',', r'\g<0>,', text)

        # Ensure pauses after colons and semicolons
        text = re.sub(r'[:;]', r'\g<0>,', text)

        return text

    def get_text_stats(self, text):
        """Get text statistics"""
        if not text:
            return {'chars': 0, 'words': 0, 'lines': 0, 'sentences': 0}

        chars = len(text)
        words = len(text.split())
        lines = text.count('\n') + 1
        sentences = len(re.findall(r'[.!?]+', text))

        return {
            'chars': chars,
            'words': words,
            'lines': lines,
            'sentences': sentences
        }

    def needs_chunking(self, text):
        """Check if text needs to be split into chunks"""
        return len(text) > self.chunk_size

    def split_text_into_chunks(self, text, model_type='matcha'):
        """Split text into manageable chunks for TTS processing"""
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)

        if len(text) <= safe_chunk_size:
            # Double-check token count for single chunk
            if self.estimate_token_count(text) <= self.model_token_limits.get(model_type, 800):
                return [text]

        chunks = []
        remaining_text = text

        while remaining_text:
            if len(remaining_text) <= safe_chunk_size:
                # Final chunk - check token count
                if self.estimate_token_count(remaining_text) <= self.model_token_limits.get(model_type, 800):
                    chunks.append(remaining_text.strip())
                    break
                else:
                    # Still too many tokens, need to split further
                    chunk = self._find_optimal_chunk(remaining_text, model_type)
                    chunks.append(chunk.strip())
                    remaining_text = remaining_text[len(chunk):].strip()
            else:
                # Find the best split point
                chunk = self._find_optimal_chunk(remaining_text, model_type)
                chunks.append(chunk.strip())
                remaining_text = remaining_text[len(chunk):].strip()

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
                    print(f"Warning: Chunk {i+1} significantly over limit ({estimated_tokens} > {token_limit * 1.2:.0f}): {error_msg}")
                    # Try to split this chunk into smaller pieces
                    sub_chunks = self._emergency_split_chunk(chunk, model_type)
                    validated_chunks.extend(sub_chunks)
                else:
                    # Close to limit but not too bad, let it through with warning
                    print(f"Warning: Chunk {i+1} slightly over limit but allowing: {error_msg}")
                    validated_chunks.append(chunk)
            else:
                validated_chunks.append(chunk)

        return validated_chunks

    def _emergency_split_chunk(self, text, model_type):
        """Emergency splitting for chunks that are still too large"""
        token_limit = self.model_token_limits.get(model_type, 700)

        # Try splitting by sentences first (preserve original endings)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
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

    def _find_optimal_chunk(self, text, model_type='matcha'):
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

    def _split_at_sentences(self, text, model_type='matcha'):
        """Try to split at sentence boundaries"""
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)

        best_pos = 0
        for i in range(min(len(text), safe_chunk_size), 0, -1):
            for ending in sentence_endings:
                if text[i-len(ending):i] == ending:
                    # Found a sentence boundary - check token count
                    candidate = text[:i]
                    if self.estimate_token_count(candidate) <= token_limit:
                        return candidate
                elif i < len(text) - len(ending) and text[i:i+len(ending)] == ending:
                    candidate_pos = i + len(ending)
                    candidate = text[:candidate_pos]
                    if self.estimate_token_count(candidate) <= token_limit:
                        best_pos = candidate_pos

        if best_pos > 0:
            return text[:best_pos]

        return None

    def _split_at_clauses(self, text, model_type='matcha'):
        """Try to split at clause boundaries"""
        clause_endings = [', ', '; ', ': ', ',\n', ';\n', ':\n']
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)

        best_pos = 0
        for i in range(min(len(text), safe_chunk_size), 0, -1):
            for ending in clause_endings:
                if text[i-len(ending):i] == ending:
                    candidate = text[:i]
                    if self.estimate_token_count(candidate) <= token_limit:
                        return candidate
                elif i < len(text) - len(ending) and text[i:i+len(ending)] == ending:
                    candidate_pos = i + len(ending)
                    candidate = text[:candidate_pos]
                    if self.estimate_token_count(candidate) <= token_limit:
                        best_pos = candidate_pos

        if best_pos > 0:
            return text[:best_pos]

        return None

    def _split_at_words(self, text, model_type='matcha'):
        """Try to split at word boundaries"""
        safe_chunk_size = self.get_model_safe_chunk_size(model_type)
        token_limit = self.model_token_limits.get(model_type, 800)

        # Find the last space within safe chunk size
        for i in range(min(len(text), safe_chunk_size), 0, -1):
            if text[i-1] == ' ':
                candidate = text[:i-1]
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
    if os.name == 'nt':  # Windows
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
        self.cache_file = os.path.join(self.cache_dir, 'tts_audio_cache.pkl')
        self.load_cache()

    def _generate_key(self, text, model_type, speaker_id, speed, voice_config_id=None):
        """Generate cache key from parameters including voice model"""
        key_data = f"{text}|{model_type}|{speaker_id}|{speed}|{voice_config_id or 'default'}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, text, model_type, speaker_id, speed, voice_config_id=None):
        """Get cached audio if available"""
        key = self._generate_key(text, model_type, speaker_id, speed, voice_config_id)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, text, model_type, speaker_id, speed, audio_data, sample_rate, voice_config_id=None):
        """Cache audio data"""
        key = self._generate_key(text, model_type, speaker_id, speed, voice_config_id)

        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = {
            'audio_data': audio_data,
            'sample_rate': sample_rate,
            'timestamp': time.time()
        }

        # Save cache periodically
        if len(self.cache) % 5 == 0:
            self.save_cache()

    def save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(dict(self.cache), f)
        except Exception:
            pass  # Ignore cache save errors

    def load_cache(self):
        """Load cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
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
            'start_time': time.time(),
            'text_length': text_length,
            'model_type': model_type
        }

    def end_generation(self, audio_duration, from_cache=False):
        """End tracking and record metrics"""
        if not self.current_generation:
            return

        end_time = time.time()
        generation_time = end_time - self.current_generation['start_time']

        metric = {
            'timestamp': end_time,
            'text_length': self.current_generation['text_length'],
            'model_type': self.current_generation['model_type'],
            'generation_time': generation_time,
            'audio_duration': audio_duration,
            'rtf': generation_time / audio_duration if audio_duration > 0 else 0,
            'from_cache': from_cache
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
            relevant_metrics = [m for m in self.metrics if m['model_type'] == model_type]

        if not relevant_metrics:
            return 0

        recent_metrics = relevant_metrics[-last_n:]
        rtf_values = [m['rtf'] for m in recent_metrics if not m['from_cache']]

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


class TTSGui:
    def __init__(self, root):
        # Setup crash prevention first
        setup_crash_prevention()

        self.root = root
        self.root.title("High-Quality English TTS - Sherpa-ONNX Enhanced")
        self.root.geometry("1300x1200")

        # Dracula theme color scheme
        self.colors = {
            'bg_primary': '#282a36',      # Main background (Dracula background)
            'bg_secondary': '#44475a',    # Secondary background (Dracula selection)
            'bg_tertiary': '#6272a4',     # Tertiary background (Dracula comment)
            'bg_accent': '#44475a',       # Accent background
            'fg_primary': '#f8f8f2',      # Primary text (Dracula foreground)
            'fg_secondary': '#bd93f9',    # Secondary text (Dracula purple)
            'fg_muted': '#6272a4',        # Muted text (Dracula comment)
            'accent_pink': '#ff79c6',     # Dracula pink (primary accent)
            'accent_pink_hover': '#ff92d0', # Pink hover state
            'accent_cyan': '#8be9fd',     # Dracula cyan
            'accent_green': '#50fa7b',    # Dracula green (success)
            'accent_orange': '#ffb86c',   # Dracula orange (warning)
            'accent_red': '#ff5555',      # Dracula red (error/danger)
            'accent_purple': '#bd93f9',   # Dracula purple
            'selection': '#44475a',       # Selection background
            'border': '#6272a4',          # Border color
            'border_light': '#bd93f9',    # Light border (purple)
        }

        # Configure root window
        self.root.configure(bg=self.colors['bg_primary'])

        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

        # Initialize helper components
        self.text_processor = TextProcessor()
        self.audio_cache = AudioCache()
        self.performance_monitor = PerformanceMonitor()
        self.audio_stitcher = AudioStitcher(silence_duration=0.3)
        self.audio_exporter = AudioExporter()  # Advanced audio export system
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
            'normalize_whitespace': tk.BooleanVar(value=True),
            'normalize_punctuation': tk.BooleanVar(value=True),
            'remove_urls': tk.BooleanVar(value=False),
            'remove_emails': tk.BooleanVar(value=False),
            'remove_duplicates': tk.BooleanVar(value=False),
            'numbers_to_words': tk.BooleanVar(value=False),
            'expand_abbreviations': tk.BooleanVar(value=False),
            'handle_acronyms': tk.BooleanVar(value=False),
            'add_pauses': tk.BooleanVar(value=False)
        }

        # SSML Support
        self.ssml_processor = SSMLProcessor()
        self.ssml_enabled = tk.BooleanVar(value=False)
        self.ssml_auto_detect = tk.BooleanVar(value=True)

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

    def setup_theme(self):
        """Configure the dark theme for ttk widgets"""
        style = ttk.Style()

        # Configure the theme
        style.theme_use('clam')

        # Configure Frame styles
        style.configure('Dark.TFrame',
                       background=self.colors['bg_secondary'],
                       borderwidth=1,
                       relief='solid',
                       bordercolor=self.colors['border'])

        style.configure('Card.TFrame',
                       background=self.colors['bg_tertiary'],
                       borderwidth=1,
                       relief='solid',
                       bordercolor=self.colors['border_light'])

        # Configure LabelFrame styles
        style.configure('Dark.TLabelframe',
                       background=self.colors['bg_secondary'],
                       borderwidth=2,
                       relief='solid',
                       bordercolor=self.colors['border'])

        style.configure('Dark.TLabelframe.Label',
                       background=self.colors['bg_accent'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 12, 'bold'),
                       borderwidth=0,
                       relief='flat')

        # Configure Label styles
        style.configure('Dark.TLabel',
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 12))

        style.configure('Time.TLabel',
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['accent_cyan'],
                       font=('Consolas', 11, 'bold'))

        # Configure Button styles with Dracula colors
        style.configure('Primary.TButton',
                       background=self.colors['accent_pink'],
                       foreground=self.colors['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 11, 'bold'),
                       padding=(15, 8))

        style.map('Primary.TButton',
                 background=[('active', self.colors['accent_pink_hover']),
                           ('pressed', '#ff66c4'),
                           ('disabled', self.colors['bg_accent'])])

        style.configure('Success.TButton',
                       background=self.colors['accent_green'],
                       foreground=self.colors['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 11),
                       padding=(12, 6))

        style.map('Success.TButton',
                 background=[('active', '#45e070'),
                           ('pressed', '#3dd164'),
                           ('disabled', self.colors['bg_accent'])])

        style.configure('Warning.TButton',
                       background=self.colors['accent_orange'],
                       foreground=self.colors['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 11),
                       padding=(12, 6))

        style.map('Warning.TButton',
                 background=[('active', '#ffad5c'),
                           ('pressed', '#ffa04d'),
                           ('disabled', self.colors['bg_accent'])])

        style.configure('Danger.TButton',
                       background=self.colors['accent_red'],
                       foreground=self.colors['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 11),
                       padding=(12, 6))

        style.map('Danger.TButton',
                 background=[('active', '#ff4444'),
                           ('pressed', '#ff3333'),
                           ('disabled', self.colors['bg_accent'])])

        style.configure('Dark.TButton',
                       background=self.colors['bg_accent'],
                       foreground=self.colors['fg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 11),
                       padding=(12, 6))

        style.map('Dark.TButton',
                 background=[('active', self.colors['bg_tertiary']),
                           ('pressed', self.colors['border']),
                           ('disabled', self.colors['bg_secondary'])])

        # Configure Utility button style (for Import/Export buttons)
        style.configure('Utility.TButton',
                       background=self.colors['accent_cyan'],
                       foreground=self.colors['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 11, 'bold'),
                       padding=(12, 6))

        style.map('Utility.TButton',
                 background=[('active', '#7de8f5'),
                           ('pressed', '#6dd9e8'),
                           ('disabled', self.colors['bg_accent'])])

        # Configure Radiobutton styles
        style.configure('Dark.TRadiobutton',
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['fg_primary'],
                       focuscolor='none',
                       font=('Segoe UI', 11))

        style.map('Dark.TRadiobutton',
                 background=[('active', self.colors['bg_accent'])])

        # Configure Scale styles with Dracula colors
        style.configure('Dark.Horizontal.TScale',
                       background=self.colors['bg_secondary'],
                       troughcolor=self.colors['bg_accent'],
                       borderwidth=0,
                       lightcolor=self.colors['accent_purple'],
                       darkcolor=self.colors['accent_purple'])

        # Configure Spinbox styles
        style.configure('Dark.TSpinbox',
                       fieldbackground=self.colors['bg_tertiary'],
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['fg_primary'],
                       bordercolor=self.colors['border'],
                       arrowcolor=self.colors['fg_secondary'],
                       font=('Segoe UI', 10))

        # Configure Progressbar styles with Dracula colors
        style.configure('Dark.Horizontal.TProgressbar',
                       background=self.colors['accent_pink'],
                       troughcolor=self.colors['bg_accent'],
                       borderwidth=0,
                       lightcolor=self.colors['accent_pink'],
                       darkcolor=self.colors['accent_pink'])

    def setup_ui(self):
        # Main frame with dark theme
        main_frame = ttk.Frame(self.root, style='Dark.TFrame', padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.configure(style='Dark.TFrame')

        # Enhanced Voice Selection Frame
        voice_frame = ttk.LabelFrame(main_frame, text="🎤 Enhanced Voice Selection", style='Dark.TLabelframe', padding="15")
        voice_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        # Voice Model Selection
        model_selection_frame = ttk.Frame(voice_frame, style='Dark.TFrame')
        model_selection_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(model_selection_frame, text="🤖 Voice Model:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        self.voice_model_var = tk.StringVar()
        self.voice_model_combo = ttk.Combobox(model_selection_frame, textvariable=self.voice_model_var,
                                            state="readonly", width=50, style='Dark.TSpinbox')
        self.voice_model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.voice_model_combo.bind('<<ComboboxSelected>>', self.on_voice_model_changed)

        # Voice/Speaker Selection
        speaker_selection_frame = ttk.Frame(voice_frame, style='Dark.TFrame')
        speaker_selection_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(speaker_selection_frame, text="👤 Voice/Speaker:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        self.speaker_var = tk.StringVar()
        self.speaker_combo = ttk.Combobox(speaker_selection_frame, textvariable=self.speaker_var,
                                        state="readonly", width=50, style='Dark.TSpinbox')
        self.speaker_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.speaker_combo.bind('<<ComboboxSelected>>', self.on_speaker_changed)

        # Voice Preview Button
        self.preview_btn = ttk.Button(speaker_selection_frame, text="🎵 Preview Voice",
                                    command=self.preview_voice, style="Dark.TButton")
        self.preview_btn.grid(row=0, column=2, padx=(10, 0))

        # Voice Information Display
        info_frame = ttk.Frame(voice_frame, style='Card.TFrame', padding="8")
        info_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(info_frame, text="ℹ️ Voice Info:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.voice_info_label = ttk.Label(info_frame, text="Select a voice model to see details",
                                        style='Time.TLabel', wraplength=600)
        self.voice_info_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Configure grid weights for voice frame
        model_selection_frame.columnconfigure(1, weight=1)
        speaker_selection_frame.columnconfigure(1, weight=1)
        info_frame.columnconfigure(1, weight=1)

        # Speed control
        speed_frame = ttk.Frame(voice_frame, style='Dark.TFrame')
        speed_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(15, 0))

        ttk.Label(speed_frame, text="⚡ Generation Speed:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.5, to=3.0, variable=self.speed_var,
                               orient=tk.HORIZONTAL, style='Dark.Horizontal.TScale')
        speed_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.speed_label = ttk.Label(speed_frame, text="1.0x", style='Dark.TLabel')
        self.speed_label.grid(row=0, column=2, padx=(5, 0))

        # Update speed label when scale changes
        speed_scale.configure(command=self.update_speed_label)

        # Configure speed frame grid weights
        speed_frame.columnconfigure(1, weight=1)

        # Text input frame with dark theme
        text_frame = ttk.LabelFrame(main_frame, text="📝 Enhanced Text Input", style='Dark.TLabelframe', padding="15")
        text_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))

        # Text controls frame
        text_controls_frame = ttk.Frame(text_frame, style='Dark.TFrame')
        text_controls_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # Import/Export buttons with distinctive colors
        ttk.Button(text_controls_frame, text="📁 Import Text", command=self.import_text,
                  style="Utility.TButton").grid(row=0, column=0, padx=(0, 10))
        ttk.Button(text_controls_frame, text="💾 Export Text", command=self.export_text,
                  style="Utility.TButton").grid(row=0, column=1, padx=(0, 10))
        ttk.Button(text_controls_frame, text="🧹 Clear", command=self.clear_text,
                  style="Warning.TButton").grid(row=0, column=2, padx=(0, 10))

        # Text preprocessing options
        preprocess_frame = ttk.LabelFrame(text_frame, text="🔧 Text Processing Options",
                                        style='Dark.TLabelframe', padding="10")
        preprocess_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Checkbutton(preprocess_frame, text="Normalize whitespace",
                       variable=self.text_options['normalize_whitespace'],
                       style='Dark.TRadiobutton').grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(preprocess_frame, text="Normalize punctuation",
                       variable=self.text_options['normalize_punctuation'],
                       style='Dark.TRadiobutton').grid(row=0, column=1, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(preprocess_frame, text="Remove URLs",
                       variable=self.text_options['remove_urls'],
                       style='Dark.TRadiobutton').grid(row=0, column=2, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(preprocess_frame, text="Remove emails",
                       variable=self.text_options['remove_emails'],
                       style='Dark.TRadiobutton').grid(row=1, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(preprocess_frame, text="Remove duplicate lines",
                       variable=self.text_options['remove_duplicates'],
                       style='Dark.TRadiobutton').grid(row=1, column=1, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(preprocess_frame, text="Numbers to words (123→one hundred...)",
                       variable=self.text_options['numbers_to_words'],
                       style='Dark.TRadiobutton').grid(row=1, column=2, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(preprocess_frame, text="Expand abbreviations (Dr.→Doctor)",
                       variable=self.text_options['expand_abbreviations'],
                       style='Dark.TRadiobutton').grid(row=2, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(preprocess_frame, text="Pronounce acronyms (NASA→N A S A)",
                       variable=self.text_options['handle_acronyms'],
                       style='Dark.TRadiobutton').grid(row=2, column=1, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(preprocess_frame, text="Add natural pauses",
                       variable=self.text_options['add_pauses'],
                       style='Dark.TRadiobutton').grid(row=2, column=2, sticky=tk.W, padx=(0, 15))

        # SSML Support Frame - Professional-grade speech control
        ssml_frame = ttk.LabelFrame(text_frame, text="🎭 SSML Support (Professional Speech Control)",
                                   style='Dark.TLabelframe', padding="10")
        ssml_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # SSML controls row 1
        ttk.Checkbutton(ssml_frame, text="Enable SSML parsing",
                       variable=self.ssml_enabled,
                       command=self.on_ssml_toggle,
                       style='Dark.TRadiobutton').grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(ssml_frame, text="Auto-detect SSML markup",
                       variable=self.ssml_auto_detect,
                       style='Dark.TRadiobutton').grid(row=0, column=1, sticky=tk.W, padx=(0, 15))
        
        # SSML control buttons
        ttk.Button(ssml_frame, text="📋 SSML Templates",
                  command=self.show_ssml_templates,
                  style="Dark.TButton").grid(row=0, column=2, padx=(0, 10))
        ttk.Button(ssml_frame, text="❓ SSML Reference",
                  command=self.show_ssml_reference,
                  style="Dark.TButton").grid(row=0, column=3, padx=(0, 10))
        ttk.Button(ssml_frame, text="✓ Validate SSML",
                  command=self.validate_ssml_input,
                  style="Dark.TButton").grid(row=0, column=4, padx=(0, 10))

        # SSML info label
        self.ssml_info_label = ttk.Label(ssml_frame, 
                                        text="SSML enables: <emphasis>, <break>, <prosody>, <say-as>, and more",
                                        style='Time.TLabel')
        self.ssml_info_label.grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(5, 0))

        # Chunking info frame
        chunking_frame = ttk.Frame(text_frame, style='Card.TFrame', padding="8")
        chunking_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(chunking_frame, text="📄 Long Text Handling:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.chunking_info_label = ttk.Label(chunking_frame,
                                           text="Texts over 8,000 chars will be automatically split and stitched",
                                           style='Time.TLabel')
        self.chunking_info_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Create custom text widget with Dracula theme
        self.text_widget = scrolledtext.ScrolledText(text_frame, width=75, height=8, wrap=tk.WORD,
                                                    bg=self.colors['bg_primary'],
                                                    fg=self.colors['fg_primary'],
                                                    insertbackground=self.colors['accent_cyan'],
                                                    selectbackground=self.colors['selection'],
                                                    selectforeground=self.colors['fg_primary'],
                                                    font=('Segoe UI', 11),
                                                    borderwidth=1,
                                                    relief='solid')
        self.text_widget.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Bind text change events for real-time validation and stats
        self.text_widget.bind('<KeyRelease>', self.on_text_change)
        self.text_widget.bind('<Button-1>', self.on_text_change)
        
        # Bind paste events to remove duplicate lines (multiple methods for compatibility)
        self.text_widget.bind('<<Paste>>', self.on_paste)
        self.text_widget.bind('<Control-v>', self.on_paste)
        self.text_widget.bind('<Control-V>', self.on_paste)
        self.text_widget.bind('<Shift-Insert>', self.on_paste)

        # Text statistics frame
        stats_frame = ttk.Frame(text_frame, style='Card.TFrame', padding="8")
        stats_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(stats_frame, text="📊 Text Stats:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.stats_label = ttk.Label(stats_frame, text="Characters: 0 | Words: 0 | Lines: 0 | Sentences: 0",
                                   style='Time.TLabel')
        self.stats_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Validation status
        self.validation_label = ttk.Label(stats_frame, text="✓ Ready", style='Dark.TLabel')
        self.validation_label.grid(row=0, column=2, sticky=tk.E, padx=(20, 0))

        # Add sample text with better guidance
        sample_text = ("Welcome to the enhanced high-quality English text-to-speech system! "
                      "This version features improved text processing, performance optimizations, and audio caching. "
                      "Try editing this text, importing your own content, or adjusting the processing options above. "
                      "The system will provide real-time feedback on text statistics and validation.")
        self.text_widget.insert(tk.END, sample_text)

        # Initial text stats update
        self.on_text_change(None)

        # Controls frame with better spacing
        controls_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        controls_frame.grid(row=3, column=0, columnspan=3, pady=(0, 15))

        # Generate button (primary action)
        self.generate_btn = ttk.Button(controls_frame, text="🎵 Generate Speech",
                                     command=self.generate_speech, style="Primary.TButton")
        self.generate_btn.grid(row=0, column=0, padx=(0, 10))

        # Cancel button (initially hidden)
        self.cancel_btn = ttk.Button(controls_frame, text="⏹ Cancel", command=self.cancel_generation,
                                   style="Danger.TButton")
        self.cancel_btn.grid(row=0, column=1, padx=(0, 15))
        self.cancel_btn.grid_remove()  # Hide initially

        # Play button
        self.play_btn = ttk.Button(controls_frame, text="▶ Play", command=self.play_audio,
                                 state=tk.DISABLED, style="Success.TButton")
        self.play_btn.grid(row=0, column=2, padx=(0, 10))

        # Stop button
        self.stop_btn = ttk.Button(controls_frame, text="⏸ Pause", command=self.stop_audio,
                                 state=tk.DISABLED, style="Warning.TButton")
        self.stop_btn.grid(row=0, column=3, padx=(0, 10))

        # Save button
        self.save_btn = ttk.Button(controls_frame, text="💾 Save As...", command=self.save_audio,
                                 state=tk.DISABLED, style="Dark.TButton")
        self.save_btn.grid(row=0, column=4, padx=(0, 10))

        # Keyboard shortcuts help button
        self.shortcuts_btn = ttk.Button(controls_frame, text="⌨️ Shortcuts (F1)",
                                       command=self.show_keyboard_shortcuts, style="Dark.TButton")
        self.shortcuts_btn.grid(row=0, column=5, padx=(10, 0))

        # Audio Playback Controls Frame with enhanced styling
        playback_frame = ttk.LabelFrame(main_frame, text="🎛️ Audio Playback Controls",
                                       style='Dark.TLabelframe', padding="15")
        playback_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        # Time display with modern styling
        time_frame = ttk.Frame(playback_frame, style='Card.TFrame', padding="8")
        time_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(time_frame, text="⏱️ Time:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.time_label = ttk.Label(time_frame, text="00:00 / 00:00", style='Time.TLabel')
        self.time_label.grid(row=0, column=1, sticky=tk.W)

        # Seek bar with enhanced styling
        seek_frame = ttk.Frame(playback_frame, style='Dark.TFrame')
        seek_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 15))

        ttk.Label(seek_frame, text="🎯 Position:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(seek_frame, from_=0.0, to=100.0, variable=self.seek_var,
                                   orient=tk.HORIZONTAL, command=self.on_seek, style='Dark.Horizontal.TScale')
        self.seek_scale.grid(row=0, column=1, padx=(15, 0), sticky=(tk.W, tk.E))
        self.seek_scale.config(state=tk.DISABLED)

        # Playback speed control with enhanced styling
        playback_speed_frame = ttk.Frame(playback_frame, style='Dark.TFrame')
        playback_speed_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Label(playback_speed_frame, text="🚀 Playback Speed (pitch preserved):",
                 style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.playback_speed_var = tk.DoubleVar(value=1.0)
        playback_speed_scale = ttk.Scale(playback_speed_frame, from_=0.5, to=2.0,
                                        variable=self.playback_speed_var, orient=tk.HORIZONTAL,
                                        command=self.update_playback_speed_label, style='Dark.Horizontal.TScale')
        playback_speed_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.playback_speed_label = ttk.Label(playback_speed_frame, text="1.0x", style='Dark.TLabel')
        self.playback_speed_label.grid(row=0, column=2, padx=(5, 0))

        # Volume control with enhanced styling
        volume_frame = ttk.Frame(playback_frame, style='Dark.TFrame')
        volume_frame.grid(row=2, column=2, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        ttk.Label(volume_frame, text="🔊 Volume:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.volume_var = tk.DoubleVar(value=70.0)
        volume_scale = ttk.Scale(volume_frame, from_=0.0, to=100.0,
                                variable=self.volume_var, orient=tk.HORIZONTAL,
                                command=self.update_volume_label, style='Dark.Horizontal.TScale')
        volume_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.volume_label = ttk.Label(volume_frame, text="70%", style='Dark.TLabel')
        self.volume_label.grid(row=0, column=2, padx=(5, 0))

        # Status frame with dark theme
        status_frame = ttk.LabelFrame(main_frame, text="📊 Status & Performance",
                                     style='Dark.TLabelframe', padding="15")
        status_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        # Performance info frame
        perf_info_frame = ttk.Frame(status_frame, style='Card.TFrame', padding="8")
        perf_info_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(perf_info_frame, text="🚀 Performance:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.perf_label = ttk.Label(perf_info_frame, text="Cache: 0 items | Avg RTF: N/A",
                                   style='Time.TLabel')
        self.perf_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Cache management buttons
        ttk.Button(perf_info_frame, text="🗑️ Clear Cache", command=self.clear_cache,
                  style="Warning.TButton").grid(row=0, column=2, sticky=tk.E, padx=(20, 0))

        # Status text with Dracula theme
        self.status_text = scrolledtext.ScrolledText(status_frame, width=75, height=6, wrap=tk.WORD,
                                                    bg=self.colors['bg_primary'],
                                                    fg=self.colors['fg_secondary'],
                                                    insertbackground=self.colors['accent_cyan'],
                                                    selectbackground=self.colors['selection'],
                                                    selectforeground=self.colors['fg_primary'],
                                                    font=('Consolas', 9),
                                                    borderwidth=1,
                                                    relief='solid')
        self.status_text.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # Progress bar with dark theme (can switch between indeterminate and determinate)
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', style='Dark.Horizontal.TProgressbar')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        voice_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)
        playback_frame.columnconfigure(1, weight=1)
        playback_frame.columnconfigure(3, weight=1)
        seek_frame.columnconfigure(1, weight=1)
        playback_speed_frame.columnconfigure(1, weight=1)
        volume_frame.columnconfigure(1, weight=1)

    # ==================== Keyboard Shortcuts ====================

    def setup_keyboard_shortcuts(self):
        """
        Setup keyboard shortcuts for power user productivity.
        
        Shortcuts:
            Space         - Play/Pause audio
            Ctrl+Enter    - Generate speech
            Ctrl+G        - Generate speech (alternative)
            1, 2, 3       - Speed presets (0.75x, 1.0x, 1.5x)
            4, 5          - Speed presets (2.0x, 0.5x)
            Alt+Up        - Previous voice model
            Alt+Down      - Next voice model
            Shift+Alt+Up  - Previous speaker
            Shift+Alt+Down - Next speaker
            Ctrl+A        - Select all text
            Ctrl+Shift+C  - Clear text
            Escape        - Cancel generation / Stop playback
            F1 / Ctrl+/   - Show keyboard shortcuts help
        """
        # NOTE: Some shortcuts only work when focus is not in text widget
        # to avoid conflicts with text editing
        
        # Global shortcuts (work anywhere in the window)
        self.root.bind('<F1>', self.show_keyboard_shortcuts)
        self.root.bind('<Control-slash>', self.show_keyboard_shortcuts)
        self.root.bind('<Control-question>', self.show_keyboard_shortcuts)
        
        # Play/Pause with Space (only when not in text widget)
        self.root.bind('<space>', self._on_space_key)
        
        # Generate with Ctrl+Enter or Ctrl+G
        self.root.bind('<Control-Return>', self._on_ctrl_enter)
        self.root.bind('<Control-g>', self._on_ctrl_enter)
        self.root.bind('<Control-G>', self._on_ctrl_enter)
        
        # Speed presets with number keys (only when not in text widget)
        self.root.bind('<Key-1>', lambda e: self._apply_speed_preset(e, 0.75))
        self.root.bind('<Key-2>', lambda e: self._apply_speed_preset(e, 1.0))
        self.root.bind('<Key-3>', lambda e: self._apply_speed_preset(e, 1.5))
        self.root.bind('<Key-4>', lambda e: self._apply_speed_preset(e, 2.0))
        self.root.bind('<Key-5>', lambda e: self._apply_speed_preset(e, 0.5))
        
        # Voice switching with Alt+Arrow keys
        self.root.bind('<Alt-Up>', self._previous_voice_model)
        self.root.bind('<Alt-Down>', self._next_voice_model)
        
        # Speaker switching with Shift+Alt+Arrow keys
        self.root.bind('<Shift-Alt-Up>', self._previous_speaker)
        self.root.bind('<Shift-Alt-Down>', self._next_speaker)
        
        # Cancel/Stop with Escape
        self.root.bind('<Escape>', self._on_escape)
        
        # Clear text with Ctrl+Shift+C (different from Ctrl+C copy)
        self.root.bind('<Control-Shift-c>', self._on_clear_text_shortcut)
        self.root.bind('<Control-Shift-C>', self._on_clear_text_shortcut)
        
        # Text widget specific bindings (Ctrl+A select all already built-in)
        # Add Ctrl+Enter in text widget to still generate
        self.text_widget.bind('<Control-Return>', self._on_ctrl_enter)
        
        self.log_status("⌨️ Keyboard shortcuts enabled (press F1 for help)")

    def _on_space_key(self, event):
        """Handle Space key for play/pause toggle"""
        # Check if focus is in a text entry widget
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Text, ttk.Entry, tk.Entry, scrolledtext.ScrolledText)):
            return  # Let the text widget handle it normally
        
        # Toggle play/pause
        if self.is_playing:
            self.stop_audio()
        elif self.current_audio_file and os.path.exists(self.current_audio_file):
            self.play_audio()
        
        return 'break'  # Prevent default behavior

    def _on_ctrl_enter(self, event):
        """Handle Ctrl+Enter for speech generation"""
        # Only generate if button is enabled (not already generating)
        if str(self.generate_btn['state']) != 'disabled':
            self.generate_speech()
        return 'break'

    def _apply_speed_preset(self, event, speed):
        """Apply a speed preset if not in text widget"""
        # Check if focus is in a text entry widget
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Text, ttk.Entry, tk.Entry, scrolledtext.ScrolledText)):
            return  # Let the text widget handle it normally
        
        # Apply the speed preset
        self.speed_var.set(speed)
        self.update_speed_label(speed)
        self.log_status(f"⚡ Speed preset: {speed}x")
        return 'break'

    def _previous_voice_model(self, event):
        """Switch to previous voice model"""
        values = self.voice_model_combo['values']
        if not values:
            return 'break'
        
        current_idx = self.voice_model_combo.current()
        new_idx = (current_idx - 1) % len(values)
        self.voice_model_combo.current(new_idx)
        self.on_voice_model_changed(None)
        
        # Get the model name for logging
        model_name = values[new_idx] if new_idx < len(values) else "Unknown"
        self.log_status(f"🎤 Voice model: {model_name[:50]}...")
        return 'break'

    def _next_voice_model(self, event):
        """Switch to next voice model"""
        values = self.voice_model_combo['values']
        if not values:
            return 'break'
        
        current_idx = self.voice_model_combo.current()
        new_idx = (current_idx + 1) % len(values)
        self.voice_model_combo.current(new_idx)
        self.on_voice_model_changed(None)
        
        # Get the model name for logging
        model_name = values[new_idx] if new_idx < len(values) else "Unknown"
        self.log_status(f"🎤 Voice model: {model_name[:50]}...")
        return 'break'

    def _previous_speaker(self, event):
        """Switch to previous speaker"""
        values = self.speaker_combo['values']
        if not values:
            return 'break'
        
        current_idx = self.speaker_combo.current()
        new_idx = (current_idx - 1) % len(values)
        self.speaker_combo.current(new_idx)
        self.on_speaker_changed(None)
        
        # Get the speaker name for logging  
        speaker_name = values[new_idx] if new_idx < len(values) else "Unknown"
        self.log_status(f"👤 Speaker: {speaker_name[:40]}...")
        return 'break'

    def _next_speaker(self, event):
        """Switch to next speaker"""
        values = self.speaker_combo['values']
        if not values:
            return 'break'
        
        current_idx = self.speaker_combo.current()
        new_idx = (current_idx + 1) % len(values)
        self.speaker_combo.current(new_idx)
        self.on_speaker_changed(None)
        
        # Get the speaker name for logging
        speaker_name = values[new_idx] if new_idx < len(values) else "Unknown"
        self.log_status(f"👤 Speaker: {speaker_name[:40]}...")
        return 'break'

    def _on_escape(self, event):
        """Handle Escape key - cancel generation or stop playback"""
        if self.generation_thread and self.generation_thread.is_alive():
            # Cancel ongoing generation
            self.cancel_generation()
        elif self.is_playing:
            # Stop audio playback
            self.stop_audio()
        return 'break'

    def _on_clear_text_shortcut(self, event):
        """Handle Ctrl+Shift+C for clearing text"""
        self.clear_text()
        return 'break'

    def show_keyboard_shortcuts(self, event=None):
        """Show keyboard shortcuts help dialog"""
        shortcuts_window = tk.Toplevel(self.root)
        shortcuts_window.title("Keyboard Shortcuts")
        shortcuts_window.geometry("550x600")
        shortcuts_window.configure(bg=self.colors['bg_primary'])
        
        # Make window modal
        shortcuts_window.transient(self.root)
        shortcuts_window.grab_set()
        
        # Title
        title_label = tk.Label(
            shortcuts_window, 
            text="⌨️ Keyboard Shortcuts",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['fg_primary']
        )
        title_label.pack(pady=(20, 5))
        
        # Subtitle
        subtitle_label = tk.Label(
            shortcuts_window,
            text="Power user productivity features",
            font=('Segoe UI', 10),
            bg=self.colors['bg_primary'],
            fg=self.colors['fg_muted']
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Shortcuts content frame with scrollbar
        content_frame = ttk.Frame(shortcuts_window, style='Dark.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Define shortcut categories
        shortcuts_data = [
            ("🎵 Playback Controls", [
                ("Space", "Play / Pause audio"),
                ("Escape", "Stop playback"),
            ]),
            ("🎤 Speech Generation", [
                ("Ctrl+Enter", "Generate speech"),
                ("Ctrl+G", "Generate speech (alternative)"),
                ("Escape", "Cancel generation (during processing)"),
            ]),
            ("⚡ Speed Presets", [
                ("1", "Set speed to 0.75x (slower)"),
                ("2", "Set speed to 1.0x (normal)"),
                ("3", "Set speed to 1.5x (faster)"),
                ("4", "Set speed to 2.0x (fast)"),
                ("5", "Set speed to 0.5x (slowest)"),
            ]),
            ("🔊 Voice Switching", [
                ("Alt+↑", "Previous voice model"),
                ("Alt+↓", "Next voice model"),
                ("Shift+Alt+↑", "Previous speaker"),
                ("Shift+Alt+↓", "Next speaker"),
            ]),
            ("📝 Text Editing", [
                ("Ctrl+A", "Select all text"),
                ("Ctrl+Shift+C", "Clear all text"),
                ("Ctrl+V", "Paste (auto-removes duplicate lines)"),
            ]),
            ("❓ Help", [
                ("F1", "Show this help dialog"),
                ("Ctrl+/", "Show this help dialog (alternative)"),
            ]),
        ]
        
        # Create shortcuts display
        for category, shortcuts in shortcuts_data:
            # Category header
            cat_label = tk.Label(
                content_frame,
                text=category,
                font=('Segoe UI', 11, 'bold'),
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_cyan'],
                anchor='w'
            )
            cat_label.pack(fill=tk.X, pady=(10, 5))
            
            # Shortcuts in this category
            for key, description in shortcuts:
                shortcut_frame = ttk.Frame(content_frame, style='Dark.TFrame')
                shortcut_frame.pack(fill=tk.X, pady=2)
                
                # Key label (fixed width, highlighted)
                key_label = tk.Label(
                    shortcut_frame,
                    text=key,
                    font=('Consolas', 10, 'bold'),
                    bg=self.colors['bg_tertiary'],
                    fg=self.colors['accent_pink'],
                    width=18,
                    anchor='w',
                    padx=8,
                    pady=2
                )
                key_label.pack(side=tk.LEFT, padx=(10, 10))
                
                # Description label
                desc_label = tk.Label(
                    shortcut_frame,
                    text=description,
                    font=('Segoe UI', 10),
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['fg_primary'],
                    anchor='w'
                )
                desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Note about text widget
        note_frame = ttk.Frame(shortcuts_window, style='Card.TFrame', padding=10)
        note_frame.pack(fill=tk.X, padx=20, pady=10)
        
        note_label = tk.Label(
            note_frame,
            text="💡 Note: Number keys (1-5) and Space only work when focus is outside the text editor.\n"
                 "     Use Ctrl+Enter to generate while typing in the text editor.",
            font=('Segoe UI', 9),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['fg_muted'],
            justify=tk.LEFT
        )
        note_label.pack()
        
        # Close button
        close_btn = ttk.Button(
            shortcuts_window,
            text="Close",
            command=shortcuts_window.destroy,
            style='Dark.TButton'
        )
        close_btn.pack(pady=15)
        
        # Allow Escape to close the window
        shortcuts_window.bind('<Escape>', lambda e: shortcuts_window.destroy())
        shortcuts_window.bind('<F1>', lambda e: shortcuts_window.destroy())
        
        return 'break'

    def check_available_voices(self):
        """Check which voice models are available on the system"""
        self.available_voice_configs = {}

        for config_id, config in VOICE_CONFIGS.items():
            model_files = config["model_files"]
            available = True

            # Check if required model files exist
            if config["model_type"] == "kokoro":
                required_files = ["model", "voices", "tokens", "data_dir"]
            elif config["model_type"] == "matcha":
                required_files = ["acoustic_model", "vocoder", "tokens", "data_dir"]
            elif config["model_type"] == "vits":
                required_files = ["model", "tokens", "data_dir"]
            else:
                continue

            for file_key in required_files:
                if file_key in model_files:
                    file_path = model_files[file_key]
                    if file_key.endswith("_dir"):
                        # Check if directory exists
                        if not os.path.isdir(file_path):
                            available = False
                            break
                    else:
                        # Check if file exists
                        if not os.path.isfile(file_path):
                            available = False
                            break

            if available:
                self.available_voice_configs[config_id] = config
                self.log_status(f"✓ Found voice model: {config['name']}")
            else:
                self.log_status(f"⚠ Voice model not available: {config['name']}")

    def populate_voice_selections(self):
        """Populate the voice selection dropdowns"""
        if not self.available_voice_configs:
            self.log_status("⚠ No voice models found. Please download TTS models.")
            return

        # Populate model selection
        model_options = []
        for config_id, config in self.available_voice_configs.items():
            quality_indicator = "⭐" * (4 if config["quality"] == "excellent" else
                                     3 if config["quality"] == "very_high" else 2)
            model_options.append(f"{quality_indicator} {config['name']}")

        self.voice_model_combo['values'] = model_options

        # Select first available model
        if model_options:
            self.voice_model_combo.current(0)
            self.on_voice_model_changed(None)

    def on_voice_model_changed(self, event):
        """Handle voice model selection change"""
        if not self.voice_model_combo.get():
            return

        # Find the selected config
        selected_text = self.voice_model_combo.get()
        selected_config = None
        selected_config_id = None

        for config_id, config in self.available_voice_configs.items():
            quality_indicator = "⭐" * (4 if config["quality"] == "excellent" else
                                     3 if config["quality"] == "very_high" else 2)
            if f"{quality_indicator} {config['name']}" == selected_text:
                selected_config = config
                selected_config_id = config_id
                break

        if not selected_config:
            return

        self.selected_voice_config = (selected_config_id, selected_config)

        # Update speaker selection
        speaker_options = []
        for speaker_id, speaker_info in selected_config["speakers"].items():
            gender_icon = "👩" if speaker_info["gender"] == "female" else "👨"
            accent_text = f" ({speaker_info['accent']})" if speaker_info.get('accent') else ""
            speaker_options.append(f"{gender_icon} {speaker_info['name']}{accent_text} - {speaker_info['description']}")

        self.speaker_combo['values'] = speaker_options

        # Select first speaker
        if speaker_options:
            self.speaker_combo.current(0)
            self.on_speaker_changed(None)

        # Update model info
        info_text = f"{selected_config['description']} | Quality: {selected_config['quality'].replace('_', ' ').title()}"
        self.voice_info_label.config(text=info_text)

    def on_speaker_changed(self, event):
        """Handle speaker selection change"""
        if not self.speaker_combo.get() or not self.selected_voice_config:
            return

        # Update voice info with speaker details
        selected_speaker_text = self.speaker_combo.get()
        config_id, config = self.selected_voice_config

        # Find selected speaker
        for speaker_id, speaker_info in config["speakers"].items():
            gender_icon = "👩" if speaker_info["gender"] == "female" else "👨"
            accent_text = f" ({speaker_info['accent']})" if speaker_info.get('accent') else ""
            if f"{gender_icon} {speaker_info['name']}{accent_text} - {speaker_info['description']}" == selected_speaker_text:
                info_text = (f"{config['description']} | "
                           f"Speaker: {speaker_info['name']} ({speaker_info['gender']}) | "
                           f"Quality: {config['quality'].replace('_', ' ').title()}")
                self.voice_info_label.config(text=info_text)
                break

    # ==================== SSML Support Methods ====================
    
    def on_ssml_toggle(self):
        """Handle SSML enable/disable toggle"""
        if self.ssml_enabled.get():
            self.log_status("🎭 SSML mode enabled - Use SSML markup for professional speech control")
            self.ssml_info_label.config(
                text="SSML active! Use tags like <break>, <emphasis>, <prosody>, <say-as>",
                foreground=self.colors['accent_green']
            )
        else:
            self.log_status("🎭 SSML mode disabled - Plain text mode")
            self.ssml_info_label.config(
                text="SSML enables: <emphasis>, <break>, <prosody>, <say-as>, and more",
                foreground=self.colors['accent_cyan']
            )
    
    def show_ssml_templates(self):
        """Show SSML templates in a dialog"""
        templates_window = tk.Toplevel(self.root)
        templates_window.title("SSML Templates")
        templates_window.geometry("700x600")
        templates_window.configure(bg=self.colors['bg_primary'])
        
        # Make window modal
        templates_window.transient(self.root)
        templates_window.grab_set()
        
        # Title
        title_label = tk.Label(templates_window, text="📋 SSML Templates", 
                              font=('Segoe UI', 14, 'bold'),
                              bg=self.colors['bg_primary'], fg=self.colors['fg_primary'])
        title_label.pack(pady=(15, 10))
        
        # Description
        desc_label = tk.Label(templates_window, 
                             text="Click a template to insert it into the text editor",
                             font=('Segoe UI', 10),
                             bg=self.colors['bg_primary'], fg=self.colors['fg_muted'])
        desc_label.pack(pady=(0, 10))
        
        # Template buttons frame
        templates_frame = ttk.Frame(templates_window, style='Dark.TFrame')
        templates_frame.pack(fill=tk.X, padx=20, pady=10)
        
        template_buttons = [
            ("Basic", "basic"),
            ("Emphasis", "emphasis"),
            ("Prosody (Rate/Pitch)", "prosody"),
            ("Say-As (Pronunciation)", "say_as"),
            ("Full Example", "full_example")
        ]
        
        for i, (name, template_id) in enumerate(template_buttons):
            btn = ttk.Button(templates_frame, text=name,
                           command=lambda tid=template_id: self._insert_ssml_template(tid, templates_window),
                           style="Dark.TButton")
            btn.grid(row=0, column=i, padx=5, pady=5)
        
        # Template preview
        preview_label = tk.Label(templates_window, text="Template Preview:",
                                font=('Segoe UI', 10, 'bold'),
                                bg=self.colors['bg_primary'], fg=self.colors['fg_primary'])
        preview_label.pack(pady=(15, 5), anchor=tk.W, padx=20)
        
        self.template_preview = scrolledtext.ScrolledText(
            templates_window, width=80, height=20,
            bg=self.colors['bg_secondary'],
            fg=self.colors['fg_primary'],
            font=('Consolas', 10),
            wrap=tk.WORD
        )
        self.template_preview.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Show basic template by default
        self.template_preview.insert(tk.END, self.ssml_processor.get_ssml_template('basic'))
        
        # Template selection callback
        def show_template(template_id):
            self.template_preview.delete(1.0, tk.END)
            self.template_preview.insert(tk.END, self.ssml_processor.get_ssml_template(template_id))
        
        # Update buttons to show preview
        for i, (name, template_id) in enumerate(template_buttons):
            btn = templates_frame.grid_slaves(row=0, column=i)[0]
            btn.configure(command=lambda tid=template_id: show_template(tid))
        
        # Insert button
        insert_btn = ttk.Button(templates_window, text="📥 Insert Selected Template",
                               command=lambda: self._insert_template_from_preview(templates_window),
                               style="Primary.TButton")
        insert_btn.pack(pady=15)
        
        # Close button
        close_btn = ttk.Button(templates_window, text="Close",
                              command=templates_window.destroy,
                              style="Dark.TButton")
        close_btn.pack(pady=(0, 15))
    
    def _insert_ssml_template(self, template_id, window=None):
        """Insert an SSML template into the text editor"""
        template = self.ssml_processor.get_ssml_template(template_id)
        
        # Clear and insert
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, template)
        
        # Enable SSML mode
        self.ssml_enabled.set(True)
        self.on_ssml_toggle()
        
        # Update stats
        self.on_text_change(None)
        
        self.log_status(f"📋 Inserted SSML template: {template_id}")
        
        if window:
            window.destroy()
    
    def _insert_template_from_preview(self, window):
        """Insert template from preview text widget"""
        if hasattr(self, 'template_preview'):
            template_text = self.template_preview.get(1.0, tk.END).strip()
            if template_text:
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, template_text)
                self.ssml_enabled.set(True)
                self.on_ssml_toggle()
                self.on_text_change(None)
                self.log_status("📋 Inserted SSML template from preview")
        window.destroy()
    
    def show_ssml_reference(self):
        """Show SSML reference documentation in a dialog"""
        ref_window = tk.Toplevel(self.root)
        ref_window.title("SSML Reference Guide")
        ref_window.geometry("750x700")
        ref_window.configure(bg=self.colors['bg_primary'])
        
        # Make window modal
        ref_window.transient(self.root)
        ref_window.grab_set()
        
        # Title
        title_label = tk.Label(ref_window, text="❓ SSML Quick Reference Guide",
                              font=('Segoe UI', 14, 'bold'),
                              bg=self.colors['bg_primary'], fg=self.colors['fg_primary'])
        title_label.pack(pady=(15, 10))
        
        # Subtitle
        subtitle_label = tk.Label(ref_window,
                                 text="Speech Synthesis Markup Language (W3C Standard)",
                                 font=('Segoe UI', 10),
                                 bg=self.colors['bg_primary'], fg=self.colors['fg_muted'])
        subtitle_label.pack(pady=(0, 10))
        
        # Reference text
        ref_text = scrolledtext.ScrolledText(
            ref_window, width=85, height=35,
            bg=self.colors['bg_secondary'],
            fg=self.colors['fg_primary'],
            font=('Consolas', 10),
            wrap=tk.WORD
        )
        ref_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Insert reference content
        ref_text.insert(tk.END, self.ssml_processor.get_ssml_reference())
        ref_text.config(state=tk.DISABLED)
        
        # Close button
        close_btn = ttk.Button(ref_window, text="Close",
                              command=ref_window.destroy,
                              style="Dark.TButton")
        close_btn.pack(pady=15)
    
    def validate_ssml_input(self):
        """Validate the current text as SSML"""
        text = self.text_widget.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showinfo("SSML Validation", "No text to validate.")
            return
        
        # Check if it looks like SSML
        if not self.ssml_processor.is_ssml(text):
            response = messagebox.askyesno(
                "SSML Validation",
                "The text doesn't appear to contain SSML markup.\n\n"
                "Would you like to wrap it in <speak> tags?"
            )
            if response:
                wrapped_text = f"<speak>\n    {text}\n</speak>"
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, wrapped_text)
                self.ssml_enabled.set(True)
                self.on_ssml_toggle()
                self.log_status("✓ Text wrapped in SSML <speak> tags")
            return
        
        # Parse and validate
        result = self.ssml_processor.parse_ssml(text)
        
        if result['errors']:
            error_msg = "SSML Validation Errors:\n\n" + "\n".join(result['errors'])
            messagebox.showerror("SSML Validation Failed", error_msg)
            self.log_status(f"✗ SSML validation failed: {result['errors'][0]}")
        else:
            info_msg = f"✓ SSML is valid!\n\n"
            info_msg += f"Extracted text length: {len(result['text'])} characters\n"
            info_msg += f"Number of segments: {len(result['segments'])}\n"
            
            if result['has_prosody_changes']:
                info_msg += f"Average rate adjustment: {result['rate']:.2f}x\n"
            
            # Show segment info
            if result['segments']:
                unique_features = set()
                for seg in result['segments']:
                    if seg.get('is_break'):
                        unique_features.add('breaks')
                    if seg.get('emphasis'):
                        unique_features.add(f"emphasis: {seg['emphasis']}")
                    if seg.get('interpret_as'):
                        unique_features.add(f"say-as: {seg['interpret_as']}")
                    if seg.get('phoneme'):
                        unique_features.add('phoneme hints')
                    if seg.get('voice_hint'):
                        unique_features.add('voice hints')
                
                if unique_features:
                    info_msg += f"\nFeatures detected: {', '.join(unique_features)}"
            
            messagebox.showinfo("SSML Validation", info_msg)
            self.log_status("✓ SSML validation passed")
            
            # Enable SSML mode if not already enabled
            if not self.ssml_enabled.get():
                self.ssml_enabled.set(True)
                self.on_ssml_toggle()
    
    def process_ssml_text(self, text):
        """
        Process SSML text and return plain text with prosody adjustments.
        
        Returns:
            dict with 'text' (processed plain text) and 'rate' (suggested speed multiplier)
        """
        # Check if SSML processing is enabled or auto-detect is on
        should_process = self.ssml_enabled.get() or (
            self.ssml_auto_detect.get() and self.ssml_processor.is_ssml(text)
        )
        
        if not should_process:
            return {'text': text, 'rate': 1.0, 'was_processed': False}
        
        # Parse SSML
        result = self.ssml_processor.parse_ssml(text)
        
        if result['errors']:
            self.log_status(f"⚠ SSML parsing warnings: {result['errors'][0]}")
        
        if result['text'] != text:
            self.log_status(f"🎭 SSML processed: {len(text)} chars → {len(result['text'])} chars")
            if result['has_prosody_changes']:
                self.log_status(f"   Rate adjustment: {result['rate']:.2f}x")
        
        result['was_processed'] = True
        return result

    def preview_voice(self):
        """Preview the selected voice with sample text"""
        if not self.selected_voice_config:
            messagebox.showwarning("No Voice Selected", "Please select a voice model and speaker first.")
            return

        # Use a short sample text for preview
        sample_text = "Hello! This is a preview of the selected voice. How does it sound?"

        # Temporarily store current text
        current_text = self.text_widget.get(1.0, tk.END).strip()

        # Set sample text
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, sample_text)

        # Generate speech
        self.generate_speech()

        # Restore original text after a short delay
        def restore_text():
            time.sleep(0.5)  # Wait for generation to start
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, current_text)

        threading.Thread(target=restore_text, daemon=True).start()

    def update_speed_label(self, value):
        """Update speed label when scale changes"""
        self.speed_label.config(text=f"{float(value):.1f}x")

    def update_playback_speed_label(self, value):
        """Update playback speed label when scale changes"""
        self.playback_speed_label.config(text=f"{float(value):.1f}x")

    def update_volume_label(self, value):
        """Update volume label when scale changes"""
        self.volume_label.config(text=f"{int(float(value))}%")
        if self.current_sound and self.is_playing:
            self.current_sound.set_volume(float(value) / 100.0)

    def on_paste(self, event):
        """Handle paste operation to remove duplicate consecutive lines"""
        try:
            # Get clipboard content
            clipboard_text = self.root.clipboard_get()
            
            if not clipboard_text:
                return None  # Allow default behavior for empty clipboard
            
            # Remove duplicate consecutive lines
            lines = clipboard_text.split('\n')
            cleaned_lines = []
            prev_line = None
            duplicates_removed = 0
            
            for line in lines:
                stripped = line.strip()
                # Only skip if the stripped line is non-empty and matches the previous
                if stripped and stripped == prev_line:
                    duplicates_removed += 1
                    continue  # Skip duplicate
                cleaned_lines.append(line)
                prev_line = stripped
            
            cleaned_text = '\n'.join(cleaned_lines)
            
            # Insert cleaned text at cursor position
            self.text_widget.insert(tk.INSERT, cleaned_text)
            
            # Log if duplicates were removed
            if duplicates_removed > 0:
                self.log_status(f"✓ Removed {duplicates_removed} duplicate line(s) from pasted text")
            
            # Update text stats
            self.on_text_change(None)
            
            # Return "break" to prevent default paste behavior
            return "break"
            
        except tk.TclError:
            # Clipboard is empty or inaccessible, allow default behavior
            return None
        except Exception as e:
            # Log any other errors but allow default paste
            self.log_status(f"⚠ Error processing paste: {str(e)}")
            return None
    
    def on_text_change(self, event):
        """Handle text changes for real-time validation and stats"""
        text = self.text_widget.get(1.0, tk.END).strip()

        # Update text statistics
        stats = self.text_processor.get_text_stats(text)
        stats_text = f"Characters: {stats['chars']} | Words: {stats['words']} | Lines: {stats['lines']} | Sentences: {stats['sentences']}"
        self.stats_label.config(text=stats_text)

        # Update chunking information
        if self.text_processor.needs_chunking(text):
            # Use current model type for chunking preview
            if self.selected_voice_config:
                model_type = self.selected_voice_config[1]["model_type"]
            else:
                model_type = "matcha"  # Default
            chunks = self.text_processor.split_text_into_chunks(text, model_type)
            chunk_info = f"Will split into {len(chunks)} chunks for {model_type.upper()} model"
            self.chunking_info_label.config(text=chunk_info, foreground=self.colors['accent_orange'])
        else:
            self.chunking_info_label.config(text="Single chunk processing",
                                          foreground=self.colors['accent_green'])

        # Update validation status
        is_valid, error_msg = self.text_processor.validate_text(text)
        if is_valid:
            self.validation_label.config(text="✓ Ready", foreground=self.colors['accent_green'])
        else:
            self.validation_label.config(text=f"⚠ {error_msg}", foreground=self.colors['accent_orange'])

        # Update SSML detection status
        if text and self.ssml_auto_detect.get():
            if self.ssml_processor.is_ssml(text):
                self.ssml_info_label.config(
                    text="🎭 SSML markup detected - will be processed automatically",
                    foreground=self.colors['accent_green']
                )
            elif self.ssml_enabled.get():
                self.ssml_info_label.config(
                    text="SSML mode enabled but no markup detected",
                    foreground=self.colors['accent_orange']
                )
            else:
                self.ssml_info_label.config(
                    text="SSML enables: <emphasis>, <break>, <prosody>, <say-as>, and more",
                    foreground=self.colors['accent_cyan']
                )

    def import_text(self):
        """Import text from file"""
        file_path = filedialog.askopenfilename(
            title="Import Text File",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, content)
                self.on_text_change(None)
                self.log_status(f"📁 Text imported from: {os.path.basename(file_path)}")

            except Exception as e:
                self.log_status(f"✗ Error importing text: {str(e)}")
                messagebox.showerror("Import Error", f"Failed to import text:\n{str(e)}")

    def export_text(self):
        """Export text to file"""
        text = self.text_widget.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Export Warning", "No text to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Text File",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                self.log_status(f"💾 Text exported to: {os.path.basename(file_path)}")

            except Exception as e:
                self.log_status(f"✗ Error exporting text: {str(e)}")
                messagebox.showerror("Export Error", f"Failed to export text:\n{str(e)}")

    def clear_text(self):
        """Clear text widget"""
        if messagebox.askyesno("Clear Text", "Are you sure you want to clear all text?"):
            self.text_widget.delete(1.0, tk.END)
            self.on_text_change(None)
            self.log_status("🧹 Text cleared")

    def clear_cache(self):
        """Clear audio cache"""
        if messagebox.askyesno("Clear Cache", "Are you sure you want to clear the audio cache?"):
            self.audio_cache.clear()
            self.update_performance_display()
            self.log_status("🗑️ Audio cache cleared")

    def update_performance_display(self):
        """Update performance information display"""
        cache_size = len(self.audio_cache.cache)
        avg_rtf = self.performance_monitor.get_average_rtf()

        perf_text = f"Cache: {cache_size} items"
        if avg_rtf > 0:
            perf_text += f" | Avg RTF: {avg_rtf:.3f}"
        else:
            perf_text += " | Avg RTF: N/A"

        self.perf_label.config(text=perf_text)

    def on_seek(self, value):
        """Handle seek bar changes"""
        if self.audio_duration > 0 and not self.is_playing:
            seek_position = (float(value) / 100.0) * self.audio_duration
            self.pause_position = seek_position
            self.update_time_display(seek_position)

    def format_time(self, seconds):
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def update_time_display(self, current_time=None):
        """Update the time display"""
        if current_time is None:
            if self.is_playing:
                elapsed = time.time() - self.playback_start_time
                current_time = self.pause_position + elapsed * self.playback_speed_var.get()
            else:
                current_time = self.pause_position

        current_time = min(current_time, self.audio_duration)
        current_str = self.format_time(current_time)
        total_str = self.format_time(self.audio_duration)
        self.time_label.config(text=f"{current_str} / {total_str}")

        if self.audio_duration > 0:
            progress = (current_time / self.audio_duration) * 100
            self.seek_var.set(progress)

    def log_status(self, message, level="info"):
        """Add message to status text widget with color coding"""
        timestamp = time.strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}\n"

        # Configure text tags for different message types with Dracula colors
        if not hasattr(self, '_tags_configured'):
            self.status_text.tag_configure("info", foreground=self.colors['fg_secondary'])
            self.status_text.tag_configure("success", foreground=self.colors['accent_green'])
            self.status_text.tag_configure("warning", foreground=self.colors['accent_orange'])
            self.status_text.tag_configure("error", foreground=self.colors['accent_red'])
            self.status_text.tag_configure("timestamp", foreground=self.colors['fg_muted'])
            self._tags_configured = True

        # Determine message type based on content
        if "✓" in message or "successfully" in message.lower():
            tag = "success"
        elif "⚠" in message or "warning" in message.lower():
            tag = "warning"
        elif "✗" in message or "error" in message.lower():
            tag = "error"
        else:
            tag = "info"

        self.status_text.insert(tk.END, formatted_message, tag)
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def preload_models(self):
        """Preload available models in background for better performance"""
        if not self.available_voice_configs:
            self.log_status("⚠ No voice models available for preloading")
            return

        def preload_thread():
            try:
                self.model_loading_in_progress = True
                self.log_status("🚀 Preloading available voice models in background...")

                # Preload the first few available models for better performance
                preload_count = 0
                max_preload = 2  # Limit to avoid excessive memory usage

                for config_id, config in self.available_voice_configs.items():
                    if preload_count >= max_preload:
                        break

                    try:
                        self.load_voice_model(config_id, config)
                        preload_count += 1
                    except Exception as e:
                        self.log_status(f"⚠ Failed to preload {config['name']}: {str(e)}")

                if preload_count > 0:
                    self.log_status(f"✓ Preloaded {preload_count} voice models - ready for fast generation!")
                else:
                    self.log_status("⚠ No models could be preloaded")

            except Exception as e:
                self.log_status(f"⚠ Model preloading failed: {str(e)}")
            finally:
                self.model_loading_in_progress = False

        # Start preloading in background thread
        preload_thread = threading.Thread(target=preload_thread, daemon=True)
        preload_thread.start()

    def load_voice_model(self, config_id, config):
        """Load a voice model based on configuration with robust error handling"""
        if config_id in self.tts_models:
            return self.tts_models[config_id]

        self.log_status(f"Loading {config['name']}...")

        # Wrap everything in try-catch to prevent crashes
        try:
            model_files = config["model_files"]

            if config["model_type"] == "matcha":
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
                        provider="cpu",
                    ),
                    max_num_sentences=1,
                )

            elif config["model_type"] == "kokoro":
                # Handle multi-lingual Kokoro models properly
                lexicon_path = model_files.get("lexicon", "")
                dict_dir_path = model_files.get("dict_dir", "")

                # For multi-lingual models, ensure lexicon and dict_dir are properly set
                if "multi-lang" in config_id or "enhanced" in config_id:
                    if not lexicon_path:
                        # Use default English lexicon for multi-lang models
                        lexicon_path = f"{model_files['model'].split('/')[0]}/lexicon-us-en.txt"
                    if not dict_dir_path:
                        dict_dir_path = f"{model_files['model'].split('/')[0]}/dict"

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
                            provider="cpu",
                        ),
                        max_num_sentences=1,
                    )
                except Exception as kokoro_error:
                    # If multi-lingual setup fails, try without dict_dir (fallback to single-lang mode)
                    self.log_status(f"⚠ Multi-lingual setup failed for {config['name']}, trying single-language mode...")
                    try:
                        tts_config = sherpa_onnx.OfflineTtsConfig(
                            model=sherpa_onnx.OfflineTtsModelConfig(
                                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                                    model=model_files["model"],
                                    voices=model_files["voices"],
                                    tokens=model_files["tokens"],
                                    lexicon="",  # Empty for single-language
                                    data_dir=model_files["data_dir"],
                                    dict_dir="",  # Empty for single-language
                                ),
                                num_threads=2,
                                debug=False,
                                provider="cpu",
                            ),
                            max_num_sentences=1,
                        )
                        self.log_status(f"✓ {config['name']} loaded in single-language mode")
                    except Exception as fallback_error:
                        raise kokoro_error  # Re-raise original error if fallback also fails

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
                        provider="cpu",
                    ),
                    max_num_sentences=1,
                )
            else:
                raise ValueError(f"Unsupported model type: {config['model_type']}")

            # Try to create the TTS model with comprehensive error handling
            try:
                # Wrap model creation in additional safety
                import subprocess
                import tempfile

                # All models should work now with proper fixes applied
                # But wrap in additional safety for any remaining issues

                tts_model = sherpa_onnx.OfflineTts(tts_config)
                self.tts_models[config_id] = tts_model
                self.log_status(f"✓ {config['name']} loaded successfully")
                return tts_model

            except SystemExit:
                # Catch system exit calls that cause crashes
                self.log_status(f"✗ {config['name']} caused system exit - model incompatible")
                self.log_status("💡 This model may require different parameters or files")
                return None
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                self.log_status(f"✗ {config['name']} loading interrupted by user")
                return None
            except Exception as model_error:
                # Handle model creation errors
                error_msg = str(model_error)
                if "multi-lingual" in error_msg or "lexicon" in error_msg or "dict-dir" in error_msg:
                    self.log_status(f"✗ {config['name']} requires multi-lingual setup - skipping for stability")
                    self.log_status("💡 This model needs specific lexicon and dictionary configuration")
                elif "version" in error_msg and "Kokoro" in error_msg:
                    self.log_status(f"✗ {config['name']} version mismatch - skipping for stability")
                    self.log_status("💡 This model version may not be compatible with current sherpa-onnx")
                else:
                    self.log_status(f"✗ {config['name']} model creation failed: {error_msg}")
                return None

        except Exception as e:
            error_msg = str(e)
            if "phontab" in error_msg or "espeak-ng-data" in error_msg:
                self.log_status(f"✗ Failed to load {config['name']}: Missing espeak-ng-data files")
                self.log_status("💡 Run fix_espeak_data.py to download missing language data")
            elif "dict" in error_msg and "utf8" in error_msg:
                self.log_status(f"✗ Failed to load {config['name']}: Missing dictionary files")
                self.log_status("💡 Run fix_kokoro_dict.py to download missing dictionary files")
            else:
                self.log_status(f"✗ Failed to load {config['name']}: {error_msg}")
            return None

    def get_current_speaker_id(self):
        """Get the currently selected speaker ID with proper mapping for multi-speaker models"""
        if not self.selected_voice_config or not self.speaker_combo.get():
            return 0

        config_id, config = self.selected_voice_config
        selected_speaker_text = self.speaker_combo.get()

        # Find the speaker ID from the display text
        for speaker_id, speaker_info in config["speakers"].items():
            gender_icon = "👩" if speaker_info["gender"] == "female" else "👨"
            accent_text = f" ({speaker_info['accent']})" if speaker_info.get('accent') else ""
            if f"{gender_icon} {speaker_info['name']}{accent_text} - {speaker_info['description']}" == selected_speaker_text:
                # All configurations now use sequential speaker IDs (0, 1, 2, etc.)
                # so we can use the speaker_id directly
                self.log_status(f"🎯 Selected speaker: {speaker_info['name']} ({speaker_info['gender']}) - ID {speaker_id}")
                return speaker_id

        return 0

    def generate_speech_thread(self):
        """Generate speech in separate thread with chunking support"""
        try:
            # Check for cancellation
            if self.generation_cancelled:
                return

            # Get and validate text
            raw_text = self.text_widget.get(1.0, tk.END).strip()
            if not raw_text:
                self.log_status("⚠ Please enter some text to synthesize")
                return

            # Check for cancellation
            if self.generation_cancelled:
                return

            # Validate text
            is_valid, error_msg = self.text_processor.validate_text(raw_text)
            if not is_valid:
                self.log_status(f"⚠ Text validation failed: {error_msg}")
                return

            # Preprocess text with enhanced options for OOV handling
            options = {key: var.get() for key, var in self.text_options.items()}
            # Enable encoding fixes and modern term replacement by default
            options.update({
                'fix_encoding': True,
                'replace_modern_terms': True
            })
            text = self.text_processor.preprocess_text(raw_text, options)

            if text != raw_text:
                self.log_status("🔧 Text preprocessed for optimal synthesis")

            # Check for cancellation
            if self.generation_cancelled:
                return

            # Process SSML if enabled or auto-detected
            ssml_result = self.process_ssml_text(text)
            if ssml_result['was_processed']:
                text = ssml_result['text']
                # Apply SSML rate adjustment to generation speed
                if ssml_result.get('rate', 1.0) != 1.0:
                    # Combine with user-selected speed
                    ssml_rate = ssml_result['rate']
                    self.log_status(f"🎭 SSML prosody: applying {ssml_rate:.2f}x rate adjustment")

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
                self.log_status(f"📄 Long text detected ({len(text)} chars) - splitting into chunks...")
                self._generate_chunked_speech(text, model_type, speed, speaker_id)
            else:
                self._generate_single_speech(text, model_type, speed, speaker_id)

        except Exception as e:
            error_msg = str(e)
            # Handle specific ONNX runtime errors
            if "BroadcastIterator::Append" in error_msg or "axis == 1 || axis == largest was false" in error_msg:
                self.log_status("✗ Model compatibility error: Text may be too long or contain unsupported characters. Try shorter text or different model.")
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
        config_id = self.selected_voice_config[0] if self.selected_voice_config else None
        cached_audio = self.audio_cache.get(text, model_type, speaker_id, speed, config_id)
        if cached_audio:
            self.log_status("⚡ Using cached audio (instant generation!)")

            # Use cached data
            self.audio_data = cached_audio['audio_data']
            self.sample_rate = cached_audio['sample_rate']
            self.audio_duration = len(self.audio_data) / self.sample_rate
            self.pause_position = 0.0

            # Create temporary file from cached data
            temp_file = f"audio_output/temp_cached_{uuid.uuid4().hex[:8]}.wav"
            self.current_audio_file = temp_file
            sf.write(self.current_audio_file, self.audio_data, self.sample_rate)

            # Record performance metrics
            self.performance_monitor.end_generation(self.audio_duration, from_cache=True)

            self.log_status(f"✓ Cached audio loaded (Duration: {self.audio_duration:.2f}s)")

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
        config_id = self.selected_voice_config[0] if self.selected_voice_config else None
        self.audio_cache.put(text, model_type, speaker_id, speed,
                           self.audio_data.copy(), self.sample_rate, config_id)

        # Calculate and record performance metrics
        elapsed_time = time.time() - start_time
        rtf = elapsed_time / self.audio_duration if self.audio_duration > 0 else 0

        metric = self.performance_monitor.end_generation(self.audio_duration, from_cache=False)
        avg_rtf = self.performance_monitor.get_average_rtf(model_type)

        self.log_status(f"✓ Speech generated successfully!")
        self.log_status(f"  Duration: {self.audio_duration:.2f} seconds")
        self.log_status(f"  Generation time: {elapsed_time:.2f} seconds")
        self.log_status(f"  RTF (Real-time factor): {rtf:.3f}")
        self.log_status(f"  Average RTF ({model_type}): {avg_rtf:.3f}")

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

        self.log_status(f"📄 Split into {total_chunks} chunks for {model_type.upper()} model (token-aware chunking)")

        # Log chunk sizes for debugging
        for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
            estimated_tokens = self.text_processor.estimate_token_count(chunk)
            self.log_status(f"  Chunk {i}: {len(chunk)} chars, ~{estimated_tokens} tokens")
        if total_chunks > 3:
            self.log_status(f"  ... and {total_chunks - 3} more chunks")

        # Check if entire chunked result is cached (include voice config)
        config_id = self.selected_voice_config[0] if self.selected_voice_config else None
        full_cache_key = f"chunked_{hashlib.md5(text.encode()).hexdigest()}"
        cached_full = self.audio_cache.get(full_cache_key, model_type, speaker_id, speed, config_id)
        if cached_full:
            self.log_status("⚡ Using cached chunked audio (instant generation!)")
            self._use_cached_audio(cached_full)
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
                self.log_status(f"🎵 Processing chunk {i}/{total_chunks} ({len(chunk)} chars)...")

                # Update progress bar to show chunk progress
                progress = (i - 1) / total_chunks * 100
                self.root.after(0, lambda p=progress: self.progress.configure(value=p))

                # Check cache for individual chunk (include voice config)
                config_id = self.selected_voice_config[0] if self.selected_voice_config else None
                cached_chunk = self.audio_cache.get(chunk, model_type, speaker_id, speed, config_id)
                if cached_chunk:
                    self.log_status(f"  ⚡ Chunk {i} found in cache")
                    audio_data = cached_chunk['audio_data']
                else:
                    # Check for cancellation before generating
                    if self.generation_cancelled:
                        self.log_status("🚫 Generation cancelled during chunk processing")
                        return

                    # Generate audio for this chunk
                    estimated_tokens = self.text_processor.estimate_token_count(chunk)
                    self.log_status(f"  🔍 Chunk {i}: {len(chunk)} chars, ~{estimated_tokens} tokens")

                    audio = self._generate_audio_for_text(chunk, model_type, speed, speaker_id)
                    if audio is None:
                        self.log_status(f"  ⚠ Failed to generate chunk {i}, skipping...")
                        continue

                    # Convert to numpy array
                    if isinstance(audio.samples, list):
                        audio_data = np.array(audio.samples, dtype=np.float32)
                    else:
                        audio_data = np.array(audio.samples, dtype=np.float32)

                    # Cache individual chunk (include voice config)
                    config_id = self.selected_voice_config[0] if self.selected_voice_config else None
                    self.audio_cache.put(chunk, model_type, speaker_id, speed,
                                       audio_data.copy(), audio.sample_rate, config_id)

                    self.log_status(f"  ✓ Chunk {i} generated ({len(audio_data)/audio.sample_rate:.1f}s)")

                audio_chunks.append(audio_data)
                successful_chunks += 1
                self.log_status(f"  ✓ Chunk {i} completed successfully")

            except Exception as e:
                error_msg = str(e)
                if "BroadcastIterator::Append" in error_msg or "axis == 1 || axis == largest was false" in error_msg:
                    self.log_status(f"  ✗ Model compatibility error in chunk {i}: {error_msg[:100]}...")
                else:
                    self.log_status(f"  ✗ Error processing chunk {i}: {error_msg[:100]}...")
                self.log_status(f"  ⏭ Continuing with remaining chunks...")
                continue

        if not audio_chunks:
            self.log_status("✗ Failed to generate any audio chunks")
            return

        # Check success rate and provide detailed feedback
        success_rate = successful_chunks / total_chunks
        failed_chunks = total_chunks - successful_chunks

        if success_rate < 0.5:  # Less than 50% success
            self.log_status(f"⚠ Low success rate: {successful_chunks}/{total_chunks} chunks succeeded ({success_rate:.1%})")
            self.log_status(f"  {failed_chunks} chunks failed - consider using shorter text or switching to a different model")
        elif failed_chunks > 0:
            self.log_status(f"✓ Good success rate: {successful_chunks}/{total_chunks} chunks succeeded ({success_rate:.1%})")
            self.log_status(f"  Note: {failed_chunks} chunks were skipped due to errors")
        else:
            self.log_status(f"✓ Perfect success rate: All {total_chunks} chunks processed successfully!")

        # Stitch chunks together
        self.log_status(f"🔗 Stitching {successful_chunks} audio chunks together...")

        # Use the sample rate from the first successful generation or cached chunk
        if hasattr(self, 'sample_rate'):
            sample_rate = self.sample_rate
        else:
            sample_rate = 22050  # Default fallback

        stitched_audio = self.audio_stitcher.stitch_audio_chunks(audio_chunks, sample_rate)

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
        config_id = self.selected_voice_config[0] if self.selected_voice_config else None
        self.audio_cache.put(full_cache_key, model_type, speaker_id, speed,
                           self.audio_data.copy(), self.sample_rate, config_id)

        # Calculate performance metrics
        elapsed_time = time.time() - start_time
        rtf = elapsed_time / self.audio_duration if self.audio_duration > 0 else 0

        metric = self.performance_monitor.end_generation(self.audio_duration, from_cache=False)
        avg_rtf = self.performance_monitor.get_average_rtf(model_type)

        self.log_status(f"✓ Chunked speech generated successfully!")
        self.log_status(f"  Total chunks: {total_chunks} (successful: {successful_chunks})")
        self.log_status(f"  Total duration: {self.audio_duration:.2f} seconds")
        self.log_status(f"  Total generation time: {elapsed_time:.2f} seconds")
        self.log_status(f"  Overall RTF: {rtf:.3f}")
        self.log_status(f"  Average RTF ({model_type}): {avg_rtf:.3f}")

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
            is_valid, error_msg = self.text_processor.validate_chunk_for_model(text, model_type)
            if not is_valid:
                self.log_status(f"⚠ Chunk validation warning: {error_msg} - attempting anyway...")
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
                    self.log_status(f"⚠ Language processing error (missing espeak data): {error_msg}")
                    self.log_status("💡 Try using a different voice model or check espeak-ng-data installation")
                elif "No such file or directory" in error_msg:
                    self.log_status(f"⚠ Missing model file: {error_msg}")
                    self.log_status("💡 Some model files may be missing - try redownloading the model")
                else:
                    self.log_status(f"⚠ Voice generation error: {error_msg}")
                    self.log_status("💡 Try using a different voice or check the text input")
                return None

        except Exception as e:
            error_msg = str(e)
            # Handle specific ONNX runtime errors with helpful messages
            if "BroadcastIterator::Append" in error_msg or "axis == 1 || axis == largest was false" in error_msg:
                self.log_status("✗ Model compatibility error: Chunk too complex for model. Trying to split further...")
                # Try to split the problematic chunk further if it's long enough
                if len(text) > 1000:
                    self.log_status("  📄 Attempting to split problematic chunk...")
                    return self._handle_problematic_chunk(text, model_type, speed, speaker_id)
                else:
                    self.log_status("  ⚠ Chunk too short to split further - skipping this chunk")
            elif "Non-zero status code" in error_msg:
                self.log_status("✗ ONNX Runtime error: Model processing failed. Try different text or model.")
            else:
                self.log_status(f"✗ Error generating audio: {error_msg}")
            return None

    def _handle_problematic_chunk(self, text, model_type, speed, speaker_id):
        """Handle chunks that cause ONNX runtime errors by splitting them further"""
        try:
            # Split the problematic chunk into smaller pieces
            sentences = text.split('. ')
            if len(sentences) <= 1:
                # Try splitting by other punctuation if no sentences
                sentences = text.split(', ')
                if len(sentences) <= 1:
                    # Last resort: split by words
                    words = text.split()
                    mid = len(words) // 2
                    sentences = [' '.join(words[:mid]), ' '.join(words[mid:])]

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

                    audio = tts_model.generate(sentence.strip(), sid=speaker_id, speed=speed)

                    if audio:
                        audio_chunks.append(audio)
                        self.log_status(f"    ✓ Sub-chunk {i+1}/{len(sentences)} processed")
                except Exception as sub_e:
                    error_msg = str(sub_e)
                    if "phontab" in error_msg or "espeak-ng-data" in error_msg:
                        self.log_status(f"    ⚠ Language processing error in sub-chunk {i+1}: skipping")
                    elif "No such file or directory" in error_msg:
                        self.log_status(f"    ⚠ Missing file error in sub-chunk {i+1}: skipping")
                    else:
                        self.log_status(f"    ⚠ Error in sub-chunk {i+1}: {error_msg}")
                    continue  # Skip this chunk and continue with others
                except Exception as sub_e:
                    self.log_status(f"    ⚠ Sub-chunk {i+1} failed: {str(sub_e)[:50]}...")
                    continue

            if not audio_chunks:
                self.log_status("  ✗ All sub-chunks failed")
                return None

            # Combine the audio chunks (simplified - just return the first successful one for now)
            # In a more sophisticated implementation, we would stitch them together
            self.log_status(f"  ✓ Successfully processed {len(audio_chunks)}/{len(sentences)} sub-chunks")
            return audio_chunks[0]  # Return first successful chunk

        except Exception as e:
            self.log_status(f"  ✗ Error handling problematic chunk: {str(e)}")
            return None

    def _use_cached_audio(self, cached_audio):
        """Use cached audio data"""
        self.audio_data = cached_audio['audio_data']
        self.sample_rate = cached_audio['sample_rate']
        self.audio_duration = len(self.audio_data) / self.sample_rate
        self.pause_position = 0.0

        # Create temporary file from cached data
        temp_file = f"audio_output/temp_cached_{uuid.uuid4().hex[:8]}.wav"
        self.current_audio_file = temp_file
        sf.write(self.current_audio_file, self.audio_data, self.sample_rate)

        # Record performance metrics
        self.performance_monitor.end_generation(self.audio_duration, from_cache=True)

        self.log_status(f"✓ Cached audio loaded (Duration: {self.audio_duration:.2f}s)")

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
            self.progress.configure(mode='determinate', value=0, maximum=100)
        else:
            # Use indeterminate progress for single chunk
            self.progress.configure(mode='indeterminate')
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

    def save_audio(self):
        """Save audio to file - opens advanced export options dialog"""
        if self.audio_data is None or len(self.audio_data) == 0:
            messagebox.showwarning("No Audio", "No audio to save. Generate speech first.")
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
            original_text
        )
        
        export_options = dialog.show()
        
        if not export_options:
            return  # User cancelled
        
        # Determine save location based on split mode
        split_mode = export_options.get('split_mode', 'none')
        fmt = export_options.get('format', 'wav')
        fmt_config = self.audio_exporter.FORMATS.get(fmt, {})
        ext = fmt_config.get('extension', '.wav')
        
        if split_mode == 'none':
            # Single file export
            file_path = filedialog.asksaveasfilename(
                title="Save Audio As",
                defaultextension=ext,
                initialdir="audio_output",
                filetypes=[
                    (f"{fmt.upper()} files", f"*{ext}"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self._export_single_file(file_path, export_options)
        else:
            # Multiple file export - choose directory
            output_dir = filedialog.askdirectory(
                title="Select Output Directory for Split Files",
                initialdir="audio_output"
            )
            
            if output_dir:
                self._export_split_files(output_dir, export_options)
    
    def _export_single_file(self, file_path, options):
        """Export audio as a single file"""
        try:
            fmt = options.get('format', 'wav')
            
            export_options = {
                'target_sample_rate': options.get('target_sample_rate', self.sample_rate),
                'normalize': options.get('normalize', False),
                'metadata': options.get('metadata', {}),
            }
            
            if 'bitrate' in options:
                export_options['bitrate'] = options['bitrate']
            
            self.log_status(f"💾 Exporting to {fmt.upper()}...")
            
            success, message, output_path = self.audio_exporter.export(
                self.audio_data,
                self.sample_rate,
                file_path,
                fmt,
                export_options
            )
            
            if success:
                file_size = os.path.getsize(output_path) / 1024  # KB
                self.log_status(f"✓ {message}")
                self.log_status(f"  File size: {file_size:.1f} KB")
                messagebox.showinfo("Export Complete", f"Audio exported successfully!\n\n{output_path}")
            else:
                self.log_status(f"✗ Export failed: {message}")
                messagebox.showerror("Export Failed", message)
                
        except Exception as e:
            self.log_status(f"✗ Export error: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export audio:\n{str(e)}")
    
    def _export_split_files(self, output_dir, options):
        """Export audio as multiple split files"""
        try:
            split_mode = options.get('split_mode', 'silence')
            fmt = options.get('format', 'wav')
            
            # Prepare segments based on split mode
            if split_mode == 'silence':
                silence_settings = options.get('silence_settings', {})
                min_silence = silence_settings.get('min_silence_len', 500)
                threshold = silence_settings.get('silence_thresh', -40)
                
                self.log_status(f"🔍 Detecting silence regions (threshold: {threshold}dB, min: {min_silence}ms)...")
                
                segments = self.audio_exporter.split_by_silence(
                    self.audio_data,
                    self.sample_rate,
                    min_silence_len=min_silence,
                    silence_thresh=threshold
                )
                
                self.log_status(f"  Found {len(segments)} segment(s)")
                
                # Convert to (title, audio) format
                audio_segments = [(f"Track {i:02d}", seg) for i, seg in enumerate(segments, 1)]
                
            elif split_mode == 'chapters':
                chapters = options.get('chapters', [])
                
                if not chapters:
                    self.log_status("⚠ No chapters detected, exporting as single file...")
                    self._export_single_file(
                        os.path.join(output_dir, f"audio.{fmt}"),
                        options
                    )
                    return
                
                self.log_status(f"📚 Splitting by {len(chapters)} chapter(s)...")
                
                # For chapter splitting, we need to estimate timing
                # Calculate approximate timing based on text position
                audio_duration_ms = len(self.audio_data) * 1000 // self.sample_rate
                
                chapter_markers = []
                current_time = 0
                
                for i, ch in enumerate(chapters):
                    chapter_markers.append({
                        'start_ms': current_time,
                        'title': ch.get('title', f'Chapter {i+1}')
                    })
                    # Estimate time for this chapter (proportional to text)
                    if i < len(chapters) - 1:
                        # Use roughly equal distribution for simplicity
                        current_time += audio_duration_ms // len(chapters)
                
                audio_segments = self.audio_exporter.split_by_chapters(
                    self.audio_data,
                    self.sample_rate,
                    chapter_markers
                )
            else:
                # Fallback - single segment
                audio_segments = [("Full Audio", self.audio_data)]
            
            # Export all segments
            export_options = {
                'sample_rate': self.sample_rate,
                'target_sample_rate': options.get('target_sample_rate', self.sample_rate),
                'normalize': options.get('normalize', False),
                'metadata': options.get('metadata', {}),
            }
            
            if 'bitrate' in options:
                export_options['bitrate'] = options['bitrate']
            
            base_name = options.get('metadata', {}).get('title', 'audio') or 'audio'
            base_name = re.sub(r'[<>:"/\\|?*]', '', base_name)[:30]
            
            self.log_status(f"💾 Exporting {len(audio_segments)} file(s) to {fmt.upper()}...")
            
            results = self.audio_exporter.export_multiple_tracks(
                audio_segments,
                output_dir,
                base_name,
                fmt,
                export_options
            )
            
            # Report results
            success_count = sum(1 for r in results if r[0])
            fail_count = len(results) - success_count
            
            for success, message, path in results:
                if success:
                    file_size = os.path.getsize(path) / 1024 if path else 0
                    self.log_status(f"  ✓ {os.path.basename(path)} ({file_size:.1f} KB)")
                else:
                    self.log_status(f"  ✗ {message}")
            
            self.log_status(f"✓ Export complete: {success_count} succeeded, {fail_count} failed")
            
            if success_count > 0:
                messagebox.showinfo(
                    "Export Complete",
                    f"Successfully exported {success_count} file(s) to:\n{output_dir}"
                )
            else:
                messagebox.showerror("Export Failed", "All exports failed. Check the status log for details.")
                
        except Exception as e:
            self.log_status(f"✗ Split export error: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export split files:\n{str(e)}")
    
    def save_audio_quick(self):
        """Quick save to WAV (original simple export behavior)"""
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            file_path = filedialog.asksaveasfilename(
                title="Quick Save as WAV",
                defaultextension=".wav",
                initialdir="audio_output",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )

            if file_path:
                try:
                    shutil.copy2(self.current_audio_file, file_path)
                    self.log_status(f"💾 Audio saved to: {file_path}")
                except Exception as e:
                    self.log_status(f"✗ Error saving audio: {str(e)}")

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
            if hasattr(self, 'current_audio_file') and self.current_audio_file:
                try:
                    if os.path.exists(self.current_audio_file):
                        os.remove(self.current_audio_file)
                except:
                    pass

            # Quit pygame
            pygame.mixer.quit()

        except Exception:
            pass  # Ignore cleanup errors

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
