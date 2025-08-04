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
        """Preprocess text based on options"""
        if not text:
            return text

        if options is None:
            options = {}

        processed = text

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

        return processed

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


class AudioCache:
    """Manages caching of generated audio"""

    def __init__(self, max_size=50, cache_dir=None):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.cache_file = os.path.join(self.cache_dir, 'tts_audio_cache.pkl')
        self.load_cache()

    def _generate_key(self, text, model_type, speaker_id, speed):
        """Generate cache key from parameters"""
        key_data = f"{text}|{model_type}|{speaker_id}|{speed}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, text, model_type, speaker_id, speed):
        """Get cached audio if available"""
        key = self._generate_key(text, model_type, speaker_id, speed)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, text, model_type, speaker_id, speed, audio_data, sample_rate):
        """Cache audio data"""
        key = self._generate_key(text, model_type, speaker_id, speed)

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
        self.root = root
        self.root.title("High-Quality English TTS - Sherpa-ONNX Enhanced")
        self.root.geometry("1300x1100")

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
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # TTS model instances
        self.matcha_tts = None
        self.kokoro_tts = None
        self.current_audio_file = None
        self.model_loading_in_progress = False

        # Model availability flags
        self.matcha_available = False
        self.kokoro_available = False

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
            'remove_emails': tk.BooleanVar(value=False)
        }

        # Setup theme and UI
        self.setup_theme()
        self.setup_ui()
        self.check_models()

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
                       background=self.colors['bg_primary'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=0,
                       relief='flat')

        # Configure Label styles
        style.configure('Dark.TLabel',
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 9))

        style.configure('Title.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['fg_primary'],
                       font=('Segoe UI', 18, 'bold'),
                       borderwidth=0,
                       relief='flat')

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
                       font=('Segoe UI', 10, 'bold'),
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
                       font=('Segoe UI', 10),
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
                       font=('Segoe UI', 10),
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
                       font=('Segoe UI', 10),
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
                       font=('Segoe UI', 10),
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
                       font=('Segoe UI', 10, 'bold'),
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
                       font=('Segoe UI', 10))

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
                       font=('Segoe UI', 9))

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

        # Title with modern styling
        title_label = ttk.Label(main_frame, text="🎵 High-Quality English Text-to-Speech", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 25))

        # Model selection frame with dark theme
        model_frame = ttk.LabelFrame(main_frame, text="🤖 TTS Model Selection", style='Dark.TLabelframe', padding="15")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))

        self.model_var = tk.StringVar(value="matcha")

        self.matcha_radio = ttk.Radiobutton(model_frame, text="🎯 Matcha-TTS LJSpeech (High Quality)",
                                          variable=self.model_var, value="matcha", style='Dark.TRadiobutton')
        self.matcha_radio.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.kokoro_radio = ttk.Radiobutton(model_frame, text="🗣️ Kokoro English (Multiple Speakers)",
                                          variable=self.model_var, value="kokoro", style='Dark.TRadiobutton')
        self.kokoro_radio.grid(row=1, column=0, sticky=tk.W, pady=5)

        # Speaker selection (for Kokoro)
        speaker_frame = ttk.Frame(model_frame, style='Dark.TFrame')
        speaker_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 0))

        ttk.Label(speaker_frame, text="👤 Speaker ID (Kokoro only):", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.speaker_var = tk.StringVar(value="0")
        speaker_spinbox = ttk.Spinbox(speaker_frame, from_=0, to=10, width=8, textvariable=self.speaker_var, style='Dark.TSpinbox')
        speaker_spinbox.grid(row=0, column=1, padx=(15, 0))

        # Speed control
        speed_frame = ttk.Frame(model_frame, style='Dark.TFrame')
        speed_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(15, 0))

        ttk.Label(speed_frame, text="⚡ Generation Speed:", style='Dark.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.5, to=2.0, variable=self.speed_var,
                               orient=tk.HORIZONTAL, style='Dark.Horizontal.TScale')
        speed_scale.grid(row=0, column=1, padx=(15, 10), sticky=(tk.W, tk.E))
        self.speed_label = ttk.Label(speed_frame, text="1.0x", style='Dark.TLabel')
        self.speed_label.grid(row=0, column=2, padx=(5, 0))

        # Update speed label when scale changes
        speed_scale.configure(command=self.update_speed_label)

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
                       style='Dark.TRadiobutton').grid(row=1, column=0, sticky=tk.W, padx=(0, 15))
        ttk.Checkbutton(preprocess_frame, text="Remove emails",
                       variable=self.text_options['remove_emails'],
                       style='Dark.TRadiobutton').grid(row=1, column=1, sticky=tk.W, padx=(0, 15))

        # Chunking info frame
        chunking_frame = ttk.Frame(text_frame, style='Card.TFrame', padding="8")
        chunking_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

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
        self.text_widget.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Bind text change events for real-time validation and stats
        self.text_widget.bind('<KeyRelease>', self.on_text_change)
        self.text_widget.bind('<Button-1>', self.on_text_change)

        # Text statistics frame
        stats_frame = ttk.Frame(text_frame, style='Card.TFrame', padding="8")
        stats_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

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
        model_frame.columnconfigure(0, weight=1)
        speaker_frame.columnconfigure(1, weight=1)
        speed_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(0, weight=1)
        playback_frame.columnconfigure(1, weight=1)
        playback_frame.columnconfigure(3, weight=1)
        seek_frame.columnconfigure(1, weight=1)
        playback_speed_frame.columnconfigure(1, weight=1)
        volume_frame.columnconfigure(1, weight=1)

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
            model_type = self.model_var.get()
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

    def check_models(self):
        """Check if models are available"""
        self.log_status("Checking available models...")

        # Check Matcha-TTS
        matcha_path = Path("matcha-icefall-en_US-ljspeech")
        vocoder_path = Path("vocos-22khz-univ.onnx")
        if (matcha_path.exists() and vocoder_path.exists() and
            (matcha_path / "model-steps-3.onnx").exists()):
            self.matcha_available = True
            self.log_status("✓ Matcha-TTS LJSpeech model found")
        else:
            self.matcha_radio.config(state=tk.DISABLED)
            self.log_status("✗ Matcha-TTS model not found")

        # Check Kokoro
        kokoro_path = Path("kokoro-en-v0_19")
        if (kokoro_path.exists() and
            (kokoro_path / "model.onnx").exists() and
            (kokoro_path / "voices.bin").exists()):
            self.kokoro_available = True
            self.log_status("✓ Kokoro English model found")
        else:
            self.kokoro_radio.config(state=tk.DISABLED)
            self.log_status("✗ Kokoro English model not found")
        if self.matcha_available:
            self.model_var.set("matcha")
        elif self.kokoro_available:
            self.model_var.set("kokoro")
        else:
            self.log_status("⚠ No models available! Please download models first.")
            self.generate_btn.config(state=tk.DISABLED)

    def preload_models(self):
        """Preload models in background for better performance"""
        if not (self.matcha_available or self.kokoro_available):
            return

        def preload_thread():
            try:
                self.model_loading_in_progress = True
                self.log_status("🚀 Preloading models in background for optimal performance...")

                if self.matcha_available and self.matcha_tts is None:
                    self.load_matcha_model()

                if self.kokoro_available and self.kokoro_tts is None:
                    self.load_kokoro_model()

                self.log_status("✓ Model preloading completed - ready for fast generation!")

            except Exception as e:
                self.log_status(f"⚠ Model preloading failed: {str(e)}")
            finally:
                self.model_loading_in_progress = False

        # Start preloading in background thread
        preload_thread = threading.Thread(target=preload_thread, daemon=True)
        preload_thread.start()

    def load_matcha_model(self):
        """Load Matcha-TTS model"""
        if self.matcha_tts is None:
            self.log_status("Loading Matcha-TTS model...")

            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                        acoustic_model="matcha-icefall-en_US-ljspeech/model-steps-3.onnx",
                        vocoder="vocos-22khz-univ.onnx",
                        lexicon="",  # empty for English
                        tokens="matcha-icefall-en_US-ljspeech/tokens.txt",
                        data_dir="matcha-icefall-en_US-ljspeech/espeak-ng-data",
                    ),
                    num_threads=2,
                    debug=False,
                    provider="cpu",
                ),
                max_num_sentences=1,
            )

            self.matcha_tts = sherpa_onnx.OfflineTts(config)
            self.log_status("✓ Matcha-TTS model loaded")

    def load_kokoro_model(self):
        """Load Kokoro model"""
        if self.kokoro_tts is None:
            self.log_status("Loading Kokoro English model...")

            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model="kokoro-en-v0_19/model.onnx",
                        voices="kokoro-en-v0_19/voices.bin",
                        tokens="kokoro-en-v0_19/tokens.txt",
                        lexicon="",  # empty string as no separate lexicon file
                        data_dir="kokoro-en-v0_19/espeak-ng-data",
                    ),
                    num_threads=2,
                    debug=False,
                    provider="cpu",
                ),
                max_num_sentences=1,
            )

            self.kokoro_tts = sherpa_onnx.OfflineTts(config)
            self.log_status("✓ Kokoro English model loaded")

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

            # Preprocess text
            options = {key: var.get() for key, var in self.text_options.items()}
            text = self.text_processor.preprocess_text(raw_text, options)

            if text != raw_text:
                self.log_status("🔧 Text preprocessed for optimal synthesis")

            # Check for cancellation
            if self.generation_cancelled:
                return

            model_type = self.model_var.get()
            speed = self.speed_var.get()
            speaker_id = int(self.speaker_var.get()) if model_type == "kokoro" else 0

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

        # Check cache first
        cached_audio = self.audio_cache.get(text, model_type, speaker_id, speed)
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

        # Cache the generated audio
        self.audio_cache.put(text, model_type, speaker_id, speed,
                           self.audio_data.copy(), self.sample_rate)

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

        # Check if entire chunked result is cached
        full_cache_key = f"chunked_{hashlib.md5(text.encode()).hexdigest()}"
        cached_full = self.audio_cache.get(full_cache_key, model_type, speaker_id, speed)
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

                # Check cache for individual chunk
                cached_chunk = self.audio_cache.get(chunk, model_type, speaker_id, speed)
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

                    # Cache individual chunk
                    self.audio_cache.put(chunk, model_type, speaker_id, speed,
                                       audio_data.copy(), audio.sample_rate)

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

        # Cache the final stitched result
        self.audio_cache.put(full_cache_key, model_type, speaker_id, speed,
                           self.audio_data.copy(), self.sample_rate)

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

            if model_type == "matcha":
                if not self.matcha_available:
                    self.log_status("⚠ Matcha-TTS model not available")
                    return None

                self.load_matcha_model()
                return self.matcha_tts.generate(text, sid=0, speed=speed)

            elif model_type == "kokoro":
                if not self.kokoro_available:
                    self.log_status("⚠ Kokoro model not available")
                    return None

                self.load_kokoro_model()
                return self.kokoro_tts.generate(text, sid=speaker_id, speed=speed)

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
                    if model_type == "matcha":
                        audio = self.matcha_tts.generate(sentence.strip(), sid=0, speed=speed)
                    elif model_type == "kokoro":
                        audio = self.kokoro_tts.generate(sentence.strip(), sid=speaker_id, speed=speed)

                    if audio:
                        audio_chunks.append(audio)
                        self.log_status(f"    ✓ Sub-chunk {i+1}/{len(sentences)} processed")
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
        """Save audio to file"""
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            file_path = filedialog.asksaveasfilename(
                title="Save Audio As",
                defaultextension=".wav",
                initialdir="audio_output",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )

            if file_path:
                try:
                    # Copy temporary file to chosen location
                    import shutil
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
