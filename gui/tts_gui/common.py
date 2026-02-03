#!/usr/bin/env python3
"""
Common imports and configurations for the TTS GUI.
Refactored to use PySide6 (Qt) instead of Tkinter.
"""

# PySide6 imports
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    QRadioButton,
    QGroupBox,
    QTextEdit,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QDialog,
    QListWidget,
    QScrollArea,
    QSizePolicy,
    QButtonGroup,
    QApplication,
    QLineEdit,
    QDialogButtonBox,
    QSplitter,
    QTabWidget,
    QPlainTextEdit,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QSize
from PySide6.QtGui import (
    QFont,
    QColor,
    QPalette,
    QTextCharFormat,
    QTextCursor,
    QShortcut,
    QKeySequence,
)

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

VOICE_CONFIGS = {
    "vits_piper_libritts": {
        "name": "LibriTTS Multi-Speaker (904 Diverse Voices) ⭐RECOMMENDED⭐",
        "model_type": "vits",
        "quality": "excellent",
        "description": "Massive collection of high-quality diverse American English voices",
        "model_files": {
            "model": "vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx",
            "tokens": "vits-piper-en_US-libritts_r-medium/tokens.txt",
            "lexicon": "vits-piper-en_US-libritts_r-medium/espeak-ng-data/en_dict",
            "data_dir": "vits-piper-en_US-libritts_r-medium/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Victoria",
                "gender": "female",
                "accent": "american",
                "description": "Warm, articulate female voice",
            },
            1: {
                "name": "Alexander",
                "gender": "male",
                "accent": "american",
                "description": "Professional male narrator",
            },
            2: {
                "name": "Rachel",
                "gender": "female",
                "accent": "american",
                "description": "Clear, engaging female voice",
            },
            3: {
                "name": "Christopher",
                "gender": "male",
                "accent": "american",
                "description": "Deep, resonant male voice",
            },
            4: {
                "name": "Amanda",
                "gender": "female",
                "accent": "american",
                "description": "Friendly, approachable female",
            },
            5: {
                "name": "Jonathan",
                "gender": "male",
                "accent": "american",
                "description": "Smooth male broadcaster",
            },
            6: {
                "name": "Michelle",
                "gender": "female",
                "accent": "american",
                "description": "Professional female voice",
            },
            7: {
                "name": "Daniel",
                "gender": "male",
                "accent": "american",
                "description": "Authoritative male speaker",
            },
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2",
    },
    "vits_piper_amy": {
        "name": "Amy - High Quality Female Voice ⭐RECOMMENDED⭐",
        "model_type": "vits",
        "quality": "excellent",
        "description": "Crystal clear American English female voice, perfect for narration",
        "model_files": {
            "model": "vits-piper-en_US-amy-medium/en_US-amy-medium.onnx",
            "tokens": "vits-piper-en_US-amy-medium/tokens.txt",
            "data_dir": "vits-piper-en_US-amy-medium/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Amy",
                "gender": "female",
                "accent": "american",
                "description": "Crystal clear, professional female narrator",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-medium.tar.bz2",
    },
    "vits_piper_lessac": {
        "name": "Lessac - Premium Female Voice",
        "model_type": "vits",
        "quality": "excellent",
        "description": "High-quality American English female voice with natural intonation",
        "model_files": {
            "model": "vits-piper-en_US-lessac-medium/en_US-lessac-medium.onnx",
            "tokens": "vits-piper-en_US-lessac-medium/tokens.txt",
            "data_dir": "vits-piper-en_US-lessac-medium/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Lessac",
                "gender": "female",
                "accent": "american",
                "description": "Premium quality female voice with excellent clarity",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2",
    },
    "vits_piper_ryan": {
        "name": "Ryan - High Quality Male Voice",
        "model_type": "vits",
        "quality": "excellent",
        "description": "Natural American English male voice, great for professional use",
        "model_files": {
            "model": "vits-piper-en_US-ryan-high/en_US-ryan-high.onnx",
            "tokens": "vits-piper-en_US-ryan-high/tokens.txt",
            "data_dir": "vits-piper-en_US-ryan-high/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Ryan",
                "gender": "male",
                "accent": "american",
                "description": "Natural, professional male voice",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-ryan-high.tar.bz2",
    },
    "vits_piper_danny": {
        "name": "Danny - Male Voice",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Clear American English male voice",
        "model_files": {
            "model": "vits-piper-en_US-danny-low/en_US-danny-low.onnx",
            "tokens": "vits-piper-en_US-danny-low/tokens.txt",
            "data_dir": "vits-piper-en_US-danny-low/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Danny",
                "gender": "male",
                "accent": "american",
                "description": "Clear male voice, optimized for speed",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-danny-low.tar.bz2",
    },
    "vits_piper_kathleen": {
        "name": "Kathleen - Female Voice",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Warm American English female voice",
        "model_files": {
            "model": "vits-piper-en_US-kathleen-low/en_US-kathleen-low.onnx",
            "tokens": "vits-piper-en_US-kathleen-low/tokens.txt",
            "data_dir": "vits-piper-en_US-kathleen-low/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Kathleen",
                "gender": "female",
                "accent": "american",
                "description": "Warm female voice, fast generation",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-kathleen-low.tar.bz2",
    },
    "vits_piper_libritts_high": {
        "name": "LibriTTS High Quality (10 Premium Speakers)",
        "model_type": "vits",
        "quality": "excellent",
        "description": "Top 10 highest quality speakers from LibriTTS dataset",
        "model_files": {
            "model": "vits-piper-en_US-libritts-high/en_US-libritts-high.onnx",
            "tokens": "vits-piper-en_US-libritts-high/tokens.txt",
            "data_dir": "vits-piper-en_US-libritts-high/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Speaker 0",
                "gender": "mixed",
                "accent": "american",
                "description": "Premium quality voice #1",
            },
            1: {
                "name": "Speaker 1",
                "gender": "mixed",
                "accent": "american",
                "description": "Premium quality voice #2",
            },
            2: {
                "name": "Speaker 2",
                "gender": "mixed",
                "accent": "american",
                "description": "Premium quality voice #3",
            },
            3: {
                "name": "Speaker 3",
                "gender": "mixed",
                "accent": "american",
                "description": "Premium quality voice #4",
            },
            4: {
                "name": "Speaker 4",
                "gender": "mixed",
                "accent": "american",
                "description": "Premium quality voice #5",
            },
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts-high.tar.bz2",
    },
    "vits_piper_alba": {
        "name": "Alba - British Female Voice",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Natural British English female voice",
        "model_files": {
            "model": "vits-piper-en_GB-alba-medium/en_GB-alba-medium.onnx",
            "tokens": "vits-piper-en_GB-alba-medium/tokens.txt",
            "data_dir": "vits-piper-en_GB-alba-medium/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Alba",
                "gender": "female",
                "accent": "british",
                "description": "Natural British English female voice",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-alba-medium.tar.bz2",
    },
    "vits_piper_jenny_dioco": {
        "name": "Jenny Dioco - British Multi-Speaker (2 Voices)",
        "model_type": "vits",
        "quality": "very_high",
        "description": "British English multi-speaker model",
        "model_files": {
            "model": "vits-piper-en_GB-jenny_dioco-medium/en_GB-jenny_dioco-medium.onnx",
            "tokens": "vits-piper-en_GB-jenny_dioco-medium/tokens.txt",
            "data_dir": "vits-piper-en_GB-jenny_dioco-medium/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Jenny",
                "gender": "female",
                "accent": "british",
                "description": "British female voice",
            },
            1: {
                "name": "Dioco",
                "gender": "male",
                "accent": "british",
                "description": "British male voice",
            },
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-jenny_dioco-medium.tar.bz2",
    },
    "vits_vctk": {
        "name": "VCTK Multi-Speaker (109 Diverse British Voices)",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Large collection of diverse British and international voices",
        "model_files": {
            "model": "vits-vctk/vits-vctk.onnx",
            "tokens": "vits-vctk/tokens.txt",
            "lexicon": "vits-vctk/lexicon.txt",
            "data_dir": "vits-vctk/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Speaker p225",
                "gender": "female",
                "accent": "british",
                "description": "British female - voice 1",
            },
            1: {
                "name": "Speaker p226",
                "gender": "male",
                "accent": "british",
                "description": "British male - voice 1",
            },
            2: {
                "name": "Speaker p227",
                "gender": "male",
                "accent": "british",
                "description": "British male - voice 2",
            },
            3: {
                "name": "Speaker p228",
                "gender": "female",
                "accent": "british",
                "description": "British female - voice 2",
            },
            4: {
                "name": "Speaker p229",
                "gender": "female",
                "accent": "british",
                "description": "British female - voice 3",
            },
            5: {
                "name": "Speaker p230",
                "gender": "female",
                "accent": "british",
                "description": "British female - voice 4",
            },
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2",
    },
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
            "data_dir": "matcha-icefall-en_US-ljspeech/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Linda",
                "gender": "female",
                "accent": "american",
                "description": "Premium quality female narrator with natural prosody",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2",
    },
    "vits_glados": {
        "name": "GLaDOS - AI Character Voice",
        "model_type": "vits",
        "quality": "high",
        "description": "Distinctive robotic/AI character voice (from Portal game)",
        "model_files": {
            "model": "vits-piper-en_US-glados/en_US-glados.onnx",
            "tokens": "vits-piper-en_US-glados/tokens.txt",
            "data_dir": "vits-piper-en_US-glados/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "GLaDOS",
                "gender": "female",
                "accent": "robotic",
                "description": "Distinctive AI/robotic character voice",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-glados.tar.bz2",
    },
    "vits_piper_zh_huayan": {
        "name": "Huayan - Chinese Female Voice (中文女声)",
        "model_type": "vits",
        "quality": "excellent",
        "description": "High-quality Mandarin Chinese female voice",
        "model_files": {
            "model": "vits-piper-zh_CN-huayan-medium/zh_CN-huayan-medium.onnx",
            "tokens": "vits-piper-zh_CN-huayan-medium/tokens.txt",
            "data_dir": "vits-piper-zh_CN-huayan-medium/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Huayan (华严)",
                "gender": "female",
                "accent": "mandarin",
                "description": "Premium Mandarin Chinese female voice",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-zh_CN-huayan-medium.tar.bz2",
    },
    "vits_piper_de_thorsten": {
        "name": "Thorsten - German Male Voice (Deutsch)",
        "model_type": "vits",
        "quality": "excellent",
        "description": "High-quality German male voice",
        "model_files": {
            "model": "vits-piper-de_DE-thorsten-high/de_DE-thorsten-high.onnx",
            "tokens": "vits-piper-de_DE-thorsten-high/tokens.txt",
            "data_dir": "vits-piper-de_DE-thorsten-high/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Thorsten",
                "gender": "male",
                "accent": "german",
                "description": "Premium German male voice",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-thorsten-high.tar.bz2",
    },
    "vits_piper_fr_siwis": {
        "name": "Siwis - French Female Voice (Français)",
        "model_type": "vits",
        "quality": "excellent",
        "description": "High-quality French female voice",
        "model_files": {
            "model": "vits-piper-fr_FR-siwis-medium/fr_FR-siwis-medium.onnx",
            "tokens": "vits-piper-fr_FR-siwis-medium/tokens.txt",
            "data_dir": "vits-piper-fr_FR-siwis-medium/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Siwis",
                "gender": "female",
                "accent": "french",
                "description": "Premium French female voice",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fr_FR-siwis-medium.tar.bz2",
    },
    "vits_piper_es_carlfm": {
        "name": "Carlfm - Spanish Male Voice (Español)",
        "model_type": "vits",
        "quality": "very_high",
        "description": "Natural Spanish male voice",
        "model_files": {
            "model": "vits-piper-es_ES-carlfm-x_low/es_ES-carlfm-x_low.onnx",
            "tokens": "vits-piper-es_ES-carlfm-x_low/tokens.txt",
            "data_dir": "vits-piper-es_ES-carlfm-x_low/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Carlfm",
                "gender": "male",
                "accent": "spanish",
                "description": "Natural Spanish male voice",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-es_ES-carlfm-x_low.tar.bz2",
    },
    "vits_piper_ru_irinia": {
        "name": "Irina - Russian Female Voice (Русский)",
        "model_type": "vits",
        "quality": "very_high",
        "description": "High-quality Russian female voice",
        "model_files": {
            "model": "vits-piper-ru_RU-irina-medium/ru_RU-irina-medium.onnx",
            "tokens": "vits-piper-ru_RU-irina-medium/tokens.txt",
            "data_dir": "vits-piper-ru_RU-irina-medium/espeak-ng-data",
        },
        "speakers": {
            0: {
                "name": "Irina (Ирина)",
                "gender": "female",
                "accent": "russian",
                "description": "Premium Russian female voice",
            }
        },
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-ru_RU-irina-medium.tar.bz2",
    },
}
