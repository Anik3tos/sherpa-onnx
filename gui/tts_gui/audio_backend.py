#!/usr/bin/env python3
"""
Optional audio playback backend for the TTS GUI.
"""

try:
    import pygame as _pygame

    pygame = _pygame
    PYGAME_AVAILABLE = True
    PYGAME_IMPORT_ERROR = ""
except Exception as exc:
    pygame = None
    PYGAME_AVAILABLE = False
    PYGAME_IMPORT_ERROR = str(exc)

try:
    from PySide6.QtCore import QUrl as _QUrl
    from PySide6.QtMultimedia import (
        QAudioOutput as _QAudioOutput,
        QMediaPlayer as _QMediaPlayer,
    )

    QUrl = _QUrl
    QAudioOutput = _QAudioOutput
    QMediaPlayer = _QMediaPlayer
    QT_AUDIO_AVAILABLE = True
    QT_AUDIO_IMPORT_ERROR = ""
except Exception as exc:
    QUrl = None
    QAudioOutput = None
    QMediaPlayer = None
    QT_AUDIO_AVAILABLE = False
    QT_AUDIO_IMPORT_ERROR = str(exc)
