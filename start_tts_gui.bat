@echo off
cd /d "%~dp0"

echo Installing required dependencies...
pip install sherpa-onnx soundfile numpy PySide6 -q

python -c "import sys; raise SystemExit(0 if sys.version_info < (3,13) else 1)"
if %ERRORLEVEL% EQU 0 (
  pip install pygame -q
) else (
  echo Skipping pygame install on Python 3.13+; playback will be disabled.
)
echo.
echo Starting High-Quality English TTS GUI...
echo.
echo Make sure you have the following models downloaded:
echo - matcha-icefall-en_US-ljspeech (for high-quality single speaker)
echo - vits-piper-en_US-amy-medium or another VITS Piper voice
echo - ASR model for transcription is auto-downloaded on first use
echo.
python gui\tts_gui.py
pause
