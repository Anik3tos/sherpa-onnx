# High-Quality English TTS GUI

A user-friendly graphical interface for the sherpa-onnx text-to-speech system, featuring high-quality English models.

## Features

- **Two Premium English Models:**

  - **Matcha-TTS LJSpeech**: Ultra-high quality single female speaker (American English)
  - **Kokoro English**: Multiple speakers (11 different voices) with excellent quality

- **Easy-to-Use Interface:**

  - Simple text input with sample content
  - Model selection (Matcha-TTS or Kokoro)
  - Speaker selection for Kokoro model (0-10)
  - Speed control (0.5x to 2.0x)
  - Real-time audio generation and playback
  - Save audio to file functionality

- **Real-Time Features:**
  - Live status updates during generation
  - Performance metrics (RTF - Real-Time Factor)
  - Integrated audio player with play/stop controls

## Requirements

- Python 3.7+
- sherpa-onnx
- pygame (for audio playback)
- soundfile
- tkinter (usually included with Python)

## Installation

1. **Install dependencies:**

   ```bash
   pip install sherpa-onnx pygame soundfile
   ```

2. **Download the models (already done if you followed the previous steps):**

   **Matcha-TTS LJSpeech (High Quality):**

   ```bash
   curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
   tar xf matcha-icefall-en_US-ljspeech.tar.bz2
   curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx
   ```

   **Kokoro English (Multiple Speakers):**

   ```bash
   curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
   tar xf kokoro-en-v0_19.tar.bz2
   ```

## Usage

### Option 1: Python Script

```bash
python tts_gui.py
```

### Option 2: Windows Batch File

```bash
start_tts_gui.bat
```

## Model Information

### Matcha-TTS LJSpeech

- **Quality:** Ultra-high (premium quality)
- **Speakers:** 1 (female, American English)
- **Size:** ~73MB + ~51MB vocoder
- **Speed:** Very fast (RTF ~0.03-0.05)
- **Best for:** Highest quality single-voice applications

### Kokoro English v0.19

- **Quality:** High
- **Speakers:** 11 different voices (IDs 0-10)
- **Size:** ~304MB
- **Speed:** Fast (RTF ~0.35-0.40)
- **Best for:** Applications needing voice variety

## Performance Metrics

**RTF (Real-Time Factor):** Lower is better

- RTF < 1.0: Faster than real-time (can generate audio faster than playback)
- RTF = 1.0: Real-time generation
- RTF > 1.0: Slower than real-time

Both models achieve excellent RTF values:

- Matcha-TTS: ~0.03-0.05 RTF (30-50x faster than real-time!)
- Kokoro: ~0.35-0.40 RTF (2.5-3x faster than real-time)

## GUI Features

1. **Model Selection:** Choose between Matcha-TTS and Kokoro
2. **Speaker Selection:** For Kokoro model, select speaker ID (0-10)
3. **Speed Control:** Adjust speech speed from 0.5x to 2.0x
4. **Text Input:** Large text area with sample content
5. **Generate:** Create audio from text
6. **Play/Stop:** Built-in audio player
7. **Save:** Export audio to WAV file
8. **Status Log:** Real-time feedback and performance metrics

## Tips

- **For highest quality:** Use Matcha-TTS LJSpeech
- **For voice variety:** Use Kokoro with different speaker IDs
- **For faster speech:** Increase speed slider above 1.0x
- **For slower, clearer speech:** Decrease speed slider below 1.0x

## File Structure

```
sherpa-onnx/
├── tts_gui.py                     # Main GUI application
├── start_tts_gui.bat             # Windows launcher
├── matcha-icefall-en_US-ljspeech/ # Matcha-TTS model
├── kokoro-en-v0_19/              # Kokoro English model
├── vocos-22khz-univ.onnx         # Vocoder for Matcha-TTS
└── *.wav                         # Generated audio files
```

## Troubleshooting

1. **"No models available" error:** Make sure models are downloaded and extracted in the correct directories
2. **Audio playback issues:** Ensure pygame is properly installed
3. **Model loading errors:** Check that all model files are present and not corrupted
4. **Performance issues:** Try reducing the number of threads or using a different model

Enjoy your high-quality English text-to-speech experience!
