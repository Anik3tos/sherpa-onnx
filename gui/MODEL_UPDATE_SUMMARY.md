# TTS GUI Model Update Summary

## Changes Made

### 1. Updated Voice Model Configurations

The GUI has been updated with modern, stable TTS models from the sherpa-onnx project.

### 2. Removed Problematic Kokoro Models

**Why?** Kokoro models were causing stability issues:
- Required complex multi-lingual lexicon configurations
- Missing dictionary files caused crashes
- Inconsistent file path requirements

**Replaced with:** Stable VITS Piper models that work out-of-the-box

### 3. New Models Added

#### American English (Recommended ⭐)
- **Amy** - Crystal clear female voice (medium quality)
- **Lessac** - Premium female voice (medium quality)
- **Ryan** - Natural male voice (high quality)
- **Danny** - Fast male voice (low quality, optimized for speed)
- **Kathleen** - Fast female voice (low quality, optimized for speed)
- **LibriTTS-R** - 904 diverse multi-speaker voices (medium quality)
- **LibriTTS High** - 10 premium voices (high quality)

#### British English
- **Alba** - Natural British female (medium quality)
- **Jenny Dioco** - 2 British voices (medium quality)
- **VCTK** - 109 diverse British/international voices

#### Advanced Models
- **Matcha-TTS LJSpeech** - State-of-the-art with natural prosody
- **GLaDOS** - AI character voice (from Portal game)

#### Multilingual Support
- **Chinese (中文)** - Huayan (female)
- **German (Deutsch)** - Thorsten (male)
- **French (Français)** - Siwis (female)
- **Spanish (Español)** - Carlfm (male)
- **Russian (Русский)** - Irina (female)
- **Japanese (日本語)** - Hiroshiba (female)

### 4. Fixed Model Path Issues

- Updated to use correct file naming conventions (e.g., `en_US-amy-medium.onnx`)
- Fixed espeak-ng-data directory references
- Corrected vocoder paths for Matcha-TTS models

### 5. Added Download URLs

Each model configuration now includes a `download_url` field pointing to the official GitHub releases.

## How to Use

### Quick Start

1. **Download recommended model:**
   ```bash
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-medium.tar.bz2
   tar -xvf vits-piper-en_US-amy-medium.tar.bz2
   ```

2. **Run the GUI:**
   ```bash
   python tts_gui.py
   # or on Windows:
   .\start_tts_gui.bat
   ```

3. **Select the model** from the dropdown menu

### Full Instructions

See `TTS_MODELS_GUIDE.md` for:
- Complete model list with download commands
- Installation instructions for Windows/Linux/macOS
- Model selection recommendations
- Troubleshooting tips

## Model Recommendations

### For Best Quality
- **Female**: Amy or Lessac
- **Male**: Ryan
- **Advanced**: Matcha-TTS LJSpeech

### For Multiple Voices
- **LibriTTS-R Medium** (904 voices)
- **VCTK** (109 voices)

### For Fast Generation
- **Danny** (male)
- **Kathleen** (female)

## Breaking Changes

### Removed Models
The following models have been **removed** due to stability issues:
- `kokoro_multi_v1_1`
- `kokoro_multi_v1_0`
- `kokoro_multi_enhanced`
- `kokoro_en_v0_19`
- Old Russian models (denis, dmitri, ruslan - replaced with piper versions)

### If You Were Using Kokoro Models
Please switch to:
- **For American voices**: Amy, Ryan, or LibriTTS
- **For British voices**: Alba, Jenny Dioco, or VCTK
- **For multi-speaker**: LibriTTS-R (904 voices) or VCTK (109 voices)

## Model File Locations

Models should be extracted in the **same directory** as `tts_gui.py` or one level up.

Example structure:
```
sherpa-onnx/
├── gui/
│   └── tts_gui.py
├── vits-piper-en_US-amy-medium/
├── vits-piper-en_US-ryan-high/
└── vits-piper-en_US-libritts_r-medium/
```

## Compatibility

✅ **All models use VITS architecture** (except Matcha-TTS)
✅ **Compatible with sherpa-onnx** latest version
✅ **Cross-platform**: Windows, Linux, macOS
✅ **No external dependencies** required (all included)

## Performance

### Model Loading
- First model load: ~2-5 seconds
- Subsequent models: ~1-2 seconds (if preloaded)
- Audio generation: Real-time Factor (RTF) typically 0.1-0.5 (faster than real-time)

### Memory Usage
- Single model: ~50-500 MB depending on quality
- GUI can preload up to 2 models simultaneously
- Audio cache: ~50 items max

## Troubleshooting

### "Model not found" error
1. Download the model using wget/curl
2. Extract to correct location
3. Restart GUI

### "espeak-ng-data" error
Download complete espeak-ng-data:
```bash
wget https://github.com/rhasspy/piper/releases/download/v0.0.2/espeak-ng-data.tar.gz
tar -xzf espeak-ng-data.tar.gz
```

### Model loads but no audio
1. Check pygame mixer initialization
2. Verify sample rate (should be 22050 Hz)
3. Try different speaker ID

## Additional Resources

- **Model Downloads**: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
- **Documentation**: https://k2-fsa.github.io/sherpa/onnx/tts/index.html
- **Piper TTS Project**: https://github.com/rhasspy/piper
- **sherpa-onnx GitHub**: https://github.com/k2-fsa/sherpa-onnx

## Future Updates

Planned improvements:
- Auto-download models from GUI
- Model quality selector
- Voice cloning support (if available)
- Additional language support

---

**Version**: 2.0 (Model Configuration Update)
**Date**: November 19, 2025
**Compatibility**: sherpa-onnx v1.10+
