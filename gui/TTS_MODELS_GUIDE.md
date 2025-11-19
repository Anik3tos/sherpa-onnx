# TTS GUI - Model Download Guide

## Overview
This guide explains how to download and set up TTS models for the enhanced TTS GUI.

## Quick Start - Recommended Models

### Best Overall (English)
**Amy - High Quality Female Voice** ⭐RECOMMENDED⭐
```bash
# Download
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-medium.tar.bz2

# Extract
tar -xvf vits-piper-en_US-amy-medium.tar.bz2
```

**Ryan - High Quality Male Voice**
```bash
# Download
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-ryan-high.tar.bz2

# Extract
tar -xvf vits-piper-en_US-ryan-high.tar.bz2
```

### Best Multi-Speaker
**LibriTTS - 904 Diverse Voices** ⭐RECOMMENDED⭐
```bash
# Download
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2

# Extract
tar -xvf vits-piper-en_US-libritts_r-medium.tar.bz2
```

## All Available Models

### American English Voices

#### Single Speaker Models
| Model | Gender | Quality | Download Command |
|-------|--------|---------|------------------|
| Amy | Female | Excellent | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-medium.tar.bz2` |
| Lessac | Female | Excellent | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2` |
| Ryan | Male | Excellent | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-ryan-high.tar.bz2` |
| Danny | Male | Very High | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-danny-low.tar.bz2` |
| Kathleen | Female | Very High | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-kathleen-low.tar.bz2` |

#### Multi-Speaker Models
| Model | Speakers | Quality | Download Command |
|-------|----------|---------|------------------|
| LibriTTS-R Medium | 904 | Excellent | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2` |
| LibriTTS High | 10 | Excellent | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts-high.tar.bz2` |

### British English Voices

| Model | Gender | Speakers | Download Command |
|-------|--------|----------|------------------|
| Alba | Female | 1 | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-alba-medium.tar.bz2` |
| Jenny Dioco | Mixed | 2 | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-jenny_dioco-medium.tar.bz2` |
| VCTK | Mixed | 109 | `wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2` |

### Advanced Models

#### Matcha-TTS (Natural Prosody)
```bash
# Premium female voice with advanced prosody
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar -xvf matcha-icefall-en_US-ljspeech.tar.bz2
```

#### Special Character Voice
```bash
# GLaDOS - AI character voice from Portal
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-glados.tar.bz2
tar -xvf vits-piper-en_US-glados.tar.bz2
```

### Multilingual Models

#### Chinese (中文)
```bash
# Huayan - Premium Chinese female voice
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-zh_CN-huayan-medium.tar.bz2
tar -xvf vits-piper-zh_CN-huayan-medium.tar.bz2
```

#### German (Deutsch)
```bash
# Thorsten - High quality German male voice
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-thorsten-high.tar.bz2
tar -xvf vits-piper-de_DE-thorsten-high.tar.bz2
```

#### French (Français)
```bash
# Siwis - Premium French female voice
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fr_FR-siwis-medium.tar.bz2
tar -xvf vits-piper-fr_FR-siwis-medium.tar.bz2
```

#### Spanish (Español)
```bash
# Carlfm - Natural Spanish male voice
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-es_ES-carlfm-x_low.tar.bz2
tar -xvf vits-piper-es_ES-carlfm-x_low.tar.bz2
```

#### Russian (Русский)
```bash
# Irina - Premium Russian female voice
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-ru_RU-irina-medium.tar.bz2
tar -xvf vits-piper-ru_RU-irina-medium.tar.bz2
```

#### Japanese (日本語)
```bash
# Hiroshiba - Natural Japanese female voice
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-ja_JP-hiroshiba-medium.tar.bz2
tar -xvf vits-piper-ja_JP-hiroshiba-medium.tar.bz2
```

## Installation Instructions

### Windows (PowerShell)
```powershell
# Create models directory
New-Item -ItemType Directory -Force -Path "models"
cd models

# Download a model (example: Amy)
Invoke-WebRequest -Uri "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-medium.tar.bz2" -OutFile "amy.tar.bz2"

# Extract using 7-Zip or tar
tar -xvf amy.tar.bz2

# Return to gui directory
cd ..
```

### Linux/macOS
```bash
# Create models directory
mkdir -p models
cd models

# Download a model (example: Amy)
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-medium.tar.bz2

# Extract
tar -xvf vits-piper-en_US-amy-medium.tar.bz2

# Return to gui directory
cd ..
```

## Folder Structure

After downloading and extracting, your folder structure should look like:
```
sherpa-onnx/
├── gui/
│   ├── tts_gui.py
│   └── TTS_MODELS_GUIDE.md (this file)
├── vits-piper-en_US-amy-medium/
│   ├── en_US-amy-medium.onnx
│   ├── tokens.txt
│   └── espeak-ng-data/
├── vits-piper-en_US-ryan-high/
│   ├── en_US-ryan-high.onnx
│   ├── tokens.txt
│   └── espeak-ng-data/
└── vits-piper-en_US-libritts_r-medium/
    ├── en_US-libritts_r-medium.onnx
    ├── tokens.txt
    └── espeak-ng-data/
```

## Model Selection Guide

### For Narration & Audiobooks
- **Best Female**: Amy or Lessac
- **Best Male**: Ryan
- **Natural Prosody**: Matcha-TTS LJSpeech

### For Multiple Characters
- **LibriTTS-R Medium** (904 diverse voices)
- **VCTK** (109 British/International voices)

### For Fast Generation
- **Danny** (male, optimized for speed)
- **Kathleen** (female, optimized for speed)

### For Non-English Content
- Choose the appropriate language-specific model from the multilingual section

## Troubleshooting

### Model Not Detected
1. Ensure the model folder is in the correct location (same directory as tts_gui.py or one level up)
2. Check that all required files exist:
   - `.onnx` model file
   - `tokens.txt`
   - `espeak-ng-data/` directory

### espeak-ng-data Missing
Some models require language-specific espeak-ng-data files. If you get errors:
```bash
# Download complete espeak-ng-data
wget https://github.com/rhasspy/piper/releases/download/v0.0.2/espeak-ng-data.tar.gz
tar -xzf espeak-ng-data.tar.gz
```

### Model Loading Errors
- Make sure you have enough RAM (models range from 50MB to 500MB)
- Close other applications if needed
- Try a smaller model (e.g., "low" quality versions)

## Model Quality Levels

- **Excellent**: Best quality, larger files, slower generation
- **Very High**: Great quality, balanced performance
- **High**: Good quality, faster generation
- **Medium**: Decent quality, fast generation
- **Low**: Basic quality, fastest generation

## Performance Tips

1. **Preloading**: The GUI automatically preloads up to 2 models for faster generation
2. **Caching**: Frequently used text is cached for instant replay
3. **Model Choice**: 
   - Use "low" or "medium" quality for real-time applications
   - Use "high" or "excellent" for final production

## Additional Resources

- **Full Model List**: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
- **Documentation**: https://k2-fsa.github.io/sherpa/onnx/tts/index.html
- **Piper TTS**: https://github.com/rhasspy/piper

## Notes About Kokoro Models

Kokoro models are currently disabled in this GUI due to stability issues:
- They require complex multi-lingual lexicon configurations
- Can cause crashes without proper dictionary files
- Use VITS Piper models instead for stable, high-quality TTS

## Updates and New Models

New models are regularly added to the sherpa-onnx releases. Check:
https://github.com/k2-fsa/sherpa-onnx/releases

To request support for additional models, open an issue on the sherpa-onnx GitHub repository.

---

**Created for sherpa-onnx TTS GUI**
Last Updated: November 2025
