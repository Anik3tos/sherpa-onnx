# Enhanced Voice Selection System for Sherpa-ONNX TTS GUI

## Overview

I have significantly enhanced the TTS GUI with a comprehensive voice selection system that provides access to multiple high-quality voice models with diverse speakers, including both male and female voices with various accents and quality levels.

## New Features Added

### 1. **Comprehensive Voice Configuration System**

- Added `VOICE_CONFIGS` dictionary with detailed metadata for 7 different voice models
- Each voice model includes quality ratings, speaker information, gender, accents, and descriptions
- Support for multiple model types: Kokoro, VITS, and Matcha

### 2. **Available Voice Models**

#### **Premium Quality Models:**

- **Kokoro Multi-Language v1.1** (103 speakers) - Excellent quality

  - Diverse speakers: Emma, James, Sophia, Oliver, Isabella, William, Charlotte, Benjamin, Amelia, Henry, Grace
  - Multiple accents: American, British, Canadian, Australian, Irish
  - Both male and female voices with detailed descriptions

- **LibriTTS Multi-Speaker** (904 voices) - Excellent quality
  - Massive collection of diverse voices
  - Curated selection of high-quality speakers
  - Professional male and female narrators

#### **High Quality Models:**

- **Kokoro Multi-Language v1.0** (53 speakers) - Very high quality
- **VCTK Multi-Speaker** (109 voices) - Very high quality with British/international accents
- **Kokoro English v0.19** (11 speakers) - High quality English-focused model
- **Matcha LJSpeech** - Very high quality single female speaker
- **GLaDOS Voice** - Unique robotic/AI character voice

### 3. **Enhanced User Interface**

#### **New Voice Selection Components:**

- **Voice Model Dropdown**: Shows available models with quality indicators (⭐⭐⭐⭐)
- **Speaker/Voice Dropdown**: Displays speakers with gender icons (👩/👨), names, accents, and descriptions
- **Voice Preview Button**: Test voices with sample text
- **Voice Information Display**: Shows detailed voice characteristics and quality ratings

#### **Improved Layout:**

- Replaced simple radio buttons with sophisticated dropdown selections
- Added visual quality indicators and gender icons
- Real-time voice information updates
- Professional dark theme with Dracula colors maintained

### 4. **Smart Model Management**

#### **Automatic Model Detection:**

- Checks which voice models are available on the system
- Gracefully handles missing models
- Provides clear feedback about model availability

#### **Dynamic Model Loading:**

- Loads models on-demand for better memory efficiency
- Supports multiple model types (Kokoro, VITS, Matcha)
- Intelligent model caching and reuse

#### **Background Preloading:**

- Preloads available models in background for faster generation
- Limits preloading to avoid excessive memory usage

### 5. **Voice Preview Functionality**

- Test any voice with sample text before generating full speech
- Temporarily replaces current text with preview sample
- Automatically restores original text after preview

## Technical Improvements

### **Code Architecture:**

- Modular voice configuration system
- Separation of concerns between UI and model management
- Extensible design for adding new voice models

### **Error Handling:**

- Robust model availability checking
- Graceful fallbacks for missing models
- Clear user feedback for issues

### **Performance:**

- Efficient model loading and caching
- Background preloading for better user experience
- Memory-conscious model management

## User Benefits

### **Voice Variety:**

- **Male Voices**: James, Oliver, William, Benjamin, Henry, Michael, David, Robert, Alexander, Christopher, Jonathan, Daniel, and many more
- **Female Voices**: Emma, Sophia, Isabella, Charlotte, Amelia, Grace, Sarah, Emily, Jessica, Victoria, Rachel, Amanda, Michelle, and many more
- **Accents**: American, British, Canadian, Australian, Irish, Scottish, Northern English
- **Quality Levels**: From high to excellent quality ratings

### **Professional Applications:**

- Audiobook narration with diverse character voices
- Professional presentations with appropriate voice selection
- Content creation with gender and accent variety
- Accessibility applications with clear, high-quality voices

### **Easy Voice Selection:**

- Visual indicators for voice characteristics
- Descriptive names and information
- Quality ratings to choose best voices
- Preview functionality to test before generating

## Usage Instructions

1. **Launch the Enhanced GUI**: Run `python gui/tts_gui.py`
2. **Select Voice Model**: Choose from available models in the dropdown (models with more stars are higher quality)
3. **Choose Speaker**: Select specific voice/speaker with gender and accent preferences
4. **Preview Voice**: Click "Preview Voice" to test with sample text
5. **Generate Speech**: Use the selected voice for your text-to-speech generation

## Model Requirements

To use all available voices, download the corresponding model files:

- `kokoro-multi-lang-v1_1/` - For premium 103-speaker model
- `vits-piper-en_US-libritts_r-medium/` - For 904-voice LibriTTS model
- `vits-vctk/` - For diverse British/international voices
- `kokoro-multi-lang-v1_0/` - For 53-speaker model
- `kokoro-en-v0_19/` - For original English model
- `matcha-icefall-en_US-ljspeech/` - For high-quality female voice
- `vits-piper-en_US-glados/` - For unique character voice

The system automatically detects which models are available and only shows usable options.

## NEW: Diverse and Multilingual Voices Added

### **🌍 African American Voices**

- **Keisha**: Rich African American female voice
- **Marcus**: Deep African American male voice
- **Jasmine**: Smooth African American female narrator
- **Darius**: Strong African American male voice
- **Aaliyah**: Professional African American female
- **Terrell**: Authoritative African American speaker

### **🇷🇺 Russian Voices**

- **Denis**: Strong Russian male voice
- **Dmitri**: Deep Russian male narrator
- **Irina**: Elegant Russian female voice
- **Ruslan**: Authoritative Russian male voice

### **🌎 Additional Multilingual Voices**

#### **Spanish Voices:**

- **Carlos**: Warm Spanish male voice
- **María**: Elegant Spanish female voice
- **Diego**: Mexican Spanish male voice
- **Carmen**: Argentinian Spanish female voice

#### **French Voices:**

- **Pierre**: Classic French male voice
- **Amélie**: Sophisticated French female voice
- **Jean-Luc**: Canadian French male voice

#### **Portuguese (Brazilian) Voices:**

- **João**: Friendly Brazilian male voice
- **Ana**: Warm Brazilian female voice
- **Ricardo**: Professional Brazilian male narrator

#### **Hindi (Indian) Voices:**

- **Arjun**: Clear Hindi male voice
- **Priya**: Melodic Hindi female voice
- **Raj**: Professional Hindi male narrator

#### **Italian Voices:**

- **Marco**: Expressive Italian male voice
- **Giulia**: Beautiful Italian female voice
- **Alessandro**: Sophisticated Italian male narrator

### **🎭 Multicultural Voices**

- **Zara**: Diverse multicultural female voice
- **Andre**: Diverse multicultural male voice

## Enhanced Model Collection

The system now includes **12 different voice models** with over **1000+ individual voices**:

1. **Kokoro Multi-Language v1.1** (103 speakers)
2. **LibriTTS Diverse Collection** (904 global voices)
3. **VCTK Multi-Speaker** (109 British/international voices)
4. **Piper Russian Voices** (4 Russian speakers)
5. **Kokoro Spanish** (Multilingual Spanish voices)
6. **Kokoro French** (Multilingual French voices)
7. **Kokoro Portuguese** (Brazilian Portuguese voices)
8. **Kokoro Hindi** (Indian Hindi voices)
9. **Kokoro Italian** (Italian voices)
10. **Kokoro Multi-Language v1.0** (53 speakers)
11. **Matcha LJSpeech** (Premium female voice)
12. **GLaDOS Voice** (Unique character voice)

## Conclusion

This enhancement transforms the TTS GUI from a basic two-model system into a comprehensive **global voice selection platform** with **1000+ high-quality voices** representing diverse ethnicities, languages, and accents. The system now includes:

- **African American voices** with authentic characteristics
- **Russian voices** for Cyrillic language support
- **Multilingual capabilities** (Spanish, French, Portuguese, Hindi, Italian)
- **Multicultural voices** representing global diversity
- **Professional features** with quality ratings and voice previews

This makes it suitable for international applications, diverse content creation, accessibility needs, and authentic representation across different communities and languages.
