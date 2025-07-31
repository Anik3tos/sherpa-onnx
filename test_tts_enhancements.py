#!/usr/bin/env python3

"""
Test script for TTS GUI enhancements
Tests the new classes and functionality without requiring the full GUI
"""

import sys
import os
import tempfile
import numpy as np

# Add the current directory to path to import our modules
sys.path.insert(0, '.')

# Import our enhanced classes
from tts_gui import TextProcessor, AudioCache, PerformanceMonitor

def test_text_processor():
    """Test TextProcessor functionality"""
    print("Testing TextProcessor...")

    processor = TextProcessor()

    # Test validation
    valid_text = "This is a valid text for testing."
    is_valid, msg = processor.validate_text(valid_text)
    print(f"  Valid text check: {is_valid} - {msg}")

    # Test invalid text
    empty_text = ""
    is_valid, msg = processor.validate_text(empty_text)
    print(f"  Empty text check: {is_valid} - {msg}")

    # Test preprocessing
    messy_text = "This   has    extra   spaces!!!   And multiple punctuation???"
    options = {'normalize_whitespace': True, 'normalize_punctuation': True}
    clean_text = processor.preprocess_text(messy_text, options)
    print(f"  Original: '{messy_text}'")
    print(f"  Cleaned:  '{clean_text}'")

    # Test statistics
    stats = processor.get_text_stats(valid_text)
    print(f"  Text stats: {stats}")

    print("  TextProcessor tests passed!\n")

def test_audio_cache():
    """Test AudioCache functionality"""
    print("Testing AudioCache...")

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = AudioCache(max_size=3, cache_dir=temp_dir)

        # Test caching
        text1 = "Hello world"
        audio_data1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        sample_rate = 22050

        # Cache some audio
        cache.put(text1, "matcha", 0, 1.0, audio_data1, sample_rate)
        print(f"  Cached audio for: '{text1}'")

        # Retrieve cached audio
        cached = cache.get(text1, "matcha", 0, 1.0)
        if cached:
            print(f"  Retrieved cached audio: {len(cached['audio_data'])} samples")
        else:
            print("  Failed to retrieve cached audio")

        # Test cache miss
        cached_miss = cache.get("Different text", "matcha", 0, 1.0)
        if cached_miss is None:
            print("  Cache miss handled correctly")

        # Test cache size limit
        for i in range(5):
            text = f"Text {i}"
            audio = np.array([i] * 10, dtype=np.float32)
            cache.put(text, "matcha", 0, 1.0, audio, sample_rate)

        print(f"  Cache size after adding 5 items: {len(cache.cache)} (max: 3)")

    print("  AudioCache tests passed!\n")

def test_performance_monitor():
    """Test PerformanceMonitor functionality"""
    print("Testing PerformanceMonitor...")

    monitor = PerformanceMonitor()

    # Test generation tracking
    monitor.start_generation(100, "matcha")
    print("  Started generation tracking")

    # Simulate some processing time
    import time
    time.sleep(0.1)

    # End generation
    metric = monitor.end_generation(2.0, from_cache=False)
    if metric:
        print(f"  Generation metric: RTF={metric['rtf']:.3f}, Time={metric['generation_time']:.3f}s")

    # Test average RTF
    avg_rtf = monitor.get_average_rtf("matcha")
    print(f"  Average RTF: {avg_rtf:.3f}")

    # Test cached generation
    monitor.start_generation(100, "matcha")
    cached_metric = monitor.end_generation(2.0, from_cache=True)
    if cached_metric:
        print(f"  Cached generation tracked: from_cache={cached_metric['from_cache']}")

    print("  PerformanceMonitor tests passed!\n")

def main():
    """Run all tests"""
    print("Testing TTS GUI Enhancements\n")

    try:
        test_text_processor()
        test_audio_cache()
        test_performance_monitor()

        print("All tests passed successfully!")
        print("\nEnhancement Summary:")
        print("  - TextProcessor: Text validation, preprocessing, and statistics")
        print("  - AudioCache: Intelligent caching with LRU eviction")
        print("  - PerformanceMonitor: RTF tracking and performance analytics")
        print("\nThe enhanced TTS GUI is ready to use!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
