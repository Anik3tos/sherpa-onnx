#!/usr/bin/env python3

"""
Demonstration of TTS GUI Improvements
Shows before/after comparison of key features
"""

import time
import numpy as np
from tts_gui import TextProcessor, AudioCache, PerformanceMonitor

def demo_text_processing():
    """Demonstrate text processing improvements"""
    print("=" * 60)
    print("TEXT PROCESSING IMPROVEMENTS DEMO")
    print("=" * 60)
    
    processor = TextProcessor()
    
    # Example of messy input text
    messy_text = """
    This   is    some   really   messy    text!!!
    
    It has   extra   spaces,,,   weird punctuation???
    
    And  even  some  URLs like https://example.com
    Plus email@example.com addresses...
    """
    
    print("BEFORE (Raw Input):")
    print(repr(messy_text))
    print()
    
    # Show text statistics
    stats = processor.get_text_stats(messy_text)
    print(f"Raw Text Stats: {stats}")
    print()
    
    # Process with different options
    options = {
        'normalize_whitespace': True,
        'normalize_punctuation': True,
        'remove_urls': True,
        'remove_word_dashes': True
    }
    
    clean_text = processor.preprocess_text(messy_text, options)
    clean_stats = processor.get_text_stats(clean_text)
    
    print("AFTER (Processed):")
    print(repr(clean_text))
    print()
    print(f"Clean Text Stats: {clean_stats}")
    print()
    
    # Validation
    is_valid, msg = processor.validate_text(clean_text)
    print(f"Validation: {'PASS' if is_valid else 'FAIL'} - {msg}")
    print()

def demo_audio_caching():
    """Demonstrate audio caching improvements"""
    print("=" * 60)
    print("AUDIO CACHING IMPROVEMENTS DEMO")
    print("=" * 60)
    
    cache = AudioCache(max_size=5)
    
    # Simulate generating audio for different texts
    texts = [
        "Hello world",
        "This is a test",
        "Welcome to the TTS system",
        "Hello world",  # Duplicate - should hit cache
        "Another test sentence"
    ]
    
    print("Simulating audio generation with caching:")
    print()
    
    for i, text in enumerate(texts, 1):
        print(f"Generation {i}: '{text}'")
        
        # Check if already cached
        cached = cache.get(text, "matcha", 0, 1.0)
        if cached:
            print("  -> CACHE HIT! Instant generation (0ms)")
        else:
            # Simulate generation time
            start_time = time.time()
            time.sleep(0.1)  # Simulate 100ms generation
            generation_time = time.time() - start_time
            
            # Create fake audio data
            audio_data = np.random.random(1000).astype(np.float32)
            sample_rate = 22050
            
            # Cache the result
            cache.put(text, "matcha", 0, 1.0, audio_data, sample_rate)
            
            print(f"  -> Generated in {generation_time*1000:.0f}ms, cached for future use")
        
        print(f"  -> Cache size: {len(cache.cache)} items")
        print()
    
    print(f"Final cache statistics:")
    print(f"  - Total cached items: {len(cache.cache)}")
    print(f"  - Cache hit rate: 20% (1 out of 5 requests)")
    print(f"  - Time saved: ~100ms on cache hits")
    print()

def demo_performance_monitoring():
    """Demonstrate performance monitoring improvements"""
    print("=" * 60)
    print("PERFORMANCE MONITORING IMPROVEMENTS DEMO")
    print("=" * 60)
    
    monitor = PerformanceMonitor()
    
    # Simulate different generation scenarios
    scenarios = [
        ("Short text", 50, 1.2, False),
        ("Medium text", 150, 3.5, False),
        ("Long text", 300, 7.8, False),
        ("Cached short", 50, 1.2, True),  # Same as first, from cache
        ("Another medium", 140, 3.2, False),
    ]
    
    print("Simulating TTS generations with performance tracking:")
    print()
    
    for desc, text_len, audio_duration, from_cache in scenarios:
        print(f"Scenario: {desc}")
        
        if not from_cache:
            # Simulate generation
            monitor.start_generation(text_len, "matcha")
            time.sleep(0.05)  # Simulate processing time
            metric = monitor.end_generation(audio_duration, from_cache=False)
            
            print(f"  - Text length: {text_len} chars")
            print(f"  - Audio duration: {audio_duration:.1f}s")
            print(f"  - Generation time: {metric['generation_time']:.3f}s")
            print(f"  - RTF: {metric['rtf']:.3f}")
        else:
            # Simulate cache hit
            metric = monitor.end_generation(audio_duration, from_cache=True)
            print(f"  - CACHED RESULT (instant)")
            print(f"  - Audio duration: {audio_duration:.1f}s")
            print(f"  - Generation time: 0.000s")
            print(f"  - RTF: 0.000 (cached)")
        
        print()
    
    # Show overall statistics
    avg_rtf = monitor.get_average_rtf("matcha")
    print(f"Performance Summary:")
    print(f"  - Total generations tracked: {len(monitor.metrics)}")
    print(f"  - Average RTF (non-cached): {avg_rtf:.3f}")
    print(f"  - Cache hits: 1 out of 5 (20%)")
    print()

def main():
    """Run all demonstrations"""
    print("TTS GUI ENHANCEMENTS DEMONSTRATION")
    print("Showing before/after improvements in key areas")
    print()
    
    demo_text_processing()
    demo_audio_caching()
    demo_performance_monitoring()
    
    print("=" * 60)
    print("SUMMARY OF IMPROVEMENTS")
    print("=" * 60)
    print()
    print("1. TEXT INPUT USABILITY:")
    print("   - Real-time text validation and statistics")
    print("   - Intelligent text preprocessing options")
    print("   - Import/export functionality")
    print("   - Enhanced user feedback")
    print()
    print("2. SPEECH GENERATION PERFORMANCE:")
    print("   - Model preloading for faster startup")
    print("   - Intelligent audio caching (instant replay)")
    print("   - Performance monitoring and analytics")
    print("   - Memory optimization and resource management")
    print()
    print("3. USER EXPERIENCE:")
    print("   - Professional UI with enhanced controls")
    print("   - Real-time feedback and progress indication")
    print("   - Better error handling and recovery")
    print("   - Comprehensive logging and status updates")
    print()
    print("The enhanced TTS GUI provides a significantly improved")
    print("experience for both casual users and power users!")

if __name__ == "__main__":
    main()
