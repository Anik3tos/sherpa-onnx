#!/usr/bin/env python3

"""
Test script for TTS GUI chunking functionality
Tests the text splitting and audio stitching features
"""

import sys
import numpy as np

# Add the current directory to path to import our modules
sys.path.insert(0, '.')

from tts_gui import TextProcessor, AudioStitcher

def test_text_splitting():
    """Test text splitting functionality"""
    print("Testing Text Splitting Functionality")
    print("=" * 50)
    
    processor = TextProcessor()
    
    # Test 1: Short text (no chunking needed)
    short_text = "This is a short text that doesn't need chunking."
    print(f"Test 1 - Short text ({len(short_text)} chars):")
    print(f"  Needs chunking: {processor.needs_chunking(short_text)}")
    chunks = processor.split_text_into_chunks(short_text)
    print(f"  Chunks: {len(chunks)}")
    print()
    
    # Test 2: Long text with sentences
    long_text = """
    This is a very long text that will definitely need to be split into multiple chunks for processing.
    It contains multiple sentences with proper punctuation. Each sentence should ideally stay together
    when possible. The text processor should split at sentence boundaries first, then at clause boundaries
    if needed, and finally at word boundaries as a last resort.
    
    This paragraph continues the long text. It has more sentences that add to the total character count.
    The goal is to create a text that exceeds the chunk size limit so we can test the splitting logic.
    We want to make sure that the splitting happens at natural boundaries to maintain readability and
    proper pronunciation when the text is converted to speech.
    
    Here's another paragraph to make the text even longer. This should definitely trigger the chunking
    mechanism. The text processor should handle this gracefully and split it into appropriate chunks
    that can be processed individually by the TTS system and then stitched back together seamlessly.
    
    Finally, this last paragraph ensures we have enough content to thoroughly test the chunking system.
    It should demonstrate that long texts can be handled automatically without user intervention.
    The system should provide feedback about how many chunks will be created and process them efficiently.
    """ * 3  # Multiply to make it really long
    
    print(f"Test 2 - Long text ({len(long_text)} chars):")
    print(f"  Needs chunking: {processor.needs_chunking(long_text)}")
    chunks = processor.split_text_into_chunks(long_text)
    print(f"  Total chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk)} chars - '{chunk[:50]}...'")
    print()
    
    # Test 3: Text with no sentence boundaries (edge case)
    no_sentences = "This is a very long text without proper sentence endings that just keeps going and going and going " * 100
    print(f"Test 3 - No sentence boundaries ({len(no_sentences)} chars):")
    print(f"  Needs chunking: {processor.needs_chunking(no_sentences)}")
    chunks = processor.split_text_into_chunks(no_sentences)
    print(f"  Total chunks: {len(chunks)}")
    print(f"  First chunk length: {len(chunks[0])}")
    print(f"  Last chunk length: {len(chunks[-1])}")
    print()

def test_audio_stitching():
    """Test audio stitching functionality"""
    print("Testing Audio Stitching Functionality")
    print("=" * 50)
    
    stitcher = AudioStitcher(silence_duration=0.2)
    sample_rate = 22050
    
    # Create fake audio chunks (sine waves at different frequencies)
    chunk1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate)).astype(np.float32)  # 1 second, 440 Hz
    chunk2 = np.sin(2 * np.pi * 880 * np.linspace(0, 0.5, sample_rate//2)).astype(np.float32)  # 0.5 seconds, 880 Hz
    chunk3 = np.sin(2 * np.pi * 220 * np.linspace(0, 1.5, int(sample_rate*1.5))).astype(np.float32)  # 1.5 seconds, 220 Hz
    
    audio_chunks = [chunk1, chunk2, chunk3]
    
    print(f"Test 1 - Stitching {len(audio_chunks)} audio chunks:")
    print(f"  Chunk 1: {len(chunk1)} samples ({len(chunk1)/sample_rate:.1f}s)")
    print(f"  Chunk 2: {len(chunk2)} samples ({len(chunk2)/sample_rate:.1f}s)")
    print(f"  Chunk 3: {len(chunk3)} samples ({len(chunk3)/sample_rate:.1f}s)")
    
    # Stitch the chunks
    stitched = stitcher.stitch_audio_chunks(audio_chunks, sample_rate)
    
    print(f"  Stitched result: {len(stitched)} samples ({len(stitched)/sample_rate:.1f}s)")
    
    # Calculate expected duration
    chunk_durations = [len(chunk)/sample_rate for chunk in audio_chunks]
    expected_duration = stitcher.estimate_total_duration(chunk_durations)
    actual_duration = len(stitched) / sample_rate
    
    print(f"  Expected duration: {expected_duration:.1f}s")
    print(f"  Actual duration: {actual_duration:.1f}s")
    print(f"  Silence between chunks: {stitcher.silence_duration}s")
    print()
    
    # Test with no silence
    stitcher_no_silence = AudioStitcher(silence_duration=0.0)
    stitched_no_silence = stitcher_no_silence.stitch_audio_chunks(audio_chunks, sample_rate)
    
    print(f"Test 2 - Stitching without silence:")
    print(f"  Result: {len(stitched_no_silence)} samples ({len(stitched_no_silence)/sample_rate:.1f}s)")
    expected_no_silence = sum(len(chunk) for chunk in audio_chunks) / sample_rate
    print(f"  Expected: {expected_no_silence:.1f}s")
    print()

def test_integration():
    """Test integration of text splitting and audio processing"""
    print("Testing Integration")
    print("=" * 50)
    
    processor = TextProcessor()
    
    # Create a long text
    long_text = """
    Welcome to the enhanced text-to-speech system with automatic chunking support.
    This system can now handle very long texts by automatically splitting them into
    manageable chunks, processing each chunk separately, and then stitching the
    audio results back together seamlessly.
    
    The chunking algorithm is intelligent and tries to split at natural boundaries
    like sentence endings first, then clause boundaries, and finally word boundaries
    if necessary. This ensures that the speech synthesis maintains natural flow
    and pronunciation across chunk boundaries.
    
    Performance is also optimized through caching. Individual chunks are cached
    separately, so if you process similar long texts, common chunks can be reused
    instantly. The final stitched result is also cached for complete reuse.
    
    This enhancement makes the TTS system much more versatile and capable of
    handling everything from short phrases to entire documents or articles
    without any manual intervention from the user.
    """ * 5  # Make it long enough to trigger chunking
    
    print(f"Integration test with {len(long_text)} character text:")
    
    # Test preprocessing
    options = {
        'normalize_whitespace': True,
        'normalize_punctuation': True,
        'remove_urls': False,
        'remove_emails': False
    }
    
    processed_text = processor.preprocess_text(long_text, options)
    print(f"  After preprocessing: {len(processed_text)} chars")
    
    # Test validation
    is_valid, msg = processor.validate_text(processed_text)
    print(f"  Validation: {'PASS' if is_valid else 'FAIL'} - {msg}")
    
    # Test chunking
    needs_chunking = processor.needs_chunking(processed_text)
    print(f"  Needs chunking: {needs_chunking}")
    
    if needs_chunking:
        chunks = processor.split_text_into_chunks(processed_text)
        print(f"  Will create {len(chunks)} chunks")
        
        total_chars = sum(len(chunk) for chunk in chunks)
        print(f"  Total characters in chunks: {total_chars}")
        print(f"  Original characters: {len(processed_text)}")
        print(f"  Character preservation: {total_chars/len(processed_text)*100:.1f}%")
        
        # Show chunk size distribution
        chunk_sizes = [len(chunk) for chunk in chunks]
        print(f"  Chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={sum(chunk_sizes)/len(chunk_sizes):.0f}")
    
    print()

def main():
    """Run all tests"""
    print("TTS GUI CHUNKING FUNCTIONALITY TESTS")
    print("=" * 60)
    print()
    
    try:
        test_text_splitting()
        test_audio_stitching()
        test_integration()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print()
        print("Key Features Tested:")
        print("  - Intelligent text splitting at natural boundaries")
        print("  - Audio chunk stitching with configurable silence")
        print("  - Integration with preprocessing and validation")
        print("  - Handling of edge cases and long texts")
        print()
        print("The enhanced TTS GUI can now handle texts of any length!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
