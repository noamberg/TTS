import torch
from TTS.api import TTS
import zipfile
from bs4 import BeautifulSoup
import os
import tempfile
import re
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import logging
from pathlib import Path
import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('epub_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Step 1: Function to Extract Text from EPUB ---
# An EPUB file is essentially a ZIP archive containing XHTML files.
# This function extracts the content and parses it to get clean text.

def extract_text_from_epub(epub_path):
    """
    Extracts text from an EPUB file by unzipping it and parsing the XHTML content.

    IMPORTANT: Verify the extraction statistics (character and word count) to ensure
    they seem reasonable for the book's size. If the count is unexpectedly low, the EPUB
    may have text stored in different HTML tags that require adjusting the parser logic
    (e.g., text in <div>, <span>, or other tags not included in the default list).
    """
    print(f"Extracting text from {epub_path}...")
    text_content = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Unzip the EPUB file to a temporary directory
        with zipfile.ZipFile(epub_path, 'r') as z:
            z.extractall(tmpdir)

        # Step 1: Collect all content file paths
        content_files = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(('.xhtml', '.html')):
                    filepath = os.path.join(root, file)
                    content_files.append(filepath)

        # Step 2: Sort the collected paths to ensure correct chronological order
        content_files.sort()

        # Step 3: Process files in sorted order
        for filepath in content_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                # Find all text within paragraph and heading tags
                for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    text_content.append(tag.get_text())

    full_text = '\n'.join(text_content)

    # Print extraction stats - IMPORTANT: Review these to verify complete extraction
    char_count = len(full_text)
    word_count = len(full_text.split())
    print(f"Text extraction complete.")
    print(f"  - Characters: {char_count:,}")
    print(f"  - Words: {word_count:,}")
    print(f"  - Estimated reading time: {word_count // 200:.1f} minutes")

    return full_text


# --- Step 2: Split text into manageable chunks ---

def split_into_sentences(text, max_length=250):
    """
    Split text into sentences, ensuring no sentence exceeds max_length characters.
    Uses a multi-level splitting approach to guarantee chunks are within limit.
    """
    # Basic sentence splitting using common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If sentence is too long, split it further
        if len(sentence) > max_length:
            # Split on commas, semicolons, or other natural breaks
            sub_chunks = re.split(r'[,;]\s+', sentence)

            # Secondary validation: ensure each sub_chunk is within limit
            for sub_chunk in sub_chunks:
                sub_chunk = sub_chunk.strip()
                if not sub_chunk:
                    continue

                # If sub_chunk is still too long, perform iterative space-based splitting
                while len(sub_chunk) > max_length:
                    # Find the last space before max_length
                    split_point = sub_chunk.rfind(' ', 0, max_length)

                    if split_point == -1:
                        # No space found, perform hard cut at max_length
                        split_point = max_length

                    # Split and add the first part to chunks
                    chunks.append(sub_chunk[:split_point].strip())
                    # Continue with the remainder
                    sub_chunk = sub_chunk[split_point:].strip()

                # Add the final remaining piece (which is now under max_length)
                if sub_chunk:
                    chunks.append(sub_chunk)
        else:
            chunks.append(sentence)

    return chunks


# --- Step 3: Split text into sections ---

def split_text_into_sections(text, num_sections=20):
    """
    Split text into roughly equal sections based on sentences.
    """
    sentences = split_into_sentences(text)
    sentences_per_section = math.ceil(len(sentences) / num_sections)

    sections = []
    for i in range(0, len(sentences), sentences_per_section):
        section_text = ' '.join(sentences[i:i + sentences_per_section])
        sections.append(section_text)

    return sections


# --- Step 4: Convert a section to audio (for parallel processing) ---

def convert_section_to_audio(args):
    """
    Convert a single section of text to audio using XTTS v2 with voice cloning.
    This function is designed to be called in parallel.

    Args:
        args: Tuple containing (section_index, section_text, output_dir, config_dict)
              config_dict includes: speed_factor, reference_audio, sentence_pause,
              bitrate, language, speaker_wav

    Returns:
        Tuple of (section_output_path, processing_duration) or (None, 0) on failure
    """
    section_index, section_text, output_dir, config = args

    # Record start time for this section
    start_time = time.time()

    try:
        # Check for available GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Section {section_index + 1}: Using device: {device}")

        # Initialize the TTS model - Using XTTS v2 for advanced, natural speech
        # XTTS v2 supports voice cloning and produces highly expressive audio
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

        try:
            tts = TTS(model_name).to(device)
            logger.info(f"Section {section_index + 1}: Loaded XTTS v2 model successfully")
        except Exception as e:
            logger.error(f"Section {section_index + 1}: Failed to load XTTS v2 model: {e}")
            logger.info(f"Section {section_index + 1}: Falling back to VITS model")
            tts = TTS("tts_models/en/ljspeech/vits").to(device)
            model_name = "vits"

        # Split section into sentences
        sentences = split_into_sentences(section_text)

        logger.info(f"Section {section_index + 1}: Processing {len(sentences)} sentences...")
        print(f"Section {section_index + 1}: Processing {len(sentences)} sentences...")

        # Create temporary directory for this section's audio files
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_files = []

            # Process each sentence
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                temp_file = os.path.join(tmpdir, f"chunk_{i:05d}.wav")

                try:
                    # Use XTTS v2 with voice cloning if reference audio is provided
                    if "xtts" in model_name.lower() and config.get("speaker_wav"):
                        if os.path.exists(config["speaker_wav"]):
                            tts.tts_to_file(
                                text=sentence,
                                file_path=temp_file,
                                speaker_wav=config["speaker_wav"],
                                language=config.get("language", "en")
                            )
                            logger.debug(f"Section {section_index + 1}, sentence {i}: Generated with voice cloning")
                        else:
                            logger.warning(f"Section {section_index + 1}: Reference audio not found, using default voice")
                            tts.tts_to_file(
                                text=sentence,
                                file_path=temp_file,
                                language=config.get("language", "en")
                            )
                    else:
                        # Standard TTS without voice cloning
                        tts.tts_to_file(
                            text=sentence,
                            file_path=temp_file
                        )

                    audio_files.append(temp_file)

                except Exception as e:
                    logger.error(f"Section {section_index + 1}, sentence {i}: Error - {e}")
                    print(f"Section {section_index + 1}, sentence {i}: Error - {e}")
                    continue

            # Combine all audio files for this section
            if audio_files:
                section_audio = AudioSegment.empty()

                for audio_file in audio_files:
                    try:
                        chunk = AudioSegment.from_wav(audio_file)

                        # Apply speed adjustment for more natural pace
                        speed_factor = config.get("speed_factor", 1.0)
                        if speed_factor != 1.0:
                            chunk = chunk._spawn(chunk.raw_data, overrides={
                                "frame_rate": int(chunk.frame_rate * speed_factor)
                            }).set_frame_rate(chunk.frame_rate)

                        section_audio += chunk

                        # Add pause between sentences (configurable)
                        sentence_pause = config.get("sentence_pause", 300)
                        section_audio += AudioSegment.silent(duration=sentence_pause)

                    except Exception as e:
                        logger.error(f"Section {section_index + 1}: Error processing audio chunk - {e}")
                        continue

                # Save section audio file with high quality
                section_output = os.path.join(output_dir, f"section_{section_index + 1:02d}.mp3")
                bitrate = config.get("bitrate", "192k")

                section_audio.export(
                    section_output,
                    format="mp3",
                    bitrate=bitrate,
                    parameters=["-q:a", "0"]  # Highest quality MP3 encoding
                )

                # Calculate processing duration
                duration = time.time() - start_time

                logger.info(f"Section {section_index + 1}: Complete! Saved to {section_output} (took {duration:.1f}s)")
                print(f"Section {section_index + 1}: Complete! Saved to {section_output} (took {duration:.1f}s)")
                return (section_output, duration)
            else:
                logger.warning(f"Section {section_index + 1}: No audio generated")
                print(f"Section {section_index + 1}: No audio generated")
                return (None, 0)

    except Exception as e:
        logger.error(f"Section {section_index + 1}: Failed with error - {e}", exc_info=True)
        print(f"Section {section_index + 1}: Failed with error - {e}")
        return (None, 0)


# --- Step 5: Main conversion function with parallel processing ---

def convert_text_to_audio_parallel(
    text,
    output_dir,
    num_sections=20,
    max_workers=5,
    speed_factor=0.85,
    speaker_wav=None,
    language="en",
    sentence_pause=300,
    bitrate="192k"
):
    """
    Convert text to audio by splitting into sections and processing in parallel.

    Args:
        text: The text content to convert to audio
        output_dir: Directory where audio files will be saved
        num_sections: Number of sections to split the book into
        max_workers: Maximum number of parallel workers for processing
        speed_factor: Speed multiplier (< 1.0 = slower, > 1.0 = faster)
                     0.85 = 15% slower for more natural pace
        speaker_wav: Path to reference audio file for voice cloning (optional)
        language: Language code for TTS (default: "en")
        sentence_pause: Pause duration between sentences in milliseconds (default: 300)
        bitrate: MP3 bitrate for output audio (default: "192k")

    Returns:
        List of generated audio file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting audiobook conversion")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of sections: {num_sections}")
    logger.info(f"Parallel workers: {max_workers}")
    logger.info(f"Speed factor: {speed_factor}")
    logger.info(f"Voice cloning: {'Enabled' if speaker_wav else 'Disabled'}")

    # Split text into sections
    print(f"Splitting text into {num_sections} sections...")
    sections = split_text_into_sections(text, num_sections)
    print(f"Created {len(sections)} sections")

    # Create configuration dictionary
    config = {
        "speed_factor": speed_factor,
        "speaker_wav": speaker_wav,
        "language": language,
        "sentence_pause": sentence_pause,
        "bitrate": bitrate
    }

    # Prepare arguments for parallel processing
    args_list = [(i, section, output_dir, config) for i, section in enumerate(sections)]

    # Process sections in parallel
    print(f"\nProcessing sections with {max_workers} parallel workers...")
    section_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_section_to_audio, args): args[0] for args in args_list}

        for future in as_completed(futures):
            section_index = futures[future]
            try:
                result = future.result()
                file_path, duration = result
                if file_path:
                    section_results.append((section_index, file_path, duration))
            except Exception as e:
                error_msg = f"Section {section_index + 1} generated an exception: {e}"
                logger.error(error_msg, exc_info=True)
                print(error_msg)

    # Sort section results by index
    section_results.sort(key=lambda x: x[0])

    print(f"\n✓ All sections processed! {len(section_results)} files created.")
    logger.info(f"Successfully processed {len(section_results)} out of {num_sections} sections")

    # Performance Report
    if section_results:
        print(f"\n{'='*70}")
        print("Performance Report")
        print(f"{'='*70}")

        total_time = 0
        for idx, file_path, duration in section_results:
            print(f"Section {idx + 1:2d}: {duration:6.1f}s - {os.path.basename(file_path)}")
            total_time += duration

        avg_time = total_time / len(section_results)
        print(f"{'-'*70}")
        print(f"Total Processing Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average Time/Section:  {avg_time:.1f}s")
        print(f"{'='*70}\n")

        logger.info(f"Total processing time: {total_time:.1f}s, Average: {avg_time:.1f}s/section")

    print(f"\nSection files saved in: {output_dir}")
    for _, file_path, _ in section_results:
        print(f"  - {os.path.basename(file_path)}")

    return [file_path for _, file_path, _ in section_results]


# --- Main Execution ---
if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION - Customize these settings for your audiobook
    # ========================================================================

    # 1. EPUB File Path
    # Path to the EPUB file you want to convert
    EPUB_FILE_PATH = "/home/noam/Documents/EPUBs/Principles for Dealing with the Changing World Order Why Nations Succeed and Fail (Ray Dalio).epub"

    # 2. Output Directory
    # Directory where audio files will be saved
    OUTPUT_DIR = "/home/noam/Documents/Audibles/Principles_for_Dealing"

    # 3. Target Sentences Per Section
    # The script will automatically calculate the number of sections needed
    # based on the book's total sentence count and this target value.
    # Lower values = more (smaller) MP3 files, Higher values = fewer (larger) MP3 files
    # Recommended: 50-150 sentences per section
    TARGET_SENTENCES_PER_SECTION = 100

    # 4. Parallel Workers
    # Number of parallel workers for processing
    # Adjust based on your GPU memory:
    #   - 3-5 workers: Good for single GPU with 8GB+ VRAM
    #   - 1-2 workers: For limited VRAM or CPU processing
    MAX_WORKERS = 3

    # 5. Speech Speed
    # Speed multiplier for audio playback:
    #   - 0.85 = 15% slower (recommended for natural, relaxed pace)
    #   - 0.75 = 25% slower (very slow, deliberate pace)
    #   - 1.0  = original speed
    #   - 1.15 = 15% faster (for experienced listeners)
    SPEED_FACTOR = 1.0

    # 6. Voice Cloning (Optional)
    # Path to reference audio file for voice cloning
    # This should be a clear 6-10 second audio sample of the target voice
    # Set to None to use the default XTTS v2 voice
    # Example: "/path/to/brian_voice_sample.wav"
    SPEAKER_REFERENCE = "/home/noam/Documents/Audibles/mike_russel_voice.wav"  # Set to your reference audio path to enable voice cloning

    # 7. Language
    # Language code for TTS generation
    # Options: "en" (English), "es" (Spanish), "fr" (French), "de" (German),
    #          "it" (Italian), "pt" (Portuguese), "pl" (Polish), "tr" (Turkish),
    #          "ru" (Russian), "nl" (Dutch), "cs" (Czech), "ar" (Arabic),
    #          "zh-cn" (Chinese), "ja" (Japanese), "ko" (Korean), "hu" (Hungarian)
    LANGUAGE = "en"

    # 8. Sentence Pause Duration
    # Pause between sentences in milliseconds
    # 300ms is natural, increase for more deliberate pacing
    SENTENCE_PAUSE_MS = 300

    # 9. Audio Quality
    # MP3 bitrate for output files
    # Options: "128k" (good), "192k" (very good), "256k" (excellent), "320k" (maximum)
    BITRATE = "192k"

    # ========================================================================
    # PROCESSING
    # ========================================================================

    # Start total execution timer
    total_start_time = time.time()

    logger.info("="*70)
    logger.info("EPUB to Audiobook Converter - Enhanced with XTTS v2")
    logger.info("="*70)

    # Extract text from the EPUB
    try:
        book_text = extract_text_from_epub(EPUB_FILE_PATH)
    except FileNotFoundError:
        logger.error(f"EPUB file not found: {EPUB_FILE_PATH}")
        print(f"\nError: EPUB file not found: {EPUB_FILE_PATH}")
        exit(1)
    except Exception as e:
        logger.error(f"Error extracting text from EPUB: {e}", exc_info=True)
        print(f"\nError extracting text from EPUB: {e}")
        exit(1)

    # Check if text was extracted
    if not book_text or len(book_text.strip()) == 0:
        logger.error("No text could be extracted from the EPUB file")
        print("\nError: No text could be extracted from the EPUB file.")
        exit(1)

    # Calculate dynamic section count based on total sentences
    print("\nAnalyzing book structure...")
    all_sentences = split_into_sentences(book_text)
    total_sentence_count = len(all_sentences)
    num_sections = math.ceil(total_sentence_count / TARGET_SENTENCES_PER_SECTION)

    print(f"  - Total sentences: {total_sentence_count:,}")
    print(f"  - Target sentences per section: {TARGET_SENTENCES_PER_SECTION}")
    print(f"  - Calculated sections: {num_sections}")

    # Display configuration summary
    print(f"\n{'='*70}")
    print("Configuration Summary")
    print(f"{'='*70}")
    print(f"EPUB File                   : {EPUB_FILE_PATH}")
    print(f"Output Dir                  : {OUTPUT_DIR}")
    print(f"Target Sentences Per Section: {TARGET_SENTENCES_PER_SECTION}")
    print(f"Calculated Sections         : {num_sections}")
    print(f"Workers                     : {MAX_WORKERS}")
    print(f"Speed Factor                : {SPEED_FACTOR}x")
    print(f"Voice Cloning               : {'Enabled (' + SPEAKER_REFERENCE + ')' if SPEAKER_REFERENCE else 'Disabled (using default XTTS v2 voice)'}")
    print(f"Language                    : {LANGUAGE}")
    print(f"Sentence Pause              : {SENTENCE_PAUSE_MS}ms")
    print(f"Bitrate                     : {BITRATE}")
    print(f"{'='*70}\n")

    # Convert the extracted text into audio files (sections)
    try:
        section_files = convert_text_to_audio_parallel(
            text=book_text,
            output_dir=OUTPUT_DIR,
            num_sections=num_sections,
            max_workers=MAX_WORKERS,
            speed_factor=SPEED_FACTOR,
            speaker_wav=SPEAKER_REFERENCE,
            language=LANGUAGE,
            sentence_pause=SENTENCE_PAUSE_MS,
            bitrate=BITRATE
        )
    except Exception as e:
        logger.error(f"Fatal error during conversion: {e}", exc_info=True)
        print(f"\nFatal error during conversion: {e}")
        exit(1)

    # Check if any sections were successfully processed
    if not section_files:
        logger.error("No audio files were generated")
        print("\nError: No audio files were generated. Check the logs for details.")
        exit(1)

    # Optionally, merge all sections into one file
    print(f"\n{'='*70}")
    merge = input("Do you want to merge all sections into one file? (y/n): ").lower().strip()

    if merge == 'y':
        try:
            print("Merging all sections into a single audiobook file...")
            logger.info("Starting merge of all sections")

            final_output = os.path.join(OUTPUT_DIR, "complete_audiobook.mp3")

            combined = AudioSegment.empty()
            for idx, audio_path in enumerate(section_files, 1):
                logger.info(f"Merging section {idx}/{len(section_files)}")
                section_audio = AudioSegment.from_mp3(audio_path)
                combined += section_audio
                # Add a longer pause between sections (1 second)
                combined += AudioSegment.silent(duration=1000)

            # Export with high quality settings
            combined.export(
                final_output,
                format="mp3",
                bitrate=BITRATE,
                parameters=["-q:a", "0"]
            )

            logger.info(f"Complete audiobook saved to: {final_output}")
            print(f"✓ Complete audiobook saved to: {final_output}")

            # Display file size
            file_size_mb = os.path.getsize(final_output) / (1024 * 1024)
            duration_minutes = len(combined) / 60000
            print(f"  File size: {file_size_mb:.1f} MB")
            print(f"  Duration: {duration_minutes:.1f} minutes")

        except Exception as e:
            logger.error(f"Error merging sections: {e}", exc_info=True)
            print(f"\nError merging sections: {e}")
            print("Individual section files are still available in the output directory.")

    # Calculate and display total execution time
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    total_elapsed_minutes = total_elapsed / 60

    print(f"\n{'='*70}")
    print("Audiobook generation complete!")
    print(f"{'='*70}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Log file: epub_conversion.log")
    print(f"\nTotal Elapsed Time: {total_elapsed:.1f}s ({total_elapsed_minutes:.1f} minutes)")
    print(f"{'='*70}\n")

    logger.info(f"Audiobook generation completed successfully in {total_elapsed:.1f}s")

