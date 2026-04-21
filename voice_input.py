"""Voice input helper for Google Calendar Agent.

Provides start/stop microphone recording (via sounddevice) and speech-to-text
transcription (via faster-whisper). The Whisper model is loaded once at import
time and cached for the lifetime of the process.

Usage:
    from voice_input import start_recording, stop_recording, transcribe_audio

    start_recording()
    # ... user speaks ...
    audio, sample_rate = stop_recording()
    text = transcribe_audio(audio, sample_rate)
"""

import io
import logging
import re
import threading
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# "medium" is meaningfully better than "small" for proper nouns and
# email addresses in short voice commands (~10–20 s), and still runs
# acceptably on CPU with int8 quantisation.
_MODEL_SIZE = "medium"
_COMPUTE_TYPE = "int8"   # int8 quantisation keeps memory low on CPU
_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Whisper initial prompt
# ---------------------------------------------------------------------------
# Priming the decoder with representative calendar command examples biases
# beam search toward correct email-address formatting, proper-noun
# capitalisation, and domain-specific vocabulary — without any fine-tuning.
_INITIAL_PROMPT = (
    "Schedule a meeting with Alice Johnson (alice.johnson@example.com) and "
    "Bob Smith (bob.smith@company.org) on Friday at 2 PM for 1 hour. "
    "Add a dentist appointment on March 30th at 10 AM. "
    "Create a task to submit the Q1 report by end of week. "
    "Move the Monday standup to 3 PM. "
    "Delete the team sync with carol@acme.com scheduled for tomorrow. "
    "Add Sarah's birthday on June 15th. "
    "Block my calendar for deep work on Tuesday from 9 AM to 12 PM. "
    "Send a meeting invite to david.lee@startup.io for next Wednesday at 4 PM."
)

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Return the shared WhisperModel, loading it on first call."""
    global _model
    if _model is None:
        logger.info("Loading Whisper '%s' model (first run may download weights)…", _MODEL_SIZE)
        _model = WhisperModel(_MODEL_SIZE, device=_DEVICE, compute_type=_COMPUTE_TYPE)
        logger.info("Whisper model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Start / stop recording
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000   # 16 kHz — Whisper's native sample rate

# Module-level recording state shared between the main thread and the
# sounddevice callback thread.
_audio_chunks: list[np.ndarray] = []
_recording: bool = False
_stream: sd.InputStream | None = None
_lock = threading.Lock()


def _audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
    """sounddevice callback — appends each chunk to the buffer."""
    if status:
        logger.warning("sounddevice status: %s", status)
    with _lock:
        if _recording:
            _audio_chunks.append(indata.copy())


def start_recording() -> None:
    """Open the microphone stream and begin buffering audio.

    Safe to call from the Streamlit main thread; audio capture runs in the
    sounddevice callback thread so it is not blocked by Streamlit reruns.
    """
    global _recording, _stream, _audio_chunks

    with _lock:
        _audio_chunks = []
        _recording = True

    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
    )
    _stream.start()
    logger.info("Recording started.")


def stop_recording() -> tuple[np.ndarray, int]:
    """Stop the microphone stream and return the captured audio.

    Returns:
        (audio_array, sample_rate) where audio_array is a 1-D float32 numpy
        array. Returns an empty array if no audio was captured.
    """
    global _recording, _stream

    with _lock:
        _recording = False

    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None

    with _lock:
        chunks = list(_audio_chunks)

    if not chunks:
        logger.warning("stop_recording called but no audio was captured.")
        return np.array([], dtype="float32"), SAMPLE_RATE

    audio = np.concatenate(chunks, axis=0).squeeze()
    logger.info("Recording stopped. Captured %.1f seconds of audio.", len(audio) / SAMPLE_RATE)
    return audio, SAMPLE_RATE


def is_recording() -> bool:
    """Return True if a recording session is currently active."""
    with _lock:
        return _recording


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

# Spoken TLD words → actual TLD strings
_TLD_MAP = {
    "dot com": ".com",
    "dot org": ".org",
    "dot net": ".net",
    "dot edu": ".edu",
    "dot gov": ".gov",
    "dot io": ".io",
    "dot co": ".co",
    "dot ai": ".ai",
    "dot us": ".us",
    "dot uk": ".uk",
    "dot de": ".de",
    "dot fr": ".fr",
    "dot ca": ".ca",
    "dot au": ".au",
}

def _reconstruct_emails(text: str) -> str:
    """Convert spoken email patterns to proper email addresses.

    Handles the most common spoken forms:
      "alice at company dot com"             → alice@company.com
      "alice dot johnson at company dot com" → alice.johnson@company.com
      "alice underscore johnson at ..."      → alice_johnson@...

    Processing order matters:
      1. Resolve "dot" / "underscore" inside local-part and domain first.
      2. Then stitch local-part and domain together with "@".
      3. Finally apply TLD word replacements that survived step 1.
    """
    result = text

    # Step 1a: "underscore" between word characters → "_"
    result = re.sub(r"(\w) underscore (\w)", r"\1_\2", result, flags=re.IGNORECASE)

    # Step 1b: "dot" between word characters → "."
    # This covers both local-part dots ("alice dot johnson") and
    # domain dots ("company dot com", "startup dot io").
    result = re.sub(r"(\w) dot (\w)", r"\1.\2", result, flags=re.IGNORECASE)

    # Step 2: " at " between an email-like local part and a domain → "@"
    # The domain must contain a dot (already converted above) so we require
    # at least one "." followed by 2+ letters — this prevents matching
    # natural-language "at" like "meet at 3 PM" or "at the office".
    result = re.sub(
        r"([\w.\-_]+) at ([\w.\-_]+\.[a-zA-Z]{2,}(?:\b|$))",
        r"\1@\2",
        result,
        flags=re.IGNORECASE,
    )

    # Step 3: any remaining spoken TLD words (edge cases where step 1b didn't
    # fire because the word boundary was at the end of the string, e.g.
    # "...company dot com<end>").  Sorted longest-first to avoid partial matches.
    for spoken, symbol in sorted(_TLD_MAP.items(), key=lambda x: -len(x[0])):
        result = re.sub(rf"\b{re.escape(spoken)}\b", symbol, result, flags=re.IGNORECASE)

    return result


def transcribe_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Transcribe a numpy audio array to text using faster-whisper.

    Args:
        audio: 1-D float32 numpy array of audio samples.
        sample_rate: Sample rate of the audio (must be 16 kHz).

    Returns:
        Transcribed text as a single string. Returns an empty string if
        nothing was detected or the audio array is empty.
    """
    if audio.size == 0:
        return ""

    model = _get_model()

    # faster-whisper accepts a WAV file-like object.
    # Write the numpy array into an in-memory WAV buffer.
    wav_buffer = io.BytesIO()
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(wav_buffer, sample_rate, audio_int16)
    wav_buffer.seek(0)

    segments, info = model.transcribe(
        wav_buffer,
        language="en",
        beam_size=5,
        initial_prompt=_INITIAL_PROMPT,   # biases decoder toward calendar/email vocabulary
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    logger.info(
        "Detected language '%s' with probability %.2f",
        info.language,
        info.language_probability,
    )

    raw = " ".join(segment.text for segment in segments).strip()
    transcript = _reconstruct_emails(raw)
    if transcript != raw:
        logger.info("Email reconstruction applied: %r → %r", raw, transcript)
    logger.info("Transcript: %s", transcript)
    return transcript
