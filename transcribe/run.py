#!/usr/bin/env -S uv run
# /// script
# dependencies = ["sherpa-onnx>=1.13.8", "soundfile>=0.14", "numpy>=2", "soxr>=0.5"]
# ///

import json
import sys
import time
from pathlib import Path

STARTED = time.monotonic()

import numpy as np  # noqa: E402
import sherpa_onnx  # noqa: E402
import soundfile  # noqa: E402
import soxr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))

from speech import MODELS_DIR, NUM_THREADS, VAD_MODEL, stt_model_directory  # noqa: E402

SAMPLE_RATE = 16000
# The plugin runner kills tools after 30 seconds. Stop starting new segments after this
# point so long recordings return a partial transcript instead of nothing.
TIME_BUDGET_SECONDS = 20


def load_audio(path: str) -> np.ndarray:
    """Read an audio file as mono float samples at the model's sample rate."""
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate == SAMPLE_RATE:
        return mono
    return soxr.resample(mono, rate, SAMPLE_RATE)


def split_into_speech_segments(audio: np.ndarray) -> list[np.ndarray]:
    """Split audio into speech segments, dropping silence."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(MODELS_DIR / VAD_MODEL)
    config.silero_vad.min_silence_duration = 0.4
    config.silero_vad.max_speech_duration = 15
    config.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=len(audio) / SAMPLE_RATE + 1)
    window = config.silero_vad.window_size
    for offset in range(0, len(audio), window):
        vad.accept_waveform(audio[offset:offset + window])
    vad.flush()
    segments = []
    while not vad.empty():
        segments.append(np.array(vad.front.samples, dtype=np.float32))
        vad.pop()
    return segments


def load_recognizer(model_directory: Path) -> sherpa_onnx.OfflineRecognizer:
    """Load the Moonshine speech-to-text model."""
    if not model_directory.exists():
        raise FileNotFoundError(f"Model {model_directory.name} is not downloaded; reinstall or update the plugin to run init.")
    return sherpa_onnx.OfflineRecognizer.from_moonshine(
        preprocessor=str(model_directory / "preprocess.onnx"),
        encoder=str(model_directory / "encode.int8.onnx"),
        uncached_decoder=str(model_directory / "uncached_decode.int8.onnx"),
        cached_decoder=str(model_directory / "cached_decode.int8.onnx"),
        tokens=str(model_directory / "tokens.txt"),
        num_threads=NUM_THREADS,
    )


def main() -> None:
    """Transcribe the audio file given on stdin."""
    params = json.load(sys.stdin)
    audio = load_audio(params["audio"])
    # The runner sends back every file left in the temp directory, which would echo
    # the voice note to the agent as if the tool had produced it.
    Path(params["audio"]).unlink()
    audio_seconds = len(audio) / SAMPLE_RATE
    recognizer = load_recognizer(stt_model_directory())
    segments = split_into_speech_segments(audio)

    texts = []
    transcribed_seconds = 0.0
    for segment in segments:
        if time.monotonic() - STARTED > TIME_BUDGET_SECONDS:
            break
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, segment)
        recognizer.decode_stream(stream)
        texts.append(stream.result.text.strip())
        transcribed_seconds += len(segment) / SAMPLE_RATE

    result = {
        "text": " ".join(text for text in texts if text),
        "audio_seconds": round(audio_seconds, 1),
        # The agent tends to forget the plugin description's instruction by the time it
        # replies, so repeat it where it is read just before replying.
        "reply_instructions": (
            "If this was the user's voice note, reply by voice: call this plugin's speak tool with your "
            "answer written as spoken sentences, then send the produced file with send_telegram_message's "
            "attachmentPath."
        ),
    }
    if not segments:
        result["note"] = "No speech was detected."
    elif len(texts) < len(segments):
        speech_seconds = sum(len(segment) for segment in segments) / SAMPLE_RATE
        result["note"] = (
            f"Transcription stopped after {transcribed_seconds:.0f} of {speech_seconds:.0f} seconds of speech "
            "to stay within the time limit; the rest is missing."
        )
    print(f"Transcribed {len(texts)} of {len(segments)} segments in {time.monotonic() - STARTED:.1f}s.", file=sys.stderr)
    json.dump(result, sys.stdout)


main()
