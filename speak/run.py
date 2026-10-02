#!/usr/bin/env -S uv run
# /// script
# dependencies = ["sherpa-onnx>=1.13.8", "soundfile>=0.14", "numpy>=2", "soxr>=0.5"]
# ///

import json
import sys
from pathlib import Path

import sherpa_onnx
import soundfile
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))

from speech import NUM_THREADS, tts_voice_directory  # noqa: E402

OUTPUT_DIR = Path("/tmp/speech")
OUTPUT_FILENAME = "reply.ogg"
# Generation runs at about 1.7 seconds per 100 characters on the low-power CPU this
# runs on, so longer text would not finish within the 30-second tool limit.
MAX_CHARACTERS = 1000
# Telegram only shows OGG Opus files as voice notes, and Opus only supports a few
# sample rates, none of which is Piper's native rate.
OPUS_SAMPLE_RATE = 48000


def load_tts(voice_directory: Path) -> sherpa_onnx.OfflineTts:
    """Load a Piper voice."""
    if not voice_directory.exists():
        raise FileNotFoundError(f"Voice {voice_directory.name} is not downloaded; update the plugin to run init.")
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(next(voice_directory.glob("*.onnx"))),
                tokens=str(voice_directory / "tokens.txt"),
                data_dir=str(voice_directory / "espeak-ng-data"),
            ),
            num_threads=NUM_THREADS,
        ),
    )
    return sherpa_onnx.OfflineTts(config)


def main() -> None:
    """Speak the text given on stdin into an OGG Opus voice note."""
    params = json.load(sys.stdin)
    text = params["text"].strip()
    if not text:
        raise ValueError("text is empty.")
    if len(text) > MAX_CHARACTERS:
        raise ValueError(f"text is {len(text)} characters; shorten it to at most {MAX_CHARACTERS}.")

    audio = load_tts(tts_voice_directory()).generate(text, sid=0, speed=1.0)
    samples = soxr.resample(audio.samples, audio.sample_rate, OPUS_SAMPLE_RATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    soundfile.write(OUTPUT_DIR / OUTPUT_FILENAME, samples, OPUS_SAMPLE_RATE, format="OGG", subtype="OPUS")
    json.dump({"file": OUTPUT_FILENAME, "audio_seconds": round(len(samples) / OPUS_SAMPLE_RATE, 1)}, sys.stdout)


main()
