#!/usr/bin/env -S uv run
# /// script
# dependencies = ["sherpa-onnx>=1.13.8", "soundfile>=0.14", "numpy>=2", "soxr>=0.5"]
# ///
# The dependencies match the tools' so their packages are already cached when the first
# tool call runs; downloading them inside a 30-second tool call is too slow.

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))

from speech import MODELS_DIR, RELEASE_URL, VAD_MODEL, stt_model_directory  # noqa: E402


def download_vad() -> None:
    """Download the voice activity detection model if missing."""
    destination = MODELS_DIR / VAD_MODEL
    if destination.exists():
        return
    partial = destination.with_suffix(".partial")
    urllib.request.urlretrieve(f"{RELEASE_URL}/{VAD_MODEL}", partial)
    partial.rename(destination)
    print(f"Downloaded {VAD_MODEL}.")


def download_stt_model(destination: Path) -> None:
    """Download and unpack a speech-to-text model if missing."""
    if destination.exists():
        return
    archive = MODELS_DIR / f"{destination.name}.tar.bz2"
    staging = MODELS_DIR / f"{destination.name}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    urllib.request.urlretrieve(f"{RELEASE_URL}/{destination.name}.tar.bz2", archive)
    with tarfile.open(archive) as tar:
        tar.extractall(staging, filter="data")
    archive.unlink()
    shutil.rmtree(staging / destination.name / "test_wavs", ignore_errors=True)
    (staging / destination.name).rename(destination)
    staging.rmdir()
    print(f"Downloaded {destination.name}.")


def main() -> None:
    """Download the models the plugin needs."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_vad()
    download_stt_model(stt_model_directory())
    print("Speech models are ready.")


main()
