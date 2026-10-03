#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "sherpa-onnx>=1.13.8",
#     "sentencepiece==0.2.0",
#     "sounddevice>=0.5",
#     "numpy>=2",
#     "soxr>=0.5",
# ]
# ///
"""Listen for a wake phrase, then send what is said next to Stavrobot and speak the reply."""
# sentencepiece is pinned because newer Windows wheels crash on import with an access
# violation on an Intel N5095, and 0.2.0 only has Windows wheels up to Python 3.12.

import base64
import json
import logging
import os
import queue
import re
import shutil
import sys
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import sentencepiece
import sherpa_onnx
import sounddevice
import soxr

LISTENER_DIR = Path(__file__).resolve().parent
CONFIG_PATH = LISTENER_DIR / "config.toml"
MODELS_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "stavrobot-listener" / "models"
RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
KWS_MODEL = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
KWS_FILES = "epoch-12-avg-2-chunk-16-left-64"
VAD_MODEL = "silero_vad.onnx"
STT_MODELS = {
    "moonshine-base": "sherpa-onnx-moonshine-base-en-int8",
    "moonshine-tiny": "sherpa-onnx-moonshine-tiny-en-int8",
}
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1
VAD_WINDOW = 512
# More threads than this is slower on the low-power CPU this runs on.
NUM_THREADS = 2
# How long to wait for speech after the wake phrase, how much silence ends a request,
# and the longest request recorded.
SPEECH_START_TIMEOUT_SECONDS = 5
END_OF_SPEECH_SILENCE_SECONDS = 1.0
MAX_REQUEST_SECONDS = 30
CHAT_TIMEOUT_SECONDS = 300
SPOKEN_PREFIX = (
    "(Said aloud to you in the kitchen; your reply will be read out by a speaker. "
    "Answer in a few short spoken sentences with no markdown, lists, links or emoji.)\n\n"
)

log = logging.getLogger("listener")


def load_config() -> dict:
    """Read config.toml next to this script."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Copy config.example.toml to {CONFIG_PATH} and fill it in.")
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in ("stavrobot_url", "password", "wake_phrase"):
        if not config.get(key):
            raise ValueError(f"{key} is missing from {CONFIG_PATH}.")
    if config.get("stt_model", "moonshine-base") not in STT_MODELS:
        raise ValueError(f"stt_model must be one of: {', '.join(STT_MODELS)}")
    return config


def download_model(release: str, name: str) -> Path:
    """Download and unpack a model archive into MODELS_DIR if missing."""
    destination = MODELS_DIR / name
    if destination.exists():
        return destination
    log.info("Downloading %s...", name)
    archive = MODELS_DIR / f"{name}.tar.bz2"
    staging = MODELS_DIR / f"{name}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    urllib.request.urlretrieve(f"{RELEASE_URL}/{release}/{name}.tar.bz2", archive)
    with tarfile.open(archive) as tar:
        tar.extractall(staging, filter="data")
    archive.unlink()
    shutil.rmtree(staging / name / "test_wavs", ignore_errors=True)
    (staging / name).rename(destination)
    staging.rmdir()
    return destination


def download_vad() -> Path:
    """Download the voice activity detection model if missing."""
    destination = MODELS_DIR / VAD_MODEL
    if not destination.exists():
        log.info("Downloading %s...", VAD_MODEL)
        partial = destination.with_suffix(".partial")
        urllib.request.urlretrieve(f"{RELEASE_URL}/asr-models/{VAD_MODEL}", partial)
        partial.rename(destination)
    return destination


def create_keyword_spotter(model_directory: Path, wake_phrase: str, sensitivity: float) -> sherpa_onnx.KeywordSpotter:
    """Load the keyword spotter for the configured wake phrase."""
    # The model is open-vocabulary: any phrase works once it is split into the model's
    # word pieces, so changing the phrase only needs a restart.
    pieces = sentencepiece.SentencePieceProcessor(model_file=str(model_directory / "bpe.model")).encode(
        wake_phrase.upper(), out_type=str
    )
    keywords_file = MODELS_DIR / "keywords.txt"
    keywords_file.write_text(" ".join(pieces) + " @" + wake_phrase.replace(" ", "_") + "\n", encoding="utf-8")
    return sherpa_onnx.KeywordSpotter(
        tokens=str(model_directory / "tokens.txt"),
        encoder=str(model_directory / f"encoder-{KWS_FILES}.int8.onnx"),
        decoder=str(model_directory / f"decoder-{KWS_FILES}.onnx"),
        joiner=str(model_directory / f"joiner-{KWS_FILES}.int8.onnx"),
        num_threads=1,
        keywords_file=str(keywords_file),
        # Higher thresholds need a more confident match, so fewer false wakes but more
        # missed ones; sensitivity is its inverse so that higher means "wakes more easily".
        keywords_threshold=1 - sensitivity,
    )


def create_vad(model_path: Path) -> sherpa_onnx.VoiceActivityDetector:
    """Load the voice activity detector used to find the end of a request."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(model_path)
    config.silero_vad.min_silence_duration = 0.25
    config.silero_vad.window_size = VAD_WINDOW
    config.sample_rate = SAMPLE_RATE
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=MAX_REQUEST_SECONDS + 5)


def create_recognizer(model_directory: Path) -> sherpa_onnx.OfflineRecognizer:
    """Load the Moonshine speech-to-text model."""
    return sherpa_onnx.OfflineRecognizer.from_moonshine(
        preprocessor=str(model_directory / "preprocess.onnx"),
        encoder=str(model_directory / "encode.int8.onnx"),
        uncached_decoder=str(model_directory / "uncached_decode.int8.onnx"),
        cached_decoder=str(model_directory / "cached_decode.int8.onnx"),
        tokens=str(model_directory / "tokens.txt"),
        num_threads=NUM_THREADS,
    )


def create_tts(voice_directory: Path) -> sherpa_onnx.OfflineTts:
    """Load a Piper voice."""
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


def find_device(name: str, kind: str) -> int | None:
    """Return the index of the first device whose name contains name, or None for the default."""
    if not name:
        return None
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    for index, device in enumerate(sounddevice.query_devices()):
        if name.lower() in device["name"].lower() and device[channels_key] > 0:
            return index
    raise ValueError(f"No {kind} device matches {name!r}; run with --list-devices to see them.")


def tone(frequencies: list[float], seconds: float = 0.12) -> np.ndarray:
    """Build a short chime from a sequence of sine tones."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    fade = np.minimum(1, np.minimum(t, seconds - t) / 0.01)
    return np.concatenate([0.3 * np.sin(2 * np.pi * f * t) * fade for f in frequencies]).astype(np.float32)


WAKE_CHIME = tone([660, 880])
CANCEL_CHIME = tone([440, 330])


def plain_spoken_text(text: str) -> str:
    """Strip markdown and links that would sound wrong when read aloud."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#>|~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences so playback can start before the whole reply is generated."""
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence]


class Listener:
    """The microphone loop: wait for the wake phrase, record a request, answer it."""

    def __init__(self, config: dict) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.config = config
        kws_directory = download_model("kws-models", KWS_MODEL)
        vad_path = download_vad()
        stt_directory = download_model("asr-models", STT_MODELS[config.get("stt_model", "moonshine-base")])
        voice_directory = download_model("tts-models", f"vits-piper-{config.get('tts_voice', 'en_US-lessac-medium')}")

        log.info("Loading models...")
        self.spotter = create_keyword_spotter(kws_directory, config["wake_phrase"], config.get("wake_sensitivity", 0.75))
        self.vad_path = vad_path
        self.recognizer = create_recognizer(stt_directory)
        self.tts = create_tts(voice_directory)

        self.input_device = find_device(config.get("input_device", ""), "input")
        self.output_device = find_device(config.get("output_device", ""), "output")
        self.input_rate = int(sounddevice.query_devices(self.input_device, "input")["default_samplerate"])
        self.blocks: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(self, indata: np.ndarray, frames: int, time_info: object, status: sounddevice.CallbackFlags) -> None:
        """Queue microphone audio from the PortAudio thread."""
        if status:
            log.warning("Audio input status: %s", status)
        self.blocks.put(indata[:, 0].copy())

    def run(self) -> None:
        """Listen forever."""
        resampler = soxr.ResampleStream(self.input_rate, SAMPLE_RATE, 1, dtype="float32")
        with sounddevice.InputStream(
            device=self.input_device,
            samplerate=self.input_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self.input_rate * BLOCK_SECONDS),
            callback=self.on_audio,
        ):
            log.info("Listening for %r.", self.config["wake_phrase"])
            stream = self.spotter.create_stream()
            while True:
                samples = resampler.resample_chunk(self.blocks.get())
                stream.accept_waveform(SAMPLE_RATE, samples)
                while self.spotter.is_ready(stream):
                    self.spotter.decode_stream(stream)
                    if self.spotter.get_result(stream):
                        log.info("Wake phrase heard.")
                        self.spotter.reset_stream(stream)
                        self.handle_request(resampler)
                        # Drop audio captured while answering, which includes the
                        # speaker's own voice, so it cannot trigger another wake.
                        self.drain_input()
                        stream = self.spotter.create_stream()
                        log.info("Listening for %r.", self.config["wake_phrase"])
                        break

    def drain_input(self) -> None:
        """Discard queued microphone audio."""
        while not self.blocks.empty():
            self.blocks.get_nowait()

    def handle_request(self, resampler: soxr.ResampleStream) -> None:
        """Record what is said after the wake phrase, send it to Stavrobot and speak the reply."""
        self.play(WAKE_CHIME)
        self.drain_input()
        audio = self.record_request(resampler)
        if audio is None:
            log.info("No request heard.")
            self.play(CANCEL_CHIME)
            return

        started = time.monotonic()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        log.info("Heard (%.1fs to transcribe): %s", time.monotonic() - started, text)
        if not text:
            self.play(CANCEL_CHIME)
            return

        try:
            reply = self.ask_stavrobot(text)
        except (urllib.error.URLError, TimeoutError) as error:
            log.error("Could not reach Stavrobot: %s", error)
            reply = "Sorry, I couldn't reach Stavrobot."
        log.info("Reply: %s", reply)
        self.speak(plain_spoken_text(reply))

    def record_request(self, resampler: soxr.ResampleStream) -> np.ndarray | None:
        """Record until the speaker stops talking, or return None if they never start."""
        vad = create_vad(self.vad_path)
        recorded: list[np.ndarray] = []
        pending = np.zeros(0, dtype=np.float32)
        heard_speech = False
        silent_seconds = 0.0
        total_seconds = 0.0
        while total_seconds < MAX_REQUEST_SECONDS:
            samples = resampler.resample_chunk(self.blocks.get())
            recorded.append(samples)
            total_seconds += len(samples) / SAMPLE_RATE
            pending = np.concatenate([pending, samples])
            while len(pending) >= VAD_WINDOW:
                vad.accept_waveform(pending[:VAD_WINDOW])
                pending = pending[VAD_WINDOW:]
                if vad.is_speech_detected():
                    heard_speech = True
                    silent_seconds = 0.0
                else:
                    silent_seconds += VAD_WINDOW / SAMPLE_RATE
            if not heard_speech and total_seconds > SPEECH_START_TIMEOUT_SECONDS:
                return None
            if heard_speech and silent_seconds >= END_OF_SPEECH_SILENCE_SECONDS:
                break
        return np.concatenate(recorded)

    def ask_stavrobot(self, text: str) -> str:
        """Send the request to Stavrobot's chat endpoint and return its reply."""
        credentials = base64.b64encode(f":{self.config['password']}".encode()).decode()
        request = urllib.request.Request(
            self.config["stavrobot_url"].rstrip("/") + "/chat",
            data=json.dumps({"message": SPOKEN_PREFIX + text, "source": "kitchen"}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Basic {credentials}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=CHAT_TIMEOUT_SECONDS) as response:
            return json.load(response)["response"]

    def speak(self, text: str) -> None:
        """Read text aloud, generating each sentence while the previous one plays."""
        for sentence in split_sentences(text):
            audio = self.tts.generate(sentence, sid=0, speed=1.0)
            sounddevice.wait()
            sounddevice.play(audio.samples, audio.sample_rate, device=self.output_device)
        sounddevice.wait()

    def play(self, samples: np.ndarray) -> None:
        """Play a chime and wait for it to finish."""
        sounddevice.play(samples, SAMPLE_RATE, device=self.output_device)
        sounddevice.wait()


def main() -> None:
    """Run the listener, or list audio devices with --list-devices."""
    if "--list-devices" in sys.argv:
        print(sounddevice.query_devices())
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LISTENER_DIR / "listener.log", encoding="utf-8")],
    )
    Listener(load_config()).run()


main()
