import json
import os
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
# HOME is a per-plugin cache directory on a Linux filesystem, which loads models much
# faster than the plugin directory and survives plugin updates.
MODELS_DIR = Path(os.environ["HOME"]) / "models"
STT_RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
TTS_RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
VAD_MODEL = "silero_vad.onnx"
STT_MODELS = {
    "moonshine-base": "sherpa-onnx-moonshine-base-en-int8",
    "moonshine-tiny": "sherpa-onnx-moonshine-tiny-en-int8",
}
DEFAULT_STT_MODEL = "moonshine-base"
DEFAULT_TTS_VOICE = "en_US-lessac-medium"
# More threads than this is slower on the low-power CPU this runs on.
NUM_THREADS = 2


def read_config() -> dict:
    """Read the plugin's config.json, or an empty config if it does not exist."""
    config_path = PLUGIN_ROOT / "config.json"
    return json.loads(config_path.read_text()) if config_path.exists() else {}


def stt_model_directory() -> Path:
    """Return the directory of the configured speech-to-text model."""
    name = read_config().get("stt_model", DEFAULT_STT_MODEL)
    if name not in STT_MODELS:
        raise ValueError(f"Unknown stt_model {name!r}; choose one of: {', '.join(STT_MODELS)}")
    return MODELS_DIR / STT_MODELS[name]


def tts_voice_directory() -> Path:
    """Return the directory of the configured Piper text-to-speech voice."""
    return MODELS_DIR / f"vits-piper-{read_config().get('tts_voice', DEFAULT_TTS_VOICE)}"
