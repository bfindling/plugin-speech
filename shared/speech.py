import json
import os
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
# HOME is a per-plugin cache directory on a Linux filesystem, which loads models much
# faster than the plugin directory and survives plugin updates.
MODELS_DIR = Path(os.environ["HOME"]) / "models"
RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
VAD_MODEL = "silero_vad.onnx"
STT_MODELS = {
    "moonshine-base": "sherpa-onnx-moonshine-base-en-int8",
    "moonshine-tiny": "sherpa-onnx-moonshine-tiny-en-int8",
}
DEFAULT_STT_MODEL = "moonshine-base"


def stt_model_directory() -> Path:
    """Return the directory of the configured speech-to-text model."""
    config_path = PLUGIN_ROOT / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    name = config.get("stt_model", DEFAULT_STT_MODEL)
    if name not in STT_MODELS:
        raise ValueError(f"Unknown stt_model {name!r}; choose one of: {', '.join(STT_MODELS)}")
    return MODELS_DIR / STT_MODELS[name]
