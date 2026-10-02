# Speech plugin for Stavrobot

Offline speech for [Stavrobot](https://github.com/skorokithakis/stavrobot). Send the
bot a voice note and it transcribes it locally, answers it like a typed message, and
replies with a voice note of its own. Everything runs on your server with
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), the Moonshine speech recognition
model and a Piper voice. No audio leaves your server and no API key is needed.

## Installation

Tell Stavrobot to install this plugin's git URL. The models (about 350 MB) download in
the background after install, which takes a minute or two.

## Configuration

- `stt_model` (optional): `moonshine-base` (default) or `moonshine-tiny`. Tiny is about
  twice as fast and slightly less accurate.
- `tts_voice` (optional): any Piper voice published in the
  [sherpa-onnx TTS models release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models),
  named without the `vits-piper-` prefix, for example `en_US-lessac-medium` (default),
  `en_US-amy-low` or `en_GB-alan-medium`.

After changing either, update the plugin so the new model downloads.

## Limits

- Speech recognition is English only.
- Tools have 30 seconds to run. On a low-power CPU (Intel N5095) about a minute of speech
  can be transcribed; longer recordings return a partial transcript with a note saying
  how much is missing. Spoken replies are limited to 1000 characters (about 50 seconds
  of audio).

## Tools

- `transcribe`: transcribe an audio file (OGG/Opus, WAV, FLAC, MP3) to text.
- `speak`: turn text into an OGG Opus voice note, which Telegram shows as a voice
  message.
