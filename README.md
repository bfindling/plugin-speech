# Speech plugin for Stavrobot

Offline speech recognition for [Stavrobot](https://github.com/skorokithakis/stavrobot).
Send the bot a voice note and it transcribes it locally with
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and the Moonshine model, then
answers it like a typed message. No audio leaves your server and no API key is needed.

## Installation

Tell Stavrobot to install this plugin's git URL. The models (about 275 MB) download in
the background after install, which takes a minute or two.

## Configuration

- `stt_model` (optional): `moonshine-base` (default) or `moonshine-tiny`. Tiny is about
  twice as fast and slightly less accurate. After changing it, update the plugin so the
  new model downloads.

## Limits

- English only.
- Tools have 30 seconds to run. On a low-power CPU (Intel N5095) about a minute of speech
  fits; longer recordings return a partial transcript with a note saying how much is
  missing.

## Tools

- `transcribe`: transcribe an audio file (OGG/Opus, WAV, FLAC, MP3) to text.
