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

## Wake word listener

`listener/listen.py` turns any computer with a microphone and speaker into a voice
terminal for Stavrobot, like a smart speaker. Say the wake phrase ("hey kitchen" by
default), wait for the chime, and ask your question. The listener transcribes it
locally, sends it to Stavrobot's `/chat` endpoint and reads the reply aloud. It does not
use the plugin or Telegram; it only needs to reach Stavrobot over the network.

Setup:

1. Install [uv](https://docs.astral.sh/uv/).
2. Copy `listener/config.example.toml` to `listener/config.toml` and set `stavrobot_url`
   and `password` (Stavrobot's `password` from its `config.toml`).
3. Run `uv run listen.py --list-devices` in the `listener` folder. If the marked default
   input or output is not the one you want, set `input_device` or `output_device` to
   part of its name.
4. Run `uv run listen.py --say "Testing the speaker."` to check the speaker.
5. Run `uv run listen.py`. The first start downloads about 350 MB of models.

To keep it running, start it with `--supervise`, which restarts the listener 10 seconds
after it exits for any reason. On Windows, a scheduled task that runs
`uvw run -q listen.py --supervise` in the `listener` folder at sign-in keeps it running
in the background with no window; Windows only gives audio devices to a signed-in user,
so that user must stay signed in.

The wake phrase can be any words: change `wake_phrase` and restart. Adjust
`wake_sensitivity` if it misses you or wakes on its own. Activity is logged to
`listener/listener.log`.

While idle the listener uses about a tenth of one CPU core on an Intel N5095. A request
takes roughly: the time you speak, a second of silence, under a second to transcribe,
however long Stavrobot takes to answer, then the spoken reply.
