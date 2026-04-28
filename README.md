# Voice → Markdown

A local, open-source voice-to-text app. Record in your browser, get clean
Markdown back. No cloud. No API keys. Audio never leaves your machine.

Built for capturing meeting notes, brainstorms, and context dumps you can
paste straight into your editor of choice.

---

## Why

Most voice-to-text tools either ship your audio to a third-party API or
output a wall of unbroken text. This does neither: transcription runs locally
via [faster-whisper][fw], and the output is structured Markdown with
paragraph breaks at natural pauses.

[fw]: https://github.com/SYSTRAN/faster-whisper

## Features

- **Fully local.** Audio is processed on your machine. The only network call
  is the one-time model download from Hugging Face.
- **Open-source stack.** Whisper (MIT) + faster-whisper (MIT) + Flask (BSD).
- **Clean Markdown output.** Pause-based paragraph breaks, light filler-word
  trimming. Editable in the browser before you copy or download.
- **No length cap.** 5–30 minute sessions work fine on a typical laptop.
- **Single-page app.** One Python file, one HTML file. Read it in five
  minutes.

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) on your PATH
- A modern browser (Chrome, Firefox, Safari, Edge)

## Quickstart

```bash
git clone https://github.com/<your-username>/voice-to-markdown.git
cd voice-to-markdown

# Install ffmpeg if you don't have it:
#   macOS:   brew install ffmpeg
#   Linux:   sudo apt install ffmpeg
#   Windows: https://ffmpeg.org/download.html

python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt

python server.py
```

Then open <http://localhost:5000> and hit **Start Recording**.

The first launch downloads the Whisper model (~500 MB for the default
`small` model). Subsequent launches load instantly from cache.

## Choosing a model

Set `WHISPER_MODEL` to trade speed for accuracy:

```bash
WHISPER_MODEL=base python server.py     # faster, less accurate
WHISPER_MODEL=medium python server.py   # slower, more accurate
```

| Model     | Size    | Speed (CPU)   | Accuracy   |
|-----------|---------|---------------|------------|
| tiny      | ~75 MB  | very fast     | rough      |
| base      | ~150 MB | fast          | okay       |
| **small** | ~500 MB | **moderate**  | **good**   |
| medium    | ~1.5 GB | slow          | very good  |
| large-v3  | ~3 GB   | slow          | best       |

Rough timings for a 5-minute recording on a modern laptop:

- `small`: ~30–60 seconds to transcribe
- `medium`: ~1.5–3 minutes to transcribe

If you have an NVIDIA GPU:

```bash
WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16 python server.py
```

## How the Markdown formatting works

The server uses Whisper's segment timestamps to detect pauses longer than
1.2 seconds and inserts paragraph breaks there. It also lightly trims
filler words (*um*, *uh*, *you know*) — light touch only, no rewriting.

The output panel in the browser is editable, so you can clean up further
before copying or downloading.

## Keyboard shortcuts

- **Space** — toggle recording (when the output panel isn't focused)

## Privacy

Everything runs on your machine. Audio is sent only to `localhost:5000`,
which is your own Python process. The temp file is deleted after
transcription. The Whisper model is downloaded once from Hugging Face and
cached locally (`~/.cache/huggingface/`).

## Project layout

```
voice-to-markdown/
├── server.py          # Flask server + faster-whisper integration
├── index.html         # Frontend (recorder UI + output panel)
├── requirements.txt   # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## Contributing

Issues and PRs welcome. Some directions worth exploring:

- LLM post-processing (clean up rambling speech into structured notes)
- Word-level timestamps
- Speaker diarization
- Push-to-talk hotkey
- Streaming transcription (transcribe while recording)

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — the underlying model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — the
  CTranslate2-optimized runtime that makes this practical on CPU
