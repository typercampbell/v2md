"""
Voice-to-Markdown local server.

Run with:
    python server.py

Requires:
    pip install flask flask-cors faster-whisper

Then open http://localhost:5000 in your browser.
"""

import os
import re
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faster_whisper import WhisperModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Model size options (trade-off: accuracy vs speed vs memory):
#   tiny    ~75MB,  fastest,  least accurate
#   base    ~150MB, fast,     decent
#   small   ~500MB, moderate, good             <-- recommended default
#   medium  ~1.5GB, slow,     very good
#   large-v3 ~3GB,  slowest,  best accuracy
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")

# "cpu" works everywhere. Use "cuda" if you have an NVIDIA GPU + CUDA installed.
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")

# int8 is fast and accurate enough on CPU. Use "float16" on GPU.
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")

PORT = int(os.environ.get("PORT", "5000"))

# -----------------------------------------------------------------------------
# Load the model once at startup
# -----------------------------------------------------------------------------

print(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})...")
print("(First run will download the model. This can take a minute.)")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded. Ready.")

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)
CORS(app)

STATIC_DIR = Path(__file__).parent


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_SIZE, "device": DEVICE})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file in request"}), 400

    audio_file = request.files["audio"]

    # Save to a temp file (faster-whisper reads from a path; this also lets
    # ffmpeg handle whatever container/codec the browser sent).
    suffix = ".webm"
    if audio_file.filename and "." in audio_file.filename:
        suffix = "." + audio_file.filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        # vad_filter trims silences -> faster transcription, cleaner output.
        # beam_size=5 is the standard quality default.
        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        # Collect segments with their timing so we can do paragraph breaks.
        seg_list = [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
            }
            for s in segments
        ]

        raw_text = " ".join(s["text"] for s in seg_list).strip()
        markdown = format_as_markdown(seg_list)

        return jsonify(
            {
                "text": raw_text,
                "markdown": markdown,
                "language": info.language,
                "duration": info.duration,
                "segments": seg_list,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Markdown formatting
# -----------------------------------------------------------------------------

# Words/phrases people use mid-stream that we can lightly trim.
FILLER_PATTERNS = [
    r"\b(um|uh|er|erm|hmm)\b[,.]?\s*",
    r"\b(you know|i mean|like, like)\b[,.]?\s*",
]


def format_as_markdown(segments):
    """
    Convert Whisper segments into clean Markdown for meeting/brainstorm capture.

    Strategy:
      - Group segments into paragraphs whenever there's a notable pause
        (>1.2s gap between segments) or a sentence-final punctuation.
      - Lightly trim filler words.
      - Capitalize and add a top-level heading.
    """
    if not segments:
        return ""

    paragraphs = []
    current = []
    last_end = None
    PAUSE_THRESHOLD = 1.2  # seconds

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        # New paragraph if there's a long pause AND we've already collected
        # at least one sentence.
        if last_end is not None and (seg["start"] - last_end) >= PAUSE_THRESHOLD:
            if current:
                paragraphs.append(" ".join(current))
                current = []

        current.append(text)
        last_end = seg["end"]

    if current:
        paragraphs.append(" ".join(current))

    # Light filler-word cleanup (case-insensitive, leaves content intact).
    cleaned = []
    for p in paragraphs:
        out = p
        for pat in FILLER_PATTERNS:
            out = re.sub(pat, "", out, flags=re.IGNORECASE)
        # Collapse double spaces created by removals.
        out = re.sub(r"\s{2,}", " ", out).strip()
        # Capitalize first letter if it got lowercased by a removal.
        if out:
            out = out[0].upper() + out[1:]
        if out:
            cleaned.append(out)

    body = "\n\n".join(cleaned)
    return body


if __name__ == "__main__":
    print(f"\nServer running at http://localhost:{PORT}")
    print("Open that URL in your browser.\n")
    app.run(host="127.0.0.1", port=PORT, debug=False)
