"""
Voice-to-Markdown local server.

Run with:
    python server.py

Requires:
    pip install flask flask-cors faster-whisper

Then open http://localhost:5000 in your browser.

Voice commands (spoken at the start of an utterance, ended by a pause):
    "heading <text>"          -> # text
    "subheading <text>"       -> ## text
    "subsubheading <text>"    -> ### text
    "bullet <text>"           -> - text
    "numbered <text>"         -> 1. text (auto-increments within a run)
    "quote <text>"            -> > text
    "horizontal rule"         -> ---
    "new paragraph"           -> forces paragraph break
    "new line"                -> forces line break

Inline commands (need explicit end):
    "bold <text> end bold"           -> **text**
    "italic <text> end italic"       -> *text*
    "inline code <text> end code"    -> `text`
    "code <text> end code"           -> ```text``` (block)
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

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
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

    suffix = ".webm"
    if audio_file.filename and "." in audio_file.filename:
        suffix = "." + audio_file.filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

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


# =============================================================================
# Markdown formatting
# =============================================================================
#
# Strategy:
#   1. Group Whisper segments into "utterances" using pause-based boundaries.
#      An utterance is a chunk of speech the user delivered without a long
#      pause. This is what voice commands like "heading" or "bullet" apply to.
#   2. For each utterance, check if it starts with a block-level voice command.
#      If so, transform it. Otherwise, treat it as a normal paragraph.
#   3. Apply inline transformations (bold/italic/code) within each utterance.
#   4. Light filler-word cleanup, then join everything together.
#
# Pause threshold for utterance boundaries: 1.2 seconds. This is the same
# threshold used for paragraph breaks in plain mode — feels natural in
# practice.
# =============================================================================

PAUSE_THRESHOLD = 1.2  # seconds

FILLER_PATTERNS = [
    r"\b(um|uh|er|erm|hmm)\b[,.]?\s*",
    r"\b(you know|i mean)\b[,.]?\s*",
]

# Block-level command aliases. Whisper occasionally mishears these, so we
# accept reasonable variants. All matching is case-insensitive and ignores
# trailing punctuation that Whisper sometimes adds (e.g., "heading,").
# Order matters: longer phrases first so "subsubheading" beats "heading".
BLOCK_COMMANDS = [
    ("heading_3", ["sub sub heading", "subsubheading", "sub-sub-heading", "sub sub header"]),
    ("heading_2", ["subheading", "sub heading", "sub-heading", "subheader", "sub header"]),
    ("heading_1", ["heading", "header"]),
    ("bullet",    ["bullet", "bullet point", "dash"]),
    ("numbered",  ["numbered", "numbered point", "numbered item"]),
    ("quote",     ["quote", "block quote", "blockquote"]),
]

# Standalone block commands (whole utterance is the command, no body).
STANDALONE_COMMANDS = [
    ("hr",            ["horizontal rule", "divider", "horizontal line"]),
    ("paragraph_brk", ["new paragraph", "paragraph break"]),
    ("line_brk",      ["new line", "line break"]),
]


def _normalize(text):
    """Lowercase, strip surrounding whitespace and trailing punctuation."""
    return text.strip().rstrip(".,;:!?").strip().lower()


def _strip_command_prefix(text, phrase):
    """
    Remove a command phrase from the start of text. Match is case-insensitive
    and tolerates punctuation right after the phrase (e.g., "heading,").
    Returns None if the text doesn't start with the phrase.
    """
    pattern = r"^\s*" + re.escape(phrase) + r"\s*[:,]?\s*"
    m = re.match(pattern, text, flags=re.IGNORECASE)
    if m:
        return text[m.end():].strip()
    return None


def _match_standalone(text):
    """Return command name if utterance is a standalone command, else None."""
    norm = _normalize(text)
    for cmd, phrases in STANDALONE_COMMANDS:
        if norm in phrases:
            return cmd
    return None


def _match_block(text):
    """
    Try to match a block-level command at the start of text.
    Returns (command_name, body) if matched, else (None, None).
    """
    for cmd, phrases in BLOCK_COMMANDS:
        for phrase in phrases:
            body = _strip_command_prefix(text, phrase)
            if body is not None and body:  # non-empty body required
                return cmd, body
    return None, None


# Inline commands. Each has open phrases and close phrases; the text between
# them gets wrapped. Order matters: "inline code" must come before "code"
# block so the longer phrase wins.
INLINE_COMMANDS = [
    (["inline code"],        ["end code", "end inline code"], "`",          "`"),
    (["code block", "code"], ["end code", "end code block"],  "\n```\n",    "\n```\n"),
    (["bold"],               ["end bold"],                    "**",         "**"),
    (["italic", "italics"],  ["end italic", "end italics"],   "*",          "*"),
]


def _apply_inline(text):
    """Apply inline command wrapping within an utterance's text."""
    out = text
    for opens, closes, wrap_open, wrap_close in INLINE_COMMANDS:
        open_alt = "|".join(re.escape(o) for o in opens)
        close_alt = "|".join(re.escape(c) for c in closes)
        # We capture trailing whitespace after the close phrase so we can
        # preserve the natural spacing — without this, "bold X end bold and"
        # would collapse to "**X**and".
        pattern = (
            r"\b(?:" + open_alt + r")\b[\s,:]*"
            r"(.+?)"
            r"[\s,.]*\b(?:" + close_alt + r")\b([\s,.]?)"
        )

        def repl(m):
            body = m.group(1).strip().rstrip(",.;:")
            trailing = m.group(2)
            # Ensure at least one space after the close wrap if more text follows.
            if trailing == "" or trailing == ".":
                trailing = trailing + " " if trailing == "." else " "
            return f"{wrap_open}{body}{wrap_close}{trailing}"

        out = re.sub(pattern, repl, out, flags=re.IGNORECASE | re.DOTALL)
    return out


def _clean_fillers(text):
    out = text
    for pat in FILLER_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if out:
        out = out[0].upper() + out[1:]
    return out


def _group_into_utterances(segments):
    """Group segments into utterances using pause-based boundaries."""
    utterances = []
    current = []
    last_end = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if last_end is not None and (seg["start"] - last_end) >= PAUSE_THRESHOLD:
            if current:
                utterances.append(current)
                current = []
        current.append(seg)
        last_end = seg["end"]

    if current:
        utterances.append(current)

    return utterances


def format_as_markdown(segments):
    """Convert Whisper segments into Markdown, applying voice commands."""
    if not segments:
        return ""

    utterances = _group_into_utterances(segments)
    blocks = []  # list of (kind, content) tuples
    numbered_run = 0

    for utt_segs in utterances:
        text = " ".join(s["text"].strip() for s in utt_segs).strip()
        if not text:
            continue

        # Standalone commands first.
        standalone = _match_standalone(text)
        if standalone == "hr":
            blocks.append(("hr", "---"))
            numbered_run = 0
            continue
        if standalone == "paragraph_brk":
            blocks.append(("brk", ""))
            numbered_run = 0
            continue
        if standalone == "line_brk":
            blocks.append(("linebrk", ""))
            continue

        # Block command at the start?
        cmd, body = _match_block(text)

        # Reset numbered run if this utterance isn't a numbered command.
        if cmd != "numbered":
            numbered_run = 0

        if cmd:
            body = _apply_inline(body)
            body = _clean_fillers(body)
            if not body:
                continue

            if cmd == "heading_1":
                blocks.append(("heading", f"# {body}"))
            elif cmd == "heading_2":
                blocks.append(("heading", f"## {body}"))
            elif cmd == "heading_3":
                blocks.append(("heading", f"### {body}"))
            elif cmd == "bullet":
                blocks.append(("bullet", f"- {body}"))
            elif cmd == "numbered":
                numbered_run += 1
                blocks.append(("numbered", f"{numbered_run}. {body}"))
            elif cmd == "quote":
                blocks.append(("quote", f"> {body}"))
        else:
            cleaned = _apply_inline(text)
            cleaned = _clean_fillers(cleaned)
            if cleaned:
                blocks.append(("para", cleaned))

    # Join blocks with appropriate spacing.
    out = []
    for i, (kind, content) in enumerate(blocks):
        if kind == "linebrk":
            if out and out[-1] == "\n\n":
                out[-1] = "\n"
            continue
        if kind == "brk":
            if out and out[-1] != "\n\n":
                out.append("\n\n")
            continue

        if i > 0:
            prev_kind = blocks[i - 1][0]
            # Tight spacing for consecutive list items of the same kind.
            if kind in ("bullet", "numbered") and prev_kind == kind:
                out.append("\n")
            else:
                out.append("\n\n")
        out.append(content)

    result = "".join(out).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


if __name__ == "__main__":
    print(f"\nServer running at http://localhost:{PORT}")
    print("Open that URL in your browser.\n")
    app.run(host="127.0.0.1", port=PORT, debug=False)
