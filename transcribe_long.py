"""Chunked transcription for long recordings (90 min+).

Why this exists: a single-pass request over 2 hours of audio makes
gemini-3.5-flash lose the thread. Observed on a 1:59:39 recording -- clean
output to ~00:34, then the model looped one Q&A block 28 times and drifted
its own clock to [09:41:00] (wall-clock, not elapsed) until it hit the
output cap. Raising max_output_tokens does not fix this; it just buys more
loop.

So: split the audio, transcribe each segment independently, and make the
timestamps deterministic.

Design decisions:
  * The model emits [MM:SS] RELATIVE to its own segment. Segments are <=20
    minutes, so MM:SS cannot overflow and the model is never asked to do
    arithmetic. Python adds the segment offset to produce absolute
    HH:MM:SS. Clock drift becomes structurally impossible.
  * Each segment is a fresh request, so a loop in one segment cannot
    poison the others, and is caught by the repetition guard below.
  * The speaker roster from earlier segments is injected into later ones
    so names stay consistent across cut points.

Usage:
    ./venv/bin/python transcribe_long.py AUDIO [--segment-seconds 1200] [--stem NAME]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from google.genai import errors as genai_errors
from google.genai import types

from src.gemini_client import get_client, upload_audio, wait_for_active, delete_file
from src.logging_config import get_logger
from src.storage import new_session_stem, save_transcript
from src.utils import get_mime_type, validate_audio_file
from transcribe_file import DEFAULT_KEYWORDS

logger = get_logger("transcribe_long")

MODEL = "gemini-3.5-flash"
# Standardized on a single model. If the model is saturated (503), the
# transcribe loop retries each segment with backoff rather than losing the
# run; a custom --model still falls back to this default.
MODEL_FALLBACKS = ["gemini-3.5-flash"]
MAX_OUTPUT_TOKENS = 65536
# 40 min ~= 77k input tokens: safely under a 250k tokens-per-minute cap,
# and only 3 requests for a 2-hour file against a 20-requests-per-day cap.
DEFAULT_SEGMENT_SECONDS = 2400
# Free-tier pacing: 5 requests/min means >=12s between calls.
MIN_REQUEST_INTERVAL = 13.0

TS_RE = re.compile(r"^\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]\s*", re.MULTILINE)

SEGMENT_PROMPT = """You are a professional transcriptionist specializing in pharmaceutical and biotech audio.

CRITICAL TERMINOLOGY LIST:
The following drug names, company names, and technical terms WILL appear in this audio.
When you hear something phonetically similar, you MUST use the exact spelling from this list:

{keywords}

CONTEXT:
This audio is segment {n} of {total} from a longer recording. It covers the
stretch from {start_label} to {end_label} of the full recording, but the audio
you are given starts at its own zero. It may begin or end mid-sentence; that is
expected -- transcribe the fragment as you hear it and do not try to complete it.
{speakers_block}
TRANSCRIPTION INSTRUCTIONS:
1. Transcribe this segment verbatim, from its first word to its last.
2. Format as: [MM:SS] Speaker: Text
3. Timestamps are ELAPSED TIME FROM THE START OF THIS SEGMENT ONLY. This
   segment begins at [00:00]. Never use the time of day. Never use the
   position in the full recording. Timestamps must increase monotonically
   and must not exceed [{max_mmss}].
4. Identify speakers where possible (use names if introduced, otherwise
   "Speaker 1:", "Speaker 2:", etc.).
5. Start a new timestamped paragraph when the speaker changes OR at a
   natural topic shift.
6. For unclear audio, use [inaudible] or [unclear: best guess].
7. Preserve numbers, percentages, p-values, dosages and financial figures
   exactly as spoken.
8. Transcribe each passage EXACTLY ONCE. Never repeat a paragraph you have
   already written. If you find yourself producing text you have already
   produced, stop and continue from the audio instead.
9. Do not summarize or abridge. Stop only when the segment's audio ends.

Begin transcription:"""


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def split_audio(audio_path: str, out_dir: Path, segment_seconds: int) -> list[Path]:
    """Split with stream copy -- no re-encode, so this is near-instant."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "part_%03d.mp3")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", audio_path,
         "-f", "segment", "-segment_time", str(segment_seconds), "-c", "copy", pattern],
        check=True,
    )
    return sorted(out_dir.glob("part_*.mp3"))


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def offset_timestamps(text: str, offset_seconds: int) -> str:
    """Rewrite segment-relative [MM:SS] / [H:MM:SS] into absolute [HH:MM:SS]."""
    def repl(m: re.Match) -> str:
        h = int(m.group(1) or 0)
        total = h * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + offset_seconds
        return f"[{hhmmss(total)}] "
    return TS_RE.sub(repl, text)


def repetition_ratio(text: str) -> tuple[float, str]:
    """Fraction of tagged paragraphs that are duplicates, plus worst offender."""
    paras = [
        TS_RE.sub("", line).strip()
        for line in text.splitlines()
        if line.startswith("[") and len(TS_RE.sub("", line).strip()) > 40
    ]
    if not paras:
        return 0.0, ""
    counts = Counter(paras)
    dupes = sum(c - 1 for c in counts.values() if c > 1)
    worst, worst_n = counts.most_common(1)[0]
    return dupes / len(paras), (worst[:70] if worst_n > 1 else "")


def known_speakers(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        m = re.match(r"^\[[0-9:]+\]\s*([A-Z][^:]{1,40}):", line)
        if m:
            name = m.group(1).strip()
            if name not in found and not name.lower().startswith("speaker "):
                found.append(name)
    return found


_last_request_at = 0.0


def _pace() -> None:
    """Sleep as needed so we never exceed the requests-per-minute allowance."""
    global _last_request_at
    wait = MIN_REQUEST_INTERVAL - (time.time() - _last_request_at)
    if wait > 0:
        print(f"  (pacing {wait:.0f}s for rate limit)", end="\r", flush=True)
        time.sleep(wait)
    _last_request_at = time.time()


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).upper()
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "QUOTA" in text


def _is_transient(exc: Exception) -> bool:
    """503/500/overload and socket trouble -- worth redoing the segment."""
    if isinstance(exc, (genai_errors.ServerError, ConnectionError, TimeoutError, OSError)):
        return True
    text = str(exc).upper()
    return any(k in text for k in ("503", "500", "UNAVAILABLE", "OVERLOADED", "INTERNAL"))


def _model_chain(primary: str) -> list[str]:
    return [primary] + [m for m in MODEL_FALLBACKS if m != primary]


def transcribe_segment(client, path: Path, n: int, total: int, offset: int,
                       seg_dur: float, keywords: str, speakers: list[str]) -> str:
    gemini_file = upload_audio(client, str(path), get_mime_type(str(path)))
    gemini_file = wait_for_active(client, gemini_file)

    speakers_block = ""
    if speakers:
        speakers_block = (
            "\nSpeakers already identified earlier in this recording -- reuse these\n"
            "exact names when you hear them again:\n"
            + "\n".join(f"- {s}" for s in speakers) + "\n"
        )

    prompt = SEGMENT_PROMPT.format(
        keywords="\n".join(f"- {k.strip()}" for k in keywords.replace("\n", ",").split(",") if k.strip()),
        n=n, total=total,
        start_label=hhmmss(offset), end_label=hhmmss(offset + seg_dur),
        max_mmss=mmss(seg_dur + 30),
        speakers_block=speakers_block,
    )

    config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    try:
        last_exc: Exception | None = None
        for model in _model_chain(MODEL):
            for attempt in range(3):
                _pace()
                chunks: list[str] = []
                try:
                    stream = client.models.generate_content_stream(
                        model=model, contents=[prompt, gemini_file], config=config)
                    for chunk in stream:
                        if chunk.text:
                            chunks.append(chunk.text)
                            print(f"\r  segment {n}/{total} [{model}]: "
                                  f"{sum(len(c) for c in chunks):,} chars",
                                  end="", flush=True)
                    return "".join(chunks)
                except Exception as exc:
                    # A stream cannot be resumed once broken; any retry redoes
                    # the whole segment, so partial `chunks` is discarded.
                    last_exc = exc
                    if not (_is_transient(exc) or _is_rate_limited(exc)):
                        raise
                    if attempt < 2:
                        delay = (30.0 if _is_rate_limited(exc) else 15.0) * (2 ** attempt)
                        kind = "rate limited" if _is_rate_limited(exc) else "unavailable"
                        print(f"\n  segment {n} {kind} on {model}; "
                              f"retry in {delay:.0f}s")
                        time.sleep(delay)
            logger.warning("Model %s failed for segment %d; trying next", model, n)
            print(f"\n  {model} unavailable for segment {n}; falling back")
        raise RuntimeError(f"segment {n}: all models failed") from last_exc
    finally:
        delete_file(client, gemini_file.name)


def main() -> int:
    global MODEL
    parser = argparse.ArgumentParser(description="Chunked transcription for long recordings.")
    parser.add_argument("audio")
    parser.add_argument("--segment-seconds", type=int, default=DEFAULT_SEGMENT_SECONDS)
    parser.add_argument("--keywords-file")
    parser.add_argument("--stem")
    parser.add_argument("--model", default=MODEL, help=f"default: {MODEL}")
    args = parser.parse_args()
    MODEL = args.model

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set (checked .env and environment).", file=sys.stderr)
        return 1

    audio_path = str(Path(args.audio).expanduser().resolve())
    validate_audio_file(audio_path, max_size_mb=4096)
    keywords = (Path(args.keywords_file).read_text(encoding="utf-8")
                if args.keywords_file else DEFAULT_KEYWORDS)
    stem = args.stem or new_session_stem()
    client = get_client(api_key)

    total_dur = duration_of(Path(audio_path))
    print(f"Source  : {audio_path}")
    print(f"Duration: {hhmmss(total_dur)}")
    print(f"Segments: {args.segment_seconds}s each")
    print(f"Model   : {MODEL}\n")

    work_dir = Path(tempfile.mkdtemp(prefix="transcribe_long_"))
    started = time.time()
    try:
        parts = split_audio(audio_path, work_dir, args.segment_seconds)
        print(f"Split into {len(parts)} segments.\n")

        pieces: list[str] = []
        offset = 0
        speakers: list[str] = []

        for i, part in enumerate(parts, start=1):
            seg_dur = duration_of(part)
            text = transcribe_segment(client, part, i, len(parts), offset,
                                      seg_dur, keywords, speakers)
            ratio, worst = repetition_ratio(text)
            flag = ""
            if ratio > 0.15:
                flag = f"  !! {ratio:.0%} repeated -- suspect loop: \"{worst}\""
                logger.warning("Segment %d repetition ratio %.2f", i, ratio)

            pieces.append(offset_timestamps(text.strip(), offset))
            speakers = known_speakers("\n".join(pieces))
            print(f"\r  segment {i}/{len(parts)}: {len(text):,} chars "
                  f"[{hhmmss(offset)}-{hhmmss(offset + seg_dur)}]{flag}")

            offset += int(round(seg_dur))
            save_transcript("\n\n".join(pieces), stem, partial=True)

        full_text = "\n\n".join(pieces)
        path = save_transcript(full_text, stem)

        ratio, _ = repetition_ratio(full_text)
        stamps = TS_RE.findall(full_text)
        last = stamps[-1] if stamps else None
        print(f"\nDone in {time.time() - started:.0f}s -> {path}")
        print(f"Words: {len(full_text.split()):,}   Repetition: {ratio:.1%}")
        if last:
            print(f"Last timestamp: [{int(last[0] or 0):02d}:{last[1]}:{last[2]}] "
                  f"(recording ends {hhmmss(total_dur)})")
        if speakers:
            print(f"Speakers: {', '.join(speakers[:12])}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
