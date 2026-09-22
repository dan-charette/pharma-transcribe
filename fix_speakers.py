"""Name the speakers on an existing transcript, from the text alone.

Command-line wrapper around src/speakers.py (which the app also runs after
each transcription). The model returns names only and Python rewrites the
labels, so the body text cannot be altered.

Use --by-voice for transcripts from gemini-3.5-transcribe, whose "Speaker N"
labels come from voice recognition (one label = one voice); the model then
maps each label to a person. Without it, every paragraph is named
individually, which suits prompt-generated labels that can't be trusted.

Usage:
    ./venv/bin/python fix_speakers.py transcripts/NAME.txt [--by-voice] [--model M] [--dry-run]
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.gemini_client import get_client
from src.speakers import MODEL, name_speakers, parse_paragraphs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--by-voice", action="store_true",
                    help="labels are voice-recognition labels (gemini-3.5-transcribe output)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set.", file=sys.stderr)
        return 1

    path = Path(args.transcript)
    original = path.read_text(encoding="utf-8")
    before = parse_paragraphs(original)
    if not before:
        print("No [HH:MM:SS] paragraphs found.", file=sys.stderr)
        return 1
    print(f"{len(before)} paragraphs -> {args.model} ({'by voice' if args.by_voice else 'per paragraph'})")

    out_text = name_speakers(get_client(api_key), original, by_voice=args.by_voice, model=args.model)
    after = parse_paragraphs(out_text)
    changed = sum(1 for (_, a, _), (_, b, _) in zip(before, after) if a != b)
    print(f"relabelled {changed}/{len(before)} paragraphs; body text verified identical")
    for name, n in Counter(label for _, label, _ in after).most_common():
        print(f"  {n:>4}  {name}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    backup = path.with_suffix(".prefix.txt")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"\noriginal preserved at {backup}")
    path.write_text(out_text, encoding="utf-8")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
