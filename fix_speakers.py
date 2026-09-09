"""Repair speaker attribution on an existing transcript, from the text alone.

The transcription pass gets the words right but the speaker labels wrong: on
the Immunic R&D day it labelled Mike Panzara's section "Daniel Vitt" and left
36% of paragraphs as "Speaker 1"/"Speaker 3". Re-transcribing would risk the
verified body text and cost 3 audio requests.

Instead: the evidence needed to fix attribution is already in the transcript --
self-introductions ("I'm Mike Panzara, the Chief Medical Officer"), handoffs
("Thank you, Hella"), and the moderator naming each analyst in Q&A. So we send
a compact digest of the paragraphs (timestamp + current label + opening words)
and ask for a MAPPING ONLY.

The model judges who is speaking; Python rewrites the labels. The prose is
never sent back through the model, so the body text cannot be altered -- this
is asserted before the file is written.

Usage:
    ./venv/bin/python fix_speakers.py transcripts/NAME.txt [--model M] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types

from src.gemini_client import get_client
from src.logging_config import get_logger

logger = get_logger("fix_speakers")

DEFAULT_MODEL = "gemini-3.5-flash"
PARA_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*([^:]{1,50}?):\s*(.*)$", re.DOTALL)
DIGEST_CHARS = 200

PROMPT = """You are correcting speaker attribution on a transcript of the Immunic
Therapeutics Virtual Relapsing MS R&D Day. The words are already correct. Your only
job is to decide WHO SPEAKS each paragraph.

Below is every paragraph: its index, timestamp, the current (unreliable) label, and
its opening words.

EVIDENCE TO USE, in priority order:
1. Self-introductions are decisive. "I'm Mike Panzara, the Chief Medical Officer"
   means THAT paragraph and the ones following it are Mike Panzara, until a handoff.
2. Handoffs name the person just finishing OR the one starting: "Thank you, Hella"
   means the PREVIOUS speaker was Hella. "I'll turn it over to Stephen Krieger"
   means the NEXT speaker is Stephen Krieger.
3. In Q&A, the moderator names the analyst before their question; the analyst then
   speaks, and the answer comes from whichever executive is addressed.
4. Continuity: a presenter usually holds the floor for many consecutive paragraphs.
   Do not flip speakers back and forth without evidence.

RULES:
- Use each person's full name consistently, exactly as it is spoken in the
  transcript (e.g. "Mike Panzara", not "Dr. Panzara" or "Mike").
- The current labels are unreliable. Correct them freely. In particular, a long
  presentation wrongly split across two names is a common error to fix.
- If a paragraph's speaker genuinely cannot be determined, use "Unidentified
  Speaker". Do not guess a name to avoid saying unknown.
- Return an entry for EVERY index, in order. Do not skip any.
- Return the speaker name only. Never return the paragraph text.

PARAGRAPHS:
{digest}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "speaker": {"type": "string"},
                },
                "required": ["index", "speaker"],
            },
        }
    },
    "required": ["speakers"],
}


def parse(text: str) -> list[tuple[str, str, str]]:
    """-> [(timestamp, label, body)] for each timestamped paragraph."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        m = PARA_RE.match(block)
        if m:
            out.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
        elif out:  # continuation of the previous paragraph
            ts, lab, body = out[-1]
            out[-1] = (ts, lab, body + "\n\n" + block)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_dotenv("/Users/danielcharette/Git/Transcriber/.env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set.", file=sys.stderr)
        return 1

    path = Path(args.transcript)
    paras = parse(path.read_text(encoding="utf-8"))
    if not paras:
        print("No timestamped paragraphs found.", file=sys.stderr)
        return 1

    digest = "\n".join(
        f"{i}\t[{ts}]\t{lab}\t{body[:DIGEST_CHARS].replace(chr(10), ' ')}"
        for i, (ts, lab, body) in enumerate(paras)
    )
    print(f"{len(paras)} paragraphs, digest {len(digest):,} chars -> {args.model}")

    client = get_client(api_key)
    resp = client.models.generate_content(
        model=args.model,
        contents=[PROMPT.format(digest=digest)],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=32768,
            response_mime_type="application/json",
            response_schema=SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    mapping = {e["index"]: e["speaker"].strip()
               for e in json.loads(resp.text)["speakers"]}
    missing = [i for i in range(len(paras)) if i not in mapping]
    if missing:
        print(f"WARNING: {len(missing)} paragraphs had no mapping; keeping "
              f"their original labels (first few: {missing[:5]})")

    rebuilt, changed = [], 0
    for i, (ts, lab, body) in enumerate(paras):
        new = mapping.get(i, lab) or lab
        if new != lab:
            changed += 1
        rebuilt.append(f"[{ts}] {new}: {body}")
    out_text = "\n\n".join(rebuilt) + "\n"

    # The model returned labels only; prove the prose survived untouched.
    before = [b for _, _, b in paras]
    after = [b for _, _, b in parse(out_text)]
    assert before == after, "body text changed -- refusing to write"

    print(f"relabelled {changed}/{len(paras)} paragraphs; body text verified identical")
    from collections import Counter
    for name, n in Counter(mapping.values()).most_common():
        print(f"  {n:>4}  {name}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    backup = path.with_suffix(".prefix.txt")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\noriginal preserved at {backup}")
    path.write_text(out_text, encoding="utf-8")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
