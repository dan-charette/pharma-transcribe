"""Headless transcription of a long audio file already on disk.

The Streamlit app is tuned for ~1-hour earnings calls uploaded through the
browser. This CLI is for long files (2h+) that already exist on disk:

  * no browser in the critical path, no triple-buffering of the bytes
  * output cap raised to the model maximum, since a 2-hour transcript
    overruns the app's 32k default
  * thinking disabled -- on 2.5 models it shares the output budget with
    the transcript itself, and transcription needs no reasoning
  * HH:MM:SS timestamps, which the app's MM:SS format cannot express
    past 59:59

Reuses src/gemini_client.py and src/storage.py unchanged.

Usage:
    ./venv/bin/python transcribe_file.py AUDIO [--keywords-file FILE] [--stem NAME]
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types

from src.gemini_client import get_client, upload_audio, wait_for_active, delete_file
from src.logging_config import get_logger
from src.storage import new_session_stem, save_transcript
from src.utils import get_mime_type, validate_audio_file

logger = get_logger("transcribe_cli")

MODEL = "gemini-3.5-flash"
# 3.5 Flash tops out here; the app's 32768 truncates a 2-hour transcript.
MAX_OUTPUT_TOKENS = 65536
# Checkpoint cadence, matching the app's behaviour.
CHECKPOINT_CHARS = 2000

PROMPT_TEMPLATE = """You are a professional transcriptionist specializing in pharmaceutical and biotech audio.

CRITICAL TERMINOLOGY LIST:
The following drug names, company names, and technical terms WILL appear in this audio.
When you hear something phonetically similar, you MUST use the exact spelling from this list:

{keywords}

TRANSCRIPTION INSTRUCTIONS:
1. Transcribe the entire audio verbatim, from the first word to the last.
2. Format as: [HH:MM:SS] Speaker: Text
3. This recording is over two hours long. Timestamps MUST use three fields
   (hours:minutes:seconds) and must keep increasing past [01:00:00] all the
   way to the end. Never reset the clock and never drop the hours field.
4. Identify speakers where possible (use names if introduced, otherwise
   "Speaker 1:", "Speaker 2:", etc.). Presenters usually introduce themselves.
5. Start a new timestamped paragraph when the speaker changes OR at a natural
   topic shift.
6. For unclear audio, use [inaudible] or [unclear: best guess].
7. Preserve numbers, percentages, p-values, dosages and financial figures
   exactly as spoken.
8. Do not summarize, abridge, or skip sections. If the audio is long, keep
   going until you reach the end of the recording.

Example format:
[00:00:12] Operator: Good morning and welcome. We will begin with opening remarks, followed by a Q&A session.

[00:01:47] Dr. Vitt: Thank you. Today we will review the Phase 3 data for vidofludimus calcium in relapsing multiple sclerosis.

[01:04:30] Analyst: A question on the EDSS endpoint -- can you walk through the confirmed disability progression numbers?

Begin transcription:"""

# Sensible default for this engagement; override with --keywords-file.
#
# NEVER put people's names in here. The prompt tells the model these terms
# "WILL appear in this audio" and to override what it hears -- which is right
# for drug names (phonetic near-misses) and catastrophic for names. Seeding
# "Daniel Vitt" made the model print it over an actual "Erik Lundgren": not a
# mishearing, an instructed substitution. Speaker names must come from the
# audio, and attribution is repaired afterwards by fix_speakers.py.
DEFAULT_KEYWORDS = """Immunic, Immunic Therapeutics, IMU-838, vidofludimus calcium, IMU-856, IMU-381,
ENSURE-1, ENSURE-2, CALLIPER, EMPhASIS, ENTRANCE,
DHODH, dihydroorotate dehydrogenase, Nrf2, teriflunomide, Aubagio, leflunomide,
relapsing multiple sclerosis, RMS, RRMS, SPMS, PPMS, progressive multiple sclerosis,
clinically isolated syndrome, CIS, PIRA, RAW, smouldering MS, chronic active lesions,
EDSS, Expanded Disability Status Scale, confirmed disability worsening, CDW,
24-week CDW, T25FW, nine-hole peg test, SDMT, annualized relapse rate, ARR,
Gd-enhancing lesions, gadolinium, T1 lesions, T2 lesions, PRL, paramagnetic rim lesions,
brain volume loss, thalamic volume, neurofilament light, NfL, GFAP,
ocrelizumab, Ocrevus, Ocrevus Zunovo, ublituximab, Briumvi, ofatumumab, Kesimpta,
natalizumab, Tysabri, Tyruko, alemtuzumab, Lemtrada, cladribine, Mavenclad,
fingolimod, Gilenya, siponimod, Mayzent, ozanimod, Zeposia, ponesimod, Ponvory,
dimethyl fumarate, Tecfidera, diroximel fumarate, Vumerity, glatiramer acetate, Copaxone,
tolebrutinib, fenebrutinib, remibrutinib, evobrutinib, orelabrutinib, BTK inhibitor,
frexalimab, CD40L, Sanofi, Roche, Genentech, Novartis, Biogen, TG Therapeutics, Merck KGaA,
FDA, EMA, CHMP, NDA, MAA, Type B meeting, breakthrough therapy, accelerated approval,
topline, readout, primary endpoint, secondary endpoint, hazard ratio, confidence interval,
p-value, statistical significance, event-driven, interim analysis, DSMB, open-label extension"""


def build_prompt(keywords: str) -> str:
    items = [k.strip() for k in keywords.replace("\n", ",").split(",") if k.strip()]
    formatted = "\n".join(f"- {k}" for k in items) if items else "(No specific terminology provided)"
    return PROMPT_TEMPLATE.format(keywords=formatted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe a long audio file via Gemini.")
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument("--keywords-file", help="File of comma/newline separated domain terms")
    parser.add_argument("--stem", help="Output filename stem (default: session_<timestamp>)")
    parser.add_argument("--keep-remote", action="store_true", help="Do not delete the uploaded file from Gemini")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set (checked .env and environment).", file=sys.stderr)
        return 1

    audio_path = str(Path(args.audio).expanduser().resolve())
    validate_audio_file(audio_path)
    mime_type = get_mime_type(audio_path)
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    keywords = DEFAULT_KEYWORDS
    if args.keywords_file:
        keywords = Path(args.keywords_file).read_text(encoding="utf-8")

    stem = args.stem or new_session_stem()
    client = get_client(api_key)

    print(f"Source     : {audio_path}")
    print(f"Size / MIME: {size_mb:.1f} MB / {mime_type}")
    print(f"Model      : {MODEL} (max_output_tokens={MAX_OUTPUT_TOKENS}, thinking off)")
    print(f"Output stem: {stem}\n")

    started = time.time()
    print("Uploading to Gemini File API...")
    gemini_file = upload_audio(client, audio_path, mime_type)

    print("Waiting for server-side processing...")
    gemini_file = wait_for_active(client, gemini_file)
    print(f"File active after {time.time() - started:.0f}s. Starting transcription.\n")

    chunks: list[str] = []
    chars_at_last_save = 0
    transcript_path = None

    try:
        stream = client.models.generate_content_stream(
            model=MODEL,
            contents=[build_prompt(keywords), gemini_file],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                # Thinking shares the output budget on 2.5 models; a
                # transcript needs none of it.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        for chunk in stream:
            if not chunk.text:
                continue
            chunks.append(chunk.text)
            total = sum(len(c) for c in chunks)
            if total - chars_at_last_save >= CHECKPOINT_CHARS:
                save_transcript("".join(chunks), stem, partial=True)
                chars_at_last_save = total
                print(f"\r  {total:,} chars written...", end="", flush=True)

        full_text = "".join(chunks)
        transcript_path = save_transcript(full_text, stem)
        print(f"\r  {len(full_text):,} chars written.   \n")
        print(f"Done in {time.time() - started:.0f}s -> {transcript_path}")

        tail = full_text.rstrip()[-200:]
        print(f"\nLast 200 chars (check it reaches ~01:59:xx):\n...{tail}")

    except BaseException:
        if chunks:
            partial = save_transcript("".join(chunks), stem, partial=True)
            print(f"\nInterrupted. Partial transcript preserved at {partial}", file=sys.stderr)
        raise
    finally:
        if not args.keep_remote:
            delete_file(client, gemini_file.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
