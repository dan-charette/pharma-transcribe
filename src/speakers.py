"""Name the speakers in a finished transcript, from the text alone.

Transcripts label speakers generically ("Speaker 1") or unreliably, but the
evidence for who is talking is usually in the words: self-introductions,
handoffs ("I'll turn it over to our CFO, Jane Smith"), and a moderator or
operator naming each analyst. The model reads the whole transcript and
returns names only; Python rewrites the labels, so the body text can never
be altered (this is asserted before returning).

Two modes:
- by_voice: labels come from acoustic diarization (gemini-3.5-transcribe),
  so each label is one voice throughout. The model maps each label to a
  person, which keeps the voice grouping and needs only a few decisions.
- per paragraph: labels were guessed by a prompt-driven model and can't be
  trusted, so the model names the speaker of every paragraph.

Returned names are only accepted if they are spoken somewhere in the
transcript (or are a role like "Operator"); anything else keeps its
original label rather than risk an invented name.
"""

import json
import re

from google import genai
from google.genai import types

from src.gemini_client import call_with_retries
from src.logging_config import get_logger

logger = get_logger("speakers")

MODEL = "gemini-2.5-flash"
PARA_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*([^:]{1,50}?):\s*(.*)$", re.DOTALL)
ROLE_NAMES = {"operator", "moderator", "host"}

_EVIDENCE = """EVIDENCE TO USE, in priority order:
1. Self-introductions are decisive: "I'm Jane Smith, Chief Financial Officer".
2. Handoffs name the person just finishing OR the one starting: "Thank you,
   Jane" means the PREVIOUS speaker was Jane. "I'll turn it over to John Doe"
   means the NEXT speaker is John Doe.
3. In Q&A, the operator or moderator names each analyst before their question;
   the answer comes from whichever executive is addressed or responds.
4. Continuity: a presenter usually holds the floor for many consecutive
   paragraphs.

RULES:
- Use each person's full name consistently, exactly as it is spoken in the
  transcript (e.g. "Jane Smith", not "Ms. Smith" or "Jane").
- A conference-call operator or event moderator who is never named may be
  called "Operator" or "Moderator".
- If a speaker genuinely cannot be determined, return an empty string. Never
  guess a name that is not supported by the transcript."""

VOICE_PROMPT = """You are identifying the speakers in a transcript. The speaker labels
("Speaker 1", "Speaker 2", ...) come from voice recognition: every paragraph with the
same label is the same voice. Your job is to decide WHICH PERSON each label is.

Below is every paragraph: its index, timestamp, voice label, and text.

""" + _EVIDENCE + """
- Return exactly one entry per distinct label.
- If one label clearly covers several different people (for example, several
  different analysts asking questions), return an empty string for it rather
  than naming only one of them.

PARAGRAPHS:
{digest}"""

PARAGRAPH_PROMPT = """You are correcting speaker attribution on a transcript. The words are
already correct. Your only job is to decide WHO SPEAKS each paragraph. The current
labels are unreliable: correct them freely (a long presentation wrongly split
across two names is a common error).

Below is every paragraph: its index, timestamp, current label, and text.

""" + _EVIDENCE + """
- Return an entry for EVERY index, in order. Do not skip any.
- Return the speaker name only. Never return the paragraph text.

PARAGRAPHS:
{digest}"""

VOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "voices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "name": {"type": "string"}},
                "required": ["label", "name"],
            },
        }
    },
    "required": ["voices"],
}

PARAGRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "speaker": {"type": "string"}},
                "required": ["index", "speaker"],
            },
        }
    },
    "required": ["speakers"],
}


def parse_paragraphs(text: str) -> list[tuple[str, str, str]]:
    """-> [(timestamp, label, body)] for each timestamped paragraph.

    Untimestamped blocks are continuations of the previous paragraph.
    """
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        m = PARA_RE.match(block)
        if m:
            out.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
        elif out:
            ts, label, body = out[-1]
            out[-1] = (ts, label, body + "\n\n" + block)
    return out


def _is_supported(name: str, transcript: str) -> bool:
    """A name is usable if it is spoken in the transcript, or is a role."""
    name = name.strip()
    if not name:
        return False
    if name.lower() in ROLE_NAMES:
        return True
    return name.lower() in transcript.lower()


def _ask(client: genai.Client, prompt: str, schema: dict, model: str) -> dict:
    response = call_with_retries(
        lambda: client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=32768,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        ),
        description="Gemini speaker-naming request",
    )
    return json.loads(response.text)


def name_speakers(client: genai.Client, text: str, by_voice: bool, model: str = MODEL) -> str:
    """Return the transcript with speaker labels replaced by names.

    Args:
        client: Gemini client instance
        text: Transcript in "[HH:MM:SS] Label: text" paragraphs
        by_voice: True when labels come from acoustic diarization (map each
            label to a person); False to name every paragraph individually
        model: Gemini model that judges the speakers

    Returns:
        The relabelled transcript. Unchanged if there is nothing to name.

    Raises:
        Any API error; RuntimeError if the body text would change.
    """
    paras = parse_paragraphs(text)
    if not paras:
        return text

    digest = "\n".join(
        f"{i}\t[{ts}]\t{label}\t{body.replace(chr(10), ' ')}"
        for i, (ts, label, body) in enumerate(paras)
    )
    bodies = "\n".join(body for _, _, body in paras)

    if by_voice:
        result = _ask(client, VOICE_PROMPT.format(digest=digest), VOICE_SCHEMA, model)
        voice_names = {}
        for entry in result.get("voices", []):
            label, name = entry.get("label", "").strip(), entry.get("name", "").strip()
            if _is_supported(name, bodies):
                voice_names[label] = name
            elif name:
                logger.warning("Rejected unsupported name %r for %s", name, label)
        names = [voice_names.get(label, label) for _, label, _ in paras]
    else:
        result = _ask(client, PARAGRAPH_PROMPT.format(digest=digest), PARAGRAPH_SCHEMA, model)
        by_index = {}
        for entry in result.get("speakers", []):
            name = entry.get("speaker", "").strip()
            if _is_supported(name, bodies):
                by_index[entry.get("index")] = name
            elif name:
                logger.warning("Rejected unsupported name %r for paragraph %s", name, entry.get("index"))
        names = [by_index.get(i, label) for i, (_, label, _) in enumerate(paras)]

    rebuilt = "\n\n".join(
        f"[{ts}] {name}: {body}" for (ts, _, body), name in zip(paras, names)
    ) + "\n\n"

    # The model returned labels only; prove the prose survived untouched.
    if [b for _, _, b in parse_paragraphs(rebuilt)] != [b for _, _, b in paras]:
        raise RuntimeError("speaker naming changed the body text")

    changed = sum(1 for (_, label, _), name in zip(paras, names) if name != label)
    logger.info("Named speakers: relabelled %d/%d paragraphs", changed, len(paras))
    return rebuilt
