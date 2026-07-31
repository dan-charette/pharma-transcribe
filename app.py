"""PharmaTranscribe AI - Streamlit application entry point."""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors

from src.audio_recorder import AudioConversionError, convert_file_to_mp3
from src.components.audio_recorder import audio_recorder
from src.gemini_client import (
    FileProcessingError,
    delete_file,
    get_client,
    transcribe,
    upload_audio,
    wait_for_active,
)
from src.logging_config import setup_logging
from src.prompts import build_transcription_prompt
from src.storage import (
    list_recordings,
    list_transcripts,
    new_session_stem,
    recording_fingerprint,
    save_recording,
    save_transcript,
)
from src.utils import get_mime_type

logger = setup_logging()

# Load environment variables
load_dotenv()

# How many characters of new transcript text trigger a partial checkpoint
TRANSCRIPT_CHECKPOINT_CHARS = 2000


def build_transcript_pdf(transcript_text: str) -> bytes:
    """Render the transcript into PDF bytes."""
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "PharmaTranscribe AI - Transcript", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 10)
    safe_text = transcript_text.encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 5, safe_text)

    return bytes(pdf.output())


# Page configuration
st.set_page_config(page_title="PharmaTranscribe AI", page_icon="💊", layout="wide")

st.title("PharmaTranscribe AI")
st.markdown("Transcribe pharmaceutical earnings calls with domain-aware accuracy.")

# --- Sidebar: API Key Handling ---
with st.sidebar:
    st.header("Configuration")

    # Check for API key in environment first
    env_api_key = os.getenv("GOOGLE_API_KEY")

    if env_api_key:
        st.success("API key loaded from environment")
        api_key = env_api_key
    else:
        api_key = st.text_input(
            "Google API Key",
            type="password",
            placeholder="Enter your Gemini API key",
            help="Get your API key from https://makersuite.google.com/app/apikey",
        )
        if not api_key:
            st.warning("Please enter your API key to continue")

# --- Main Content ---
if not api_key:
    st.info("Please configure your Google API key in the sidebar to get started.")
    st.stop()

# Store API key in session state
st.session_state["api_key"] = api_key

# --- Input Section ---
st.header("Audio Input")

# Create tabs for input methods
tab_upload, tab_record = st.tabs(["Upload File", "Record Audio"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload Earnings Call Audio",
        type=["mp3", "wav", "m4a", "mpeg", "webm"],
        help="Supported formats: MP3, WAV, M4A, MPEG, WebM (up to 200MB)",
    )

with tab_record:
    st.markdown("Record audio directly from your microphone.")
    st.caption("Speaker audio capture enabled (echo cancellation disabled)")

    audio_recording = audio_recorder(key="audio_recorder")

    if audio_recording:
        audio_bytes = audio_recording.getvalue()
        fingerprint = recording_fingerprint(audio_bytes)

        # Persist each new recording to disk immediately so no crash can lose it
        if st.session_state.get("recording_fingerprint") != fingerprint:
            stem = new_session_stem()
            try:
                saved_path = save_recording(audio_bytes, audio_recording.format, stem=stem)
            except OSError as e:
                logger.exception("Failed to save recording to disk")
                st.error(f"Could not save recording to disk: {e}")
                saved_path = None
            if saved_path:
                st.session_state["recording_fingerprint"] = fingerprint
                st.session_state["recording_path"] = str(saved_path)
                st.session_state["session_stem"] = stem
                st.session_state.pop("mp3_path", None)

        recording_path = st.session_state.get("recording_path")
        if recording_path:
            size_mb = len(audio_bytes) / (1024 * 1024)
            st.success(f"Recording saved to `{recording_path}` ({size_mb:.1f} MB)")

            # Convert to MP3 once per recording; remember failure so we don't
            # retry (and re-spike memory/CPU) on every rerun.
            if "mp3_path" not in st.session_state:
                try:
                    with st.spinner("Preparing MP3..."):
                        mp3_path = convert_file_to_mp3(Path(recording_path))
                    st.session_state["mp3_path"] = str(mp3_path)
                except AudioConversionError as e:
                    st.session_state["mp3_path"] = None
                    logger.error("MP3 conversion failed: %s", e)

            mp3_path = st.session_state.get("mp3_path")
            if mp3_path and Path(mp3_path).exists():
                st.download_button(
                    label="Download Recording as MP3",
                    data=Path(mp3_path).read_bytes(),
                    file_name=Path(mp3_path).name,
                    mime="audio/mpeg",
                )
            elif mp3_path is None and "mp3_path" in st.session_state:
                st.warning(
                    "Could not prepare MP3 download; the original recording "
                    f"is safe at `{recording_path}`."
                )

# Determine which audio source to use
audio_source = None
audio_source_type = None

if uploaded_file:
    audio_source = uploaded_file
    audio_source_type = "upload"
elif audio_recording:
    audio_source = audio_recording
    audio_source_type = "recording"

st.header("Context Keywords")

keywords = st.text_area(
    "Enter domain-specific terminology",
    placeholder="Enter drug names, tickers, separated by commas (e.g., Keytruda, VRTX, pembrolizumab)",
    help="These terms will be used to improve transcription accuracy for pharmaceutical terminology",
)

# --- Transcription ---
if st.button("Transcribe", type="primary", disabled=audio_source is None):
    if audio_source is None:
        st.error("Please upload an audio file or record audio first.")
        st.stop()

    # Initialize variables for cleanup
    client = None
    gemini_file = None
    temp_file_path = None
    owns_temp_file = False
    completed = False
    chunks = []

    # Recordings reuse their save-time stem so the transcript pairs with the
    # audio file; uploads get a fresh stem per transcription run.
    if audio_source_type == "recording":
        stem = st.session_state.get("session_stem") or new_session_stem()
        st.session_state["session_stem"] = stem
    else:
        stem = new_session_stem()

    try:
        with st.status("Processing...", expanded=True) as status:
            # Prepare audio file based on source type
            if audio_source_type == "upload":
                status.update(label="Preparing uploaded file...")
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_file_path = tmp.name
                owns_temp_file = True
                logger.info(
                    "Upload received: %s (%d bytes)", uploaded_file.name, uploaded_file.size
                )
            elif audio_source_type == "recording":
                status.update(label="Preparing recording...")
                # The recording is already durably on disk -- reuse it directly
                temp_file_path = st.session_state["recording_path"]

            # Get MIME type and initialize client
            mime_type = get_mime_type(temp_file_path)
            client = get_client(api_key)

            # Upload to Gemini
            status.update(label="Uploading audio to Gemini...")
            gemini_file = upload_audio(client, temp_file_path, mime_type)

            # Wait for processing
            status.update(label="Processing audio (this may take a few minutes)...")
            gemini_file = wait_for_active(client, gemini_file)

            # Build prompt and transcribe
            status.update(label="Generating transcript...")
            prompt = build_transcription_prompt(keywords)

            # Stream transcription, checkpointing partial text to disk
            transcript_container = st.empty()
            chars_at_last_save = 0

            for chunk in transcribe(client, gemini_file, prompt):
                chunks.append(chunk)
                text_so_far = "".join(chunks)
                transcript_container.markdown(text_so_far)
                if len(text_so_far) - chars_at_last_save >= TRANSCRIPT_CHECKPOINT_CHARS:
                    save_transcript(text_so_far, stem, partial=True)
                    chars_at_last_save = len(text_so_far)

            full_text = "".join(chunks)
            transcript_path = save_transcript(full_text, stem)
            completed = True
            logger.info("Transcription complete: %d chars", len(full_text))

            status.update(label="Complete!", state="complete")

        # Store transcript in session state for download
        st.session_state["transcript"] = full_text
        st.session_state.pop("transcript_pdf", None)

        st.success(f"Transcription complete! Saved to `{transcript_path}`")

        # Display transcript outside the status container so it's visible after status collapses
        if full_text:
            st.subheader("Transcript")
            st.markdown(full_text)

    except (google_exceptions.InvalidArgument, genai_errors.ClientError):
        logger.exception("Transcription failed: invalid key or request")
        st.error("Invalid API key or request. Please check your Google API key and try again.")
    except (google_exceptions.ResourceExhausted, genai_errors.APIError) as e:
        logger.exception("Transcription failed: API error")
        if "rate" in str(e).lower() or "quota" in str(e).lower():
            st.error("Rate limit exceeded. Please wait a few minutes and try again.")
        else:
            st.error(f"API error: {e}")
    except TimeoutError:
        logger.exception("Transcription failed: processing timed out")
        st.error("Audio processing timed out. Try a smaller file or try again later.")
    except FileProcessingError as e:
        logger.exception("Transcription failed: file processing error")
        st.error(f"Could not process audio file. {e}")
    except Exception as e:
        logger.exception("Transcription failed: unexpected error")
        st.error(f"An unexpected error occurred: {e}")

    finally:
        # Preserve any partial transcript before anything else
        if chunks and not completed:
            try:
                partial_path = save_transcript("".join(chunks), stem, partial=True)
                logger.warning("Preserved partial transcript at %s", partial_path)
                st.warning(f"A partial transcript was saved to `{partial_path}`.")
            except OSError:
                logger.exception("Failed to preserve partial transcript")

        # Cleanup: Delete Gemini file
        if gemini_file and client:
            delete_file(client, gemini_file.name)

        # Cleanup: Delete local temp file (uploads only -- never the saved recording)
        if owns_temp_file and temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# --- Download Buttons ---
if "transcript" in st.session_state and st.session_state["transcript"]:
    st.divider()
    st.subheader("Download Transcript")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download as TXT",
            data=st.session_state["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )

    with col2:
        # Generate the PDF once per transcript, not on every rerun
        if "transcript_pdf" not in st.session_state:
            st.session_state["transcript_pdf"] = build_transcript_pdf(
                st.session_state["transcript"]
            )

        st.download_button(
            label="Download as PDF",
            data=st.session_state["transcript_pdf"],
            file_name="transcript.pdf",
            mime="application/pdf",
        )

# --- Saved Sessions ---
st.divider()
with st.expander("Saved sessions on disk"):
    st.caption(
        "Recordings and transcripts are saved automatically and survive "
        "crashes and restarts."
    )
    recordings = list_recordings()[:10]
    transcripts = list_transcripts()[:10]

    if not recordings and not transcripts:
        st.write("No saved sessions yet.")

    if recordings:
        st.markdown("**Recordings**")
        for rec in recordings:
            rec_mb = rec.stat().st_size / (1024 * 1024)
            st.markdown(f"- `{rec}` ({rec_mb:.1f} MB)")

    if transcripts:
        st.markdown("**Transcripts**")
        for i, txt in enumerate(transcripts):
            entry_col, btn_col = st.columns([4, 1])
            entry_col.markdown(f"- `{txt.name}`")
            btn_col.download_button(
                "Download",
                data=txt.read_text(encoding="utf-8"),
                file_name=txt.name,
                key=f"session_dl_{i}",
            )
