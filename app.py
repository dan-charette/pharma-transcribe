"""PharmaTranscribe AI - Streamlit application entry point."""

import io
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors

from src.audio_recorder import AudioConversionError, save_recording_as_mp3
from src.gemini_client import (
    FileProcessingError,
    delete_file,
    get_client,
    transcribe,
    upload_audio,
    wait_for_active,
)
from src.prompts import build_transcription_prompt
from src.utils import get_mime_type

# Load environment variables
load_dotenv()

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
        type=["mp3", "wav", "m4a", "mpeg"],
        help="Supported formats: MP3, WAV, M4A, MPEG (up to 200MB)",
    )

with tab_record:
    st.markdown("Record audio directly from your microphone.")

    audio_recording = st.audio_input(
        "Click to start recording",
        help="Click the microphone icon to start/stop recording",
    )

    if audio_recording:
        st.audio(audio_recording)
        recording_size = len(audio_recording.getvalue()) / 1024
        st.success(f"Recording captured ({recording_size:.1f} KB)")

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
            elif audio_source_type == "recording":
                status.update(label="Converting recording to MP3...")
                try:
                    temp_file_path = save_recording_as_mp3(audio_source.getvalue())
                except AudioConversionError as e:
                    st.error(f"Failed to process recording: {e}")
                    st.stop()

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

            # Stream transcription
            transcript_container = st.empty()
            full_text = ""

            for chunk in transcribe(client, gemini_file, prompt):
                full_text += chunk
                transcript_container.markdown(full_text)

            status.update(label="Complete!", state="complete")

        # Store transcript in session state for download
        st.session_state["transcript"] = full_text

        st.success("Transcription complete!")

    except (google_exceptions.InvalidArgument, genai_errors.ClientError) as e:
        st.error("Invalid API key or request. Please check your Google API key and try again.")
    except (google_exceptions.ResourceExhausted, genai_errors.APIError) as e:
        if "rate" in str(e).lower() or "quota" in str(e).lower():
            st.error("Rate limit exceeded. Please wait a few minutes and try again.")
        else:
            st.error(f"API error: {e}")
    except TimeoutError:
        st.error("Audio processing timed out. Try a smaller file or try again later.")
    except FileProcessingError as e:
        st.error(f"Could not process audio file. {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    finally:
        # Cleanup: Delete Gemini file
        if gemini_file and client:
            delete_file(client, gemini_file.name)

        # Cleanup: Delete local temp file
        if temp_file_path and os.path.exists(temp_file_path):
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
        # Generate PDF
        pdf = FPDF()
        pdf.set_margins(15, 15, 15)
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "PharmaTranscribe AI - Transcript", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(10)

        # Content
        pdf.set_font("Helvetica", "", 10)

        # Process transcript - handle long lines properly
        transcript_text = st.session_state["transcript"]
        # Replace problematic characters for PDF encoding
        safe_text = transcript_text.encode("latin-1", errors="replace").decode("latin-1")

        # Use multi_cell for the entire text to handle wrapping
        pdf.multi_cell(0, 5, safe_text)

        # Output PDF to bytes
        pdf_bytes = pdf.output()

        st.download_button(
            label="Download as PDF",
            data=bytes(pdf_bytes),
            file_name="transcript.pdf",
            mime="application/pdf",
        )
