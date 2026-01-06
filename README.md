# PharmaTranscribe AI

Transcribe pharmaceutical earnings calls with domain-aware accuracy using Gemini AI.

## Overview

Standard speech-to-text models frequently misinterpret pharmaceutical terminology (drug names, chemical compounds, MOA acronyms). PharmaTranscribe solves this with a "Context Injection" system where you supply a keyword list that primes the model before transcription, significantly reducing phonetic hallucinations.

## Features

- Upload audio files up to 200MB (MP3, WAV, M4A, MPEG)
- Record audio directly from your microphone with **speaker audio capture** (captures audio playing through your speakers, perfect for transcribing earnings calls from another browser tab)
- Inject domain-specific keywords for improved accuracy
- Real-time streaming transcription display
- Download transcripts as text or PDF files

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pharma-transcribe.git
cd pharma-transcribe
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install system dependencies

For audio recording functionality, you need ffmpeg installed:

```bash
# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API key

Copy the example environment file and add your Google API key:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_api_key_here` with your actual key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 6. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## Usage

1. Choose input method:
   - **Upload File**: Select an existing audio file (MP3, WAV, M4A, or MPEG up to 200MB)
   - **Record Audio**: Click the record button to capture audio. This mode has echo cancellation disabled, so it will capture audio playing through your computer's speakers (e.g., an earnings call playing in another browser tab)
2. Enter domain keywords (drug names, tickers, technical terms) separated by commas
3. Click **Transcribe**
4. Download the completed transcript as TXT or PDF

## Testing

```bash
pytest tests/
```

## Project Structure

```
pharma-transcribe/
├── app.py                 # Streamlit entry point
├── src/
│   ├── audio_recorder.py  # Audio recording/conversion
│   ├── gemini_client.py   # Gemini API wrapper
│   ├── prompts.py         # Prompt templates
│   ├── utils.py           # File handling utilities
│   └── components/        # Custom Streamlit components
│       └── audio_recorder/  # Speaker audio capture component
├── tests/
│   ├── conftest.py        # Pytest fixtures
│   ├── test_audio_recorder.py  # Audio recording tests
│   ├── test_prompts.py    # Prompt tests
│   └── test_gemini.py     # Integration tests
├── requirements.txt
├── .env.example
└── README.md
```
