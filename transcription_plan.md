# Revised Documents: PharmaTranscribe AI


# REVISED plan.md

```markdown
# Project Plan: PharmaTranscribe AI

## 1. Executive Summary

Build a web application that generates high-fidelity transcripts of pharmaceutical earnings calls (typically 1 hour, up to 200MB audio files). Standard speech-to-text models frequently misinterpret domain-specific terminology (drug names, chemical compounds, MOA acronyms).

**Core differentiator:** A "Context Injection" system where users supply a keyword list (drug names, tickers, technical terms) that primes the model before transcription, significantly reducing phonetic hallucinations.

## 2. Technical Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.10+ | Ecosystem, Streamlit compatibility |
| AI Engine | Gemini 2.5 Flash | Native audio ingestion, 1M token context window, cost-efficient |
| SDK | `google-generativeai` | Official Python client (pip install google-generativeai) |
| UI Framework | Streamlit | Rapid prototyping, native file upload, streaming text display |
| Config | `python-dotenv` | Environment variable management |
| Deployment | Localhost (MVP) | Docker-ready for future deployment |

## 3. Project Structure

```
pharma-transcribe/
├── app.py                 # Streamlit entry point (run with: streamlit run app.py)
├── src/
│   ├── __init__.py        # Makes src a package
│   ├── gemini_client.py   # Gemini API wrapper
│   ├── prompts.py         # Prompt templates
│   └── utils.py           # File handling utilities
├── tests/
│   ├── conftest.py        # Pytest fixtures and mocks
│   ├── test_prompts.py    # Unit tests for prompt construction
│   └── test_gemini.py     # Integration tests with mocked API
├── requirements.txt
├── .env.example
└── README.md
```

## 4. Core Functional Requirements

### 4.1 Audio Ingestion

- Accept audio files: MP3, WAV, M4A, MPEG (up to 200MB)
- **MUST use Gemini File API** for upload (not inline bytes)
- Implement state polling: wait until file status = `ACTIVE` before generation
- MIME type mapping:
  - `.mp3` → `audio/mpeg`
  - `.wav` → `audio/wav`
  - `.m4a` → `audio/mp4`
  - `.mpeg` → `audio/mpeg`

### 4.2 Context Priming System

Users input keywords via:
- Text area (comma-separated values), OR
- CSV file upload (single column)

Keywords are injected into the system prompt to resolve phonetic ambiguity.

### 4.3 Transcription Engine

**Model:** `gemini-3.5-flash`

**Generation Config:**
```python
generation_config = {
    "temperature": 0.1,          # Low for accuracy
    "max_output_tokens": 32768,  # ~25,000 words capacity
}
```

**Streaming:** Enabled for real-time UI feedback during long transcriptions.

### 4.4 Prompt Template

```
SYSTEM PROMPT:
You are a professional transcriptionist specializing in pharmaceutical and biotech earnings calls.

CRITICAL TERMINOLOGY LIST:
The following drug names, company names, and technical terms WILL appear in this audio.
When you hear something phonetically similar, you MUST use the exact spelling from this list:

{keywords}

TRANSCRIPTION INSTRUCTIONS:
1. Transcribe the entire audio verbatim
2. Preserve speaker turns where detectable (mark as "Speaker 1:", "Speaker 2:", etc. or use names if introduced)
3. Include filler words only if they affect meaning
4. For unclear audio, use [inaudible] or [unclear: best guess]
5. Preserve numbers, percentages, and financial figures exactly as spoken
6. Do not summarize or omit any content

Begin transcription:
```

### 4.5 Lifecycle Management

1. Upload audio to Gemini File API
2. Poll for `ACTIVE` status (timeout: 300 seconds, poll interval: 3 seconds)
3. Generate transcript with streaming
4. **Cleanup:** Delete file from Gemini storage after completion (both success and failure paths)
5. **Cleanup:** Delete local temp file

### 4.6 Error Handling

| Error | Handling |
|-------|----------|
| `google.api_core.exceptions.InvalidArgument` | Invalid API key or malformed request → show user error |
| `google.api_core.exceptions.ResourceExhausted` | Rate limit hit → show "try again in a few minutes" |
| `TimeoutError` (custom) | File didn't become ACTIVE in 120s → show upload error |
| File state = `FAILED` | Gemini couldn't process file → show format/corruption error |

## 5. Testing Requirements

### 5.1 Unit Tests

- `test_prompts.py`: Verify keyword injection formats correctly
- `test_utils.py`: Verify MIME type mapping, file extension validation

### 5.2 Integration Tests (Mocked)

Mock the `genai.Client` to avoid API costs:

```python
# Expected mock responses
MOCK_FILE_RESPONSE = {
    "name": "files/abc123",
    "state": "ACTIVE",
    "uri": "https://generativelanguage.googleapis.com/v1/files/abc123"
}

MOCK_GENERATION_RESPONSE = {
    "text": "Good morning everyone, welcome to the Q3 earnings call for Vertex Pharmaceuticals..."
}
```

### 5.3 User Acceptance Criteria

1. Successfully transcribe a 5-minute sample with 3+ drug names from keyword list
2. Handle empty keyword list without crashing (proceeds with generic transcription)
3. Display meaningful error if API key is invalid
4. Clean up both remote and local files after completion
```

---

# REVISED tasks.md

```markdown
# Task List: PharmaTranscribe AI

## Phase 1: Project Setup

- [ ] **Create project directory structure**
  ```
  mkdir -p pharma-transcribe/{src,tests}
  cd pharma-transcribe
  touch app.py src/__init__.py src/gemini_client.py src/prompts.py src/utils.py
  touch tests/conftest.py tests/test_prompts.py tests/test_gemini.py
  ```

- [ ] **Create requirements.txt**
  ```
  streamlit>=1.28.0
  google-generativeai>=0.8.0
  python-dotenv>=1.0.0
  pytest>=7.4.0
  pytest-mock>=3.12.0
  ```

- [ ] **Create .env.example**
  ```
  GOOGLE_API_KEY=your_api_key_here
  ```

- [ ] **Create .gitignore**
  ```
  .env
  __pycache__/
  *.pyc
  .pytest_cache/
  ```

## Phase 2: Core Logic

### 2.1 Utility Functions (`src/utils.py`)

- [ ] **Implement MIME type mapping**
  ```python
  def get_mime_type(file_path: str) -> str:
      """Map file extension to MIME type. Raise ValueError for unsupported types."""
      # Supported: .mp3 → audio/mpeg, .wav → audio/wav, .m4a → audio/mp4, .mpeg → audio/mpeg
  ```

- [ ] **Implement file validation**
  ```python
  def validate_audio_file(file_path: str, max_size_mb: int = 200) -> bool:
      """Check file exists, has valid extension, and is under size limit."""
  ```

### 2.2 Prompt Construction (`src/prompts.py`)

- [ ] **Implement prompt builder**
  ```python
  SYSTEM_PROMPT_TEMPLATE = '''You are a professional transcriptionist...'''  # Full template from plan.md

  def build_transcription_prompt(keywords: str) -> str:
      """
      Args:
          keywords: Comma-separated string of domain terms (can be empty)
      Returns:
          Formatted system prompt with keywords injected
      """
  ```

### 2.3 Gemini Client (`src/gemini_client.py`)

- [ ] **Initialize client**
  ```python
  import google.generativeai as genai
  from google.api_core import exceptions as google_exceptions

  def get_client(api_key: str) -> genai.Client:
      """Initialize and return Gemini client."""
  ```

- [ ] **Implement file upload**
  ```python
  def upload_audio(client: genai.Client, file_path: str, mime_type: str) -> genai.File:
      """
      Upload file to Gemini File API.
      Returns: File object with .name and .state attributes
      Raises: google_exceptions.InvalidArgument if API key invalid
      """
  ```

- [ ] **Implement state polling**
  ```python
  class FileProcessingError(Exception):
      """Raised when file processing fails on Gemini side."""

  def wait_for_active(client: genai.Client, file_name: str, timeout_seconds: int = 120, poll_interval: int = 3) -> genai.File:
      """
      Poll file status until ACTIVE.
      Raises:
          TimeoutError if not ACTIVE within timeout
          FileProcessingError if state becomes FAILED
      """
  ```

- [ ] **Implement transcription**
  ```python
  def transcribe(client: genai.Client, file: genai.File, system_prompt: str) -> Iterator[str]:
      """
      Generate transcript using gemini-3.5-flash with streaming.
      Args:
          file: Active Gemini File object
          system_prompt: Formatted prompt from prompts.py
      Yields: Text chunks as they stream in
      """
      model = client.get_model("gemini-3.5-flash")
      # Use generation_config: temperature=0.1, max_output_tokens=32768
  ```

- [ ] **Implement cleanup**
  ```python
  def delete_file(client: genai.Client, file_name: str) -> bool:
      """
      Delete file from Gemini storage.
      Returns: True if deleted, False if file not found (already deleted)
      Silently handles errors to not interrupt user flow.
      """
  ```

## Phase 3: Streamlit UI (`app.py`)

- [ ] **Basic page setup**
  ```python
  import streamlit as st
  st.set_page_config(page_title="PharmaTranscribe AI", page_icon="💊")
  st.title("PharmaTranscribe AI")
  ```

- [ ] **API key handling**
  - Check for `GOOGLE_API_KEY` in environment
  - If missing, show `st.text_input` in sidebar (type="password")
  - Store in `st.session_state`

- [ ] **Input section**
  - `st.file_uploader`: Accept `["mp3", "wav", "m4a", "mpeg"]`, label="Upload Earnings Call Audio"
  - `st.text_area`: Label="Context Keywords", placeholder="Enter drug names, tickers, separated by commas (e.g., Keytruda, VRTX, pembrolizumab)"

- [ ] **Transcribe button and orchestration**
  ```python
  if st.button("Transcribe", type="primary"):
      # 1. Validate inputs
      # 2. Write uploaded file to tempfile.NamedTemporaryFile(delete=False)
      # 3. try:
      #        with st.status("Processing...") as status:
      #            status.update(label="Uploading audio...")
      #            file = upload_audio(...)
      #            status.update(label="Processing audio...")
      #            file = wait_for_active(...)
      #            status.update(label="Generating transcript...")
      #            transcript_container = st.empty()
      #            full_text = ""
      #            for chunk in transcribe(...):
      #                full_text += chunk
      #                transcript_container.markdown(full_text)
      #            status.update(label="Complete!", state="complete")
      #    finally:
      #        delete_file(...)  # Always cleanup Gemini file
      #        os.unlink(temp_file_path)  # Always cleanup local file
  ```

- [ ] **Error display**
  - Wrap orchestration in try/except
  - Catch `google_exceptions.InvalidArgument` → `st.error("Invalid API key")`
  - Catch `google_exceptions.ResourceExhausted` → `st.error("Rate limit exceeded. Wait a few minutes.")`
  - Catch `TimeoutError` → `st.error("Audio processing timed out. Try a smaller file.")`
  - Catch `FileProcessingError` → `st.error("Could not process audio. Check file format.")`

- [ ] **Download button**
  - After successful transcription, show `st.download_button` for .txt export

## Phase 4: Testing

### 4.1 Fixtures (`tests/conftest.py`)

- [ ] **Create mock fixtures**
  ```python
  import pytest
  from unittest.mock import Mock, MagicMock

  @pytest.fixture
  def mock_genai_client():
      """Return a mocked genai.Client"""

  @pytest.fixture
  def mock_active_file():
      """Return a mock File object with state=ACTIVE"""
      file = Mock()
      file.name = "files/test123"
      file.state.name = "ACTIVE"
      return file
  ```

### 4.2 Prompt Tests (`tests/test_prompts.py`)

- [ ] **Test keyword injection**
  ```python
  def test_build_prompt_with_keywords():
      result = build_transcription_prompt("Keytruda, pembrolizumab, VRTX")
      assert "Keytruda" in result
      assert "pembrolizumab" in result
      assert "CRITICAL TERMINOLOGY LIST" in result

  def test_build_prompt_empty_keywords():
      result = build_transcription_prompt("")
      # Should still return valid prompt, just with empty keyword section
      assert "TRANSCRIPTION INSTRUCTIONS" in result
  ```

### 4.3 Integration Tests (`tests/test_gemini.py`)

- [ ] **Test upload flow**
  ```python
  def test_upload_calls_api(mock_genai_client):
      # Verify upload_audio calls client.files.upload with correct params

  def test_wait_for_active_success(mock_genai_client, mock_active_file):
      # Verify immediate return when file is already ACTIVE

  def test_wait_for_active_timeout(mock_genai_client):
      # Verify TimeoutError raised after timeout

  def test_cleanup_always_called(mock_genai_client):
      # Verify delete_file called even if transcription fails
  ```

## Phase 5: Documentation

- [ ] **Create README.md**
  ```markdown
  # PharmaTranscribe AI

  Transcribe pharmaceutical earnings calls with domain-aware accuracy.

  ## Setup
  1. `pip install -r requirements.txt`
  2. Copy `.env.example` to `.env` and add your Google API key
  3. `streamlit run app.py`

  ## Usage
  1. Upload an audio file (MP3, WAV, M4A up to 200MB)
  2. Enter domain keywords (drug names, tickers) for improved accuracy
  3. Click Transcribe

  ## Testing
  `pytest tests/`
  ```
```