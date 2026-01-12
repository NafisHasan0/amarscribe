"""
AmarScribe Ultimate - World's Most Accurate Bengali Transcription Tool
======================================================================
A Streamlit application using Soniox API for high-accuracy Bengali transcription
with speaker diarization and detailed segment-level timestamps.
"""

import streamlit as st
import io
import tempfile
import os
import time
import re
import requests
from datetime import timedelta

# MoviePy for video processing
from moviepy.editor import VideoFileClip


# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="AmarScribe Ultimate - Bengali Transcription",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Transcript segment styling */
    .transcript-segment {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        border-left: 4px solid #2d5a87;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .transcript-segment:hover {
        transform: translateX(5px);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }
    
    .timestamp {
        color: #2d5a87;
        font-weight: bold;
        font-family: 'Monaco', 'Consolas', monospace;
        background: #e8f4fc;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    
    .speaker-label {
        color: #1e3a5f;
        font-weight: 600;
        margin-left: 10px;
        background: #d4edda;
        padding: 3px 10px;
        border-radius: 5px;
        font-size: 0.85rem;
    }
    
    .bengali-text {
        font-size: 1.2rem;
        line-height: 1.8;
        color: #2c3e50;
        margin-top: 0.5rem;
        font-family: 'Noto Sans Bengali', 'Kalpurush', sans-serif;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2d5a87;
    }
    
    .stat-label {
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* Success/Error messages */
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 10px;
        color: #155724;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 10px;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SONIOX API CONFIGURATION
# ============================================================================

SONIOX_API_BASE = "https://api.soniox.com"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    if seconds is None:
        return "00:00"
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp_full(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format for longer videos."""
    if seconds is None:
        return "00:00:00"
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_speaker_name(speaker_id) -> str:
    """Generate a friendly speaker name from speaker ID."""
    if speaker_id is None or speaker_id == 0 or speaker_id == "0":
        return "বক্তা ১"  # "Speaker 1" in Bengali
    try:
        return f"বক্তা {int(speaker_id)}"  # "Speaker N" in Bengali
    except:
        return f"বক্তা {speaker_id}"


def extract_audio_to_buffer(video_file) -> io.BytesIO:
    """
    Extract audio from video file and return as in-memory buffer.
    Uses MoviePy with temporary file for video (required by MoviePy),
    but audio is exported directly to BytesIO buffer.
    """
    audio_buffer = io.BytesIO()
    
    # MoviePy requires a file path, so we use a temp file for the video only
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        tmp_video.write(video_file.read())
        tmp_video_path = tmp_video.name
    
    try:
        # Load video and extract audio
        video_clip = VideoFileClip(tmp_video_path)
        audio_clip = video_clip.audio
        
        if audio_clip is None:
            raise ValueError("No audio track found in the video file.")
        
        # Create a temporary file for audio export (MoviePy limitation)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_audio:
            tmp_audio_path = tmp_audio.name
        
        # Export audio to temp file
        audio_clip.write_audiofile(
            tmp_audio_path,
            codec='libmp3lame',
            bitrate='128k',
            verbose=False,
            logger=None
        )
        
        # Read the audio file into buffer
        with open(tmp_audio_path, 'rb') as f:
            audio_buffer.write(f.read())
        
        # Clean up
        audio_clip.close()
        video_clip.close()
        os.unlink(tmp_audio_path)
        
    finally:
        # Always clean up the temp video file
        if os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)
    
    # Reset buffer position to beginning
    audio_buffer.seek(0)
    return audio_buffer


def upload_file_to_soniox(session: requests.Session, audio_buffer: io.BytesIO) -> str:
    """Upload audio file to Soniox Files API and return file_id."""
    audio_buffer.seek(0)
    
    # Read the buffer content
    audio_data = audio_buffer.read()
    audio_buffer.seek(0)
    
    files = {
        'file': ('audio.mp3', audio_data, 'audio/mpeg')
    }
    
    response = session.post(
        f"{SONIOX_API_BASE}/v1/files",
        files=files
    )
    
    if not response.ok:
        error_text = response.text
        raise Exception(f"File upload failed ({response.status_code}): {error_text}")
    
    result = response.json()
    return result['id']


def create_transcription(session: requests.Session, file_id: str = None, audio_url: str = None) -> str:
    """Create a transcription job and return transcription_id."""
    payload = {
        "model": "stt-async-preview",
        "language_hints": ["bn"],  # Bengali language hint
    }
    
    if file_id:
        payload["file_id"] = file_id
    elif audio_url:
        payload["audio_url"] = audio_url
    
    response = session.post(
        f"{SONIOX_API_BASE}/v1/transcriptions",
        json=payload
    )
    
    if not response.ok:
        error_text = response.text
        raise Exception(f"Create transcription failed ({response.status_code}): {error_text}")
    
    result = response.json()
    return result['id']


def wait_for_transcription(session: requests.Session, transcription_id: str, progress_callback=None) -> dict:
    """Poll for transcription status until completed."""
    max_attempts = 300  # 5 minutes max wait
    attempt = 0
    
    while attempt < max_attempts:
        response = session.get(f"{SONIOX_API_BASE}/v1/transcriptions/{transcription_id}")
        response.raise_for_status()
        transcription = response.json()
        
        status = transcription.get('status', 'unknown')
        
        if progress_callback:
            progress_callback(status, attempt)
        
        if status == 'completed':
            return transcription
        elif status == 'error':
            error_msg = transcription.get('error_message') or 'Unknown error'
            error_type = transcription.get('error_type') or 'unknown'
            # Return the full transcription object so we can see all error details
            raise Exception(f"Transcription error ({error_type}): {error_msg}\nFull response: {transcription}")
        
        time.sleep(1)
        attempt += 1
    
    raise Exception("Transcription timed out")


def get_transcript(session: requests.Session, transcription_id: str) -> dict:
    """Get the transcript for a completed transcription."""
    response = session.get(f"{SONIOX_API_BASE}/v1/transcriptions/{transcription_id}/transcript")
    response.raise_for_status()
    return response.json()


def delete_file(session: requests.Session, file_id: str):
    """Delete uploaded file from Soniox."""
    try:
        session.delete(f"{SONIOX_API_BASE}/v1/files/{file_id}")
    except:
        pass  # Ignore errors on cleanup


def parse_transcript_tokens(transcript: dict) -> list:
    """Parse transcript tokens into structured segments with speaker info."""
    tokens = transcript.get('tokens', [])
    
    segments = []
    current_segment = {
        'text': [],
        'start_time': None,
        'end_time': None,
        'speaker': None
    }
    
    current_speaker = None
    
    for token in tokens:
        text = token.get('text', '')
        start_ms = token.get('start_ms', 0)
        end_ms = token.get('end_ms', 0)
        
        # Check for speaker tag (format: spk:N)
        speaker_match = re.match(r'^spk:(\d+)$', text)
        if speaker_match:
            # Save current segment if it has content
            if current_segment['text']:
                current_segment['text'] = ''.join(current_segment['text']).strip()
                if current_segment['text']:
                    segments.append(current_segment.copy())
            
            # Start new segment with new speaker
            current_speaker = speaker_match.group(1)
            current_segment = {
                'text': [],
                'start_time': None,
                'end_time': None,
                'speaker': current_speaker
            }
            continue
        
        # Add text to current segment
        if text:
            if current_segment['start_time'] is None:
                current_segment['start_time'] = start_ms / 1000.0
                current_segment['speaker'] = current_speaker or '1'
            
            current_segment['text'].append(text)
            current_segment['end_time'] = end_ms / 1000.0
    
    # Add final segment
    if current_segment['text']:
        current_segment['text'] = ''.join(current_segment['text']).strip()
        if current_segment['text']:
            segments.append(current_segment)
    
    return segments


def group_into_sentences(segments: list) -> list:
    """Group segments into sentence-level chunks."""
    if not segments:
        return []
    
    # Bengali sentence-ending punctuation
    sentence_enders = {'।', '?', '!', '.', '।।'}
    
    sentences = []
    current_sentence = {
        'text': [],
        'start_time': None,
        'end_time': None,
        'speaker': None
    }
    
    for segment in segments:
        text = segment['text']
        words = text.split()
        
        for i, word in enumerate(words):
            # Handle speaker change
            if current_sentence['speaker'] is not None and segment['speaker'] != current_sentence['speaker']:
                if current_sentence['text']:
                    current_sentence['text'] = ' '.join(current_sentence['text'])
                    sentences.append(current_sentence.copy())
                current_sentence = {
                    'text': [],
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'speaker': segment['speaker']
                }
            
            # Initialize
            if current_sentence['start_time'] is None:
                current_sentence['start_time'] = segment['start_time']
                current_sentence['speaker'] = segment['speaker']
            
            current_sentence['text'].append(word)
            current_sentence['end_time'] = segment['end_time']
            
            # Check for sentence end
            if any(word.endswith(p) for p in sentence_enders):
                current_sentence['text'] = ' '.join(current_sentence['text'])
                sentences.append(current_sentence.copy())
                current_sentence = {
                    'text': [],
                    'start_time': None,
                    'end_time': None,
                    'speaker': segment['speaker']
                }
    
    # Add remaining text
    if current_sentence['text']:
        current_sentence['text'] = ' '.join(current_sentence['text'])
        sentences.append(current_sentence)
    
    # If no sentence boundaries found, return original segments
    if not sentences:
        return segments
    
    return sentences


def transcribe_with_soniox(audio_buffer: io.BytesIO, api_key: str, progress_placeholder) -> dict:
    """
    Transcribe audio using Soniox REST API.
    """
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"
    
    file_id = None
    
    try:
        # Step 1: Upload file
        progress_placeholder.info("📤 Uploading audio to Soniox...")
        file_id = upload_file_to_soniox(session, audio_buffer)
        
        # Step 2: Create transcription
        progress_placeholder.info("🎯 Starting transcription with Bengali language...")
        transcription_id = create_transcription(session, file_id=file_id)
        
        # Step 3: Wait for completion
        def update_progress(status, attempt):
            status_emoji = {
                'queued': '⏳',
                'processing': '🔄',
                'transcribing': '🧠',
                'completed': '✅',
                'error': '❌'
            }.get(status.lower(), '🔄')
            progress_placeholder.info(f"{status_emoji} Status: {status.upper()} (waiting {attempt}s)...")
        
        progress_placeholder.info("🧠 Sending to Soniox AI... Transcribing for highest accuracy...")
        transcription = wait_for_transcription(session, transcription_id, update_progress)
        
        # Step 4: Get transcript
        progress_placeholder.info("📥 Retrieving transcript...")
        transcript = get_transcript(session, transcription_id)
        
        return {
            'success': True,
            'transcript': transcript
        }
        
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 0
        error_text = e.response.text if e.response else str(e)
        
        if status_code == 401 or status_code == 403:
            return {
                'success': False,
                'error': 'Invalid API Key. Please check your Soniox API key and try again.',
                'error_type': 'auth'
            }
        elif status_code == 402:
            return {
                'success': False,
                'error': 'API quota exceeded. Please check your Soniox account balance.',
                'error_type': 'quota'
            }
        elif 'language' in error_text.lower():
            return {
                'success': False,
                'error': f'Language configuration error: {error_text}',
                'error_type': 'language'
            }
        else:
            return {
                'success': False,
                'error': f'API Error ({status_code}): {error_text}',
                'error_type': 'general'
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f'Transcription failed: {str(e)}',
            'error_type': 'general'
        }
    
    finally:
        # Cleanup: delete uploaded file
        if file_id:
            delete_file(session, file_id)


def generate_transcript_text(sentences: list) -> str:
    """Generate downloadable transcript text from sentences."""
    lines = []
    lines.append("=" * 60)
    lines.append("AmarScribe Ultimate - Bengali Transcription")
    lines.append("=" * 60)
    lines.append("")
    
    current_speaker = None
    for sentence in sentences:
        timestamp = format_timestamp_full(sentence.get('start_time', 0))
        speaker = get_speaker_name(sentence.get('speaker'))
        text = sentence.get('text', '')
        
        # Add speaker header if changed
        if sentence.get('speaker') != current_speaker:
            lines.append("")
            lines.append(f"--- {speaker} ---")
            current_speaker = sentence.get('speaker')
        
        lines.append(f"[{timestamp}] {text}")
    
    lines.append("")
    lines.append("=" * 60)
    lines.append("End of Transcript")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ AmarScribe Ultimate</h1>
        <p>World's Most Accurate Bengali Transcription Tool</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Powered by Soniox AI • Speaker Diarization • Segment Timestamps</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # API Key input
        st.markdown("### 🔑 API Authentication")
        api_key = st.text_input(
            "Soniox API Key",
            type="password",
            placeholder="Enter your Soniox API key...",
            help="Get your API key from https://console.soniox.com"
        )
        
        if api_key:
            st.success("✓ API Key provided")
        else:
            st.warning("⚠️ Please enter your API key")
        
        st.markdown("---")
        
        # Language setting (default Bengali)
        st.markdown("### 🌐 Language Settings")
        st.info("🇧🇩 Optimized for Bengali (বাংলা)")
        st.caption("Soniox uses language hints to optimize for Bengali transcription")
        
        st.markdown("---")
        
        # Info section
        st.markdown("### ℹ️ About")
        st.markdown("""
        **AmarScribe Ultimate** uses Soniox's 
        state-of-the-art AI for the highest 
        accuracy Bengali transcription.
        
        **Features:**
        - 🎯 High-accuracy Bengali ASR
        - 👥 Speaker diarization
        - ⏱️ Segment timestamps
        - 📥 Export to TXT
        - 🔒 Secure processing
        """)
        
        st.markdown("---")
        st.markdown("### 💰 Pricing")
        st.caption("~$0.10/hour (async transcription)")
        st.caption("[Get API Key](https://console.soniox.com)")
    
    # Main content area
    st.markdown("### 📤 Upload Your Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
        help="Supported formats: MP4, MOV, AVI, MKV, WEBM"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">📹</div>
                <div class="stat-label">{uploaded_file.name[:20]}{'...' if len(uploaded_file.name) > 20 else ''}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{file_size_mb:.1f} MB</div>
                <div class="stat-label">File Size</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">🇧🇩</div>
                <div class="stat-label">Bengali</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Transcribe button
        if st.button("🚀 Start Transcription", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ Please enter your Soniox API key in the sidebar.")
            else:
                # Process the video
                progress_placeholder = st.empty()
                
                try:
                    # Step 1: Extract audio
                    progress_placeholder.info("🎬 Extracting audio from video...")
                    uploaded_file.seek(0)  # Reset file position
                    audio_buffer = extract_audio_to_buffer(uploaded_file)
                    progress_placeholder.success("✓ Audio extracted successfully")
                    
                    # Step 2: Transcribe with Soniox
                    result = transcribe_with_soniox(audio_buffer, api_key, progress_placeholder)
                    
                    if not result['success']:
                        error_type = result.get('error_type', 'general')
                        if error_type == 'auth':
                            st.error(f"🔐 {result['error']}")
                        elif error_type == 'quota':
                            st.error(f"💳 {result['error']}")
                        elif error_type == 'language':
                            st.error(f"🌐 {result['error']}")
                        else:
                            st.error(f"❌ {result['error']}")
                    else:
                        progress_placeholder.success("✅ Transcription completed!")
                        
                        # Parse the transcript
                        transcript = result['transcript']
                        segments = parse_transcript_tokens(transcript)
                        
                        if not segments:
                            st.warning("⚠️ No speech detected in the audio.")
                        else:
                            # Group into sentences
                            sentences = group_into_sentences(segments)
                            
                            # Store in session state
                            st.session_state['sentences'] = sentences
                            st.session_state['transcript_ready'] = True
                            
                            # Calculate stats
                            total_duration = max(s.get('end_time', 0) for s in sentences) if sentences else 0
                            unique_speakers = len(set(s.get('speaker') for s in sentences if s.get('speaker')))
                            total_words = sum(len(s.get('text', '').split()) for s in sentences)
                            
                            # Display stats
                            st.markdown("### 📊 Transcription Statistics")
                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                            with stat_col1:
                                st.metric("Duration", format_timestamp_full(total_duration))
                            with stat_col2:
                                st.metric("Speakers", unique_speakers if unique_speakers > 0 else 1)
                            with stat_col3:
                                st.metric("Segments", len(sentences))
                            with stat_col4:
                                st.metric("Words", total_words)
                
                except ValueError as ve:
                    st.error(f"❌ Video Error: {str(ve)}")
                except Exception as e:
                    st.error(f"❌ Processing Error: {str(e)}")
                    st.info("Please ensure the video file is valid and contains audio.")
    
    # Display transcript if available
    if st.session_state.get('transcript_ready', False):
        sentences = st.session_state.get('sentences', [])
        
        st.markdown("---")
        st.markdown("### 📜 Transcript Feed")
        
        # Download button
        transcript_text = generate_transcript_text(sentences)
        st.download_button(
            label="📥 Download Transcript as TXT",
            data=transcript_text,
            file_name="amarscribe_transcript.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Display each segment
        for i, sentence in enumerate(sentences):
            timestamp = format_timestamp_full(sentence.get('start_time', 0))
            speaker = get_speaker_name(sentence.get('speaker'))
            text = sentence.get('text', '')
            
            st.markdown(f"""
            <div class="transcript-segment">
                <span class="timestamp">[{timestamp}]</span>
                <span class="speaker-label">{speaker}</span>
                <div class="bengali-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.85rem;">
        <p>AmarScribe Ultimate © 2024 | Powered by Soniox AI</p>
        <p>Built with ❤️ for Bengali Language Transcription</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
