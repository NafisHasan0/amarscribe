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
from datetime import timedelta

# Soniox SDK imports - using flexible import structure
try:
    from soniox.speech_service import SpeechClient
    from soniox.transcribe_file import transcribe_file_short
except ImportError:
    try:
        from soniox.client import Client as SpeechClient
    except ImportError:
        SpeechClient = None

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
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
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
    
    /* File uploader enhancement */
    .stFileUploader {
        border: 2px dashed #2d5a87;
        border-radius: 15px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    if seconds is None:
        return "00:00"
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp_full(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format for longer videos."""
    if seconds is None:
        return "00:00:00"
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_speaker_name(speaker_id: int) -> str:
    """Generate a friendly speaker name from speaker ID."""
    if speaker_id is None or speaker_id < 0:
        return "বক্তা"  # "Speaker" in Bengali
    return f"বক্তা {speaker_id + 1}"  # "Speaker 1", "Speaker 2", etc.


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


def group_words_into_sentences(words: list) -> list:
    """
    Group individual words into meaningful sentence chunks.
    Uses punctuation and timing gaps to determine sentence boundaries.
    """
    if not words:
        return []
    
    sentences = []
    current_sentence = {
        'text': [],
        'start_time': None,
        'end_time': None,
        'speaker': None
    }
    
    # Bengali sentence-ending punctuation
    sentence_enders = {'।', '?', '!', '.', '।।', '?', '!'}
    
    for word in words:
        word_text = word.get('text', '').strip()
        word_start = word.get('start_time', 0)
        word_end = word.get('end_time', 0)
        word_speaker = word.get('speaker', 0)
        
        if not word_text:
            continue
        
        # Start new sentence if speaker changes
        if current_sentence['speaker'] is not None and word_speaker != current_sentence['speaker']:
            if current_sentence['text']:
                current_sentence['text'] = ' '.join(current_sentence['text'])
                sentences.append(current_sentence.copy())
            current_sentence = {
                'text': [],
                'start_time': word_start,
                'end_time': word_end,
                'speaker': word_speaker
            }
        
        # Initialize sentence timing
        if current_sentence['start_time'] is None:
            current_sentence['start_time'] = word_start
            current_sentence['speaker'] = word_speaker
        
        current_sentence['text'].append(word_text)
        current_sentence['end_time'] = word_end
        
        # Check for sentence-ending punctuation
        if any(word_text.endswith(p) for p in sentence_enders):
            current_sentence['text'] = ' '.join(current_sentence['text'])
            sentences.append(current_sentence.copy())
            current_sentence = {
                'text': [],
                'start_time': None,
                'end_time': None,
                'speaker': None
            }
    
    # Add any remaining words as final sentence
    if current_sentence['text']:
        current_sentence['text'] = ' '.join(current_sentence['text'])
        sentences.append(current_sentence)
    
    return sentences


def transcribe_with_soniox(audio_buffer: io.BytesIO, api_key: str) -> dict:
    """
    Transcribe audio using Soniox API with Bengali language and speaker diarization.
    Returns structured transcription result.
    """
    try:
        # Import Soniox modules dynamically for flexibility
        from soniox.speech_service import SpeechClient, set_api_key
        from soniox.transcribe_file import transcribe_file_short
        
        # Set the API key
        set_api_key(api_key)
        
        # Save buffer to temp file (Soniox SDK requires file path)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_file.write(audio_buffer.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Perform transcription with speaker diarization
            result = transcribe_file_short(
                tmp_file_path,
                model="en_v2",  # Soniox's multilingual model
                language_hints=["bn"],  # Bengali hint
                enable_speaker_diarization=True,
                min_num_speakers=1,
                max_num_speakers=10
            )
            
            return {
                'success': True,
                'result': result
            }
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
    except ImportError:
        # Try alternative import pattern for newer SDK versions
        try:
            import soniox
            from soniox import transcribe
            
            # Save buffer to temp file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                audio_buffer.seek(0)
                tmp_file.write(audio_buffer.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Initialize client with API key
                client = soniox.Client(api_key=api_key)
                
                # Transcribe with Bengali and diarization
                result = client.transcribe(
                    tmp_file_path,
                    language="bn",
                    enable_speaker_diarization=True
                )
                
                return {
                    'success': True,
                    'result': result
                }
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as inner_e:
            return {
                'success': False,
                'error': f'SDK import error. Please ensure soniox is properly installed: {str(inner_e)}',
                'error_type': 'import'
            }
            
    except Exception as e:
        error_msg = str(e).lower()
        
        if 'invalid' in error_msg and ('key' in error_msg or 'api' in error_msg or 'auth' in error_msg):
            return {
                'success': False,
                'error': 'Invalid API Key. Please check your Soniox API key and try again.',
                'error_type': 'auth'
            }
        elif 'unauthorized' in error_msg or '401' in error_msg:
            return {
                'success': False,
                'error': 'Invalid API Key. Please check your Soniox API key and try again.',
                'error_type': 'auth'
            }
        elif 'quota' in error_msg or 'limit' in error_msg or 'exceeded' in error_msg:
            return {
                'success': False,
                'error': 'API quota exceeded. Please check your Soniox account balance.',
                'error_type': 'quota'
            }
        elif 'language' in error_msg:
            return {
                'success': False,
                'error': 'Language configuration error. Bengali (bn) may require specific model.',
                'error_type': 'language'
            }
        else:
            return {
                'success': False,
                'error': f'Transcription failed: {str(e)}',
                'error_type': 'general'
            }


def parse_soniox_result(result) -> list:
    """
    Parse Soniox transcription result into structured word list.
    Handles different response formats from the API.
    """
    words = []
    
    try:
        # Handle different result structures
        if hasattr(result, 'words'):
            for word in result.words:
                words.append({
                    'text': getattr(word, 'text', ''),
                    'start_time': getattr(word, 'start_ms', 0) / 1000.0,
                    'end_time': getattr(word, 'end_ms', 0) / 1000.0,
                    'speaker': getattr(word, 'speaker', 0)
                })
        elif hasattr(result, 'segments'):
            for segment in result.segments:
                segment_speaker = getattr(segment, 'speaker', 0)
                if hasattr(segment, 'words'):
                    for word in segment.words:
                        words.append({
                            'text': getattr(word, 'text', ''),
                            'start_time': getattr(word, 'start_ms', 0) / 1000.0,
                            'end_time': getattr(word, 'end_ms', 0) / 1000.0,
                            'speaker': getattr(word, 'speaker', segment_speaker)
                        })
                else:
                    # Segment-level text without word timestamps
                    words.append({
                        'text': getattr(segment, 'text', ''),
                        'start_time': getattr(segment, 'start_ms', 0) / 1000.0,
                        'end_time': getattr(segment, 'end_ms', 0) / 1000.0,
                        'speaker': segment_speaker
                    })
        elif hasattr(result, 'text'):
            # Fallback for simple text response
            words.append({
                'text': result.text,
                'start_time': 0,
                'end_time': 0,
                'speaker': 0
            })
        elif isinstance(result, dict):
            # Handle dictionary response
            if 'words' in result:
                for word in result['words']:
                    words.append({
                        'text': word.get('text', ''),
                        'start_time': word.get('start_ms', word.get('start_time', 0)) / 1000.0 if 'start_ms' in word else word.get('start_time', 0),
                        'end_time': word.get('end_ms', word.get('end_time', 0)) / 1000.0 if 'end_ms' in word else word.get('end_time', 0),
                        'speaker': word.get('speaker', 0)
                    })
            elif 'segments' in result:
                for segment in result['segments']:
                    segment_speaker = segment.get('speaker', 0)
                    if 'words' in segment:
                        for word in segment['words']:
                            words.append({
                                'text': word.get('text', ''),
                                'start_time': word.get('start_ms', 0) / 1000.0,
                                'end_time': word.get('end_ms', 0) / 1000.0,
                                'speaker': word.get('speaker', segment_speaker)
                            })
                    else:
                        words.append({
                            'text': segment.get('text', ''),
                            'start_time': segment.get('start_ms', 0) / 1000.0,
                            'end_time': segment.get('end_ms', 0) / 1000.0,
                            'speaker': segment_speaker
                        })
    except Exception as e:
        st.error(f"Error parsing transcription result: {str(e)}")
    
    return words


def generate_transcript_text(sentences: list) -> str:
    """Generate downloadable transcript text from sentences."""
    lines = []
    lines.append("=" * 60)
    lines.append("AmarScribe Ultimate - Bengali Transcription")
    lines.append("=" * 60)
    lines.append("")
    
    current_speaker = None
    for sentence in sentences:
        timestamp = format_timestamp_full(sentence['start_time'])
        speaker = get_speaker_name(sentence['speaker'])
        text = sentence['text']
        
        # Add speaker header if changed
        if sentence['speaker'] != current_speaker:
            lines.append("")
            lines.append(f"--- {speaker} ---")
            current_speaker = sentence['speaker']
        
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
        language = st.selectbox(
            "Transcription Language",
            options=["bn (Bengali)", "bn-BD (Bengali - Bangladesh)", "bn-IN (Bengali - India)"],
            index=0,
            help="Bengali is the default language for AmarScribe"
        )
        st.info("🇧🇩 Optimized for Bengali (বাংলা)")
        
        st.markdown("---")
        
        # Advanced settings
        st.markdown("### 🎛️ Advanced Options")
        enable_diarization = st.checkbox(
            "Enable Speaker Diarization",
            value=True,
            help="Identify and label different speakers in the audio"
        )
        
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
                <div class="stat-label">{uploaded_file.name[:20]}...</div>
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
                try:
                    # Step 1: Extract audio
                    with st.spinner("🎬 Extracting audio from video..."):
                        uploaded_file.seek(0)  # Reset file position
                        audio_buffer = extract_audio_to_buffer(uploaded_file)
                        st.success("✓ Audio extracted successfully")
                    
                    # Step 2: Transcribe with Soniox
                    with st.spinner("🧠 Sending to Soniox AI... Transcribing for highest accuracy..."):
                        transcription_result = transcribe_with_soniox(audio_buffer, api_key)
                    
                    if not transcription_result['success']:
                        if transcription_result['error_type'] == 'auth':
                            st.error(f"🔐 {transcription_result['error']}")
                        elif transcription_result['error_type'] == 'quota':
                            st.error(f"💳 {transcription_result['error']}")
                        else:
                            st.error(f"❌ {transcription_result['error']}")
                    else:
                        st.success("✓ Transcription completed!")
                        
                        # Parse the result
                        words = parse_soniox_result(transcription_result['result'])
                        
                        if not words:
                            st.warning("⚠️ No speech detected in the audio.")
                        else:
                            # Group into sentences
                            sentences = group_words_into_sentences(words)
                            
                            # Store in session state
                            st.session_state['sentences'] = sentences
                            st.session_state['transcript_ready'] = True
                            
                            # Calculate stats
                            total_duration = max(s['end_time'] for s in sentences) if sentences else 0
                            unique_speakers = len(set(s['speaker'] for s in sentences if s['speaker'] is not None))
                            total_words = sum(len(s['text'].split()) for s in sentences)
                            
                            # Display stats
                            st.markdown("### 📊 Transcription Statistics")
                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                            with stat_col1:
                                st.metric("Duration", format_timestamp_full(total_duration))
                            with stat_col2:
                                st.metric("Speakers", unique_speakers)
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
            timestamp = format_timestamp_full(sentence['start_time'])
            speaker = get_speaker_name(sentence['speaker'])
            text = sentence['text']
            
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
