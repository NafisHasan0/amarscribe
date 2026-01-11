"""
AmarScribe Pro – Advanced Bengali Video-to-Text Utility
A Streamlit application for transcribing Bengali videos with word-level timestamps.
Uses ElevenLabs Scribe v2 API with RAM-only processing.
"""

import streamlit as st
import io
import tempfile
import os
from datetime import timedelta

# Page configuration
st.set_page_config(
    page_title="AmarScribe Pro",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional look with readable Bengali text
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    /* Transcript feed styling */
    .transcript-container {
        background: #FAFBFC;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .transcript-line {
        padding: 1rem;
        margin-bottom: 0.75rem;
        background: white;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .timestamp {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-family: 'Consolas', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.75rem;
        cursor: pointer;
    }
    
    .timestamp:hover {
        background: #DBEAFE;
    }
    
    .speaker-name {
        color: #059669;
        font-weight: 600;
        font-size: 0.9rem;
        margin-right: 0.5rem;
    }
    
    .bengali-text {
        font-size: 1.35rem;
        line-height: 2;
        color: #1F2937;
        font-family: 'Noto Sans Bengali', 'Kalpurush', sans-serif;
        display: block;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #F8FAFC;
    }
    
    /* Progress styling */
    .stSpinner > div {
        border-color: #3B82F6;
    }
    
    /* Button styling */
    .stDownloadButton > button {
        background: #3B82F6;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: #2563EB;
    }
    
    /* Info box */
    .info-box {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Error styling */
    .error-box {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 1rem;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def chunk_words_into_sentences(words: list, max_words: int = 10) -> list:
    """
    Group words into meaningful chunks based on punctuation or word count.
    Returns list of dicts with start_time, end_time, speaker, and text.
    """
    if not words:
        return []
    
    chunks = []
    current_chunk = {
        "words": [],
        "start_time": None,
        "end_time": None,
        "speaker": None
    }
    
    sentence_enders = {"।", "?", "!", ".", "।।"}
    
    for word_info in words:
        word_text = word_info.get("text", "").strip()
        start = word_info.get("start", 0)
        end = word_info.get("end", 0)
        speaker = word_info.get("speaker", "Speaker")
        
        if not word_text:
            continue
        
        # Initialize chunk timing
        if current_chunk["start_time"] is None:
            current_chunk["start_time"] = start
            current_chunk["speaker"] = speaker
        
        current_chunk["words"].append(word_text)
        current_chunk["end_time"] = end
        
        # Check if we should end this chunk
        is_sentence_end = any(word_text.endswith(p) for p in sentence_enders)
        reached_max_words = len(current_chunk["words"]) >= max_words
        
        if is_sentence_end or reached_max_words:
            chunks.append({
                "text": " ".join(current_chunk["words"]),
                "start_time": current_chunk["start_time"],
                "end_time": current_chunk["end_time"],
                "speaker": current_chunk["speaker"] or "Speaker"
            })
            current_chunk = {
                "words": [],
                "start_time": None,
                "end_time": None,
                "speaker": None
            }
    
    # Don't forget remaining words
    if current_chunk["words"]:
        chunks.append({
            "text": " ".join(current_chunk["words"]),
            "start_time": current_chunk["start_time"],
            "end_time": current_chunk["end_time"],
            "speaker": current_chunk["speaker"] or "Speaker"
        })
    
    return chunks


def extract_audio_to_buffer(video_file) -> io.BytesIO:
    """
    Extract audio from video file directly to BytesIO buffer.
    Uses a temporary file for video (required by MoviePy) but audio stays in RAM.
    """
    from moviepy.editor import VideoFileClip
    
    audio_buffer = io.BytesIO()
    temp_video_path = None
    
    try:
        # MoviePy needs a file path, so we use a temp file for the video only
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file.read())
            temp_video_path = tmp.name
        
        # Extract audio
        video = VideoFileClip(temp_video_path)
        
        # Export audio to a temporary file first, then read into buffer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio:
            temp_audio_path = tmp_audio.name
        
        video.audio.write_audiofile(
            temp_audio_path,
            codec='mp3',
            bitrate='128k',
            verbose=False,
            logger=None
        )
        video.close()
        
        # Read audio into buffer
        with open(temp_audio_path, 'rb') as f:
            audio_buffer.write(f.read())
        
        # Clean up temp audio file
        os.unlink(temp_audio_path)
        
        audio_buffer.seek(0)
        return audio_buffer
        
    finally:
        # Clean up temp video file
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)


def transcribe_audio(audio_buffer: io.BytesIO, api_key: str, language: str) -> dict:
    """
    Send audio buffer to ElevenLabs Scribe API for transcription.
    """
    from elevenlabs import ElevenLabs
    
    client = ElevenLabs(api_key=api_key)
    
    # Reset buffer position
    audio_buffer.seek(0)
    
    # Call the transcribe endpoint
    result = client.speech_to_text.convert(
        file=("audio.mp3", audio_buffer, "audio/mpeg"),
        model_id="scribe_v1",
        language_code=language,
        tag_audio_events=True,
        diarize=True
    )
    
    return result


def format_transcript_for_download(chunks: list) -> str:
    """Format transcript chunks for TXT download."""
    lines = []
    lines.append("=" * 60)
    lines.append("AmarScribe Pro - Transcript Export")
    lines.append("=" * 60)
    lines.append("")
    
    for chunk in chunks:
        timestamp = format_timestamp(chunk["start_time"])
        speaker = chunk["speaker"]
        text = chunk["text"]
        lines.append(f"[{timestamp}] {speaker}: {text}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("End of Transcript")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    # Header
    st.markdown('<h1 class="main-header">🎙️ AmarScribe Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Bengali Video-to-Text Utility with Word-Level Timestamps</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        # API Key input
        api_key = st.text_input(
            "ElevenLabs API Key",
            type="password",
            placeholder="Enter your API key...",
            help="Get your API key from elevenlabs.io"
        )
        
        # Language selection
        language_options = {
            "Bengali": "bn",
            "Hindi": "hi",
            "English": "en",
            "Tamil": "ta",
            "Telugu": "te",
            "Marathi": "mr",
            "Gujarati": "gu",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Punjabi": "pa",
            "Urdu": "ur"
        }
        
        selected_language = st.selectbox(
            "Target Language",
            options=list(language_options.keys()),
            index=0,
            help="Select the primary language in your video"
        )
        
        language_code = language_options[selected_language]
        
        st.markdown("---")
        st.markdown("### 📝 About")
        st.markdown("""
        **AmarScribe Pro** uses ElevenLabs Scribe v2 
        for highly accurate Bengali transcription with:
        
        - 🎯 Word-level timestamps
        - 👥 Speaker diarization
        - 🔒 RAM-only processing
        - 📱 Clean, readable output
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'],
            help="Supported formats: MP4, MOV, AVI, MKV, WebM, M4V"
        )
    
    with col2:
        st.markdown("### 📊 Status")
        status_placeholder = st.empty()
        status_placeholder.info("Waiting for video upload...")
    
    # Process video when uploaded
    if uploaded_file is not None:
        if not api_key:
            st.error("⚠️ Please enter your ElevenLabs API Key in the sidebar.")
            return
        
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(f"**File:** {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Transcribe button
        if st.button("🚀 Start Transcription", type="primary", use_container_width=True):
            try:
                # Step 1: Extract audio
                status_placeholder.warning("🎵 Extracting audio from video...")
                
                with st.spinner("Extracting audio... This may take a moment."):
                    audio_buffer = extract_audio_to_buffer(uploaded_file)
                
                status_placeholder.warning("🔄 Analyzing Bengali Audio... This may take a moment.")
                
                # Step 2: Transcribe
                with st.spinner("Analyzing Bengali Audio... This may take a moment."):
                    try:
                        result = transcribe_audio(audio_buffer, api_key, language_code)
                    except Exception as api_error:
                        error_msg = str(api_error).lower()
                        if "invalid" in error_msg and "key" in error_msg:
                            st.error("❌ Invalid API Key. Please check your ElevenLabs API key.")
                        elif "unauthorized" in error_msg or "401" in error_msg:
                            st.error("❌ Invalid API Key. Please verify your ElevenLabs API key is correct.")
                        else:
                            st.error(f"❌ Transcription Error: {api_error}")
                        return
                
                status_placeholder.success("✅ Transcription Complete!")
                
                # Step 3: Process and display results
                st.markdown("---")
                st.markdown("### 📜 Transcript Feed")
                
                # Get words from result
                words = []
                if hasattr(result, 'words') and result.words:
                    words = [
                        {
                            "text": w.text if hasattr(w, 'text') else str(w),
                            "start": w.start if hasattr(w, 'start') else 0,
                            "end": w.end if hasattr(w, 'end') else 0,
                            "speaker": getattr(w, 'speaker', 'Speaker')
                        }
                        for w in result.words
                    ]
                
                # If no word-level data, fall back to full text
                if not words and hasattr(result, 'text'):
                    # Create a single chunk from full text
                    chunks = [{
                        "text": result.text,
                        "start_time": 0,
                        "end_time": 0,
                        "speaker": "Speaker"
                    }]
                else:
                    # Chunk words into meaningful segments
                    chunks = chunk_words_into_sentences(words, max_words=10)
                
                if not chunks:
                    st.warning("No transcript content was generated. The audio may be unclear or empty.")
                    return
                
                # Display transcript
                st.markdown('<div class="transcript-container">', unsafe_allow_html=True)
                
                for i, chunk in enumerate(chunks):
                    timestamp = format_timestamp(chunk["start_time"])
                    speaker = chunk["speaker"] or "Speaker"
                    text = chunk["text"]
                    
                    st.markdown(f"""
                    <div class="transcript-line">
                        <span class="timestamp">[{timestamp}]</span>
                        <span class="speaker-name">{speaker}:</span>
                        <span class="bengali-text">{text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Export section
                st.markdown("---")
                st.markdown("### 💾 Export")
                
                transcript_text = format_transcript_for_download(chunks)
                
                col_export1, col_export2 = st.columns([1, 2])
                with col_export1:
                    st.download_button(
                        label="📥 Download Transcript as TXT",
                        data=transcript_text.encode('utf-8'),
                        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_transcript.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col_export2:
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>✓ Transcription Summary:</strong><br>
                        • Total segments: {len(chunks)}<br>
                        • Language: {selected_language}<br>
                        • Processing: RAM-only (zero disk storage)
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                error_message = str(e)
                
                # Handle specific error types
                if "codec" in error_message.lower() or "format" in error_message.lower():
                    st.markdown("""
                    <div class="error-box">
                        <strong>❌ Unsupported Video Format</strong><br>
                        The video format could not be processed. Please try:
                        <ul>
                            <li>Converting to MP4 format</li>
                            <li>Using a different video file</li>
                            <li>Checking if the file is corrupted</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Error: {error_message}")
                
                status_placeholder.error("❌ Processing failed")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>"
        "AmarScribe Pro • Built with Streamlit & ElevenLabs Scribe v2"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
