"""
AmarScribe Ultimate - Free, Unlimited Bengali Transcription
Uses OpenAI Whisper for local, unlimited transcription with zero API costs.
"""

import streamlit as st
import io
import tempfile
import os
from pathlib import Path

# Page configuration with Bengali-inspired theming
st.set_page_config(
    page_title="AmarScribe Ultimate",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a distinctive Bengali-inspired aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tiro+Bangla:ital@0;1&family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
    
    /* Root variables */
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --accent-gold: #d4a853;
        --accent-rust: #c9513d;
        --accent-teal: #3fb9a8;
        --text-primary: #f0f3f6;
        --text-secondary: #8b949e;
        --border-color: #30363d;
    }
    
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a1f26 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #d4a853 0%, #f0c674 50%, #d4a853 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .main-header .subtitle {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 300;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    
    .bengali-accent {
        font-family: 'Tiro Bangla', serif;
        color: var(--accent-teal);
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
    
    /* Feature badges */
    .feature-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .badge {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.85rem;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .badge.gold { border-color: var(--accent-gold); color: var(--accent-gold); }
    .badge.teal { border-color: var(--accent-teal); color: var(--accent-teal); }
    .badge.rust { border-color: var(--accent-rust); color: var(--accent-rust); }
    
    /* Upload section */
    .upload-section {
        background: var(--bg-secondary);
        border: 2px dashed var(--border-color);
        border-radius: 1rem;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: var(--accent-gold);
        box-shadow: 0 0 30px rgba(212, 168, 83, 0.1);
    }
    
    /* Transcript container */
    .transcript-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 1rem;
        padding: 2rem;
        margin: 2rem 0;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .transcript-chunk {
        display: flex;
        gap: 1rem;
        padding: 1rem;
        border-bottom: 1px solid var(--border-color);
        transition: background 0.2s ease;
    }
    
    .transcript-chunk:hover {
        background: var(--bg-tertiary);
    }
    
    .timestamp {
        font-family: 'Source Sans 3', monospace;
        color: var(--accent-teal);
        font-size: 0.9rem;
        white-space: nowrap;
        padding: 0.25rem 0.75rem;
        background: rgba(63, 185, 168, 0.1);
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .timestamp:hover {
        background: rgba(63, 185, 168, 0.2);
        transform: scale(1.05);
    }
    
    .transcript-text {
        font-family: 'Tiro Bangla', serif;
        font-size: 1.2rem;
        color: var(--text-primary);
        line-height: 1.8;
    }
    
    /* Stats display */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem 2rem;
        background: var(--bg-tertiary);
        border-radius: 0.75rem;
        border: 1px solid var(--border-color);
    }
    
    .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: var(--accent-gold);
        font-weight: 600;
    }
    
    .stat-label {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-gold) 0%, #c49843 100%);
        color: #0d1117;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(212, 168, 83, 0.3);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent-teal) 0%, #2d9488 100%);
        color: white;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: var(--bg-tertiary);
        border-color: var(--border-color);
    }
    
    /* Progress styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-gold), var(--accent-teal));
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* File uploader */
    .stFileUploader > div {
        background: var(--bg-tertiary);
        border: 2px dashed var(--border-color);
        border-radius: 1rem;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--accent-gold);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent-gold) !important;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(63, 185, 168, 0.1);
        border-left: 4px solid var(--accent-teal);
        padding: 1rem 1.5rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(212, 168, 83, 0.1);
        border-left: 4px solid var(--accent-gold);
        padding: 1rem 1.5rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def extract_audio_from_video(video_bytes: bytes, file_extension: str) -> bytes:
    """Extract audio from video using MoviePy, returning audio bytes."""
    from moviepy.editor import VideoFileClip
    
    # Write video to temp file (MoviePy needs file path)
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_video:
        tmp_video.write(video_bytes)
        tmp_video_path = tmp_video.name
    
    try:
        # Extract audio
        video = VideoFileClip(tmp_video_path)
        
        # Create temp file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_audio_path = tmp_audio.name
        
        # Write audio
        video.audio.write_audiofile(
            tmp_audio_path, 
            fps=16000,  # Whisper expects 16kHz
            nbytes=2,
            codec='pcm_s16le',
            verbose=False,
            logger=None
        )
        video.close()
        
        # Read audio bytes
        with open(tmp_audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        return audio_bytes, tmp_audio_path
        
    finally:
        # Clean up video temp file
        if os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)


def transcribe_with_whisper(audio_path: str, model_name: str, language: str) -> dict:
    """Transcribe audio using local Whisper model."""
    import whisper
    
    # Load model
    model = whisper.load_model(model_name)
    
    # Transcribe with word timestamps
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        word_timestamps=True,
        verbose=False
    )
    
    return result


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>AmarScribe Ultimate</h1>
        <p class="subtitle">Free • Unlimited • Local Processing</p>
        <p class="bengali-accent">আমার স্ক্রাইব — বাংলা ট্রান্সক্রিপশন</p>
        <div class="feature-badges">
            <span class="badge gold">✨ Zero API Cost</span>
            <span class="badge teal">🔒 100% Private</span>
            <span class="badge rust">⚡ Powered by Whisper</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration columns
    col1, col2 = st.columns(2)
    
    with col1:
        target_language = st.selectbox(
            "🌐 Target Language",
            options=["Bengali", "English", "Hindi", "Arabic", "Chinese", "Spanish", "French", "German", "Japanese", "Korean"],
            index=0,
            help="Select the primary language in your video"
        )
    
    # Map language names to Whisper codes
    language_codes = {
        "Bengali": "bn",
        "English": "en", 
        "Hindi": "hi",
        "Arabic": "ar",
        "Chinese": "zh",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Japanese": "ja",
        "Korean": "ko"
    }
    
    with col2:
        model_size = st.selectbox(
            "🧠 Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=2,  # Default to "small"
            help="Larger models are more accurate but slower"
        )
    
    # Model info
    model_info = {
        "tiny": ("~1GB VRAM", "Fastest, least accurate"),
        "base": ("~1GB VRAM", "Fast, basic accuracy"),
        "small": ("~2GB VRAM", "Balanced speed/accuracy"),
        "medium": ("~5GB VRAM", "High accuracy"),
        "large": ("~10GB VRAM", "Best accuracy, slowest")
    }
    
    st.markdown(f"""
    <div class="info-box">
        <strong>{model_size.capitalize()} Model:</strong> {model_info[model_size][0]} • {model_info[model_size][1]}
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown("### 📤 Upload Your Video")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Supported formats: MP4, MOV, AVI, MKV, WEBM"
    )
    
    # Initialize session state
    if 'transcript_result' not in st.session_state:
        st.session_state.transcript_result = None
    if 'transcript_text' not in st.session_state:
        st.session_state.transcript_text = ""
    
    if uploaded_file is not None:
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-value">{uploaded_file.name[:20]}{'...' if len(uploaded_file.name) > 20 else ''}</div>
                <div class="stat-label">File Name</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{file_size_mb:.1f} MB</div>
                <div class="stat-label">File Size</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Transcribe button
        if st.button("🎯 Start Transcription", use_container_width=True):
            try:
                # Get file extension
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                # Read video bytes into memory
                video_bytes = uploaded_file.read()
                
                # Step 1: Extract audio
                with st.status("🎵 Extracting audio from video...", expanded=True) as status:
                    st.write("Processing video file...")
                    audio_bytes, audio_path = extract_audio_from_video(video_bytes, file_ext)
                    st.write("✅ Audio extracted successfully!")
                    status.update(label="Audio extraction complete!", state="complete")
                
                # Step 2: Load model and transcribe
                with st.status(f"🧠 Loading Whisper {model_size} model...", expanded=True) as status:
                    st.write("This may take a moment on first run as the model downloads...")
                    
                    progress_bar = st.progress(0)
                    st.write("📥 Downloading/loading model...")
                    progress_bar.progress(30)
                    
                    # Perform transcription
                    st.write("🎤 Transcribing audio...")
                    progress_bar.progress(50)
                    
                    result = transcribe_with_whisper(
                        audio_path,
                        model_size,
                        language_codes[target_language]
                    )
                    
                    progress_bar.progress(100)
                    st.write("✅ Transcription complete!")
                    status.update(label="Transcription complete!", state="complete")
                
                # Clean up audio temp file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                
                # Store results
                st.session_state.transcript_result = result
                
                # Build formatted transcript
                transcript_lines = []
                for segment in result.get("segments", []):
                    timestamp = format_timestamp(segment["start"])
                    text = segment["text"].strip()
                    transcript_lines.append(f"[{timestamp}] {text}")
                
                st.session_state.transcript_text = "\n\n".join(transcript_lines)
                
            except Exception as e:
                st.error(f"❌ Error during transcription: {str(e)}")
                st.markdown("""
                <div class="warning-box">
                    <strong>Troubleshooting Tips:</strong><br>
                    • Ensure ffmpeg is installed<br>
                    • Try a smaller model size<br>
                    • Check that the video file isn't corrupted
                </div>
                """, unsafe_allow_html=True)
    
    # Display results
    if st.session_state.transcript_result:
        result = st.session_state.transcript_result
        
        st.markdown("---")
        st.markdown("### 📝 Transcript")
        
        # Stats
        total_duration = result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0
        word_count = len(result.get("text", "").split())
        segment_count = len(result.get("segments", []))
        
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-value">{format_timestamp(total_duration)}</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{word_count:,}</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{segment_count}</div>
                <div class="stat-label">Segments</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Transcript display
        st.markdown('<div class="transcript-container">', unsafe_allow_html=True)
        
        for segment in result.get("segments", []):
            timestamp = format_timestamp(segment["start"])
            text = segment["text"].strip()
            
            col_ts, col_text = st.columns([1, 9])
            with col_ts:
                st.markdown(f'<span class="timestamp">[{timestamp}]</span>', unsafe_allow_html=True)
            with col_text:
                st.markdown(f'<span class="transcript-text">{text}</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download button
        st.markdown("### 💾 Export")
        
        # Prepare download content
        download_content = f"""AmarScribe Ultimate - Transcript
{'='*50}
Language: {target_language}
Model: Whisper {model_size}
Duration: {format_timestamp(total_duration)}
Words: {word_count:,}
{'='*50}

{st.session_state.transcript_text}

{'='*50}
Generated by AmarScribe Ultimate
Free, Unlimited Bengali Transcription
"""
        
        st.download_button(
            label="📥 Download Transcript as TXT",
            data=download_content,
            file_name=f"transcript_{uploaded_file.name.rsplit('.', 1)[0]}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: var(--text-secondary);">
        <p style="font-family: 'Source Sans 3', sans-serif; font-size: 0.9rem;">
            Built with ❤️ using OpenAI Whisper • 100% Free & Private
        </p>
        <p style="font-family: 'Tiro Bangla', serif; color: var(--accent-teal);">
            বাংলা ভাষার জন্য তৈরি
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
