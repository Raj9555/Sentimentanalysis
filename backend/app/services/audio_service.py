import os

# Hard cap CPU/thread usage to reduce memory spikes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import whisper

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip


_whisper_model = None


def _get_whisper_model():
    """
    Lazy-load Whisper ONLY when transcription is requested.
    Must not load at import time.
    """
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("large-v3", device="cpu")
    return _whisper_model


def extract_audio_from_video(video_path, audio_path):
    clip = VideoFileClip(video_path)
    try:
        if clip.audio is None:
            return False
        clip.audio.write_audiofile(audio_path)
        return True
    finally:
        clip.close()


def transcribe_audio(audio_path):
    model = _get_whisper_model()

    result = model.transcribe(
        audio_path,
        language="en",
        prompt="This is clear spoken English audio. Transcribe exactly as spoken.",
        fp16=False,
        beam_size=1,
    )
    return result.get("text", "")
