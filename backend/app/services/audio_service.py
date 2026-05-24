import whisper
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

model = whisper.load_model("large-v3")

def extract_audio_from_video(video_path, audio_path):
    clip = VideoFileClip(video_path)
    audio_extracted = False
    if clip.audio is not None:
        clip.audio.write_audiofile(audio_path)
        audio_extracted = True
    clip.close()
    return audio_extracted

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path, language="en", prompt="This is clear spoken English audio. Transcribe exactly as spoken.")
    return result["text"]
