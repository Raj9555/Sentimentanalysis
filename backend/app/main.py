from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

from app.services.audio_service import extract_audio_from_video, transcribe_audio
from app.services.sentiment_service import analyze_sentiment
from app.services.video_service import detect_video_emotion

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Video Sentiment API Running"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)

    video_path = f"temp/{file.filename}"
    audio_path = f"temp/{file.filename}.mp3"

    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    audio_extracted = extract_audio_from_video(video_path, audio_path)

    if audio_extracted and os.path.exists(audio_path):
        transcript = transcribe_audio(audio_path)
        if transcript.strip() and len(transcript.split()) > 1:  # Require >1 word
            sentiment = analyze_sentiment(transcript)
        else:
            sentiment = None
    else:
        transcript = None
        sentiment = None
    emotion = detect_video_emotion(video_path)

    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    result = {
        "video_emotion": emotion
    }
    if transcript is not None:
        result["transcript"] = transcript
    if sentiment is not None:
        result["sentiment"] = sentiment
    return result

@app.post("/api/analyze-video")
async def analyze_video_api(file: UploadFile = File(...)):
    return await analyze_video(file)
