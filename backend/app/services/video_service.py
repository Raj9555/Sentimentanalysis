import cv2
import numpy as np
import sys
import os

# moviepy v2 compatibility shim
try:
    import moviepy
    if not hasattr(moviepy, 'editor'):
        moviepy.editor = moviepy
        sys.modules['moviepy.editor'] = moviepy
except ImportError:
    pass

# Paths to face detector model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_PROTO = os.path.join(BASE_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Lazy-load heavy ML models
_face_net = None
_emotion_pipe = None


def _get_face_net():
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    return _face_net


def _get_emotion_pipe():
    global _emotion_pipe
    if _emotion_pipe is None:
        from transformers import pipeline
        from PIL import Image
        _emotion_pipe = pipeline(
            'image-classification',
            model='dima806/facial_emotions_image_detection',
            device='cpu'
        )
    return _emotion_pipe


def _detect_faces(frame, conf_threshold=0.5):
    """Return list of (x1, y1, x2, y2) face boxes."""
    h, w = frame.shape[:2]
    face_net = _get_face_net()
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))
    return faces


def _predict_emotion(face_roi):
    """Given a BGR face ROI, return the top emotion label."""
    from PIL import Image
    pipe = _get_emotion_pipe()
    # Convert BGR to RGB
    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    results = pipe(pil_img)
    # results = [{'label': 'happy', 'score': 0.95}, ...]
    top = results[0]
    label = top['label']
    # Normalize label to common emotion names
    label_map = {
        'angry': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happiness',
        'neutral': 'neutral',
        'sad': 'sadness',
        'surprise': 'surprise',
    }
    return label_map.get(label, label)


def detect_video_emotion(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "No face detected"

    frame_count = 0
    detected_emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue

        faces = _detect_faces(frame, conf_threshold=0.5)
        for (x1, y1, x2, y2) in faces:
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            emotion = _predict_emotion(face_roi)
            detected_emotions.append(emotion)

    cap.release()

    if not detected_emotions:
        return "No face detected"

    # Aggregate by frequency
    emotion_frequency = {}
    for emotion in detected_emotions:
        emotion_frequency[emotion] = emotion_frequency.get(emotion, 0) + 1

    final_emotion = max(emotion_frequency, key=emotion_frequency.get)
    return final_emotion
