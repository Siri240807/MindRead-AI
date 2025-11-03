from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from transformers import pipeline
import uvicorn
import tempfile
import numpy as np
import librosa
import random
import cv2


app = FastAPI()

# Allow React frontend to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load text emotion (sentiment) model
sentiment_pipeline = pipeline("sentiment-analysis")

@app.get("/")
def root():
    return {"message": "🚀 MindRead API is running successfully!"}

# --- Text Emotion Endpoint ---
@app.post("/analyze_text/")
async def analyze_text(text: str = Form(...)):
    result = sentiment_pipeline(text)[0]
    return {"emotion": result['label']}

# --- Face Emotion Endpoint ---
@app.post("/analyze_face/")
async def analyze_face(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    analysis = DeepFace.analyze(img_path=tmp_path, actions=['emotion'])
    emotion = analysis[0]['dominant_emotion']
    return {"emotion": emotion}

# --- Voice Emotion Endpoint (simple version) ---
@app.post("/analyze_voice/")
async def analyze_voice(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    y, sr = librosa.load(tmp_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    fake_emotions = ['happy', 'sad', 'angry', 'neutral']
    emotion = random.choice(fake_emotions)
    return {"emotion": emotion}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





from fastapi import File, UploadFile
from deepface import DeepFace
import shutil

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    import traceback
    try:
        contents = await file.read()
        image = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Could not decode image!")
            return {"error": "Invalid image format"}

        print("✅ Image successfully decoded")

        # Try detecting faces first
        try:
            faces = DeepFace.extract_faces(img_path=img, detector_backend='retinaface', enforce_detection=False)
            print(f"🔍 Faces detected: {len(faces)}")
        except Exception as e:
            print("⚠️ Face detection error:", e)
            faces = []

        if not faces:
            return {"error": "No face detected in the image. Try a clearer front-facing photo."}

        # Analyze emotions
        try:
            result = DeepFace.analyze(
                img_path=img,
                actions=['emotion'],
                detector_backend='retinaface',
                enforce_detection=False
            )
            print("🎯 DeepFace result:", result)

            emotion = result[0]['dominant_emotion']
            final_response = {"dominant_emotion": emotion}

            print("📦 Final Response Sent to Frontend:", final_response)
            return final_response

        except Exception as e:
            print("⚠️ Emotion analysis error:", e)
            print(traceback.format_exc())
            return {"error": "Failed to analyze emotion. Check backend logs."}

    except Exception as e:
        print("🔥 Unexpected error:", e)
        print(traceback.format_exc())
        return {"error": str(e)}
