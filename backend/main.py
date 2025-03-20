from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import torch
import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
from audiocraft.models import MusicGen
from scipy.io.wavfile import write as write_wav
import asyncio
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],  # Explicitly allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

musicgen_model = MusicGen.get_pretrained('facebook/musicgen-medium')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Global variable to store progress
progress_status = {"step": "idle", "message": "Waiting for upload..."}

@app.get("/")
async def root():
    return {"message": "Hello World", "device": DEVICE}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    global progress_status
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_path = f"uploads/{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        progress_status = {"step": "uploaded", "message": "Video uploaded!"}
        output_path = process_video(video_path)
        download_url = f"http://localhost:8000/download/{os.path.basename(output_path)}"
        progress_status = {"step": "completed", "message": "Music video generated!"}
        return {"download_url": download_url}
    except Exception as e:
        progress_status = {"step": "error", "message": f"Error: {str(e)}"}
        return {"error": str(e)}

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='video/mp4', filename=filename)
    return {"error": "File not found"}

@app.get("/progress")
async def progress():
    async def event_stream():
        while True:
            yield f"data: {json.dumps(progress_status)}\n\n"
            await asyncio.sleep(1)  # Send updates every second

    return StreamingResponse(event_stream(), media_type="text/event-stream")

def process_video(video_path):
    global progress_status
    print("Starting video processing...")

    # Step 1: Extract emotions
    progress_status = {"step": "extracting_emotions", "message": "Detecting emotions..."}
    print(f"Progress: {progress_status}")  # Log progress
    emotion = extract_emotions(video_path)
    print(f"Detected Emotion: {emotion}")
    progress_status = {"step": "extracted_emotions", "message": f"Detected Emotion: {emotion}"}
    print(f"Progress: {progress_status}")  # Log progress

    # Step 2: Detect scene
    progress_status = {"step": "detecting_scene", "message": "Detecting scene..."}
    print(f"Progress: {progress_status}")  # Log progress
    scene = detect_scene(video_path)
    print(f"Detected Scene: {scene}")
    progress_status = {"step": "detected_scene", "message": f"Detected Scene: {scene}"}
    print(f"Progress: {progress_status}")  # Log progress

    # Step 3: Generate music
    progress_status = {"step": "generating_music", "message": "Generating AI music..."}
    print(f"Progress: {progress_status}")  # Log progress
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)
    cap.release()
    music_path = generate_music(emotion, scene, duration)
    print(f"Music generated at: {music_path}")
    progress_status = {"step": "generated_music", "message": "Music generated!"}
    print(f"Progress: {progress_status}")  # Log progress

    # Step 4: Combine video and music
    progress_status = {"step": "combining_video_music", "message": "Combining video and music..."}
    print(f"Progress: {progress_status}")  # Log progress
    output_path = "outputs/final_video.mp4"
    combine_video_music(video_path, music_path, output_path)
    print(f"Final video saved at: {output_path}")
    progress_status = {"step": "combined_video_music", "message": "Music video generated!"}
    print(f"Progress: {progress_status}")  # Log progress

    return output_path

def extract_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    face_detector = YOLO('yolov8n-face.pt').to(DEVICE)
    frame_count = 0
    emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector(frame_rgb, verbose=False)

        if len(face_results[0].boxes) == 0:
            continue

        largest_face = max(face_results[0].boxes, key=lambda x: x.xywh[0][2] * x.xywh[0][3])
        x1, y1, x2, y2 = map(int, largest_face.xyxy[0].cpu().numpy())
        face_img = frame_rgb[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        face_img = cv2.resize(face_img, (224, 224))
        try:
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            emotions.append(result[0]['dominant_emotion'])
        except Exception as e:
            print(f"Emotion analysis error: {e}")

    cap.release()
    return max(set(emotions), key=emotions.count) if emotions else "neutral"

def detect_scene(video_path):
    model = YOLO('yolov8n.pt').to('cpu')
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_positions = [0, total_frames // 2, total_frames - 1]

    objects = set()
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            results = model.predict(frame)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    objects.add(model.names[class_id])

    cap.release()
    return list(objects)

def generate_music(emotion, scene, duration):
    try:
        prompt = f"{emotion} music, inspired by {scene}, lasting {duration} seconds"
        musicgen_model.set_generation_params(duration=duration)
        with torch.no_grad():
            audio_array = musicgen_model.generate([prompt], progress=True)

        audio_np = audio_array[0].cpu().numpy().squeeze()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        music_path = "outputs/generated_music.wav"
        write_wav(music_path, 32000, audio_int16)
        return music_path

    except Exception as e:
        print(f"Error generating music: {e}")
        return None

def combine_video_music(video_path, music_path, output_path):
    try:
        command = f"ffmpeg -i {video_path} -i {music_path} -c:v libx264 -map 0:v:0 -map 1:a:0 -shortest {output_path}"
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
