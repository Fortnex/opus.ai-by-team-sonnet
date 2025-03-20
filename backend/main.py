from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import os
import cv2
from ultralytics import YOLO
from bark import SAMPLE_RATE, generate_audio, preload_models  # Updated Bark imports
from scipy.io.wavfile import write as write_wav
import mediapipe as mp

app = FastAPI()

# Preload Bark models
preload_models()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}

# File upload endpoint
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    # Save the uploaded video
    video_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process the video
    output_path = process_video(video_path)

    # Return the final video
    return FileResponse(output_path)

# Video processing logic
def process_video(video_path):
    print("Starting video processing...")
    
    # Ensure the outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Extract emotions
    print("Extracting emotions...")
    emotion = extract_emotions(video_path)
    print(f"Detected Emotion: {emotion}")
    
    # Step 2: Detect scene
    print("Detecting scene...")
    scene = detect_scene(video_path)
    print(f"Detected Scene: {scene}")
    
    # Step 3: Generate music
    print("Generating music...")
    duration = 30  # Extract duration using FFmpeg (we'll add this later)
    music_path = generate_music(emotion, scene, duration)
    print(f"Music generated at: {music_path}")
    
    # Step 4: Combine video and music
    print("Combining video and music...")
    output_path = "outputs/final_video.mp4"
    combine_video_music(video_path, music_path, output_path)
    print(f"Final video saved at: {output_path}")
    
    return output_path

# Emotion extraction using MediaPipe Face Mesh
def extract_emotions(video_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    emotions = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # For simplicity, assume the emotion is "happy" if a face is detected
            # You can add more sophisticated logic here to infer emotions
            emotions.append("happy")
        else:
            emotions.append("neutral")  # Fallback emotion

    cap.release()
    print("Emotion extraction complete.")
    return max(set(emotions), key=emotions.count) if emotions else "neutral"

# Scene detection using YOLO
def detect_scene(video_path):
    model = YOLO('yolov8n.pt')
    results = model.predict(video_path)
    detected_objects = set()
    for result in results:
        for box in result.boxes:
            detected_objects.add(model.names[int(box.cls)])

    # Define the items to check for (convert to a set)
    items = {"person", "car", "tree"}  # Example items
    items = set(items)  # Ensure items is a set

    # Check for intersection
    if detected_objects & items:  # Now both are sets
        print("Detected relevant objects:", detected_objects & items)
    else:
        print("No relevant objects detected.")

    return list(detected_objects)  # Return as a list if needed

# Music generation with structured prompt
def generate_music(emotion, scene, duration):
    # Convert scene list to a tuple
    scene_key = tuple(scene)

    # Define scene descriptions
    scene_descriptions = {
        ("beach", "ocean", "sunset"): "calm beach waves with seagulls",
        ("forest", "trees"): "peaceful forest sounds with birds chirping",
        ("city", "cars", "buildings"): "urban city ambiance with traffic noise",
    }

    # Get the scene description (default to "ambient instrumental music")
    scene_text = scene_descriptions.get(scene_key, "ambient instrumental music")
    print(f"Scene Description: {scene_text}")

    # Create a structured prompt
    prompt = f"{emotion} music, {scene_text}, {duration} seconds"
    print(f"Generated Prompt: {prompt}")

    try:
        # Generate music using the prompt
        audio_array = generate_audio(prompt)
        music_path = "outputs/generated_music.wav"
        write_wav(music_path, SAMPLE_RATE, audio_array)
        print(f"Music generated successfully at {music_path}")
        return music_path
    except Exception as e:
        print(f"Error generating music: {e}")
        raise RuntimeError("Failed to generate music. Please check the prompt and Bark configuration.")

# Combine video and music using FFmpeg
def combine_video_music(video_path, music_path, output_path):
    # Ensure the outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # FFmpeg command to combine video and audio
    command = f"ffmpeg -i {video_path} -i {music_path} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {output_path}"
    print(f"Running FFmpeg command: {command}")

    try:
        # Run the FFmpeg command
        subprocess.run(command, shell=True, check=True)
        print("Video and music combined successfully.")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise RuntimeError("Failed to combine video and music. Please check if FFmpeg is installed and the input files are valid.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)