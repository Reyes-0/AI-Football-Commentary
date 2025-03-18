import os
import cv2
import base64
import openai
from moviepy.editor import VideoFileClip, AudioFileClip
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import supervision as sv
from flask import Flask, request, send_from_directory, redirect, render_template,render_template_string, jsonify, send_file 
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok
import os
import cv2
import base64
import openai
from moviepy.editor import VideoFileClip, AudioFileClip
import threading
import time

# Load configuration
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize ngrok
run_with_ngrok(app)
ngrok.set_auth_token(app.config['NGROK_AUTH_TOKEN'])
public_url = ngrok.connect(app.config['PORT_NO']).public_url

# Initialize OpenAI client
client = openai.OpenAI(api_key=app.config['OPENAI_API_KEY'])

# Initialize annotator
annotator = sv.BoxAnnotator()

# Ensure upload/download folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

# Global variables
ready = False
processing = False
task_status = {"completed": False}

def long_running_task(raw_vid_file_path):
    """Simulate a long-running task"""
    annotated_frames = []

    def my_custom_sink(predictions: dict, video_frame: VideoFrame):
        labels = [p["class"] for p in predictions["predictions"]]
        detections = sv.Detections.from_roboflow(predictions)
        image = annotator.annotate(
            scene=video_frame.image.copy(), detections=detections, labels=labels
        )
        annotated_frames.append(image)

    pipeline = InferencePipeline.init(
        model_id=app.config['YOLOV8_MODEL_ID'],
        video_reference=raw_vid_file_path,
        on_prediction=my_custom_sink,
        api_key=app.config['YOLOV8_API_KEY'],
        max_fps=app.config['MAX_FPS'],
        confidence=app.config['CONFIDENCE'],
        iou_threshold=app.config['IOU_THRESHOLD'],
    )

    pipeline.start()
    pipeline.join()

    output_video_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "output.mp4")
    height, width, _ = annotated_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

    for frame in annotated_frames:
        output_video_writer.write(frame)

    output_video_writer.release()
    print(f"Annotated video saved at: {output_video_path}")

    video = cv2.VideoCapture(output_video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_length_seconds = length / fps
    print(f'Video length: {video_length_seconds:.2f} seconds')

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": [
                    "You are provided annotated images of a FIFA football match. Only use the annotations to help you if needed.",
                    f"Your task is to create a short voiceover script in the style of a football commentator for 2.5 words per second for ONLY {video_length_seconds:.2f} seconds.",
                    "Only include the narration. Don't talk about the view. Don't rush the output! Accuracy is utmost crucial.",
                ],
            },
            {
                "role": "user",
                "content": [
                    f"Commentate the game in 1.5 words per second for ONLY {video_length_seconds / 2:.2f} seconds.",
                    *map(lambda x: {"image": x, "resize": 1080}, base64Frames[0::60]),
                ],
            },
        ],
        max_tokens=1000,
    )

    script = response.choices[0].message.content
    time.sleep(2)

    speech_file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "football.mp3")
    response = client.audio.speech.create(
        model=app.config['TEXT_TO_SPEECH_MODEL'],
        voice=app.config['TEXT_TO_SPEECH_MODEL_VOICE'],
        input=script,
        speed=1.18,
    )
    response.stream_to_file(speech_file_path)

    video_clip = VideoFileClip(raw_vid_file_path)
    audio_clip = AudioFileClip(speech_file_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "Commentated Video.m4v")
    final_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac')
    video_clip.close()
    audio_clip.close()
    final_clip.close()

    global ready, processing
    ready = True
    processing = False
    task_status["completed"] = True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            filename = file.filename
            raw_vid_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(raw_vid_file_path)

            global processing
            processing = True
            global ready
            ready = False

            if not ready and processing:
                threading.Thread(target=long_running_task, args=(raw_vid_file_path,)).start()

            return render_template('processing.html')

    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify(task_status)

@app.route('/download')
def download_file():
    path = os.path.join(app.config['DOWNLOAD_FOLDER'], "Commentated Video.m4v")
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    print('Click the link:', public_url)
    app.run()