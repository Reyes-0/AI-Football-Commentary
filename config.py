import os

class Config:
    PORT_NO = 5000
    NGROK_AUTH_TOKEN = "your_ngrok_auth_token"
    OPENAI_API_KEY = "your_openai_api_key"
    YOLOV8_API_KEY = "f8dQGPo6uoSyOXPjpa2U"
    YOLOV8_MODEL_ID = "pls-work-fifa-cv/5"
    MAX_FPS = 10
    CONFIDENCE = 0.55
    IOU_THRESHOLD = 0.5
    TEXT_TO_SPEECH_MODEL = "tts-1"
    TEXT_TO_SPEECH_MODEL_VOICE = "echo"
    UPLOAD_FOLDER = "/content/uploads"
    DOWNLOAD_FOLDER = "/content/downloads"