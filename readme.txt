# Moodly Inference

Moodly Inference is a FastAPI based AI inference service for the Moodly wellness ecosystem. It provides APIs for text embeddings, image emotion detection, and voice emotion recognition.

This service is designed to work as a separate AI microservice with the Moodly backend. It handles machine learning tasks independently so the main backend can stay focused on authentication, bookings, users, hosts, recommendations, and business logic.

## Features

* Generate text embeddings using Sentence Transformers
* Analyze image emotions using DeepFace
* Analyze voice emotions using HuBERT
* Return dominant emotion from uploaded image files
* Return dominant emotion and confidence scores from uploaded WAV audio files
* FastAPI based REST API
* Temporary file handling for uploaded media
* Structured error responses
* Logging for easier debugging and monitoring

## Tech Stack

* Python
* FastAPI
* Sentence Transformers
* DeepFace
* Hugging Face Transformers
* HuBERT
* PyTorch
* Torchaudio
* SoundFile
* Pydantic

## API Endpoints

## 1. Generate Text Embedding

```http
POST /embed
```

### Request Body

```json
{
  "text": "I feel relaxed and happy today"
}
```

### Response

```json
{
  "embedding": [0.123, -0.456, 0.789]
}
```

This endpoint converts input text into a vector embedding using the `all-MiniLM-L6-v2` Sentence Transformer model.

## 2. Analyze Image Emotion

```http
POST /analyze-image-emotion
```

### Request

Upload an image file using form-data.

| Field | Type  | Required |
| ----- | ----- | -------- |
| file  | image | Yes      |

### Response

```json
{
  "dominant_emotion": "happy"
}
```

This endpoint analyzes the uploaded image and returns the dominant facial emotion using DeepFace.

## 3. Analyze Voice Emotion

```http
POST /analyze-voice-emotion
```

### Request

Upload a WAV audio file using form-data.

| Field | Type      | Required |
| ----- | --------- | -------- |
| file  | wav audio | Yes      |

### Response

```json
{
  "dominant_emotion": "happy",
  "all_emotions": {
    "happy": {
      "confidence": 0.82,
      "label": "HAPPY"
    },
    "neutral": {
      "confidence": 0.10,
      "label": "NEUTRAL"
    }
  },
  "possible_emotions": [
    "angry",
    "happy",
    "sad",
    "neutral"
  ]
}
```

This endpoint processes a WAV audio file and detects the speaker emotion using the `superb/hubert-large-superb-er` model.

## Project Structure

```txt
moodly-inference/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/M-Talha-Atif/moodly-inference.git
cd moodly-inference
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

For Windows:

```bash
venv\Scripts\activate
```

For macOS or Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

This project loads AI models directly from Hugging Face and other Python libraries. Make sure your environment has enough memory because models such as HuBERT and DeepFace can be resource intensive.

Recommended Python version:

```txt
Python 3.10+
```

## Run Locally

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```txt
http://localhost:8000
```

FastAPI docs will be available at:

```txt
http://localhost:8000/docs
```

## Example cURL Requests

### Text Embedding

```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"I feel calm and peaceful today\"}"
```

### Image Emotion Detection

```bash
curl -X POST "http://localhost:8000/analyze-image-emotion" \
  -F "file=@face.jpg"
```

### Voice Emotion Detection

```bash
curl -X POST "http://localhost:8000/analyze-voice-emotion" \
  -F "file=@voice.wav"
```

## Error Handling

The API returns structured error responses for invalid files and processing failures.

Example invalid image response:

```json
{
  "detail": {
    "reason": "Invalid file type. Please upload an image file.",
    "errorType": "INVALID_FILE_TYPE"
  }
}
```

Example invalid audio response:

```json
{
  "detail": {
    "reason": "Only WAV files are supported.",
    "errorType": "INVALID_FILE_TYPE"
  }
}
```

## Supported Inputs

### Text

Any valid string input.

### Image

Supported image formats depend on the uploaded file content type. The file must be sent as an image.

### Audio

Only `.wav` files are supported. Audio is automatically converted to mono if needed and resampled to 16 kHz before inference.

## Use Case in Moodly

Moodly uses this inference service to understand user mood signals from different input types.

Text embeddings can be used for recommendation matching and semantic search.

Image emotion detection can help identify mood from facial expressions.

Voice emotion recognition can help detect emotional tone from audio input.

Together, these AI signals can support mood based wellness recommendations and personalized user experiences.

## Models Used

### Text Embedding Model

```txt
sentence-transformers/all-MiniLM-L6-v2
```

### Voice Emotion Model

```txt
superb/hubert-large-superb-er
```

### Image Emotion Model

```txt
DeepFace emotion analysis
```

## Notes

* The voice emotion endpoint currently supports only WAV files.
* Uploaded files are stored temporarily and deleted after processing.
* The first request may take longer because AI models need to load into memory.
* For production deployment, run the service with a production ASGI server setup and proper resource limits.

## License

This project is part of the Moodly ecosystem.
