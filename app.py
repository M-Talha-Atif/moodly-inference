from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from deepface import DeepFace
import tempfile, os, logging, torchaudio, torch
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from typing import Dict, Any
from datetime import datetime
import soundfile as sf

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ---------------- Models ----------------
# Text embedding model
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Voice emotion model (HuBERT)
voice_model_id = "superb/hubert-large-superb-er"
voice_processor = Wav2Vec2FeatureExtractor.from_pretrained(voice_model_id)
voice_model = AutoModelForAudioClassification.from_pretrained(voice_model_id)

# HuggingFace label mapping
id2label = voice_model.config.id2label

# ---------------- DTOs ----------------
class EmbedRequest(BaseModel):
    text: str

# ---------------- Endpoints ----------------
@app.post("/embed")
async def embed(req: EmbedRequest):
    vector = text_model.encode(req.text).tolist()
    return {"embedding": vector}


@app.post("/analyze-image-emotion")
async def analyze_image_emotion(file: UploadFile = File(...)) -> Dict[str, Any]:
    logger.info(f"Received image emotion analysis request at {datetime.now()}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "Invalid file type. Please upload an image file.",
                "errorType": "INVALID_FILE_TYPE",
            },
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        analysis = DeepFace.analyze(img_path=tmp_path, actions=["emotion"])
        dominant_emotion = analysis[0].get("dominant_emotion")
        return {"dominant_emotion": dominant_emotion}
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "reason": str(e),
                "errorType": "IMAGE_PROCESSING_ERROR",
            },
        )
    finally:
        os.unlink(tmp_path)


@app.post("/analyze-voice-emotion")
async def analyze_voice_emotion(file: UploadFile = File(...)) -> Dict[str, Any]:
    # Emotion label mapping (short to full)
    emotion_map = {
        "ang": "angry",
        "hap": "happy",
        "sad": "sad",
        "neu": "neutral",
        "fea": "fear",
        "sur": "surprise",
        "dis": "disgust",
    }

    # 1. Validate file type and size
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "Only WAV files are supported.",
                "errorType": "INVALID_FILE_TYPE",
            },
        )

    # 2. Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        await file.seek(0)  # Rewind to ensure full read
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 3. Load and process audio
        waveform, sr = sf.read(tmp_path)
        waveform = torch.from_numpy(waveform).float()

        # Convert stereo to mono if needed
        if len(waveform.shape) > 1:
            waveform = torch.mean(waveform, dim=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Process with HuBERT
        inputs = voice_processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = voice_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            
            # Get all emotions with their confidence scores
            all_emotions = {
                emotion_map.get(id2label[i], id2label[i]): {  # Map to full name
                    "confidence": float(score),
                    "label": emotion_map.get(id2label[i], id2label[i]).upper()  # Full name in uppercase
                }
                for i, score in enumerate(scores)
            }
            
            # Sort emotions by confidence (highest first)
            sorted_emotions = dict(sorted(
                all_emotions.items(),
                key=lambda item: item[1]["confidence"],
                reverse=True
            ))
            
            # Get dominant emotion
            dominant_emotion = next(iter(sorted_emotions))
            dominant_confidence = sorted_emotions[dominant_emotion]["confidence"]
            print(f"Dominant Emotion: {dominant_emotion} ({dominant_confidence:.2f})")

        return {
            "dominant_emotion": dominant_emotion,
            "all_emotions": sorted_emotions,
            "possible_emotions": [emotion_map.get(e, e) for e in id2label.values()]  # All possible labels with full names
        }

    except sf.LibsndfileError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "reason": "Invalid WAV file format",
                "errorType": "INVALID_AUDIO_FILE",
            },
        )
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "reason": "Audio processing failed",
                "errorType": "PROCESSING_ERROR",
            },
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
   