from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import json
from src.core.audio_processing import process_audio_pipeline

router = APIRouter()

@router.post("/v1/transcribe")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    config: Optional[str] = Form('{}')
):
    """
    Accepts an audio file and optional configuration to transcribe speech.

    - **file**: The audio file (wav, mp3, m4a, flac, ogg).
    - **config**: A JSON string with optional parameters:
        - `language_hint` (str): e.g., "en", "es".
        - `enable_separation` (bool): `true` to isolate vocals first.
        - `diarize` (bool): (Not implemented yet)
        - `model_size` (str): (Not implemented yet, uses 'small')
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")

    try:
        config_data = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'config' field.")

    # Extract parameters from config with defaults
    enable_separation = config_data.get("enable_separation", True)
    language_hint = config_data.get("language_hint", None)

    try:
        # The file object from UploadFile has a `file` attribute which is a SpooledTemporaryFile
        # It behaves like a file-like object, which is what our pipeline expects.
        result = process_audio_pipeline(
            file=file.file,
            enable_separation=enable_separation,
            language_hint=language_hint
        )
        return result
    except ValueError as e:
        # This catches the audio loading error
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Generic catch-all for other unexpected errors during processing
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during processing.")