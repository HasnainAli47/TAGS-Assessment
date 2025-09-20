import torch
import torchaudio
import whisper
import io
import time
import uuid
import ffmpeg 
import numpy as np 
from typing import BinaryIO, Dict, Any, Tuple
from demucs.apply import apply_model
from demucs.pretrained import get_model as get_demucs_model

# --- Model Management ---
_MODELS = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Application starting. Using device: {DEVICE}")

def _get_model(model_name: str) -> Any:
    if model_name not in _MODELS:
        print(f"Loading '{model_name}' model for the first time...")
        if model_name == 'demucs':
            model = get_demucs_model('htdemucs')
            model.to(DEVICE)
            _MODELS['demucs'] = model
        elif model_name == 'whisper':
            model = whisper.load_model("small", device=DEVICE)
            _MODELS['whisper'] = model
        print(f"'{model_name}' model loaded and cached.")
    return _MODELS[model_name]


def load_audio_with_ffmpeg(file_bytes: bytes, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Decodes an in-memory audio file using FFmpeg into a raw PCM format and
    then loads it into a PyTorch tensor. This is highly robust.
    """
    try:
        out, _ = (
            ffmpeg
            .input('pipe:')
            .output('pipe:', format='s16le', ac=1, ar=str(target_sr))
            .run(input=file_bytes, capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as e:
        # This catches errors if FFmpeg can't decode the file (e.g., it's corrupted)
        print("FFmpeg error:", e.stderr.decode())
        raise ValueError("Failed to decode audio file with FFmpeg.")

    waveform_np = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    waveform = waveform.to(DEVICE)
    
    return waveform, target_sr


def separate_vocals(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    demucs_model = _get_model('demucs')
    demucs_sr = demucs_model.samplerate
    resampler_to_demucs = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=demucs_sr).to(DEVICE)
    resampler_from_demucs = torchaudio.transforms.Resample(orig_freq=demucs_sr, new_freq=sample_rate).to(DEVICE)
    resampled_waveform = resampler_to_demucs(waveform.cpu()).to(DEVICE)
    if resampled_waveform.dim() == 2:
        resampled_waveform = resampled_waveform.unsqueeze(0)
    sources = apply_model(demucs_model, resampled_waveform, device=DEVICE, progress=False)[0]
    vocals = sources[3]
    vocals_resampled = resampler_from_demucs(vocals)
    return vocals_resampled

def transcribe_audio(waveform: torch.Tensor) -> Dict[str, Any]:
    whisper_model = _get_model('whisper')
    waveform_squeezed = waveform.squeeze(0).cpu()
    result = whisper_model.transcribe(waveform_squeezed, fp16=torch.cuda.is_available())
    return result


def process_audio_pipeline(
    file: BinaryIO,
    enable_separation: bool,
    language_hint: str = None
) -> Dict[str, Any]:
    """
    Orchestrates the full audio processing pipeline using the robust FFmpeg loader.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    timings = {}

    file_bytes = file.read()

    load_start = time.time()
    try:
        waveform, sample_rate = load_audio_with_ffmpeg(file_bytes, target_sr=16000)
    except ValueError as e:
        raise e
    duration_sec = waveform.shape[-1] / sample_rate
    timings["load"] = int((time.time() - load_start) * 1000)

    separation_start = time.time()
    pipeline_info = {
        "separation": {"enabled": enable_separation, "method": "none"},
        "transcription": {"model": "whisper-small"}
    }
    if enable_separation:
        try:
            waveform = separate_vocals(waveform, sample_rate)
            pipeline_info["separation"]["method"] = "demucs"
        except Exception as e:
            print(f"Source separation failed: {e}. Falling back to direct transcription.")
            pipeline_info["separation"]["enabled"] = False
            pipeline_info["separation"]["method"] = "failed_fallback"
    timings["separation"] = int((time.time() - separation_start) * 1000)

    # Transcription
    transcription_start = time.time()
    transcription_result = transcribe_audio(waveform)
    timings["transcription"] = int((time.time() - transcription_start) * 1000)

    total_time = time.time() - start_time
    timings["total"] = int(total_time * 1000)

    # Format the final JSON response
    response = {
        "request_id": request_id,
        "duration_sec": round(duration_sec, 2),
        "sample_rate": sample_rate,
        "pipeline": pipeline_info,
        "segments": [
            {"start": round(seg["start"], 2), "end": round(seg["end"], 2), "text": seg["text"].strip()}
            for seg in transcription_result.get("segments", [])
        ],
        "text": transcription_result.get("text", "").strip(),
        "language": transcription_result.get("language"),
        "timings_ms": timings
    }
    return response