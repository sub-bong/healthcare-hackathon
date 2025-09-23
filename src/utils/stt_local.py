# src/utils/stt_local.py
import os, tempfile, subprocess
from transformers import pipeline
import torch

def extract_audio_pcm16(input_video: str) -> str:
    td = tempfile.mkdtemp(prefix="stt_")
    wav_path = os.path.join(td, "audio.wav")
    cmd = ["ffmpeg", "-y", "-i", input_video, "-vn", "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_path

ASR_MODEL  = "openai/whisper-small"  # 필요시 medium/large
ASR_DEVICE = 0 if torch.cuda.is_available() else -1
_asr = pipeline(
    "automatic-speech-recognition",
    model=ASR_MODEL,
    device=ASR_DEVICE,
    chunk_length_s=30,
    generate_kwargs={"task":"transcribe", "language":"ko"}
)

def transcribe_video(video_path: str) -> str:
    wav_path = extract_audio_pcm16(video_path)
    try:
        return _asr(wav_path)["text"]
    finally:
        try:
            os.remove(wav_path)
            os.rmdir(os.path.dirname(wav_path))
        except Exception:
            pass
