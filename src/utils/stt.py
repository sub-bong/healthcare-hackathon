# src/utils/stt.py# STT: 오디오 파일 -> 한국어 텍스트

from openai import OpenAI

# OPENAI_API_KEY는 .env에서 로드됨 (utils/config.py가 load_dotenv 호출한다고 가정)
client = OpenAI()

def transcribe_ko(audio_path: str) -> str:
    """
    audio_path: wav/mp3/m4a/webm 등 파일 경로
    return: 한국어 텍스트 (양끝 공백 제거)
    """
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",            
            file=f,
            language="ko"
        )
    return (resp.text or "").strip()
