# STT/TTS API (폼/파일은 Request에서 직접 파싱)
import base64
import re
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.db.session import SessionLocal
from app.models.db_models import Story
from app.services.chains import tts, stt

router = APIRouter(tags=["audio"])


def clean_hotword(text):
    # "가가야" 변형(공백/문장부호 포함) 제거
    if not text:
        return text
    # 문장 앞쪽의 호출 제거 (예: '가가야, ...' / '가  가야 ...')
    text = re.sub(r'^\s*[“"\'(]*가\s*가\s*야[”"\'\s),.:;!?-]*',
                  '', text, flags=re.IGNORECASE)
    # 문장 중간의 단독 호출 제거 (토큰 양쪽이 공백/문장부호)
    text = re.sub(
        r'(?:(?<=\s)|(?<=^))가\s*가\s*야(?=\s|$|[”,.:;!?\')])', ' ', text, flags=re.IGNORECASE)
    # 공백 정리
    return ' '.join(text.split())

# 텍스트를 받아 MP3 base64 반환


@router.post("/tts", response_class=JSONResponse)
async def tts_api(request: Request):
    form = await request.form()
    text = form.get("text", "")
    audio = await tts(text)
    return {"b64": base64.b64encode(audio).decode()}

# 오디오 파일을 받아 텍스트로 변환하고 선택적으로 DB 저장


@router.post("/stt", response_class=JSONResponse)
async def stt_api(request: Request):
    form = await request.form()
    file = form.get("audio")
    user_id = form.get("user_id")
    data = await file.read() if file else b""
    text = await stt(data) if data else ""
    text = clean_hotword(text)

    if user_id and text.strip():
        db = SessionLocal()
        try:
            db.add(Story(user_id=user_id, text=text.strip()))
            db.commit()
        finally:
            db.close()
    return {"text": text}
