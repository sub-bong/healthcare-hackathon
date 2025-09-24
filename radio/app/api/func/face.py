# 표정 저장 -> 1. /face-label(MediaPipe 라벨 저장) 2. /face(서버추론: 크롭 이미지 업로드)
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.db.session import SessionLocal
from app.models.db_models import EmotionCapture
from app.services.chains_face_torch import classify_expression
import io

router = APIRouter(tags=["face"])

# user_id, label, score 폼을 받아 DB 저장


@router.post("/face-label", response_class=JSONResponse)
async def face_label(request: Request):

    form = await request.form()
    user_id = form.get("user_id")
    label = form.get("label")
    score = float(form.get("score", "0.7"))
    db = SessionLocal()
    try:
        db.add(EmotionCapture(user_id=user_id, label=label, score=score))
        db.commit()
    finally:
        db.close()
    return {"label": label, "score": score}

# 웹캠 스냅샷 이미지를 받아 Torch 모델로 분류 후 저장


@router.post("/face", response_class=JSONResponse)
async def face_infer(user_id: str = Form(...), frame: UploadFile = None):
    img_bytes = io.BytesIO(await frame.read())
    emo = await classify_expression(img_bytes)
    db = SessionLocal()
    try:
        db.add(EmotionCapture(user_id=user_id,
               label=emo["label"], score=emo.get("score")))
        db.commit()
    finally:
        db.close()
    return emo
