# 스케줄 상호작용: 최근 표정 라벨로 지지/미션 생성 + TTS
import base64
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.db.session import SessionLocal
from app.models.db_models import EmotionCapture
from app.services.chains import support_message, cognitive_missions, tts

router = APIRouter(tags=["schedule"])

# user_id 쿼리로 최근 표정 라벨을 읽어 지지/미션/TTS 생성


@router.get("/schedule-interact", response_class=JSONResponse)
async def schedule_interact(request: Request):
    user_id = request.query_params.get("user_id")
    db = SessionLocal()
    try:
        last = (db.query(EmotionCapture)
                  .filter(EmotionCapture.user_id == user_id)
                  .order_by(EmotionCapture.created_at.desc()).first())
        label = last.label if last else "neutral"
    finally:
        db.close()
    sup = await support_message(label)
    missions = await cognitive_missions(label)
    b64 = base64.b64encode(await tts(sup)).decode()
    return {"emotion": label, "support": sup, "missions": missions, "tts_b64": b64}
