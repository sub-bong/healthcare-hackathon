# 라디오 피드: 랜덤 이야기 반환
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func
from app.db.session import SessionLocal
from app.models.db_models import Story

router = APIRouter(tags=["radio"])


# 랜덤 스토리 1개 텍스트 반환
@router.get("/radio-feed", response_class=JSONResponse)
async def radio_feed():
    db = SessionLocal()
    try:
        row = db.query(Story).order_by(func.random()).first()
        text = row.text if row else "아직 공유된 근황이 없습니다."
        return {"text": text}
    finally:
        db.close()

# 랜덤 스토리 연속 텍스트 반환


@router.get("/radio-feed-batch", response_class=JSONResponse)
async def radio_feed_batch(request: Request):
    count = int(request.query_params.get("count", 5))
    count = max(1, min(count, 20))
    db = SessionLocal()
    try:
        rows = (db.query(Story)
                  .order_by(func.random())
                  .limit(count).all())
        texts = [r.text for r in rows] or ["아직 공유된 근황이 없습니다."]
        return {"texts": texts}
    finally:
        db.close()
