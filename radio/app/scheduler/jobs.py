# APScheduler: 매일 9/14/20시에 schedule-interact 트리거(데모 버전)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.core.config import TIMEZONE
from fastapi import FastAPI
import httpx

# 앱 부팅 시 간단 크론 등록(데모: user-001 고정)


def start_scheduler(app: FastAPI):
    sch = AsyncIOScheduler(timezone=TIMEZONE)

    async def tick():
        async with httpx.AsyncClient() as cli:
            await cli.get("http://127.0.0.1:8000/api/func/schedule-interact", params={"user_id": "user-001"})

    for h in (9, 14, 20):
        sch.add_job(tick, "cron", hour=h, minute=0)
    sch.start()
