# 앱 엔트리: 라우터/정적/템플릿/DB 초기화, 페이지 라우트, 스케줄러 시작
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.db.session import init_db
from app.api.func.audio import router as audio_router
from app.api.func.face import router as face_router
from app.api.func.radio import router as radio_router
from app.api.func.schedule import router as schedule_router
from app.scheduler.jobs import start_scheduler

app = FastAPI(title="인공지능 헬스케어")
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")
templates = Jinja2Templates(directory="app/web/templates")

init_db()
app.include_router(audio_router, prefix="/api/func")
app.include_router(face_router, prefix="/api/func")
app.include_router(radio_router, prefix="/api/func")
app.include_router(schedule_router, prefix="/api/func")
start_scheduler(app)  # 9/14/20시 데모 트리거


@app.get("/", response_class=HTMLResponse)
def root_page(request: Request):
    return templates.TemplateResponse("radio_schedule.html", {"request": request})


@app.get("/face", response_class=HTMLResponse)
def face_page(request: Request):
    return templates.TemplateResponse("face.html", {"request": request})


@app.get("/radio-schedule", response_class=HTMLResponse)
def radio_schedule_page(request: Request):
    return templates.TemplateResponse("radio_schedule.html", {"request": request})
