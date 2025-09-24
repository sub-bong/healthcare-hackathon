from .audio import router as audio_router
from .face import router as face_router
from .radio import router as radio_router
from .schedule import router as schedule_router

__all__ = ["audio_router", "face_router", "radio_router", "schedule_router"]
