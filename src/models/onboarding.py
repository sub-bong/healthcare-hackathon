# src/models/onboarding.py
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class QuizOutput(BaseModel):
    question: str = Field(description="사용자에게 보여줄 문제")
    answer: str = Field(description="정답")

class GradingOutput(BaseModel):
    correct: bool
    feedback: str

def build_time_options(step: int = 30):
    return [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, step)]

TIME_OPTIONS = set(build_time_options())

class UserProfile(BaseModel):
    """온보딩에서 받는 기본 사용자 정보"""
    mobility_issue: bool = Field(default=True, description="거동 불편 여부")
    living_arrangement: Literal["alone", "with_family"] = Field(default="alone", description="거주 형태")
    wake_up_time: str = Field(default="07:00", description="기상 HH:MM")
    bed_time: str = Field(default="21:00", description="취침 HH:MM")

    @field_validator("wake_up_time", "bed_time")
    @classmethod
    def _v_time(cls, v: str) -> str:
        if v not in TIME_OPTIONS:
            raise ValueError(f"시간은 30분 단위여야 합니다: {v}")
        return v
