# SQLAlchemy 모델: 이야기(라디오), 표정 결과
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, func

Base = declarative_base()


class Story(Base):
    __tablename__ = "stories"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(64), index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EmotionCapture(Base):
    __tablename__ = "emotion_captures"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(64), index=True)
    label = Column(String(32), index=True)
    score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
