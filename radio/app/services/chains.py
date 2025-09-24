# ChatGPT STT/TTS + 정서적 지지/미션 생성
import io
from app.core.config import OPENAI_API_KEY
import asyncio
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

client = OpenAI(api_key=OPENAI_API_KEY)
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# 텍스트 -> 음성(MP3 바이트)
async def tts(text):
    def _run():
        a = client.audio.speech.create(
            model="gpt-4o-mini-tts", voice="alloy", input=text)
        return a.read()
    return await asyncio.to_thread(_run)


# 오디오 바이트 -> 텍스트
async def stt(b):
    def _run():
        r = client.audio.transcriptions.create(
            model="whisper-1", language="ko", file=("audio.wav", io.BytesIO(b)))
        return r.text
    return await asyncio.to_thread(_run)


# 표정라벨 기반 정서적 지지 멘트(1~2문장)
async def support_message(label):
    sys = "공감 동반자. 존댓말, 1~2문장, 과한 조언 금지."
    user = "표정: " + label + ". 슬픔/불안/외로움은 수용/위로 중심."
    msgs = ChatPromptTemplate.from_messages(
        [('system', sys), ('human', user)]).format_messages()
    return (await chat.ainvoke(msgs)).content


# 표정라벨 기반 인지 미션 3개(2~5분)
async def cognitive_missions(label):
    sys = "인지 미션 생성기. 난이도 낮음, 3개, 2~5분, 음성응답 전제."
    user = "표정: " + label + ". 유형: 분류, 역순 기억, 문자 세기."
    msgs = ChatPromptTemplate.from_messages(
        [('system', sys), ('human', user)]).format_messages()
    return (await chat.ainvoke(msgs)).content
