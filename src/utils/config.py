import os
from dotenv import load_dotenv
load_dotenv()  # .env 읽기

OPENAI_MODEL = "gpt-4o-mini"
TIMEZONE = "Asia/Seoul"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 .env에 없습니다.")
