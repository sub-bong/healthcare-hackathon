from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

from ..utils.config import OPENAI_MODEL, TIMEZONE
from ..utils.time_utils import parse_hhmm
from ..models.onboarding import UserProfile

def build_mission_prompt(p: UserProfile):
    now = datetime.now(ZoneInfo(TIMEZONE))
    if now.time() >= parse_hhmm(p.bed_time):
        return None
    hour = now.hour
    tp = "오전" if hour < 12 else ("오후" if hour < 18 else "저녁")
    cur = now.strftime("%H:%M")
    return f"""
당신은 노인 복지 전문가입니다. 아래 정보를 반영하여 {tp} 시간대에 맞는 맞춤 미션 1개를 따뜻하게 제안하세요.

[사용자 정보]
    - 거동 상태: {'거동 불편' if p.mobility_issue else '거동 가능'}
    - 거주 형태: {'독거' if p.living_arrangement == 'alone' else '가족 동거'}
    - 기상 시간: {p.wake_up_time}
    - 취침 시간: {p.bed_time}
    - 현재 시각(KST): {cur}
    - 현재 시간대: {tp}

    [미션 생성 원칙]
    - 안전: 넘어짐/과로 위험 활동 금지. 건강진단 등 의학적 조언 금지.
    - 실행성: 집/근처에서 바로 할 수 있는 활동, 준비물 0~1개.
    - 구체성: 무엇을(명사) 어떻게(동사) 할지, 장소/방법을 명확히.
    - 시간: 10~20분 내 완료 가능(필요시 더 짧은 대안 제시).
    - 대안: 거동 불편 시 앉아서/벽 짚고/짧게 가능한 버전 함께 제시.
    - 사회성: 독거면 노인정 방문/이웃 인사 등 사회 연결 1가지 포함 권장.

    [시간대 고려사항]
    - 오전: 햇볕/환기/가벼운 정리·기지개 등으로 시작
    - 오후: 취미·정리·짧은 외출·사회 활동
    - 저녁: 스트레칭·호흡·감사일기·내일 준비

    [주의 사항]
    - 특수문자·이모지·번호·불릿 사용 금지. 과도한 명령/금지 표현 지양.
    - 날씨/교통 등 외부 조건은 안전을 우선해 실내 대안 함께 제공.
    - 비용/예약이 필요한 활동은 피하고, 무료/무예약 대안 우선.

    [출력 형식 - 매우 중요]
    - 1~2문장으로 전체 60자 이내의 자연스러운 구어체 한국어.
    - 예) 창가에 앉아 가벼운 목, 어깨 스트레칭을 해보세요.
    """.strip()

_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
_prompt = PromptTemplate.from_template("{prompt}")
_parser = StrOutputParser()

mission_chain = (
    RunnableLambda(lambda profile: build_mission_prompt(profile))
    | RunnableBranch(
        (lambda x: x is None, RunnableLambda(lambda _: None)),
        (lambda x: isinstance(x, str),
            RunnableLambda(lambda s: {"prompt": s}) | _prompt | _llm | _parser
        ),
        RunnableLambda(lambda _: None)
    )
)
