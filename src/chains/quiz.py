# src/chains/quiz.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate   # ← core 사용
from langchain_core.output_parsers import StrOutputParser  # (안 쓰면 삭제)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..utils.config import OPENAI_MODEL
from ..models.onboarding import QuizOutput, GradingOutput  # 스키마는 onboarding.py에 있음

# ===== 파서 =====
quiz_parser = PydanticOutputParser(pydantic_object=QuizOutput)
grading_parser = PydanticOutputParser(pydantic_object=GradingOutput)

# ===== 프롬프트(네가 붙여준 텍스트 그대로) =====
quiz_prompt_text = quiz_prompt_text = '''
[역할]
당신은 고령자 친화형 ‘인지 자극 문제’ 출제자입니다.

[출제 규칙 - 매우 중요]
- 문제는 딱 1개만 만드세요.
- 대상: 고령자. 쉬운 어휘, 존댓말. TTS로 읽힙니다.
- 길이: 문제문은 최대 60자, 줄바꿈·번호·불릿·괄호·특수기호·이모지 금지.
- 의료·법률·위험 활동·개인정보 요구 금지.
- 계산은 초등 1~2학년 수준(한 자리 덧셈/뺄셈 정도).
- 난이도: 기본 very_easy, 필요 시 easy.

[유형 가이드]
- memory: 숫자 거꾸로 말하기, 단어 3개 기억 후 말하기
- attention: 글자 세기(‘가’가 몇 번?), 홀수 찾기(보기로 제시)
- problem_solving: 규칙찾기 또는 순서 중 '가장 먼저 할 일' 고르기

[유형 선택 규칙]
- 아래 3가지 중 **한 가지를 무작위로** 선택해 문제를 1개만 출제하세요.
  memory / attention / problem_solving 중 하나만 사용.

[나쁜 예시]
- “세 자리 곱셈을 풀어보세요”
- “병원에 가서 검사를 받아보세요”
- “①, ②, ( ), #, * 사용”

[좋은 예시]
- question: 숫자 3개를 읽고 거꾸로 말해보세요. 4 7 2
  answer: 2 7 4
- question: 다음 중 과일이 아닌 것은 무엇일까요? 사과 배 우산 바나나
  answer: 우산
- question: 아래 글에서 ‘가’는 몇 번 나오나요? 가나가다라가
  answer: 3
- question: 1, 3, 5 다음 숫자는 무엇일까요?
  answer: 7

[출력 형식]
{format_instructions}
'''

grading_prompt_text = '''
[역할]
당신은 고령자의 음성 답변을 채점하고 따뜻한 피드백을 제공하는 평가자입니다.

[채점 규칙]
- STT로 변환된 답변이므로 발음 변이를 고려하세요
- 의미가 같으면 정답 처리
- 숫자는 다양한 표현 인정 (예: "둘" = "2" = "이")
- 조사, 어미, 띄어쓰기 차이는 무시 (예: "우산" = "우 산" = "우산이요")
- 고령자 특성상 추가 설명이 있어도 핵심 답이 맞으면 정답

[입력]
문제: {question}
정답: {correct_answer}
사용자 답변: {user_answer}

[평가 기준]
1. 숫자 답변: 값이 정확히 일치하는가?
2. 단어 답변: 핵심 단어가 포함되어 있는가?
3. 순서 답변: 순서가 정확한가?

[피드백 작성 규칙]
- 고령자 대상이므로 존댓말 사용
- 정답일 때: 칭찬과 격려 (20-30자)
- 오답일 때: 격려하며 정답 알려주기 (30-40자)
- 따뜻하고 긍정적인 어조 유지
- TTS로 읽히므로 자연스러운 문장
- 출력은 correct(Boolean)과 feedback 두 필드만 작성하며, 정답 텍스트는 다시 출력하지 마세요.

[출력 형식]
{format_instructions}
'''

# ===== 체인 =====
quiz_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
grading_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

quiz_prompt = PromptTemplate(
    template=quiz_prompt_text,
    input_variables=[],
    partial_variables={"format_instructions": quiz_parser.get_format_instructions()}
)
grading_prompt = PromptTemplate(
    template=grading_prompt_text,
    input_variables=["question", "correct_answer", "user_answer"],
    partial_variables={"format_instructions": grading_parser.get_format_instructions()}
)

quiz_chain = quiz_prompt | quiz_llm | quiz_parser
grading_chain = grading_prompt | grading_llm | grading_parser

# ===== 서비스 클래스 =====
class CognitiveQuizService:
    def __init__(self):
        self.current_quiz = None
        self.quiz_chain = quiz_chain
        self.grading_chain = grading_chain

    def generate_quiz(self):
        output = self.quiz_chain.invoke({})
        self.current_quiz = {'question': output.question, 'answer': output.answer}
        return output.question

    def grade_answer(self, user_answer: str):
        if not self.current_quiz:
            return {'error': '진행 중인 문제가 없습니다'}
        result = self.grading_chain.invoke({
            'question': self.current_quiz['question'],
            'correct_answer': self.current_quiz['answer'],
            'user_answer': user_answer
        })
        return {
            "is_correct": result.correct,
            "feedback": result.feedback,
            "user_answer": user_answer,
            "correct_answer": self.current_quiz["answer"]
        }
