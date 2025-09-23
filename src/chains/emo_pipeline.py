
# src/chains/emo_pipeline.py
from typing import Dict, List, Optional
import json
from openai import OpenAI

from ..models.onboarding import UserProfile
from ..utils.stt_local import transcribe_video

client = OpenAI()  # .env에서 키 로드된다고 가정

EMO_LABELS = ["happy","neutral","sad","angry","fear","disgust","surprise"]

EMO_LABELS = ["happy","neutral","sad","angry","fear","disgust","surprise"]

def classify_text_emotion(text: str) -> Dict:
    prompt = f"""
다음 한국어 텍스트의 감정을 분석하세요.
- 주 감정(primary): 가장 강하게 나타나는 감정 1개
- 보조 감정(secondary): 함께 감지되는 부가적인 감정 1개
  * 신뢰도 0.3 이상인 경우만 포함
  * 주 감정과 충분히 구별되는 경우만 포함
  * 없으면 빈 리스트

가능한 감정: {", ".join(EMO_LABELS)}

JSON으로만 답하세요:
{{
  "primary": {{"label": "감정명", "confidence": 0.0~1.0}},
  "secondary": [
    {{"label": "감정명", "confidence": 0.0~1.0}},
    ...
  ]
}}

텍스트: {text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a precise emotion classifier that detects multiple emotional layers."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    data = json.loads(content)

    # 응답 검증 및 기본값 설정
    if "primary" not in data:
        data["primary"] = {"label": "neutral", "confidence": 0.5}
    if "secondary" not in data:
        data["secondary"] = []

    # 보조 감정 필터링 (신뢰도 낮은 것 제거)
    data["secondary"] = [s for s in data["secondary"] if s.get("confidence", 0) >= 0.3][:2]

    return data

# === 2) 얼굴 감정 ===
def run_face_emotion(video_path: str) -> List[Dict]:
    """
    TODO: 실제 얼굴/표정 모델 연결.
    지금은 빈 리스트 반환 (텍스트만 사용).
    """
    return []  # 예: [{"label":"happy","confidence":0.72}]

# === 3) 융합 로직 (얼굴 우선, 근소 차이는 텍스트 허용) ===
def fuse_emotion(video_emotions: Dict, text_pred: Dict) -> Dict:
    if isinstance(video_emotions, list):
        ve_sorted = sorted(
            [x for x in video_emotions if isinstance(x, dict) and "label" in x and "confidence" in x],
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True
        )
        video_emotions = {
            "first_emotion": ve_sorted[0] if len(ve_sorted) > 0 else None,
            "second_emotion": ve_sorted[1] if len(ve_sorted) > 1 else None,
        }
    elif video_emotions is None:
        video_emotions = {"first_emotion": None, "second_emotion": None}

    # 1. 텍스트에서 상위 2개 추출
    text_emotions = []
    if text_pred.get("primary"):
        text_emotions.append(text_pred["primary"])
    text_emotions.extend(text_pred.get("secondary", []))
    text_emotions.sort(key=lambda x: x["confidence"], reverse=True)
    text_top2 = text_emotions[:2]
    
    # 2. 결과 구조 초기화
    result = {
        "first_emotion": video_emotions.get("first_emotion"),
        "second_emotion": video_emotions.get("second_emotion"),
        "text_support": [],
        "source": "video" if video_emotions.get("first_emotion") else "text"
    }
    
    # 3. 영상 감정이 없으면 텍스트로 대체
    if not result["first_emotion"] and text_top2:
        result["first_emotion"] = {
            "label": text_top2[0]["label"],
            "confidence": round(text_top2[0]["confidence"], 2),
            "source": "text"
        }
        result["source"] = "text_fallback"
        
        if len(text_top2) > 1:
            result["second_emotion"] = {
                "label": text_top2[1]["label"],
                "confidence": round(text_top2[1]["confidence"], 2),
                "source": "text"
            }
    
    # 4. 텍스트 보조 감정 추가 (중복 제거)
    existing_labels = set()
    if result["first_emotion"]:
        existing_labels.add(result["first_emotion"]["label"])
    if result["second_emotion"]:
        existing_labels.add(result["second_emotion"]["label"])
    
    for t_emo in text_top2:
        if t_emo["label"] not in existing_labels:
            result["text_support"].append({
                "label": t_emo["label"],
                "confidence": round(t_emo["confidence"], 2)
            })
    
    # 5. 기본값 처리
    if not result["first_emotion"]:
        result["first_emotion"] = {"label": "neutral", "confidence": 0.5, "source": "default"}
    
    return result




# === 4) 공감 멘트 ===
def build_empathy(fused: Dict, text: str) -> str:
    """
    first_emotion, second_emotion, text_support를 활용한 공감 메시지 생성
    """
    # 첫 번째 감정 (메인)
    first = fused['first_emotion']
    emotion_desc = f"주 감정: {first['label']} (신뢰도 {first['confidence']})"
    
    # 두 번째 감정
    if fused.get('second_emotion'):
        second = fused['second_emotion']
        emotion_desc += f"\n부 감정: {second['label']} (신뢰도 {second['confidence']})"
    
    # 텍스트 보조 감정
    if fused.get('text_support'):
        support_labels = [s['label'] for s in fused['text_support']]
        emotion_desc += f"\n보조 감정: {', '.join(support_labels)}"
    
    # 감정 소스 힌트
    if fused.get('source') == 'video':
        source_hint = "표정에서 읽히는"
    elif fused.get('source') == 'text_fallback':
        source_hint = "말씀에서 느껴지는"
    else:
        source_hint = "전반적으로 느껴지는"
    
    prompt = f"""
아래 정보를 바탕으로 고령자에게 공감과 격려를 전하는 짧은 문단을 만드세요.

지침:
- 존댓말, 쉬운 어휘 사용, 200자 이내
- 이모지/특수기호/괄호 사용 금지
- 구조: 감사 → 감정 인식 → 공감/격려 → 맞춤 제안
- {source_hint} 감정을 중심으로 응답
- 주 감정을 가장 중요하게, 부 감정도 자연스럽게 언급

감정 분석 결과:
{emotion_desc}

말씀 요지: {text[:120]}
"""
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a warm, emotionally perceptive Korean caregiver assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    return resp.choices[0].message.content.strip()


# === 5) 감정 기반 미션 ===
def make_emo_mission(fused: Dict, text: str, user_profile: Optional[UserProfile] = None) -> Dict:
    """감정 기반 미션 생성 (선택적으로 사용자 프로필 고려)"""

    # 감정 설명 - 새로운 구조에 맞게 수정
    first = fused['first_emotion']
    emotion_desc = f"주 감정: {first['label']} (신뢰도 {first['confidence']})"
    
    if fused.get('second_emotion'):
        second = fused['second_emotion']
        emotion_desc += f"\n부 감정: {second['label']} (신뢰도 {second['confidence']})"
    
    # 텍스트 보조 감정도 포함
    if fused.get('text_support'):
        support_labels = [s['label'] for s in fused['text_support']]
        emotion_desc += f"\n보조 감정: {', '.join(support_labels)}"

    # 사용자 프로필이 있으면 추가
    user_context = ""
    if user_profile:
        user_context = f"""

사용자 상황:
- 거동 불편 여부: {'예 (이동 제한, 앉아서 가능한 활동 위주)' if user_profile.mobility_issue else '아니오 (자유로운 이동 가능)'}
- 거주 형태: {'독거 (혼자서 안전하게 할 수 있는 활동)' if user_profile.living_arrangement == 'alone' else '가족과 함께 (가족 참여 가능)'}

위 상황을 반드시 고려하여 적합한 미션을 제안하세요.
"""

    prompt = f"""
고령자 친화 '감정 기반 미션' 1개를 만들어주세요.
- 존댓말, 쉬운 어휘, 이모지/특수기호/괄호 금지
- 의료·법률·위험 활동·개인정보 요구 금지
- 출력은 JSON만: title(12자 이내), steps(60자 이내), duration(예: 3분), difficulty(very_easy|easy)
- 첫 번째 감정을 중심으로, 두 번째 감정도 고려하여 미션 설계
{user_context}
{emotion_desc}
말씀 요지: {text[:120]}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You write short, safe, elder-friendly tasks in Korean."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    data = json.loads(content)
    return data

# === 6) 전체 실행 ===
# === 기존의 run_full_pipeline을 두 개로 분리 ===

def run_core_pipeline(video_path: str) -> Dict:
    """
    핵심 파이프라인
    """
    # 1) STT
    text = transcribe_video(video_path)
    
    # 2) 감정 분석
    text_pred = classify_text_emotion(text)
    video_emotions = run_face_emotion(video_path)  # {"first_emotion": ..., "second_emotion": ...}
    
    # 3) 융합
    fused = fuse_emotion(video_emotions, text_pred)
    
    # 4) 공감 메시지
    empathy = build_empathy(fused, text)
    
    return {
        "stt_text": text,
        "text_emotion": text_pred,
        "video_emotions": video_emotions,
        "fused_emotion": fused,
        "empathy": empathy
    }

def generate_emo_mission(core_result: Dict, user_profile: Optional[UserProfile] = None) -> Dict:
    """
    미션 생성 (사용자 프로필 선택적 고려)
    """
    mission = make_emo_mission(
        core_result["fused_emotion"],
        core_result["stt_text"],
        user_profile  # 그대로 전달
    )
    return mission