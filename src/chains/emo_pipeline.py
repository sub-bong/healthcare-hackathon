
# src/chains/emo_pipeline.py
from typing import Dict, List, Optional
import json
from openai import OpenAI

from ..models.onboarding import UserProfile
from ..utils.stt_local import transcribe_video

client = OpenAI()  # .env에서 키 로드된다고 가정

EMO_LABELS = ["happy","neutral","sad","angry","fear","disgust","surprise"]

def classify_text_emotion(text: str) -> Dict:
    prompt = f"""
다음 한국어 텍스트의 감정을 분석하세요.
- 주 감정(primary): 가장 강하게 나타나는 감정 1개
- 보조 감정(secondary): 함께 감지되는 부가적인 감정 0~2개
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
    data = json.loads(resp.choices[0].message.content)
    data.setdefault("primary", {"label":"neutral","confidence":0.5})
    secs = data.get("secondary", [])
    data["secondary"] = [s for s in secs if s.get("confidence",0)>=0.3][:2]
    return data

def run_face_emotion(video_path: str) -> List[Dict]:
    """
    TODO: 실제 얼굴/표정 모델 연결.
    지금은 빈 리스트 반환 (텍스트만 사용).
    """
    return []

def fuse_emotion(video_preds: List[Dict], text_pred: Dict,
                 min_conf=0.35, max_video_emotions=4, max_total_emotions=6) -> Dict:

    valid_video_emotions = [v for v in video_preds if v["confidence"] >= min_conf]
    valid_video_emotions.sort(key=lambda x: x["confidence"], reverse=True)
    top_video_emotions = valid_video_emotions[:max_video_emotions]

    text_emotions = []
    if text_pred.get("primary"):
        text_emotions.append(text_pred["primary"])
    text_emotions.extend(text_pred.get("secondary", []))
    
    emotion_dict = {}

    for v_emo in top_video_emotions:
        label = v_emo["label"]
        if label not in emotion_dict:
            emotion_dict[label] = {
                "label": label,
                "confidence": round(v_emo["confidence"], 2),
                "source": "video"
            }
    
    for t_emo in text_emotions:
        label = t_emo["label"]
        if label not in emotion_dict and t_emo["confidence"] >= min_conf * 0.8:  
            emotion_dict[label] = {
                "label": label,
                "confidence": round(t_emo["confidence"], 2),
                "source": "text"
            }
    
    all_emotions = list(emotion_dict.values())
    all_emotions.sort(key=lambda x: x["confidence"], reverse=True)
    all_emotions = all_emotions[:max_total_emotions]

    if all_emotions:
        primary = all_emotions[0]
        secondary = all_emotions[1:]
    else:
        primary = {"label": "neutral", "confidence": 0.5, "source": "default"}
        secondary = []
    
    return {
        "primary": primary,
        "secondary": secondary,
        "metadata": {
            "video_count": len(top_video_emotions),
            "text_count": len([e for e in all_emotions if e["source"] == "text"]),
            "total_count": len(all_emotions)
        }
    }

def build_empathy(fused: Dict, text: str) -> str:

    primary = fused['primary']
    emotion_desc = f"주 감정: {primary['label']} (신뢰도 {primary['confidence']})"
    
    if fused.get('secondary'):
        secondary_labels = [f"{s['label']}({s['confidence']})" for s in fused['secondary']]
        emotion_desc += f"\n보조 감정: {', '.join(secondary_labels[:3])}"  
    
    video_dominant = primary.get('source') == 'video'
    source_hint = "표정에서 주로 읽히는" if video_dominant else "말씀에서 주로 느껴지는"
    
    prompt = f"""
아래 정보를 바탕으로 고령자에게 공감과 격려를 전하는 짧은 문단을 만드세요.

지침:
- 존댓말, 쉬운 어휘 사용, 200자 이내
- 이모지/특수기호/괄호 사용 금지
- 구조: 감사 → 감정 인식 → 공감/격려 → 맞춤 제안
- {source_hint} 감정을 중심으로 응답

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

def make_emo_mission(fused: Dict, text: str, user_profile: Optional[UserProfile]=None) -> Dict:
    emotion_desc = f"주 감정: {fused['primary']['label']} (신뢰도 {fused['primary']['confidence']})"
    if fused.get('secondary'):
        emotion_desc += " / 보조: " + ", ".join([s['label'] for s in fused['secondary']])

    user_ctx = ""
    if user_profile:
        user_ctx = f"""
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
{user_ctx}
{emotion_desc}
말씀 요지: {text[:120]}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type":"json_object"},
        temperature=0.3,
        messages=[
            {"role":"system","content":"You write short, safe, elder-friendly tasks in Korean."},
            {"role":"user","content":prompt},
        ],
    )
    return json.loads(resp.choices[0].message.content)

# 기존 run_core_pipeline 교체
def run_core_pipeline(video_path: str) -> Dict:
    text = transcribe_video(video_path)
    t_pred = classify_text_emotion(text)
    fused = fuse_emotion(vid_pred=None, txt_pred=t_pred)  # ← 지금은 비디오 감정 없음(폴백)

    empathy = build_empathy(fused, text)
    return {
        "stt_text": text,
        "text_emotion": t_pred,
        "fused_emotion": fused,     # ← 누락됐던 키 추가
        "empathy": empathy
    }


def generate_emo_mission(core_result: Dict, user_profile: Optional[UserProfile]=None) -> Dict:
    return make_emo_mission(core_result["fused_emotion"], core_result["stt_text"], user_profile)
