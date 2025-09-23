# src/demo.py
from .models.onboarding import UserProfile
from .chains.missions import mission_chain
from .chains.quiz import CognitiveQuizService
from .utils.stt import transcribe_ko
from .chains.emo_pipeline import run_core_pipeline, generate_emo_mission
from pathlib import Path
import os
import tempfile
import subprocess
import shutil

# ---- demo 전용 ffmpeg 경로 주입 (Windows) ----
FFMPEG_BIN = r"C:\Tools\ffmpeg\ffmpeg-8.0-essentials_build\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
assert shutil.which("ffmpeg"), "ffmpeg not found: FFMPEG_BIN 경로를 확인하세요."

def extract_audio_pcm16(input_video: str) -> str:
    """비디오에서 오디오 추출"""
    td = tempfile.mkdtemp(prefix="stt_")
    wav_path = os.path.join(td, "audio.wav")
    ffmpeg_path = r"C:\Tools\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
    cmd = [ffmpeg_path, "-y", "-i", input_video, "-vn", "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_path

def demo_mission():
    """미션 데모"""
    profile = UserProfile(
        mobility_issue=True,
        living_arrangement="alone",
        wake_up_time="07:00",
        bed_time="22:00",
    )
    mission = mission_chain.invoke(profile)
    print("[미션]")
    print(mission if mission else "취침 이후라 미션 없음")

def demo_quiz():
    """퀴즈 데모"""
    svc = CognitiveQuizService()
    question = svc.generate_quiz()
    print("\n[문제]")
    print(question)

    # 답변 채점
    audio_path = "sample_answer.wav"
    if os.path.exists(audio_path):
        stt_text = transcribe_ko(audio_path)
        print("\n[STT 결과]")
        print(stt_text or "(빈 텍스트)")
        result = svc.grade_answer(stt_text or "")
    else:
        result = svc.grade_answer("9 2 5")

    print("\n[채점]")
    print("정답여부:", result["is_correct"])
    print("피드백:", result["feedback"])
    print("정답:", result["correct_answer"])

def demo_emotion():
    """감정 분석 데모"""
    video_path = Path("assets") / "test_vid.mp4"
    
    if not video_path.exists():
        print(f"[오류] 비디오 파일이 없습니다: {video_path}")
        return
    
    core = run_core_pipeline(str(video_path))
    print("\n[STT]", core["stt_text"])
    print("[융합감정]", core["fused_emotion"])
    print("[공감]", core["empathy"])

    # 감정 기반 미션 생성
    profile = None
    emo_mission = generate_emo_mission(core, profile)
    print("[감정 기반 미션]", emo_mission)

def main():
    """메인 데모 함수"""
    print("=== 온라인 해커톤 데모 ===")
    
    # 1) 미션 데모
    demo_mission()
    
    # 2) 퀴즈 데모
    demo_quiz()
    
    # 3) 감정 분석 데모
    demo_emotion()

if __name__ == "__main__":
    main()