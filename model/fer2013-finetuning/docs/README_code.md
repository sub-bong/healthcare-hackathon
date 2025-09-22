# MobileNetv4

## 🎯 주요 기능

### 감정 인식 모델
- **백본 모델**: MobileNetV4 (효율적인 모바일 최적화 모델)
- **감정 클래스**: 7가지 (angry, disgust, fear, happy, neutral, sad, surprise)
- **학습 전략**: 2단계 전이 학습
  1. 분류기 레이어만 학습 (백본 고정)
  2. 전체 네트워크 미세조정

### 모델 성능 특징
- **입력 크기**: 224x224 RGB 이미지
- **출력**: 7개 감정 클래스에 대한 확률 분포
- **최적화**: GPU 메모리 효율성 및 추론 속도 최적화

## 🚀 빠른 시작

### 1. 환경 설정
```bash
cd model/fer2013-finetuning
pip install -r requirements.txt
```

### 2. 통합 코드 실행
```bash
cd model
python integrated_model.py
```

```

## 📊 모델 아키텍처

```
MobileNetV4 Backbone (사전 훈련된 가중치)
    ↓
Classifier Head:
    Linear(1280 → 512) + BatchNorm + ReLU + Dropout(0.5)
    ↓
    Linear(512 → 128) + BatchNorm + ReLU + Dropout(0.5)
    ↓
    Linear(128 → 7) [감정 클래스]
```

## 🔧 설정 옵션

주요 하이퍼파라미터는 `fer2013-finetuning/configs/config.py`에서 관리됩니다:

- **배치 크기**: 256
- **학습률**: 분류기 1e-3, 백본 1e-5
- **에포크**: 분류기 10, 전체 15
- **데이터 증강**: 수평 뒤집기, 회전, 색상 조정

## 📈 학습 과정

### 1단계: 분류기 학습
- 백본 네트워크 가중치 고정
- 새로운 분류기 레이어만 학습
- 빠른 수렴 및 안정적인 초기화

### 2단계: 전체 네트워크 미세조정
- 모든 레이어 학습 가능
- 백본과 분류기에 다른 학습률 적용
- 성능 향상을 위한 세밀한 조정
