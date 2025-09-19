# FER2013 감정 인식 모델 학습

이 디렉토리는 FER2013 데이터셋을 사용하여 감정 인식 모델을 학습하기 위한 코드를 포함하고 있습니다.

## 디렉토리 구조

```
fer2013-finetuning/
├── configs/              # 모델 및 학습 설정 파일
├── docs/                           # 문서, 이해 관련 폴더
│   ├── model_weights_link.txt      # 학습된 모델 가중치 다운로드 링크
│   ├── README_code.md              # 모델 학습 관련 코드
│   └── 통합코드_이해_목적.py          # 통합 코드 및 코드 이해를 위한 파일
├── models/              # 모델 아키텍처 정의
│   └── classifier.py    # 감정 분류기 모델 구현
├── utils/                     # 유틸리티 함수들
│   ├── dataset.py             # 데이터셋 처리 관련 코드
│   └── trainer.py             # 모델 학습 관련 코드
├── MobileNet_v4_init_train.ipynb    # 초기 모델 학습 노트북
├── MobileNet_v4_extra-train.ipynb   # 추가 모델 학습 노트북
└── requirements.txt                 # 필요한 파이썬 패키지 목록
```

## 주요 파일 설명

### 학습 노트북
- `MobileNet_v4_init_train.ipynb`: MobileNet 기반 감정 인식 모델의 초기 학습을 위한 노트북
- `MobileNet_v4_extra-train.ipynb`: 추가적인 모델 성능 개선을 위한 학습 노트북

### 모델 가중치
- `model_weights_link.txt`: 학습된 모델의 가중치 파일이 저장된 구글 드라이브 링크를 포함

### 설정 및 유틸리티
- `configs/`: 모델 학습에 필요한 하이퍼파라미터 및 설정 파일
- `models/`: 모델 아키텍처 구현 코드
- `utils/`: 데이터 전처리, 학습 로직 등 유틸리티 함수

## 환경 설정

필요한 파이썬 패키지 설치:
```bash
pip install -r requirements.txt
```

주요 의존성:
- torch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- pillow >= 9.0.0
- matplotlib >= 3.5.0
- tqdm >= 4.65.0

## 모델 가중치 다운로드

학습된 모델 가중치는 다음 구글 드라이브 링크에서 다운로드할 수 있습니다:
[모델 가중치 다운로드](https://drive.google.com/drive/folders/1THUE1pM99-4yTcUZeqkjjcNPhpP4yzE_?usp=drive_link)
