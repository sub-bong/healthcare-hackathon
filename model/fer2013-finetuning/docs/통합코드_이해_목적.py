#!/usr/bin/env python3
"""
FER2013 감정 인식 모델 통합 실행 코드
헬스케어 해커톤 - 감정 분석 모델

이 파일은 fer2013-finetuning의 모든 모듈을 통합하여
단일 파일로 실행 가능하도록 구성되었습니다.

또한 각 기능들을 상세히 설명하여 코드 이해를 돕기 위한 파일입니다.

FER2013 (Facial Expression Recognition 2013) 데이터셋:
- Kaggle에서 제공하는 얼굴 감정 인식 데이터셋
- 7가지 감정 클래스: 화남, 혐오, 두려움, 행복, 중립, 슬픔, 놀람
- 48x48 픽셀 그레이스케일 이미지로 구성
"""

# 필요한 라이브러리 임포트
import os                    # 파일 시스템 작업용
import sys                   # 시스템 관련 파라미터와 함수
import torch                 # PyTorch 메인 라이브러리
import torch.nn as nn        # 신경망 레이어와 함수들
import torch.optim as optim  # 최적화 알고리즘들 (Adam, SGD 등)
import timm                  # PyTorch Image Models - 사전 훈련된 모델 라이브러리
from torch.utils.data import Dataset, DataLoader  # 데이터 로딩 유틸리티
from torchvision import transforms                 # 이미지 전처리 변환 함수들
from PIL import Image        # Python Imaging Library - 이미지 처리
from tqdm import tqdm        # 진행률 표시 바
import matplotlib.pyplot as plt  # 그래프 시각화

# ===============================
# Configuration - 모델 학습에 필요한 모든 설정값들
# ===============================
class Config:
    """
    모델 및 학습 설정 클래스
    모든 하이퍼파라미터와 설정값을 한 곳에서 관리
    """
    
    # ========== 데이터 관련 설정 ==========
    DATA_DIR = './fer2013'  # FER2013 데이터셋이 저장된 디렉토리 경로
    BATCH_SIZE = 256        # 한 번에 처리할 이미지 개수 (GPU 메모리에 따라 조정)
    NUM_WORKERS = 4         # 데이터 로딩에 사용할 CPU 프로세스 개수
    
    # ========== 모델 관련 설정 ==========
    # MobileNetV4 모델명 해석:
    # - hf_hub: Hugging Face Hub에서 제공하는 모델
    # - timm: PyTorch Image Models 라이브러리
    # - mobilenetv4: Google의 MobileNet 아키텍처 4번째 버전
    # - conv_medium: 중간 크기의 컨볼루션 버전 (small < medium < large)
    # - e500: 500 에포크 동안 사전 훈련됨
    # - r224: 입력 해상도 224x224 픽셀
    # - in1k: ImageNet-1K 데이터셋으로 사전 훈련됨 (1000개 클래스)
    MODEL_NAME = "hf_hub:timm/mobilenetv4_conv_medium.e500_r224_in1k"
    NUM_CLASSES = 7         # FER2013의 감정 클래스 개수 (7가지 감정)
    
    # ========== 학습 관련 설정 ==========
    # GPU가 사용 가능하면 GPU를, 없으면 CPU를 사용
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== 1단계: 분류기 학습 설정 ==========
    # Transfer Learning의 첫 번째 단계
    # 사전 훈련된 백본은 고정하고 새로운 분류기만 학습
    CLASSIFIER_EPOCHS = 10  # 분류기 학습 에포크 수
    CLASSIFIER_LR = 1e-3    # 분류기 학습률 (0.001)
    
    # ========== 2단계: 전체 네트워크 미세조정 설정 ==========
    # 전체 네트워크를 미세하게 조정하는 단계
    FULL_NETWORK_EPOCHS = 15    # 전체 네트워크 학습 에포크 수
    BACKBONE_LR = 1e-5          # 백본 네트워크 학습률 (0.00001) - 매우 작게 설정
    CLASSIFIER_LR_FINETUNE = 1e-4  # 분류기 미세조정 학습률 (0.0001)

# ===============================
# Model Architecture - 감정 분류 모델 구조 정의
# ===============================
def create_emotion_classifier(num_classes=7, backbone_name="hf_hub:timm/mobilenetv4_conv_medium.e500_r224_in1k"):
    """
    감정 분류를 위한 딥러닝 모델 생성 함수
    
    Transfer Learning 방식을 사용:
    1. ImageNet으로 사전 훈련된 MobileNetV4 백본 사용
    2. 마지막 분류기 레이어를 7개 감정 클래스용으로 교체
    
    Args:
        num_classes (int): 분류할 감정 클래스 개수 (기본값: 7)
        backbone_name (str): 사용할 백본 모델명
    
    Returns:
        torch.nn.Module: 감정 분류용으로 수정된 모델
    """
    
    # timm 라이브러리에서 사전 훈련된 MobileNetV4 모델 로드
    # pretrained=True: ImageNet으로 사전 훈련된 가중치 사용
    model = timm.create_model(backbone_name, pretrained=True)
    
    # 기존 분류기의 입력 특성 수 확인 (MobileNetV4의 경우 1280개)
    in_features = model.classifier.in_features
    
    # 기존 분류기를 새로운 3층 분류기로 교체
    # 구조: 1280 → 512 → 128 → 7 (감정 클래스)
    model.classifier = nn.Sequential(
        # 첫 번째 층: 특성 차원 축소 (1280 → 512)
        nn.Linear(in_features, 512),    # 완전 연결층 (Fully Connected)
        nn.BatchNorm1d(512),            # 배치 정규화 (학습 안정화)
        nn.ReLU(),                      # ReLU 활성화 함수 (비선형성 추가)
        nn.Dropout(0.5),                # 드롭아웃 50% (과적합 방지)
        
        # 두 번째 층: 추가 특성 축소 (512 → 128)
        nn.Linear(512, 128),            # 완전 연결층
        nn.BatchNorm1d(128),            # 배치 정규화
        nn.ReLU(),                      # ReLU 활성화 함수
        nn.Dropout(0.5),                # 드롭아웃 50%
        
        # 출력 층: 최종 분류 (128 → 7개 감정 클래스)
        nn.Linear(128, num_classes)     # 최종 분류를 위한 완전 연결층
        # 참고: 마지막 층에는 활성화 함수 없음 (CrossEntropyLoss에서 처리)
    )
    
    return model

# ===============================
# Dataset and DataLoader - 데이터 로딩 및 전처리
# ===============================
class FERDataset(Dataset):
    """
    FER2013 감정 인식 데이터셋 클래스
    PyTorch의 Dataset 클래스를 상속받아 커스텀 데이터셋 구현
    
    데이터셋 구조:
    fer2013/
    ├── train/          # 학습용 데이터
    │   ├── angry/      # 화난 표정 이미지들
    │   ├── disgust/    # 혐오 표정 이미지들
    │   ├── fear/       # 두려운 표정 이미지들
    │   ├── happy/      # 행복한 표정 이미지들
    │   ├── neutral/    # 중립 표정 이미지들
    │   ├── sad/        # 슬픈 표정 이미지들
    │   └── surprise/   # 놀란 표정 이미지들
    └── test/           # 검증용 데이터 (같은 구조)
    """
    
    def __init__(self, root_dir, is_train=True, transform=None):
        """
        데이터셋 초기화
        
        Args:
            root_dir (str): 데이터셋 루트 디렉토리 경로
            is_train (bool): True면 학습용, False면 검증용 데이터
            transform (callable): 이미지 전처리 변환 함수
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        
        # 학습/검증 데이터 디렉토리 경로 설정
        data_dir = os.path.join(root_dir, 'train' if is_train else 'test')
        
        # 이미지 경로와 레이블을 저장할 리스트
        self.image_paths = []
        self.labels = []
        
        # 감정 클래스명을 숫자 레이블로 매핑하는 딕셔너리
        # 딥러닝 모델은 숫자 레이블로 학습하므로 문자열을 숫자로 변환
        self.emotion_map = {
            'angry': 0,     # 화남
            'disgust': 1,   # 혐오
            'fear': 2,      # 두려움
            'happy': 3,     # 행복
            'neutral': 4,   # 중립
            'sad': 5,       # 슬픔
            'surprise': 6   # 놀람
        }
        
        # 각 감정 폴더에서 이미지 파일들을 찾아서 경로와 레이블 수집
        for emotion in self.emotion_map.keys():
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                # 해당 감정 폴더의 모든 이미지 파일 처리
                for img_name in os.listdir(emotion_dir):
                    # 이미지 파일 확장자 확인 (jpg, jpeg, png만 처리)
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        # 이미지 전체 경로 저장
                        self.image_paths.append(os.path.join(emotion_dir, img_name))
                        # 해당 감정의 숫자 레이블 저장
                        self.labels.append(self.emotion_map[emotion])
    
    def __len__(self):
        """
        데이터셋의 총 샘플 개수 반환
        PyTorch DataLoader에서 자동으로 호출됨
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        주어진 인덱스의 데이터 샘플 반환
        PyTorch DataLoader에서 배치 생성 시 자동으로 호출됨
        
        Args:
            idx (int): 가져올 데이터의 인덱스
            
        Returns:
            tuple: (이미지 텐서, 감정 레이블) 쌍
        """
        # 인덱스에 해당하는 이미지 파일 경로 가져오기
        img_path = self.image_paths[idx]
        
        # PIL을 사용하여 이미지 로드 및 RGB 형식으로 변환
        # convert('RGB'): 그레이스케일이나 다른 형식을 RGB로 통일
        image = Image.open(img_path).convert('RGB')
        
        # 해당 이미지의 감정 레이블 가져오기
        label = self.labels[idx]
        
        # 이미지 전처리 변환 적용 (있는 경우)
        # transform에는 크기 조정, 정규화, 데이터 증강 등이 포함됨
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(is_train=True):
    """
    이미지 전처리 변환 함수들을 정의
    
    Args:
        is_train (bool): True면 학습용 변환, False면 검증용 변환
        
    Returns:
        transforms.Compose: 전처리 변환들의 조합
    """
    if is_train:
        # 학습용 변환: 데이터 증강(Data Augmentation) 포함
        return transforms.Compose([
            # 1. 이미지 크기를 224x224로 조정 (MobileNetV4 입력 크기)
            transforms.Resize((224, 224)),
            
            # 2. 데이터 증강 기법들 (학습 데이터 다양성 증가)
            transforms.RandomHorizontalFlip(),  # 50% 확률로 좌우 반전
            transforms.RandomRotation(10),      # ±10도 범위에서 무작위 회전
            transforms.ColorJitter(             # 색상, 밝기 무작위 변경
                brightness=0.2,  # 밝기 ±20% 변경
                contrast=0.2     # 대비 ±20% 변경
            ),
            
            # 3. PIL 이미지를 PyTorch 텐서로 변환 (0-255 → 0-1 정규화)
            transforms.ToTensor(),
            
            # 4. ImageNet 통계값으로 정규화 (사전 훈련된 모델과 일치시키기 위해)
            # RGB 각 채널별 평균과 표준편차
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet RGB 평균값
                std=[0.229, 0.224, 0.225]    # ImageNet RGB 표준편차
            )
        ])
    else:
        # 검증용 변환: 데이터 증강 없이 기본 전처리만
        return transforms.Compose([
            # 1. 이미지 크기 조정
            transforms.Resize((224, 224)),
            
            # 2. 텐서 변환
            transforms.ToTensor(),
            
            # 3. 정규화 (학습용과 동일)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    학습용과 검증용 데이터로더 생성
    
    Args:
        data_dir (str): 데이터셋 루트 디렉토리 경로
        batch_size (int): 배치 크기 (한 번에 처리할 이미지 수)
        num_workers (int): 데이터 로딩에 사용할 CPU 프로세스 수
        
    Returns:
        tuple: (학습용 DataLoader, 검증용 DataLoader)
    """
    
    # 학습용 데이터셋 생성 (데이터 증강 포함)
    train_dataset = FERDataset(
        root_dir=data_dir,
        is_train=True,                          # 학습 모드
        transform=get_transforms(is_train=True) # 데이터 증강 포함된 변환
    )
    
    # 검증용 데이터셋 생성 (데이터 증강 없음)
    val_dataset = FERDataset(
        root_dir=data_dir,
        is_train=False,                          # 검증 모드
        transform=get_transforms(is_train=False) # 기본 변환만
    )
    
    # 학습용 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 배치 크기
        shuffle=True,           # 학습 시 데이터 순서를 무작위로 섞음 (중요!)
        num_workers=num_workers,# 멀티프로세싱으로 데이터 로딩 속도 향상
        pin_memory=True         # GPU 전송 속도 향상 (CUDA 사용 시)
    )
    
    # 검증용 데이터로더 생성
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # 배치 크기
        shuffle=False,          # 검증 시에는 순서 유지 (재현성 위해)
        num_workers=num_workers,# 멀티프로세싱
        pin_memory=True         # GPU 전송 속도 향상
    )
    
    return train_loader, val_loader

# ===============================
# Training Logic - 모델 학습 로직
# ===============================
class EmotionTrainer:
    """
    감정 인식 모델 학습을 담당하는 클래스
    2단계 Transfer Learning 방식으로 학습 진행:
    1단계: 분류기만 학습 (백본 고정)
    2단계: 전체 네트워크 미세조정
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        """
        트레이너 초기화
        
        Args:
            model: 학습할 모델
            train_loader: 학습용 데이터로더
            val_loader: 검증용 데이터로더
            device: 학습에 사용할 디바이스 ('cuda' 또는 'cpu')
        """
        self.model = model.to(device)  # 모델을 지정된 디바이스로 이동
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def train_classifier_only(self, epochs=10, lr=1e-3):
        """
        1단계: 분류기 레이어만 학습 (백본 네트워크는 고정)
        
        Transfer Learning의 첫 번째 단계로, 사전 훈련된 특성 추출기는 
        그대로 사용하고 새로운 작업(감정 분류)에 맞는 분류기만 학습
        
        Args:
            epochs (int): 학습 에포크 수
            lr (float): 학습률
            
        Returns:
            dict: 학습 히스토리 (손실, 정확도 기록)
        """
        
        # 백본 네트워크의 모든 파라미터를 고정 (gradient 계산 안함)
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:  # 분류기가 아닌 모든 레이어
                param.requires_grad = False  # 가중치 업데이트 비활성화
        
        # 손실 함수: 다중 클래스 분류용 CrossEntropy
        criterion = nn.CrossEntropyLoss()
        
        # 최적화 알고리즘: Adam (분류기 파라미터만 대상)
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=lr)
        
        # 학습률 스케줄러: 5 에포크마다 학습률을 50% 감소
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        return self._train(criterion, optimizer, scheduler, epochs)
    
    def train_full_network(self, epochs=15, backbone_lr=1e-5, classifier_lr=1e-4):
        """
        2단계: 전체 네트워크 미세조정 (Fine-tuning)
        
        1단계에서 학습된 분류기를 바탕으로 전체 네트워크를 미세하게 조정
        백본과 분류기에 서로 다른 학습률을 적용하여 안정적인 학습 진행
        
        Args:
            epochs (int): 학습 에포크 수
            backbone_lr (float): 백본 네트워크 학습률 (작게 설정)
            classifier_lr (float): 분류기 학습률 (상대적으로 크게 설정)
            
        Returns:
            dict: 학습 히스토리 (손실, 정확도 기록)
        """
        
        # 모든 파라미터를 학습 가능하도록 설정
        for param in self.model.parameters():
            param.requires_grad = True
            
        # 백본과 분류기 파라미터를 분리하여 다른 학습률 적용
        backbone_params = []    # 백본 네트워크 파라미터들
        classifier_params = []  # 분류기 파라미터들
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # 차별적 학습률을 가진 옵티마이저 설정
        # 백본은 이미 좋은 특성을 학습했으므로 작은 학습률 사용
        # 분류기는 새로 학습하므로 상대적으로 큰 학습률 사용
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': backbone_lr},      # 백본: 작은 학습률
            {'params': classifier_params, 'lr': classifier_lr}   # 분류기: 큰 학습률
        ])
        
        # 손실 함수
        criterion = nn.CrossEntropyLoss()
        
        # 학습률 스케줄러: 7 에포크마다 학습률을 30% 감소
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
        
        return self._train(criterion, optimizer, scheduler, epochs)
    
    def _train(self, criterion, optimizer, scheduler, epochs):
        """
        실제 학습 루프를 실행하는 내부 메서드
        
        Args:
            criterion: 손실 함수
            optimizer: 최적화 알고리즘
            scheduler: 학습률 스케줄러
            epochs: 학습할 에포크 수
            
        Returns:
            dict: 학습 히스토리 (각 에포크별 손실과 정확도 기록)
        """
        
        best_val_acc = 0.0  # 최고 검증 정확도 추적용
        
        # 학습 기록을 저장할 딕셔너리
        history = {
            'train_loss': [],  # 학습 손실 기록
            'train_acc': [],   # 학습 정확도 기록
            'val_loss': [],    # 검증 손실 기록
            'val_acc': []      # 검증 정확도 기록
        }
        
        # 각 에포크별 학습 진행
        for epoch in range(epochs):
            
            # ========== 훈련 단계 ==========
            self.model.train()  # 모델을 학습 모드로 설정 (dropout, batchnorm 활성화)
            train_metrics = self._train_epoch(criterion, optimizer)
            
            # ========== 검증 단계 ==========
            self.model.eval()   # 모델을 평가 모드로 설정 (dropout, batchnorm 비활성화)
            val_metrics = self._validate_epoch(criterion)
            
            # ========== 메트릭 기록 ==========
            # 학습 메트릭 저장
            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)
            # 검증 메트릭 저장
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)
            
            # ========== 진행 상황 출력 ==========
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}%")
            
            # ========== 최고 성능 모델 저장 ==========
            # 검증 정확도가 이전 최고 기록을 갱신하면 모델 저장
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                torch.save(self.model.state_dict(), 'best_model.pth')  # 모델 가중치 저장
                print(f"  새로운 최고 성능! 모델 저장됨 (Val Acc: {val_metrics['acc']:.2f}%)")
            
            # ========== 학습률 업데이트 ==========
            scheduler.step()  # 스케줄러에 따라 학습률 조정
            print()
        
        return history
    
    def _train_epoch(self, criterion, optimizer):
        """
        한 에포크 동안의 학습을 수행하는 메서드
        
        Args:
            criterion: 손실 함수
            optimizer: 최적화 알고리즘
            
        Returns:
            dict: 학습 손실과 정확도를 담은 딕셔너리
        """
        
        running_loss = 0.0  # 누적 손실값
        correct = 0         # 올바르게 예측한 샘플 수
        total = 0           # 전체 샘플 수
        
        # 학습 데이터를 배치 단위로 처리
        for data, target in tqdm(self.train_loader, desc='Training'):
            
            # ========== 데이터를 GPU로 이동 ==========
            # non_blocking=True: CPU-GPU 데이터 전송을 비동기적으로 수행 (속도 향상)
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # ========== 기울기 초기화 ==========
            # 이전 배치의 기울기를 제거 (PyTorch는 기울기를 누적하므로)
            optimizer.zero_grad()
            
            # ========== 순전파 (Forward Pass) ==========
            # 모델에 입력 데이터를 통과시켜 예측값 계산
            output = self.model(data)

            # ========== 손실 계산 ==========
            # 예측값과 실제값 사이의 차이(손실) 계산
            loss = criterion(output, target)
            
            # ========== 역전파 (Backward Pass) ==========
            # 손실에 대한 각 파라미터의 기울기 계산
            loss.backward()
            
            # ========== 파라미터 업데이트 ==========
            # 계산된 기울기를 사용하여 모델 파라미터 업데이트
            optimizer.step()
            
            # ========== 메트릭 계산 ==========
            running_loss += loss.item()  # 배치 손실을 누적 손실에 추가
            
            # 예측 클래스 계산 (가장 높은 확률의 클래스)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)  # 전체 샘플 수 증가
            correct += (predicted == target).sum().item()  # 정답 개수 증가
        
        # 평균 손실과 정확도 반환
        return {
            'loss': running_loss / len(self.train_loader),  # 평균 손실
            'acc': 100. * correct / total                   # 정확도 (%)
        }
    
    def _validate_epoch(self, criterion):
        """
        한 에포크 동안의 검증을 수행하는 메서드
        
        Args:
            criterion: 손실 함수
            
        Returns:
            dict: 검증 손실과 정확도를 담은 딕셔너리
        """
        
        running_loss = 0.0  # 누적 손실값
        correct = 0         # 올바르게 예측한 샘플 수
        total = 0           # 전체 샘플 수
        
        # ========== 검증 모드 ==========
        # torch.no_grad(): 기울기 계산을 비활성화하여 메모리 절약 및 속도 향상
        # 검증 시에는 파라미터 업데이트가 없으므로 기울기 계산 불필요
        with torch.no_grad():
            
            # 검증 데이터를 배치 단위로 처리
            for data, target in tqdm(self.val_loader, desc='Validating'):
                
                # ========== 데이터를 GPU로 이동 ==========
                data, target = data.to(self.device), target.to(self.device)
                
                # ========== 순전파 (Forward Pass) ==========
                # 모델에 입력 데이터를 통과시켜 예측값 계산
                output = self.model(data)
                
                # ========== 손실 계산 ==========
                # 예측값과 실제값 사이의 차이(손실) 계산
                loss = criterion(output, target)
                
                # ========== 메트릭 계산 ==========
                running_loss += loss.item()  # 배치 손실을 누적 손실에 추가
                
                # 예측 클래스 계산 (가장 높은 확률의 클래스)
                _, predicted = torch.max(output.data, 1)
                
                total += target.size(0)  # 전체 샘플 수 증가
                correct += (predicted == target).sum().item()  # 정답 개수 증가
        
        # 평균 손실과 정확도 반환
        return {
            'loss': running_loss / len(self.val_loader),  # 평균 손실
            'acc': 100. * correct / total                 # 정확도 (%)
        }
    
    @staticmethod
    def plot_history(history):
        """
        학습 과정의 손실과 정확도 변화를 시각화하는 정적 메서드
        
        Args:
            history (dict): 학습 히스토리 딕셔너리
                - 'train_loss': 에포크별 학습 손실 리스트
                - 'val_loss': 에포크별 검증 손실 리스트  
                - 'train_acc': 에포크별 학습 정확도 리스트
                - 'val_acc': 에포크별 검증 정확도 리스트
        """
        
        # 2개의 서브플롯을 가진 그래프 생성 (가로 12, 세로 4 인치)
        plt.figure(figsize=(12, 4))
        
        # ========== 손실 그래프 (왼쪽) ==========
        plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째
        plt.plot(history['train_loss'], label='Train Loss')  # 학습 손실 곡선
        plt.plot(history['val_loss'], label='Val Loss')      # 검증 손실 곡선
        plt.title('Loss History')    # 그래프 제목
        plt.xlabel('Epoch')          # x축 레이블
        plt.ylabel('Loss')           # y축 레이블
        plt.legend()                 # 범례 표시
        
        # ========== 정확도 그래프 (오른쪽) ==========
        plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째
        plt.plot(history['train_acc'], label='Train Acc')    # 학습 정확도 곡선
        plt.plot(history['val_acc'], label='Val Acc')        # 검증 정확도 곡선
        plt.title('Accuracy History') # 그래프 제목
        plt.xlabel('Epoch')           # x축 레이블
        plt.ylabel('Accuracy (%)')    # y축 레이블
        plt.legend()                  # 범례 표시
        
        # 서브플롯들 간의 간격 자동 조정
        plt.tight_layout()
        
        # 그래프 화면에 출력
        plt.show()

# ===============================
# Inference Functions - 학습된 모델을 사용한 추론 함수들
# ===============================
def load_trained_model(model_path, device='cuda'):
    """
    저장된 모델 가중치를 로드하여 추론용 모델 생성
    
    Args:
        model_path (str): 저장된 모델 파일 경로 (.pth 파일)
        device (str): 모델을 로드할 디바이스 ('cuda' 또는 'cpu')
        
    Returns:
        torch.nn.Module: 로드된 모델 (평가 모드로 설정됨)
    """
    
    # 동일한 구조의 새 모델 생성
    model = create_emotion_classifier()
    
    # 저장된 가중치 로드 (디바이스 호환성을 위해 map_location 사용)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 모델을 지정된 디바이스로 이동
    model.to(device)
    
    # 평가 모드로 설정 (dropout, batchnorm 비활성화)
    model.eval()
    
    return model

def predict_emotion(model, image_path, device='cuda'):
    """
    단일 이미지에 대한 감정 예측 수행
    
    Args:
        model: 학습된 감정 분류 모델
        image_path (str): 예측할 이미지 파일 경로
        device (str): 추론에 사용할 디바이스
        
    Returns:
        dict: 예측 결과를 담은 딕셔너리
            - 'emotion': 예측된 감정 클래스명
            - 'confidence': 예측 신뢰도 (0~1)
            - 'all_probabilities': 모든 감정 클래스별 확률
    """
    
    # 7가지 감정 클래스 레이블 (숫자 인덱스에 대응)
    emotion_labels = [
        'angry',     # 0: 화남
        'disgust',   # 1: 혐오
        'fear',      # 2: 두려움
        'happy',     # 3: 행복
        'neutral',   # 4: 중립
        'sad',       # 5: 슬픔
        'surprise'   # 6: 놀람
    ]
    
    # ========== 이미지 전처리 ==========
    # 검증용 전처리 변환 적용 (데이터 증강 없음)
    transform = get_transforms(is_train=False)
    
    # 이미지 로드 및 RGB 변환
    image = Image.open(image_path).convert('RGB')
    
    # 전처리 적용 후 배치 차원 추가 (1, 3, 224, 224)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # ========== 예측 수행 ==========
    # 기울기 계산 비활성화 (추론 시에는 불필요)
    with torch.no_grad():
        
        # 모델 순전파로 예측값 계산
        output = model(image_tensor)
        
        # 로짓을 확률로 변환 (softmax 적용)
        probabilities = torch.softmax(output, dim=1)
        
        # 가장 높은 확률의 클래스 인덱스 찾기
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # 예측된 클래스의 신뢰도 (확률값)
        confidence = probabilities[0][predicted_class].item()
    
    # ========== 결과 반환 ==========
    return {
        'emotion': emotion_labels[predicted_class],  # 예측된 감정명
        'confidence': confidence,                    # 신뢰도
        'all_probabilities': {                       # 모든 클래스별 확률
            emotion_labels[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
    }

# ===============================
# Main Execution - 프로그램 메인 실행부
# ===============================
def main():
    """
    FER2013 감정 인식 모델의 전체 학습 파이프라인을 실행하는 메인 함수
    
    실행 단계:
    1. 환경 설정 및 데이터 확인
    2. 데이터로더 생성 
    3. 모델 생성
    4. 1단계 학습: 분류기만 학습 (백본 고정)
    5. 2단계 학습: 전체 네트워크 미세조정
    6. 학습 결과 시각화
    """
    
    # ========== 시작 메시지 및 환경 정보 출력 ==========
    print("🎭 FER2013 감정 인식 모델 학습 시작")
    print(f"사용 디바이스: {Config.DEVICE}")
    print(f"PyTorch 버전: {torch.__version__}")
    
    # ========== 데이터셋 존재 여부 확인 ==========
    if not os.path.exists(Config.DATA_DIR):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {Config.DATA_DIR}")
        print("FER2013 데이터셋을 다운로드하고 올바른 경로에 배치해주세요.")
        print("\n데이터셋 구조:")
        print("fer2013/")
        print("├── train/")
        print("│   ├── angry/")
        print("│   ├── disgust/")
        print("│   ├── fear/")
        print("│   ├── happy/")
        print("│   ├── neutral/")
        print("│   ├── sad/")
        print("│   └── surprise/")
        print("└── test/ (같은 구조)")
        return
    
    try:
        # ========== 1. 데이터 로더 생성 ==========
        print("\n📊 데이터 로더 생성 중...")
        train_loader, val_loader = create_dataloaders(
            data_dir=Config.DATA_DIR,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
        print(f"✅ 데이터 로더 생성 완료 (배치 크기: {Config.BATCH_SIZE})")
        
        # ========== 2. 모델 생성 ==========
        print("\n🤖 모델 생성 중...")
        model = create_emotion_classifier(
            num_classes=Config.NUM_CLASSES,
            backbone_name=Config.MODEL_NAME
        )
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ 모델 생성 완료")
        print(f"   전체 파라미터: {total_params:,}")
        print(f"   학습 가능 파라미터: {trainable_params:,}")
        
        # ========== 3. 트레이너 초기화 ==========
        print(f"\n🔧 트레이너 초기화 중... (디바이스: {Config.DEVICE})")
        trainer = EmotionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=Config.DEVICE
        )
        print("✅ 트레이너 초기화 완료")
        
        # ========== 4. 1단계 학습: 분류기만 학습 ==========
        print(f"\n🎯 1단계 학습 시작: 분류기 레이어만 학습 ({Config.CLASSIFIER_EPOCHS} 에포크)")
        print("   - 백본 네트워크: 고정 (가중치 업데이트 안함)")
        print("   - 분류기 레이어: 학습 (새로운 작업에 맞게 조정)")
        
        history1 = trainer.train_classifier_only(
            epochs=Config.CLASSIFIER_EPOCHS,
            lr=Config.CLASSIFIER_LR
        )
        print("✅ 1단계 학습 완료!")
        
        # ========== 5. 2단계 학습: 전체 네트워크 미세조정 ==========
        print(f"\n🔬 2단계 학습 시작: 전체 네트워크 미세조정 ({Config.FULL_NETWORK_EPOCHS} 에포크)")
        print(f"   - 백본 네트워크: 미세조정 (학습률: {Config.BACKBONE_LR})")
        print(f"   - 분류기 레이어: 계속 학습 (학습률: {Config.CLASSIFIER_LR_FINETUNE})")
        
        history2 = trainer.train_full_network(
            epochs=Config.FULL_NETWORK_EPOCHS,
            backbone_lr=Config.BACKBONE_LR,
            classifier_lr=Config.CLASSIFIER_LR_FINETUNE
        )
        print("✅ 2단계 학습 완료!")
        
        # ========== 6. 학습 결과 시각화 ==========
        print("\n📈 학습 결과 시각화...")
        print("1단계 학습 결과:")
        EmotionTrainer.plot_history(history1)
        
        print("2단계 학습 결과:")
        EmotionTrainer.plot_history(history2)
        
        # ========== 학습 완료 메시지 ==========
        print("\n🎉 모든 학습이 성공적으로 완료되었습니다!")
        print("\n📁 저장된 파일:")
        print("  - best_model.pth: 최고 성능을 달성한 모델 가중치")
        print("\n💡 모델 사용 예시:")
        print("  model = load_trained_model('best_model.pth')")
        print("  result = predict_emotion(model, 'image.jpg')")
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류가 발생했습니다: {str(e)}")
        print("\n🔍 상세 오류 정보:")
        import traceback
        traceback.print_exc()
        print("\n💡 해결 방법:")
        print("  1. 데이터셋 경로가 올바른지 확인")
        print("  2. GPU 메모리가 충분한지 확인 (배치 크기 줄이기)")
        print("  3. 필요한 라이브러리가 모두 설치되었는지 확인")

if __name__ == "__main__":
    main()
