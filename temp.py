import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm
model = timm.create_model("hf_hub:timm/mobilenetv4_conv_medium.e500_r224_in1k", pretrained=True)

# 분류기 형태 확인
# 1280개 -> 1000개로 out
model.classifier

in_features = model.classifier.in_features # 1280개

# 학습 가능한 분류기 만듦
# out_feature가 최종 7로 되도록
model.classifier = nn.Sequential(
    nn.Linear(1280, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 7)
)

model.classifier # (8): Linear(in_features=128, out_features=7, bias=True)

# 모델 파라미터 이름 확인
# 맨 하단에 분류기는 'classifier'라 명명됨
for name, param in model.named_parameters():
  print(name)

# 백본 고정 시
# if 'classifier' not in name:
# 쓸 예정

# 데이터로더
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class FERDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        """
        Args:
            root_dir (str): FER2013 데이터셋의 루트 디렉토리 경로
            is_train (bool): 학습/테스트 데이터 구분
            transform (callable, optional): 이미지에 적용할 변환
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        
        # 데이터 경로 설정
        data_dir = os.path.join(root_dir, 'train' if is_train else 'test')
        
        self.image_paths = []
        self.labels = []
        
        # 감정 클래스 매핑 (디렉토리명 → 숫자 레이블)
        self.emotion_map = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'neutral': 4,
            'sad': 5,
            'surprise': 6
        }
        
        # 이미지 경로와 레이블 수집
        for emotion in self.emotion_map.keys():
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(emotion_dir, img_name))
                        self.labels.append(self.emotion_map[emotion])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 데이터 변환 정의
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),  # MobileNetV4 입력 크기
            transforms.RandomHorizontalFlip(),  # 데이터 증강
            transforms.RandomRotation(10),      # 데이터 증강
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # 데이터 증강
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

# 데이터 로더 생성 함수
def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    데이터 로더 생성
    Args:
        data_dir (str): 데이터셋 경로
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩에 사용할 워커 수
    """
    # 데이터셋 생성
    train_dataset = FERDataset(
        root_dir=data_dir,
        is_train=True,
        transform=get_transforms(is_train=True)
    )
    
    val_dataset = FERDataset(
        root_dir=data_dir,
        is_train=False,
        transform=get_transforms(is_train=False)
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

data_dir = './fer2013'
train_loader, val_loader = create_dataloaders(
    data_dir=data_dir,
    batch_size=32,
    num_workers=4
)

def train_classifier_only(model, train_loader, val_loader, 
                          epochs=10, lr=1e-3, device='cuda'):
    '''분류기 레이어만 훈련하는 함수(백본 고정)'''
    model.to(device)

    # 백본 네트워크 파라미터 고정
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    # 분류기 레이어만 훈련 가능하도록 설정
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    # train_per_epoch 함수 호출하여 학습
    best_val_acc, history = train_per_epoch(model, train_loader, val_loader, criterion, 
                   optimizer, scheduler, epochs, device, best_val_acc)

    return model, history

def train_per_epoch(model, train_loader, val_loader, criterion, 
                   optimizer, scheduler, epochs, device, best_val_acc):
    '''에포크별 학습을 수행하는 함수'''
    
    # 학습 히스토리 저장용 딕셔너리
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # 훈련 단계
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Train')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Val'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # 에포크별 손실과 정확도 계산
        epoch_train_loss = train_loss/len(train_loader)
        epoch_val_loss = val_loss/len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # 히스토리에 저장
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_classifier_model.pth')
            print(f'  새로운 최고 성능! 모델 저장됨 (Val Acc: {val_acc:.2f}%)')
        
        scheduler.step()
        print()
    
    return best_val_acc, history

# 학습 결과 시각화를 위한 함수
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 학습 실행
model, history = train_classifier_only(model, train_loader, val_loader)

# 학습 과정 시각화
plot_training_history(history)

def train_full_network(model, train_loader, val_loader, 
                       epochs=15, backbone_lr=1e-5, classifier_lr=1e-4, device='cuda'):
    """전체 네트워크 미세 조정 (백본 + 분류기)"""
    model.to(device)
    
    # 모든 파라미터를 훈련 가능하도록 설정
    for param in model.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    
    # 백본과 분류기에 다른 학습률 적용
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': classifier_params, 'lr': classifier_lr}
    ])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
    
    best_val_acc = 0.0
    
    # train_per_epoch 함수 호출하여 학습
    best_val_acc, history = train_per_epoch(model, train_loader, val_loader, criterion, 
                   optimizer, scheduler, epochs, device, best_val_acc)

    return model, history

# 학습 실행
model, history = train_full_network(model, train_loader, val_loader)

# 학습 과정 시각화
plot_training_history(history)