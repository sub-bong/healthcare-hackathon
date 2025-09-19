import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FERDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        
        data_dir = os.path.join(root_dir, 'train' if is_train else 'test')
        
        self.image_paths = []
        self.labels = []
        
        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2,
            'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
        }
        
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

def get_transforms(is_train=True, is_norm=True):
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(10) if is_train else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(brightness=0.2, contrast=0.2) if is_train else transforms.Lambda(lambda x: x),
        transforms.ToTensor()  # 이미지를 [0, 1] 범위의 텐서로 변환
    ]
    
    if is_norm:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                   std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)

def create_dataloaders(data_dir, batch_size=32, num_workers=4, is_norm=True):
    train_dataset = FERDataset(
        root_dir=data_dir,
        is_train=True,
        transform=get_transforms(is_train=True, is_norm=is_norm)
    )
    
    val_dataset = FERDataset(
        root_dir=data_dir,
        is_train=False,
        transform=get_transforms(is_train=False, is_norm=is_norm)
    )
    
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

