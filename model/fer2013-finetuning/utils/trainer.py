import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class EmotionTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def train_classifier_only(self, epochs=10, lr=1e-3):
        # 백본 고정
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        return self._train(criterion, optimizer, scheduler, epochs)
    
    def train_full_network(self, epochs=15, backbone_lr=1e-5, classifier_lr=1e-4):
        for param in self.model.parameters():
            param.requires_grad = True
            
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': classifier_params, 'lr': classifier_lr}
        ])
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
        
        return self._train(criterion, optimizer, scheduler, epochs)
    
    def _train(self, criterion, optimizer, scheduler, epochs):
        best_val_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # 훈련 단계
            self.model.train()
            train_metrics = self._train_epoch(criterion, optimizer)
            
            # 검증 단계
            self.model.eval()
            val_metrics = self._validate_epoch(criterion)
            
            # 메트릭 저장
            for k, v in train_metrics.items():
                history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)
            
            # 출력
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}%")
            
            # 모델 저장
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"  새로운 최고 성능! 모델 저장됨 (Val Acc: {val_metrics['acc']:.2f}%)")
            
            scheduler.step()
            print()
        
        return history
    
    def _train_epoch(self, criterion, optimizer):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in tqdm(self.train_loader, desc='Training'):
            # 데이터를 GPU로 이동
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            # 순전파
            output = self.model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        return {
            'loss': running_loss/len(self.train_loader),
            'acc': 100. * correct / total
        }
    
    def _validate_epoch(self, criterion):
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return {
            'loss': running_loss/len(self.val_loader),
            'acc': 100. * correct / total
        }
    
    @staticmethod
    def plot_history(history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
