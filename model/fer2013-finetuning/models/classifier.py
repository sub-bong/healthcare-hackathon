import torch.nn as nn
import timm

def create_emotion_classifier(num_classes=7, backbone_name="hf_hub:timm/mobilenetv4_conv_medium.e500_r224_in1k"):
    """감정 분류 모델 생성"""
    model = timm.create_model(backbone_name, pretrained=True)
    
    # 분류기 교체
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    
    return model
