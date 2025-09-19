import torch

class Config:
    # 데이터 관련
    DATA_DIR = '/content/fer2013'
    BATCH_SIZE = 256
    NUM_WORKERS = 4
    
    # 모델 관련
    MODEL_NAME = "hf_hub:timm/mobilenetv4_conv_medium.e500_r224_in1k"
    NUM_CLASSES = 7
    
    # 학습 관련
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 분류기 학습
    CLASSIFIER_EPOCHS = 10
    CLASSIFIER_LR = 1e-3
    
    # 전체 네트워크 학습
    FULL_NETWORK_EPOCHS = 15
    BACKBONE_LR = 1e-5
    CLASSIFIER_LR_FINETUNE = 1e-4
