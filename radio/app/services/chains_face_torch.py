# TorchScript(.pt) 표정 분류 (크롭 얼굴 이미지 바이트 → 라벨/점수)
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from io import BytesIO
import numpy as np

MODEL_PATH = "weights/ViT_model.pth"
DETECTOR_PATH = "weights/yolov12n-face.pt"
# LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
DEVICE = "cpu"

_model = None
_detector = None


def _load_detector():
    global _detector
    if _detector is None:
        _detector = torch.hub.load(
            "ultralytics/yolov5", "custom", path=DETECTOR_PATH)
    return _detector


def _load_model():
    global _model
    if _model is None:
        _model = torch.load(MODEL_PATH, map_location="cpu")
        _model.eval()
    return _model


def _preprocess(img_bytes):
    im = Image.open(BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr = np.asarray(im).astype("float32")/255.0
    # 학습 시 mean/std 정규화가 있었다면 동일 적용:
    # mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]; arr=(arr-mean)/std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)[None, ...]


async def classify_expression(img_bytes):
    detector = _load_detector()
    model = _load_model()

    img = Image.open(img_bytes).convert("RGB")
    results = detector(img)
    if len(results.xyxy[0]) == 0:
        return "unknown"

    # 가장 큰 얼굴 선택
    x1, y1, x2, y2, conf, cls = results.xyxy[0][0]
    face = img.crop((x1, y1, x2, y2))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    timg = transform(face).unsqueeze(0)

    with torch.no_grad():
        out = model(timg)
        pred = out.argmax(dim=1).item()

    classes = ['Angry', 'Disgust', 'Fear',
               'Happy', 'Neutral', 'Sad', 'Surprise']
    return classes[pred]
