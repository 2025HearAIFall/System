from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prediction(BaseModel):
    text: str
    emotion: str

# CNN_GRU_Model 직접 정의
class CNN_GRU_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 2, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)           # [B, 64, 40, T/2]
        x = x.mean(dim=2)         # [B, 64, T/2]
        x = x.permute(0, 2, 1)    # [B, T/2, 64]
        out, _ = self.gru(x)      # [B, T/2, 256]
        out = out[:, -1, :]
        return self.fc(out)

# 모델 로딩
model_path = Path("C:/github/System/Voice_Emotion_classification/model/cnn_gru.pt")
num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_GRU_Model(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"[DEBUG] 모델 로드 완료\ndevice: {device}")

label_map = {
    0: ("angry", "분노"),
    1: ("disgust", "혐오"),
    2: ("fear", "공포"),
    3: ("happiness", "행복"),
    4: ("neutral", "중립"),
    5: ("sadness", "슬픔"),
    6: ("surprise", "놀람"),
}

def predict_emotion_from_file(file_path: str, max_len=128):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # 정규화 추가!

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(mfcc_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
        pred_id = int(np.argmax(probs))
        eng, kor = label_map.get(pred_id, ("unknown", "알 수 없음"))

        print(f"[DEBUG] MFCC shape: {mfcc.shape}, Pred ID: {pred_id}, Probs: {probs}")
        return f"{eng} ({kor})"

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    emotion = predict_emotion_from_file("temp.wav")
    return Prediction(text="감정 분석 결과입니다", emotion=emotion)

# 실행: 1. cd C:\github\System\emotion_webpage_test
# 2. uvicorn emotion_api_server:app --reload
# 3. 브라우저에서 http://127.0.0.1:8000/docs