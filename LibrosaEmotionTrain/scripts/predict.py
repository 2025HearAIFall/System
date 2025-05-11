import os
import numpy as np
import torch
import torch.nn as nn
import librosa

# librosa / numpy 호환 이슈 대응
np.complex = complex

# 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "emotion_cnn.pth")
emotion_labels = ['angry', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# CNN + GRU 모델 정의
class CNN_GRU_EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gru = nn.GRU(input_size=32 * 20, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):  # x: (B, 1, 40, 100)
        x = torch.relu(self.conv1(x))   # (B, 32, 40, 100)
        x = self.pool(x)                # (B, 32, 20, 50)
        x = x.permute(0, 3, 1, 2)       # (B, 50, 32, 20)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # (B, 50, 640)
        _, h = self.gru(x)              # h: (1, B, 128)
        x = torch.relu(self.fc1(h[-1])) # (B, 64)
        return self.fc2(x)              # (B, num_classes)

# 예측 함수
def predict_emotion(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_GRU_EmotionClassifier(num_classes=len(emotion_labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 특징 추출 및 정규화
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)  # 정규화 추가
    mfcc = mfcc[:, :100] if mfcc.shape[1] >= 100 else np.pad(mfcc, ((0,0), (0,100 - mfcc.shape[1])), mode='constant')
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 40, 100)

    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
        return emotion_labels[pred]

# 테스트 실행
if __name__ == "__main__":
    test_wav = os.path.join(base_dir, "data", "wav", "5fb8e09444697678c497afe0.wav")  # 예시
    if not os.path.exists(test_wav):
        print(f"테스트 파일 없음: {test_wav}")
    else:
        result = predict_emotion(test_wav)
        print(f"예측된 감정: {result}")
