import os
import numpy as np
import torch
import torch.nn as nn
import librosa

# librosa / numpy 호환 이슈 대응
np.complex = complex

# 📁 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "emotion_cnn.pth")
emotion_labels = ['angry', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# 🧠 모델 정의 (학습 때 사용한 것과 동일해야 함)
class CNNEmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNEmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool2d((9, 23))
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 9 * 23, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 🧪 예측 함수
def predict_emotion(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNEmotionClassifier(num_classes=len(emotion_labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 특징 추출
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc[:, :100] if mfcc.shape[1] >= 100 else np.pad(mfcc, ((0,0), (0,100 - mfcc.shape[1])), mode='constant')
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 40, 100)

    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
        return emotion_labels[pred]

# 🔧 테스트
if __name__ == "__main__":
    test_wav = os.path.join(base_dir, "data", "wav", "5f41426d9dd513131eacee35.wav")  # 예시
    if not os.path.exists(test_wav):
        print(f"⚠️ 테스트 파일 없음: {test_wav}")
    else:
        result = predict_emotion(test_wav)
        print(f"🔊 예측된 감정: {result}")
