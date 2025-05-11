import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 정의 (CNN + GRU)
class CNN_GRU_EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gru = nn.GRU(input_size=32 * 20, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        _, h = self.gru(x)
        x = torch.relu(self.fc1(h[-1]))
        return self.fc2(x)

# 감정 라벨
emotion_labels = ['angry', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
emotion_to_idx = {label: i for i, label in enumerate(emotion_labels)}

# 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "emotion_cnn.pth")
csv_path = os.path.join(base_dir, "data", "5차년도_2차.csv")
wav_dir = os.path.join(base_dir, "data", "wav")

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_GRU_EmotionClassifier(num_classes=len(emotion_labels)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 데이터 로드 및 필터
df = pd.read_csv(csv_path, encoding='cp949')
df = df[df["상황"].isin(emotion_labels)]
df = df.head(200)  # 상위 200개만 평가

y_true = []
y_pred = []

for _, row in df.iterrows():
    wav_id = row["wav_id"]
    label = row["상황"]
    wav_path = os.path.join(wav_dir, f"{wav_id}.wav")
    if not os.path.exists(wav_path): continue
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
        mfcc = mfcc[:, :100] if mfcc.shape[1] >= 100 else np.pad(mfcc, ((0,0), (0,100 - mfcc.shape[1])), mode='constant')
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item()
        y_true.append(emotion_to_idx[label])
        y_pred.append(pred)
    except Exception:
        continue

# 평가 결과
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=emotion_labels, digits=4))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Top 200 samples)")
plt.tight_layout()
plt.show()
