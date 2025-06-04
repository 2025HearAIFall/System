import os, glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Config ===
wav_dir = Path("C:/github/System/Voice_Emotion_classification/data/wav_test")
model_path = Path("C:/github/System/Voice_Emotion_classification/model/cnn_gru_final.pt")
label_names = ['Happiness', 'Surprise', 'Neutral', 'Fear', 'Disgust', 'Anger', 'Sadness']

# === 라벨 추출 ===
def extract_label_from_filename(filename):
    sentence_num = int(filename.split('-')[1].split('.')[0])
    if 1 <= sentence_num <= 50: return 0
    elif 51 <= sentence_num <= 100: return 1
    elif 101 <= sentence_num <= 150: return 2
    elif 151 <= sentence_num <= 200: return 3
    elif 201 <= sentence_num <= 250: return 4
    elif 251 <= sentence_num <= 300: return 5
    elif 301 <= sentence_num <= 350: return 6
    else: return -1

def extract_mfcc(filepath, max_len=128):
    y, sr = librosa.load(filepath, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# === 평가용 Dataset ===
class MFCCDataset(torch.utils.data.Dataset):
    def __init__(self, wav_dir):
        self.files = sorted(Path(wav_dir).glob("*.wav"))
        self.X, self.y = [], []
        for f in self.files:
            label = extract_label_from_filename(f.name)
            if label == -1:
                print(f"[SKIP] Invalid label: {f.name}")
                continue
            mfcc = extract_mfcc(f)
            self.X.append(mfcc)
            self.y.append(label)

        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(1)  # [B, 1, 40, T]
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# === 모델 정의 ===
class CNN_GRU_Model(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2))  # Time 축만 절반으로
        )
        self.gru = torch.nn.GRU(input_size=16, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x):  # [B, 1, 40, T]
        x = self.cnn(x)              # [B, 16, 40, T/2]
        x = x.mean(dim=2)            # [B, 16, T/2]
        x = x.permute(0, 2, 1)       # [B, T/2, 16]
        _, h = self.gru(x)           # h: [2, B, 64]
        h = torch.cat([h[0], h[1]], dim=1)  # [B, 128]
        return self.fc(h)

# === 실행 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
print(f"[INFO] Loaded model from: {model_path}")

dataset = MFCCDataset(wav_dir)
if len(dataset) == 0:
    print("[ERROR] No valid wav files found.")
    exit()

loader = DataLoader(dataset, batch_size=32)

model = CNN_GRU_Model(num_classes=len(label_names)).to(device)
model.load_state_dict(torch.load(str(model_path), map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())

# === 결과 출력 ===
acc = accuracy_score(all_labels, all_preds)
print(f"\n[Test Accuracy] {acc * 100:.2f}%\n")
print("[Classification Report]")
print(classification_report(all_labels, all_preds, target_names=label_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()