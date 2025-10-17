# evaluate_cnn_gru_5mb.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1) Dataset
# -------------------------
class MFCCDataset(Dataset):
    def __init__(self, mfcc_dir, label_csv, max_len=128):
        self.data = []
        self.labels = []
        self.label_map = {}
        self.mfcc_dir = Path(mfcc_dir)
        self.max_len = max_len

        df = pd.read_csv(label_csv)
        classes = sorted(df['emotion'].unique())
        self.label_map = {label: idx for idx, label in enumerate(classes)}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        for _, row in df.iterrows():
            path = self.mfcc_dir / f"{row['file']}.npy"
            if path.exists():
                self.data.append(path)
                self.labels.append(self.label_map[row['emotion']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mfcc = np.load(self.data[idx])  # [40, T]

        # pad/trim to max_len
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_len]

        # 학습과 동일하게 per-sample 정규화
        m = mfcc.mean()
        s = mfcc.std() + 1e-5
        mfcc = (mfcc - m) / s

        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, 40, T]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc, label

# -------------------------
# 2) Model (5MB 버전)
# -------------------------
class CNN_GRU_5MB(nn.Module):
    """
    CNN: 1→64→128, MaxPool(1,2)
    GRU: input=128, hidden=204, num_layers=2, bidirectional=True
    FC : 408→num_classes
    """
    def __init__(self, num_classes, dropout=0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))  # [B, 128, 40, T/2]
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=204,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(204 * 2, num_classes)  # 408 -> C
        )

    def forward(self, x):
        x = self.cnn(x)         # [B, 128, 40, T/2]
        x = x.mean(dim=2)       # [B, 128, T/2]
        x = x.permute(0, 2, 1)  # [B, T/2, 128]
        out, _ = self.gru(x)    # [B, T/2, 408]
        out = out[:, -1, :]
        return self.fc(out)

# -------------------------
# 3) Evaluation
# -------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # 프로젝트 경로 맞게 수정
    PROJECT_ROOT = "C:/github/System/Voice_Emotion_classification"
    MFCC_DIR = f"{PROJECT_ROOT}/data/mfcc"
    TEST_CSV = f"{PROJECT_ROOT}/data/test.csv"
    MODEL_PATH = f"{PROJECT_ROOT}/model/cnn_gru_5mb.pt"

    # 데이터셋/로더
    dataset = MFCCDataset(MFCC_DIR, TEST_CSV, max_len=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 모델 로드
    num_classes = len(dataset.label_map)
    model = CNN_GRU_5MB(num_classes=num_classes, dropout=0.25).to(device)

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 평가
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    # 지표
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ [Test Accuracy] {acc*100:.2f}%\n")

    # 레이블 이름(정렬된 감정명)
    target_names = [dataset.inv_label_map[i] for i in range(num_classes)]
    print("[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    # 혼동행렬
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix (Test Set) - CNN_GRU_5MB")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()