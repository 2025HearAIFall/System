# 모델 평가 --> 정확도, classification report, confusion matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------- Dataset ---------
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
        mfcc = np.load(self.data[idx])
        if mfcc.shape[1] < self.max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_len]
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, 40, T]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc, label

# --------- Model (train.py와 동일) ---------
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
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)          # [B, 64, 40, T/2]
        x = x.mean(dim=2)        # [B, 64, T/2]
        x = x.permute(0, 2, 1)   # [B, T/2, 64]
        out, _ = self.gru(x)     # [B, T/2, 256]
        out = out[:, -1, :]
        return self.fc(out)

# --------- Evaluation ---------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    mfcc_dir = "C:/github/System/Voice_Emotion_classification/data/mfcc"
    test_csv = "C:/github/System/Voice_Emotion_classification/data/test.csv"
    model_path = "C:/github/System/Voice_Emotion_classification/model/cnn_gru_final.pt"

    dataset = MFCCDataset(mfcc_dir, test_csv)
    loader = DataLoader(dataset, batch_size=32)

    model = CNN_GRU_Model(num_classes=len(dataset.label_map)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ [Test Accuracy] {acc*100:.2f}%\n")
    print("[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=list(dataset.label_map.keys())))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(dataset.label_map.keys()),
                yticklabels=list(dataset.label_map.keys()))
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
