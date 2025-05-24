# 실행 3번째
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --------- 1. Dataset ---------
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

        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_len]

        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, 40, T]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc, label

# --------- 2. Model ---------
class CNN_GRU_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )
        self.gru = nn.GRU(input_size=16, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)              # [B, 16, 40, T/2]
        x = x.mean(dim=2)            # [B, 16, T/2]
        x = x.permute(0, 2, 1)       # [B, T/2, 16]
        out, _ = self.gru(x)         # [B, T/2, 128]
        out = out[:, -1, :]
        return self.fc(out)

# --------- 3. Train ---------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    dataset = MFCCDataset(
        "C:/github/System/Voice_Emotion_classification/data/mfcc",
        "C:/github/System/Voice_Emotion_classification/data/labels.csv"
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = CNN_GRU_Model(num_classes=len(dataset.label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Accuracy: {acc:.2%}")

    # Confusion Matrix
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            output = model(x)
            preds = output.argmax(1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(dataset.label_map.keys()))
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix (Train Set)")
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "C:/github/System/Voice_Emotion_classification/model/cnn_gru_final.pt")
    print("최종 모델 저장 완료: cnn_gru_final.pt")

if __name__ == "__main__":
    train()
