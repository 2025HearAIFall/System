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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))  # [B, 64, 40, T/2]
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

# --------- 3. EarlyStopping ---------
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.best_acc = 0
        self.counter = 0
        self.should_stop = False

    def step(self, current_acc):
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

# --------- 4. Train ---------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    mfcc_dir = "C:/github/System/Voice_Emotion_classification/data/mfcc"
    train_csv = "C:/github/System/Voice_Emotion_classification/data/train.csv"
    val_csv   = "C:/github/System/Voice_Emotion_classification/data/val.csv"

    train_set = MFCCDataset(mfcc_dir, train_csv)
    val_set   = MFCCDataset(mfcc_dir, val_csv)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=32)

    model = CNN_GRU_Model(num_classes=len(train_set.label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    early_stopping = EarlyStopping(patience=7)
    best_model_state = None

    train_accs, val_accs = [], []

    for epoch in range(1, 100):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        train_accs.append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        val_accs.append(val_acc)

        print(f"[Epoch {epoch:02d}] Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

        if val_acc > early_stopping.best_acc:
            best_model_state = model.state_dict()

        early_stopping.step(val_acc)
        if early_stopping.should_stop:
            print(f"🛑Early stopping at epoch {epoch} (Val Acc stagnated)")
            break

    torch.save(best_model_state, "C:/github/System/Voice_Emotion_classification/model/cnn_gru.pt")
    print("✅모델 저장 완료: cnn_gru.pt")

    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()
