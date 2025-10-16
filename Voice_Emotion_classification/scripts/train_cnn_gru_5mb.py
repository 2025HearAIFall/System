# train_cnn_gru_5mb_final_v2.py
import os, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# 0. Config & Utils
# =========================
SEED = 42
BATCH_SIZE = 32
MAX_LEN = 128

# ← 조정(완화/상향)
LR = 8e-4                  # 3e-4 → 8e-4
WEIGHT_DECAY = 5e-5        # 1e-4 → 5e-5
DROPOUT = 0.25             # 0.35 → 0.25
LABEL_SMOOTH = 0.05        # 0.10 → 0.05
CLIP_NORM = 1.0
PATIENCE = 12              # 10 → 12
EPOCHS = 100

AUG_PROB = 0.3             # 0.7 → 0.3
FREQ_MASK_PARAM = 6        # 8 → 6
TIME_MASK_PARAM = 12       # 16 → 12

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)

def ensure_exists(p: str, kind: str):
    if not Path(p).exists():
        print(f"⚠️ {kind} 경로가 존재하지 않습니다: {p}")

# -------------------------
# SpecAugment (time/freq mask)
# -------------------------
def spec_augment(mfcc: torch.Tensor,
                 freq_mask_param: int = FREQ_MASK_PARAM,
                 time_mask_param: int = TIME_MASK_PARAM,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2):
    """
    mfcc: [1, 40, T] tensor (float)
    """
    x = mfcc.clone()
    _, F, T = x.shape
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param + 1)
        if f == 0: continue
        f0 = np.random.randint(0, max(1, F - f + 1))
        x[:, f0:f0+f, :] = 0
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_param + 1)
        if t == 0: continue
        t0 = np.random.randint(0, max(1, T - t + 1))
        x[:, :, t0:t0+t] = 0
    return x

# =========================
# 1. Dataset
# =========================
class MFCCDataset(Dataset):
    def __init__(self, mfcc_dir, label_csv, max_len=MAX_LEN, train=False, aug_prob=AUG_PROB):
        self.data = []
        self.labels = []
        self.label_map = {}
        self.mfcc_dir = Path(mfcc_dir)
        self.max_len = max_len
        self.train = train
        self.aug_prob = aug_prob

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

        # per-sample normalize
        m = mfcc.mean()
        s = mfcc.std() + 1e-5
        mfcc = (mfcc - m) / s

        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, 40, T]

        # SpecAugment (train만)
        if self.train and random.random() < self.aug_prob:
            mfcc = spec_augment(mfcc)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc, label

# =========================
# 2. Model (≈5MB FP32 target)
# =========================
class CNN_GRU_5MB(nn.Module):
    """
    안정성 강화 모델(≈5MB):
      - CNN: 1→64→128
      - GRU: input_size=128, hidden_size=204, num_layers=2, bidirectional=True
      - FC: 408→num_classes
    """
    def __init__(self, num_classes, dropout=DROPOUT):
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
        x = self.cnn(x)           # [B, 128, 40, T/2]
        x = x.mean(dim=2)         # [B, 128, T/2]
        x = x.permute(0, 2, 1)    # [B, T/2, 128]
        out, _ = self.gru(x)      # [B, T/2, 408]
        out = out[:, -1, :]
        return self.fc(out)

# =========================
# 3. EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=PATIENCE):
        self.patience = patience
        self.best_acc = 0.0
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

# =========================
# 4. Plot helpers
# =========================
def plot_accuracy(train_accs, val_accs, save_path, title="Train vs Val Accuracy"):
    import numpy as np
    epochs = np.arange(1, len(train_accs) + 1)
    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.2, 1.0)   # 비교 그래프와 유사 축 범위
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"📈 Saved: {save_path}")

# =========================
# 5. Train
# =========================
def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # --- 프로젝트 루트와 경로 ---
    PROJECT_ROOT = "C:/github/System/Voice_Emotion_classification"
    DATA_DIR = f"{PROJECT_ROOT}/data"
    MFCC_DIR = f"{DATA_DIR}/mfcc"
    TRAIN_CSV = f"{DATA_DIR}/train.csv"
    VAL_CSV   = f"{DATA_DIR}/val.csv"
    MODEL_DIR = f"{PROJECT_ROOT}/model"
    SAVE_PATH = f"{MODEL_DIR}/cnn_gru_5mb.pt"   # 최종 1개만 저장

    # --- figure 저장 경로 ---
    FIG_DIR = f"{PROJECT_ROOT}/figures"
    os.makedirs(FIG_DIR, exist_ok=True)
    ACC_PNG = f"{FIG_DIR}/train_val_acc_5mb_final_v2.png"
    CM_PNG  = f"{FIG_DIR}/cm_val_5mb_final_v2.png"

    ensure_exists(MFCC_DIR, "MFCC_DIR")
    ensure_exists(TRAIN_CSV, "TRAIN_CSV")
    ensure_exists(VAL_CSV, "VAL_CSV")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Dataset / Loader
    train_set = MFCCDataset(MFCC_DIR, TRAIN_CSV, train=True, aug_prob=AUG_PROB)
    val_set   = MFCCDataset(MFCC_DIR, VAL_CSV, train=False)
    num_classes = len(train_set.label_map)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model / Optim / Loss / Scheduler
    model = CNN_GRU_5MB(num_classes=num_classes).to(device)
    print(f"📦 Trainable params: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=False, min_lr=1e-5
    )

    early_stopping = EarlyStopping(patience=PATIENCE)
    best_model_state = None
    best_epoch = -1

    # 최신 AMP API
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    train_accs, val_accs = [], []
    prev_lr = optimizer.param_groups[0]['lr']

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)

        train_acc = correct / total if total > 0 else 0.0
        train_accs.append(train_acc)

        # ---- Validate ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)
        val_acc = correct / total if total > 0 else 0.0
        val_accs.append(val_acc)

        print(f"[Epoch {epoch:03d}] Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

        # Scheduler & Early stopping
        scheduler.step(val_acc)
        cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr != prev_lr:
            print(f"⚙️ LR decayed: {prev_lr:.2e} → {cur_lr:.2e}")
            prev_lr = cur_lr

        if val_acc >= early_stopping.best_acc:
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch

        early_stopping.step(val_acc)
        if early_stopping.should_stop:
            print(f"🛑 Early stopping at epoch {epoch} (no val acc improvement for {early_stopping.patience} checks)")
            break

    # ✅ 최종: 베스트 스냅샷 저장
    if best_model_state is not None:
        torch.save(best_model_state, SAVE_PATH)
    else:
        torch.save(model.state_dict(), SAVE_PATH)

    size_mb = bytes_to_mb(os.path.getsize(SAVE_PATH))
    total_params = count_parameters(model)
    approx_fp32_mb = total_params * 4 / (1024 * 1024)

    print(f"✅ 모델 저장 완료: {SAVE_PATH}  (파일 크기: {size_mb:.2f} MB, best epoch: {best_epoch})")
    print(f"📊 파라미터 수: {total_params:,} (~{approx_fp32_mb:.2f} MB if raw FP32 tensor)")

    # ---- 학습 곡선 PNG 저장 ----
    try:
        plot_accuracy(
            train_accs, val_accs,
            ACC_PNG,
            title="Train vs Val Accuracy (5MB Model - Final v2)"
        )
    except Exception as e:
        print("(그래프 저장 생략) 이유:", e)

    # ---- 혼동행렬 PNG 저장(선택) ----
    try:
        inv_map = val_set.inv_label_map
        all_preds, all_labels = [], []
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                all_labels.extend(y.numpy().tolist())
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[inv_map[i] for i in range(num_classes)]
        )
        disp.plot(xticks_rotation=45, cmap='Blues', colorbar=False)
        plt.title("Validation Confusion Matrix (5MB Model - Final v2)")
        plt.tight_layout()
        plt.savefig(CM_PNG, bbox_inches="tight")
        print(f"🧩 Saved: {CM_PNG}")
    except Exception as e:
        print("혼동행렬 생략 이유:", e)

if __name__ == "__main__":
    train()
