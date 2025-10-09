import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
feature_dir = os.path.join(base_dir, "features")
label_path = os.path.join(base_dir, "labels.npy")
model_path = os.path.join(base_dir, "models", "emotion_cnn.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# 하이퍼파라미터
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 7

# 데이터 불러오기
file_ids = [fname.replace(".npy", "") for fname in sorted(os.listdir(feature_dir)) if fname.endswith(".npy")]
labels = np.load(label_path)

# 라벨 매핑
label_map = {fid: label for fid, label in zip(file_ids, labels)}

# 훈련/검증 분할 (ID만 나눔)
train_ids, val_ids = train_test_split(file_ids, test_size=0.2, random_state=42)

# 데이터셋 클래스
class EmotionDataset(Dataset):
    def __init__(self, feature_dir, file_list, label_map):
        self.feature_dir = feature_dir
        self.file_list = file_list
        self.label_map = label_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        feature = np.load(os.path.join(self.feature_dir, f"{file_id}.npy"))
        feature = (feature - feature.mean()) / (feature.std() + 1e-6)  # 정규화 추가
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # (1, 40, 100)
        label = torch.tensor(self.label_map[file_id], dtype=torch.long)
        return feature_tensor, label

train_dataset = EmotionDataset(feature_dir, train_ids, label_map)
val_dataset = EmotionDataset(feature_dir, val_ids, label_map)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 모델 정의 (CNN + GRU)
class CNN_GRU_EmotionClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN_GRU_EmotionClassifier, self).__init__()
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

# 모델 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 학습 디바이스: {device}")
model = CNN_GRU_EmotionClassifier(num_classes=NUM_CLASSES).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 검증
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f}")

# 모델 저장
torch.save(model.state_dict(), model_path)
print(f"모델 저장 완료: {model_path}")
