# train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from features import extract_features
from model import CNNEmotionClassifier

# 감정 클래스와 폴더 매핑
EMOTION_LABELS = {
    "happy": 0,
    "sad": 1,
    "angry": 2
}

class EmotionDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = []
        self.labels = []
        for emotion, label in EMOTION_LABELS.items():
            emotion_dir = os.path.join(data_dir, emotion)
            for file_name in os.listdir(emotion_dir):
                if file_name.endswith(".wav"):
                    self.file_paths.append(os.path.join(emotion_dir, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        feature = extract_features(self.file_paths[idx])  # (40, 100)
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # (1, 40, 100)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def train():
    dataset = EmotionDataset(data_dir="data")  # data/happy/, data/sad/, data/angry/
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = CNNEmotionClassifier(num_classes=len(EMOTION_LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        total_loss = 0
        for x, y in dataloader:
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    # 학습된 모델 저장
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/emotion_cnn.pth")

if __name__ == "__main__":
    train()
