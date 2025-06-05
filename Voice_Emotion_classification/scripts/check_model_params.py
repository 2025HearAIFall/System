# 모델 파라미터 확인 코드
import torch
import torch.nn as nn

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

model = CNN_GRU_Model(num_classes=7)
state = torch.load("C:/github/System/Voice_Emotion_classification/model/cnn_gru_final.pt")

total_params = sum(v.numel() for v in state.values())
print(f"총 파라미터 수: {total_params}")
