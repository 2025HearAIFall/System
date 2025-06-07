# 모델 파라미터 확인 코드
import torch
import torch.nn as nn

# --------- 모델 정의 ---------
class CNN_GRU_Model(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))  # [B, 64, 40, T/2]
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# --------- 파라미터 계산 함수 ---------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")

# --------- 실행 ---------
if __name__ == "__main__":
    model = CNN_GRU_Model(num_classes=7)
    model.load_state_dict(torch.load("C:/github/System/Voice_Emotion_classification/model/cnn_gru_final.pt", map_location='cpu'))
    count_parameters(model)