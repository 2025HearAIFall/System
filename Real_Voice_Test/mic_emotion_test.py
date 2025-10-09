# 음성 데이터 셋이 아닌, 실제 목소리로 모델 테스트 하는 코드
import numpy as np
import torch, torch.nn as nn
import librosa, pyaudio

# ====== 학습 스펙 ======
MODEL_PATH  = r"C:\github\System\Voice_Emotion_classification\model\cnn_gru.pt"
TARGET_SR   = 16000
N_MFCC      = 40
N_FFT       = 2048          # librosa 기본(학습에서도 기본 사용)
HOP_LENGTH  = 512           # librosa 기본(학습에서도 기본 사용)
MAX_T       = 128           # train.MFCCDataset(max_len)

EMOTIONS = ['anger','disgust','fear','happiness','neutral','sadness','surprise']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 모델 구조 (train.py와 동일) ======
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
    def forward(self, x):                 # x: [B,1,40,T]
        x = self.cnn(x)                   # [B,64,40,T/2]
        x = x.mean(dim=2)                 # [B,64,T/2]
        x = x.permute(0,2,1)              # [B,T/2,64]
        out,_ = self.gru(x)               # [B,T/2,256]
        out = out[:, -1, :]
        return self.fc(out)

# ====== 전처리 함수 (extract_mfcc.py와 동일 로직) ======
def make_mfcc(y, sr):
    # 필요시 리샘플
    if sr != TARGET_SR:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=TARGET_SR, res_type="kaiser_fast")
        sr = TARGET_SR
    # float32 -1~1
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0: y = y / (np.max(np.abs(y)) + 1e-9)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)  # [40,T]
    # 샘플 단위 z-score (extract_mfcc.py와 동일)
    m = mfcc.mean(); s = mfcc.std() if mfcc.std() > 1e-9 else 1.0
    mfcc = (mfcc - m) / s

    # pad/trim to MAX_T (train.MFCCDataset와 동일)
    T = mfcc.shape[1]
    if T < MAX_T:
        pad = MAX_T - T
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_T]
    return mfcc  # [40,128]

# ====== 마이크 ======
CHUNK = 1024
RECORD_SECONDS = 3

def record_audio(seconds=RECORD_SECONDS, rate=TARGET_SR):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                     input=True, frames_per_buffer=CHUNK)
    print(f"🎙️ {seconds}초 동안 말해보세요...")
    frames = []
    for _ in range(int(rate/CHUNK*seconds)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    stream.stop_stream(); stream.close(); pa.terminate()
    y = np.concatenate(frames).astype(np.float32) / 32768.0
    return y, rate

def main():
    # 모델 로드
    model = CNN_GRU_Model(num_classes=len(EMOTIONS)).to(DEVICE).eval()
    state = torch.load(MODEL_PATH, map_location=DEVICE)  # 저장은 순수 state_dict
    model.load_state_dict(state)

    # 녹음 → MFCC
    y, sr = record_audio()
    mfcc = make_mfcc(y, sr)                              # [40,128]
    x = torch.from_numpy(mfcc[None,None,:,:]).to(DEVICE) # [1,1,40,128]

    # 추론
    with torch.no_grad():
        logits = model(x)                                # [1,7]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    top = int(np.argmax(probs))
    print(f"🧠 결과: {EMOTIONS[top]} (p={probs[top]:.2f})")
    top3 = probs.argsort()[-3:][::-1]
    print("Top-3:", [(EMOTIONS[i], float(probs[i])) for i in top3])

if __name__ == "__main__":
    main()
