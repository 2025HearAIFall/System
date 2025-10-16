# MFCC 시각화 코드
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

# ====== 학습 스펙 (사용자 모델과 동일) ======
AUDIO_PATH = r"C:\github\System\Voice_Emotion_Classification\data\wav\000-001.wav"  # 🎯 여기에 파일 경로만 바꿔주세요
TARGET_SR   = 16000
N_MFCC      = 40
N_FFT       = 2048
HOP_LENGTH  = 512
MAX_T       = 128  # 학습에서 사용한 고정 길이

def make_mfcc(y, sr):
    """사용자 학습 파이프라인과 동일한 전처리 + pad/trim."""
    # 리샘플
    if sr != TARGET_SR:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=TARGET_SR, res_type="kaiser_fast")
        sr = TARGET_SR

    # float32 -1~1 정규화
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-9)

    # MFCC 추출 [40, T]
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # 샘플 단위 z-score
    m = mfcc.mean()
    s = mfcc.std() if mfcc.std() > 1e-9 else 1.0
    mfcc = (mfcc - m) / s

    # pad/trim → [40, MAX_T]
    T = mfcc.shape[1]
    if T < MAX_T:
        pad = MAX_T - T
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_T]

    return mfcc, sr

def plot_mfcc(mfcc, sr, hop_length, title="MFCC (z-score, padded/trimmed)"):
    """MFCC 시각화"""
    num_frames = mfcc.shape[1]
    times = np.arange(num_frames) * (hop_length / sr)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        mfcc,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1] if len(times) > 1 else 0, 0, mfcc.shape[0]]
    )
    plt.colorbar(label='z-score')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coeff Index (0..39)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    audio_path = Path(AUDIO_PATH)
    if not audio_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")

    # 파일 로드 (모노로)
    y, sr = sf.read(str(audio_path), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)  # 스테레오 → 모노

    # MFCC 계산
    mfcc, sr = make_mfcc(y, sr)

    # 시각화
    plot_mfcc(mfcc, sr, HOP_LENGTH, title=f"MFCC: {audio_path.name}")

if __name__ == "__main__":
    main()
