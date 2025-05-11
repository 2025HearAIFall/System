import os
import pandas as pd
import numpy as np
import librosa

# librosa와 numpy 버전 호환 이슈 대응
np.complex = complex

# 경로 설정
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(base_dir, "data", "5차년도_2차.csv")
wav_dir = os.path.join(base_dir, "data", "wav")
feature_dir = os.path.join(base_dir, "features")
label_path = os.path.join(base_dir, "labels.npy")

# 출력 폴더 생성
os.makedirs(feature_dir, exist_ok=True)

# CSV 불러오기
df = pd.read_csv(csv_path, encoding='cp949')
emotion_column = '상황'
wav_column = 'wav_id'

# 감정 라벨 매핑 (정렬된 순서 보장)
emotion_labels = sorted(df[emotion_column].unique())
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
print(f"감정 라벨 매핑: {emotion_to_idx}")

# 전처리 시작
labels = []
missing = []

for i, row in df.iterrows():
    wav_id = row[wav_column]
    emotion = row[emotion_column]
    wav_path = os.path.join(wav_dir, f"{wav_id}.wav")

    try:
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = mfcc[:, :100] if mfcc.shape[1] >= 100 else np.pad(mfcc, ((0,0), (0,100 - mfcc.shape[1])), mode='constant')
        np.save(os.path.join(feature_dir, f"{wav_id}.npy"), mfcc)
        labels.append(emotion_to_idx[emotion])
    except Exception as e:
        missing.append(wav_id)
        print(f"[오류] {wav_id}: {e}")

# 라벨 저장
np.save(label_path, np.array(labels))
print(f"전처리 완료: {len(labels)}개 저장, 실패 {len(missing)}개")
