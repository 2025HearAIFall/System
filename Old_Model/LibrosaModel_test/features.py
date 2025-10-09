import librosa
import numpy as np

def extract_features(file_path, n_mfcc=40, max_len=100):
    """
    오디오 파일에서 MFCC 특징을 추출하고 고정 길이로 패딩 처리함.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc