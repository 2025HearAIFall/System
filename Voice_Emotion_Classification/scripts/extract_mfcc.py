# 실행 2번째, data/mfcc/*.npy + labels.csv 생성
import librosa
import numpy as np
import os
from pathlib import Path
import csv

def get_emotion_from_sentence_id(sid: int) -> str:
    if 1 <= sid <= 50:
        return 'happiness'
    elif 51 <= sid <= 100:
        return 'surprise'
    elif 101 <= sid <= 150:
        return 'neutral'
    elif 151 <= sid <= 200:
        return 'fear'
    elif 201 <= sid <= 250:
        return 'disgust'
    elif 251 <= sid <= 300:
        return 'anger'
    elif 301 <= sid <= 350:
        return 'sadness'
    return 'unknown'

def extract_and_save_mfcc(wav_path: Path, out_dir: Path, n_mfcc=40):
    name = wav_path.stem  # ex: 000-001
    sentence_id = int(name.split('-')[1])
    emotion = get_emotion_from_sentence_id(sentence_id)

    # load audio
    y, sr = librosa.load(wav_path, sr=16000)
    
    # extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # normalize
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    
    # save
    out_path = out_dir / f"{name}.npy"
    np.save(out_path, mfcc)

    return name, emotion

if __name__ == "__main__":
    wav_dir = Path("C:/github/System/Voice_Emotion_classification/data/wav")
    mfcc_dir = Path("C:/github/System/Voice_Emotion_classification/data/mfcc")
    label_file = Path("C:/github/System/Voice_Emotion_classification/data/labels.csv")

    mfcc_dir.mkdir(exist_ok=True, parents=True)

    records = []

    wav_files = sorted(wav_dir.glob("*.wav"))
    print(f"[INFO] Found {len(wav_files)} wav files.")

    for wav_path in wav_files:
        name, emotion = extract_and_save_mfcc(wav_path, mfcc_dir)
        records.append((name, emotion))
        print(f"[OK] {name}.wav → MFCC → {emotion}")

    # Save labels.csv
    with open(label_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "emotion"])
        writer.writerows(records)

    print(f"[INFO] labels.csv saved with {len(records)} entries.")
