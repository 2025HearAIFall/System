# 모델 test에 쓸 임시 코드. 확인용
from pathlib import Path
import subprocess

def convert_m2ts_to_wav(m2ts_path: Path, wav_path: Path, sr: int = 16000):
    cmd = [
        "ffmpeg",
        "-i", str(m2ts_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", "1",
        str(wav_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print(f"[OK] {m2ts_path.name} → {wav_path.name}")
    else:
        print(f"[FAIL] {m2ts_path.name}")
        print(result.stderr.decode())


if __name__ == "__main__":
    raw_dir = Path("C:/github/System/Voice_Emotion_classification/data/raw_test")
    wav_dir = Path("C:/github/System/Voice_Emotion_classification/data/wav_test")
    wav_dir.mkdir(exist_ok=True, parents=True)

    files = list(raw_dir.glob("*.m2ts"))
    print(f"[INFO] Found {len(files)} m2ts files.")

    for m2ts_file in files:
        name = m2ts_file.stem
        wav_path = wav_dir / f"{name}.wav"
        print(f"[INFO] Converting {m2ts_file.name} → {wav_path.name}")
        convert_m2ts_to_wav(m2ts_file, wav_path)
