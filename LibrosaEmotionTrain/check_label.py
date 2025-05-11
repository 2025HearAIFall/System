import os
import numpy as np
from collections import Counter

# 경로 설정 (현재 스크립트 기준)
base_dir = os.path.abspath(os.path.dirname(__file__))
label_path = os.path.join(base_dir, "labels.npy")

# 라벨 로딩
labels = np.load(label_path)
counter = Counter(labels)

# 출력
print("라벨 분포 (index: count):")
for idx, count in sorted(counter.items()):
    print(f"  {idx}: {count}")

# 비율로 보기
total = sum(counter.values())
print("\n비율 (%)")
for idx, count in sorted(counter.items()):
    print(f"  {idx}: {count / total * 100:.2f}%")
