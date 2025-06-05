# 실행 3번째, train/val/test.csv 생성
import pandas as pd
from pathlib import Path

# 1. 경로 설정
base_dir = Path("C:/github/System/Voice_Emotion_classification/data")
label_path = base_dir / "labels.csv"

# 2. labels.csv 로드
df = pd.read_csv(label_path)

# 3. 참가자 번호 추출
df['speaker'] = df['file'].apply(lambda x: int(x.split('-')[0]))

# 4. 화자 기반 분할 (예: 70/15/15%)
train_ids = list(range(0, 70))     # 000~069
val_ids   = list(range(70, 85))    # 070~084
test_ids  = list(range(85, 100))   # 085~099

# 5. 각 split 추출
train_df = df[df['speaker'].isin(train_ids)].drop(columns=['speaker'])
val_df   = df[df['speaker'].isin(val_ids)].drop(columns=['speaker'])
test_df  = df[df['speaker'].isin(test_ids)].drop(columns=['speaker'])

# 6. 저장
train_df.to_csv(base_dir / "train.csv", index=False)
val_df.to_csv(base_dir / "val.csv", index=False)
test_df.to_csv(base_dir / "test.csv", index=False)

print(f"[완료] train.csv: {len(train_df)}개, val.csv: {len(val_df)}개, test.csv: {len(test_df)}개")
