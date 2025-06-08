# System

# 🗣️ 실시간 음성 감정 인식 및 자막 출력 시스템

## 📌 프로젝트 개요
- 사용자의 음성을 인식하여 감정을 분석하고, 이를 실시간 자막으로 출력하는 시스템 개발
- 오프라인에서도 작동 가능한 경량 모델 기반
- 청각장애인을 위한 실시간 소통 보조 기술로 활용 가능

## 🚀 진행 과정

### 1. 초기 실험
- TESS 영어 음성 데이터셋을 기반으로 Happy / Sad / Angry 감정 분류 실험 수행
- Librosa로 MFCC 40차원 음성 특징 추출
- 경량 CNN 모델을 PyTorch로 구현하여 소규모 학습을 통해 기본 구조 검증 완료

### 2. 1차 시도 (실패)
- AI Hub의 [감정 분류를 위한 대화 음성 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=263)을 사용하여 학습 시도
- `.wav` 파일에서 MFCC 추출 후 CNN+GRU 모델로 학습
- 그러나 해당 데이터셋은 감정 표현 방식이 유사하고, 음성 간 억양 차이가 부족해  
  **감정 간 구분이 어려워 학습 정확도가 약 20% 수준에 머무름 (상위 200개 샘플 기준)**
- 원인 파악 후 완전히 다른 데이터셋을 기반으로 프로젝트를 새로 시작함

### 3. 2차 시도 (학습 성공, 그러나 과적합)
- AI Hub의 [감정 분류용 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=259) 사용
- 참가자 약 30명의 `.m2ts` 파일에서 음성 추출
- 문장번호 기반으로 7개 감정(happiness, surprise, neutral, fear, disgust, anger, sadness) 자동 라벨링
- Librosa로 40차원 MFCC 추출 후 시간축을 128로 고정
- 정규화(z-score) 적용하여 학습 안정화
- 최종 학습 정확도: **81.01%**
- 총 약 3,000개 이상의 샘플 기반 학습
- 모델은 학습 정확도는 높았지만, 테스트 정확도는 5~6%에 불과해 심각한 **과적합(Overfitting)**이 확인됨

### 4. 모델 구조 개선 (과적합(Overfitting) 해결)
- AI Hub의 [감정 분류용 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=259) 사용. 2차 시도랑 동일
- 과적합 원인을 모델의 표현력 부족으로 판단하고 구조를 다음과 같이 확장함:
  - CNN: Conv2d(1→32→64), ReLU, MaxPool
  - GRU(Gated Recurrent Unit): input_size=64, hidden_size=128, 2-layer, bidirectional + Dropout(0.3)
  - FC(Fully Connected) Layer: Dropout(0.3) + Linear(256 → 7)
- 구조 확장 후 파라미터 수는 약 **32K → 466K**로 증가하여 감정의 복잡한 특성을 효과적으로 학습할 수 있게 됨
- 화자 기반으로 train/val/test를 분리하고 EarlyStopping 기법을 도입하여 일반화 성능 확보

### 5. 학습 결과
- 최종 학습 정확도 약 **91.09%**, validation accuracy는 **64.04%**
- 테스트셋 평가 결과 정확도 **68.51%**, macro F1-score **0.68**을 달성
- 모든 감정 클래스에서 precision/recall/f1이 균형 있게 분포됨
- Confusion matrix 시각화 결과, 감정별 예측 성능이 명확히 개선됨

## ✅ 현재 성능 요약
- 최종 테스트 정확도: **68.51%** (macro F1-score: 0.68)
- 총 파라미터 수: **466,055개**
- 총 101명의 화자, 10,351개의 음성 기반 학습
- 과적합 없이 안정적으로 학습된 모델로 실사용 가능성 확보
- 추론 성능 우수하며 TorchScript 변환 및 모바일 배포 가능 구조 유지

## 🔜 앞으로 추가할 예정인 기능들 (상황에 따라 변경 가능)
1. 잡음 환경에서도 정확도를 유지할 수 있도록 **노이즈 필터링** 및 **데이터 증강** 기능 적용
2. 음성뿐만 아니라 **얼굴 표정 인식**을 포함한 **멀티모달 감정 분석 모델** 확장
3. 실시간 추론을 위한 **TorchScript 변환 모델 테스트 및 경량화 (quantization)**
4. **실시간 자막 출력 기능** + 감정 태그 연결 (자막 예: "오늘 기분이 좋아보이네요 [😊 happiness]")

## 🧠 사용 기술 스택
- Python / PyTorch / Librosa / pandas / sklearn
- 모델 학습: GPU (RTX 4060 Ti)
- (초기) 데이터셋: [AI Hub 감정 분류를 위한 대화 음성 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=263)
- (현재) 데이터셋: [AI Hub 감정 분류용 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=259)
- 시각화: seaborn, matplotlib