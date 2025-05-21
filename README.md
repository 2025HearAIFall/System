# System

# 🗣️ 실시간 음성 감정 인식 및 자막 출력 시스템

## 📌 프로젝트 개요
- 사용자의 음성을 인식하여 감정을 분석하고, 이를 실시간 자막으로 출력하는 시스템 개발
- 오프라인에서도 작동 가능한 경량 모델 기반
- 청각장애인을 위한 실시간 소통 보조 기술로 활용 가능

## 🚀 진행 과정
1. **초기 실험**
   - TESS 데이터셋에서 Happy / Sad / Angry 감정의 오디오 수집
   - Librosa를 사용해 MFCC 음성 특징 추출
   - 직접 설계한 경량 CNN 모델을 PyTorch로 구현하고 소규모 데이터셋으로 테스트용 학습 진행
   - 기본 모델이 정상 작동함을 확인함

2. **본격적 개발 단계 (한국어 기반)**
   - AI Hub의 [5차년도_2차 한국어 음성 감정 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=263) 사용
   - 약 19,000개 이상의 `.wav`와 `csv` 파일 기반 학습
   - 감정 라벨은 '상황' 컬럼 기준 (angry, happiness, etc.)
   - Librosa로 MFCC 40차원 특징 추출 후 (40, 100) 크기로 고정

3. **모델 학습**
   - CNN + GRU 기반 모델 구조 구현
   - 정규화된 MFCC 입력 사용 (평균 0, 표준편차 1)
   - CrossEntropyLoss + Adam Optimizer 기반 학습
   - Epoch 수 20 → 50으로 확장 후 성능 개선 실험

4. **모델 평가**
   - 200개 샘플 대상으로 confusion matrix 및 classification report 생성
   - 감정별 분류 정확도 시각화
   - 정확도가 높은 감정(happiness, angry)과 낮은 감정(disgust, fear) 구분

## 🔍 현재 성능 요약
- 최종 모델 정확도 (상위 200 샘플 기준): **약 19.5%**
- 다양한 감정을 구분할 수 있으나, 감정 간 경계가 모호한 경우 성능 저하
- GRU가 추가된 구조보다 단순 CNN 구조로 성능 비교 필요

## 🔜 앞으로 추가할 기능들
1. 잡음 환경에서도 정확도를 유지할 수 있도록 **노이즈 필터링** 추가
2. 음성뿐만 아니라 **얼굴 표정 인식**을 포함한 다중 입력 감정 분석
3. 음성을 인식해 **실시간 자막 출력 기능** 구현
4. 감정을 7개 → 4개 그룹으로 축소해 분류 정확도 개선 실험

## 🧠 사용 기술 스택
- Python / PyTorch / Librosa / pandas / sklearn
- 모델 학습: GPU (RTX 4060 Ti)
- (舊)데이터셋: [AI Hub 감정 분류를 위한 대화 음성 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=263)
- (新)데이터셋: [AI Hub 감정 분류용 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=259)
- 시각화: seaborn, matplotlib
