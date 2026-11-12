# Mission 2 결과 및 효율

## 모델
- WavLM-base-plus 앞 4층 층별 평균/표준편차 → 층 가중합 → 분류기
- 최종 선택: 미세조정 epoch 4
- 학습 데이터: 서울 Training 28,200통화, 337,883발화 (내부 dev는 별도 1,000통화, 30,356발화)
- 내부 dev Accuracy 0.9209, Validation Accuracy 0.9173 (111,947발화), threshold 0.5 고정
- Validation 클래스별 Accuracy: 119대원 0.8692, 신고자 0.9618

## 파라미터
- Total 38,070,742 / Active 38,070,742 (dense 모델이라 둘이 같음)
- 이번 데이터로 학습한 파라미터: 14,573,614 (분류기 396,806 + WavLM Transformer 위쪽 2층)
- 체크포인트 크기 152MB

## 실행 환경
- GPU NVIDIA A100-SXM4-40GB, CPU 12코어, Python 3.13.15, torch 2.11.0+cu130
- 학습 시간 36.8분

## Validation 추론 (inference.py 1회)
- batch size 32 (padding 후 길이가 같은 발화끼리 묶음), wav 읽기 프로세스 2개
- 발화 111,947개 처리에 128.7초, 발화당 1.15ms (wav 읽기, 전처리, 모델 포함)
- 이 중 모델 forward만 61.04초, 발화당 0.545ms
- 모델 로드까지 포함한 전체 시간 134.41초

## 학습 기록 (내부 dev Accuracy@0.5)
| 단계 | dev Accuracy | train loss |
|---|---|---|
| 1단계 동결 | 0.9143 | - |
| 미세조정 epoch 1 | 0.9176 | 0.1463 |
| 미세조정 epoch 2 | 0.9180 | 0.1347 |
| 미세조정 epoch 3 | 0.9195 | 0.1242 |
| 미세조정 epoch 4 | 0.9209 | 0.1168 |
