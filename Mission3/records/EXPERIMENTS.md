# Mission 3 Experiment Report

## 1. 문제 정의

Mission 3은 119 신고자와 119대원의 대화 transcript를 입력으로 받아
다음 9개 증상을 예측하는 multi-label classification 문제이다.

- 고열
- 구토
- 두통
- 복통
- 어지러움
- 열상
- 오심
- 전신쇠약
- 호흡곤란

하나의 대화에서 여러 증상이 동시에 나타날 수 있으므로
각 class를 독립적인 binary classification 문제로 처리하였다.
모델의 9개 logit에는 sigmoid를 적용하며,
최종 prediction threshold는 모든 class에서 0.5로 고정하였다.

평가 지표는 class별 F1-score를 계산한 뒤 평균하는 Macro-F1이다.
따라서 빈도가 낮거나 판별이 어려운 일부 class의 성능 저하가
전체 점수에 직접 영향을 줄 수 있다.

---

## 2. 실험 전략

처음부터 많은 하이퍼파라미터를 동시에 변경하기보다,
먼저 단순한 baseline을 만들고
Validation 결과의 class별 Precision / Recall / F1 및
False Negative 사례를 분석한 뒤
관찰된 문제에 대응하는 실험을 순차적으로 진행하였다.

전체 실험 흐름은 다음과 같다.

```text
Baseline 구축
    ↓
Class별 성능 분석
    ↓
오심의 심한 under-prediction 확인
    ↓
오심 positive weight 실험
    ↓
Residual FN 분석
    ↓
두통의 명시적 증상 표현 miss 확인
    ↓
두통 positive weight 추가
    ↓
다른 class도 동일한 방식으로 분석
    ↓
필요한 class만 개별 실험
    ↓
입력 truncation 방식 실험
    ↓
Seed 안정성 검증
    ↓
최종 checkpoint 선택
```

핵심 원칙은
"성능이 낮은 class에 무조건 weight를 추가하지 않고,
실제 오류 사례를 확인한 뒤 개입 여부를 결정한다"는 것이었다.

---

## 3. Base Model 선정

### KLUE RoBERTa Base

Base model로 `klue/roberta-base`를 사용하였다.

선정 이유는 다음과 같다.

1. 입력이 한국어 119 신고 대화 transcript이므로
   한국어에 특화된 pretrained language model이 적합하다고 판단하였다.
2. 증상 표현은 한 문장에 직접적으로 등장하기도 하지만,
   여러 발화의 문맥을 종합해야 판단 가능한 경우도 있으므로
   문맥 기반 sequence classification 모델이 필요하였다.
3. Base 규모는 성능과 계산 비용 사이의 균형이 좋고,
   제한된 GPU 환경에서도 전체 fine-tuning이 가능하였다.
4. Hugging Face 기반이므로
   학습과 inference preprocessing을 동일하게 유지하기 용이하였다.

---

## 4. 공통 학습 설정

최종 실험에서 공통으로 유지한 설정은 다음과 같다.

- Model: `klue/roberta-base`
- Task: Multi-label Sequence Classification
- Number of Labels: 9
- Max Length: 512
- Input Strategy: Basic truncation
- Train Batch Size: 8
- Validation Batch Size: 16
- Gradient Accumulation Steps: 2
- Effective Batch Size: 16
- Optimizer: AdamW
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Warmup Ratio: 0.1
- Maximum Epochs: 8
- Early Stopping Patience: 2
- Decision Threshold: 0.5
- Metric: Macro-F1

Training data는 29,200개,
Validation data는 3,640개를 사용하였다.

Validation data는 모델 선택과 성능 평가에만 사용하였으며
backpropagation에는 사용하지 않았다.

---

## 5. Baseline

### 5.1 목적

첫 실험의 목적은 복잡한 보정 없이
KLUE/RoBERTa가 현재 데이터에서 어느 정도의 성능을 내는지 확인하고,
어떤 class가 실제 병목인지 파악하는 것이었다.

따라서 baseline에서는 다음과 같이 단순한 설정을 사용하였다.

- Basic truncation
- BCE loss
- class weight 없음
- threshold = 0.5
- learning rate = 2e-5

### 5.2 결과

**Macro-F1 = 0.611252**

주요 class별 결과 중 가장 눈에 띈 문제는 오심이었다.

- 오심: Precision 0.4558 / Recall 0.1562 / F1 0.2326
- 두통: Precision 0.6609 / Recall 0.4118 / F1 0.5074

오심은 precision보다 recall이 매우 낮았고,
실제 positive 429건에 비해 prediction positive가 147건에 불과하였다.

### 5.3 해석

이 결과로부터
모델이 오심을 잘못 많이 예측하는 것이 아니라,
실제 오심 sample을 지나치게 보수적으로 예측하는
under-prediction 문제가 크다고 판단하였다.

따라서 다음 실험은
전체 sampling 구조를 바꾸기보다
오심 positive sample의 loss 기여도만 제한적으로 높이는 방향으로 계획하였다.

---

## 6. Experiment 1 - 오심 Positive Weight

### 6.1 가설

오심은 False Positive보다 False Negative 문제가 더 크므로,
오심 positive sample에 더 높은 loss weight를 부여하면
recall을 높이면서 Macro-F1을 개선할 수 있을 것으로 예상하였다.

### 6.2 설정

- 오심 `pos_weight = 2.0`
- 나머지 class `pos_weight = 1.0`
- 나머지 학습 조건은 baseline과 동일

### 6.3 결과

**Macro-F1 = 0.627050**

Baseline 대비:

**+0.015798**

오심 성능:

- Recall: 0.1562 → 0.3193
- F1: 0.2326 → 0.3526

### 6.4 해석

오심 recall이 약 두 배 증가했고
전체 Macro-F1도 유의하게 상승하였다.

따라서 오심에 대한 제한적인 positive weighting은
현재 데이터에서 효과적인 보정 방법이라고 판단하였다.

하지만 오심 weight를 적용한 뒤에도
두통을 포함한 일부 class에서 False Negative가 많이 남아 있었기 때문에,
다음 단계에서는 단순 수치만 보고 weight를 추가하지 않고
실제 FN transcript를 직접 확인하였다.

---

## 7. False Negative 분석

False Negative를 두 종류로 나누어 분석하였다.

### Type A - 입력 텍스트에서 target 증상 표현을 찾기 어려운 경우

모델의 예측 확률이 매우 낮은 FN 중에는
해당 증상이 transcript에 직접적으로 드러나지 않는 sample이 다수 존재하였다.

이 경우 모델이 해당 label을 텍스트만으로 복원하기 어렵기 때문에
단순히 class weight를 크게 높이면
False Positive만 증가할 가능성이 있다고 판단하였다.

### Type B - 증상 표현이 명확하지만 threshold 아래로 예측된 경우

반면 probability가 0.5에 가까운 borderline FN에서는
실제 증상 표현이 transcript에 직접 등장하는 sample도 확인되었다.

예를 들어 두통의 경우
"두통", "머리가 아프다"와 같이
명시적인 표현이 있는데도 prediction이 0으로 나온 사례가 있었다.

이러한 Type B 오류는
모델이 관련 표현을 전혀 학습하지 못한 것이 아니라
confidence가 조금 부족한 문제로 볼 수 있으므로,
제한적인 positive weighting의 근거가 될 수 있다고 판단하였다.

---

## 8. Experiment 2 - 두통 Positive Weight 추가

### 8.1 가설

두통 borderline FN에서 명시적인 두통 표현이 확인되었으므로,
오심과 마찬가지로 두통 positive sample의 loss 비중을 조금 높이면
recall 개선이 가능할 것으로 예상하였다.

단, 오심만큼 심한 under-prediction은 아니었기 때문에
오심보다 낮은 weight를 사용하였다.

### 8.2 설정

- 오심 `pos_weight = 2.0`
- 두통 `pos_weight = 1.5`
- 나머지 class `pos_weight = 1.0`

### 8.3 결과

**Macro-F1 = 0.635403**

오심 weight 단독 실험 대비:

**+0.008353**

Baseline 대비:

**+0.024151**

### 8.4 해석

두통 weight 추가 후 Macro-F1이 다시 상승하였다.

따라서 최종 loss 방향은
전체 class에 큰 weight를 일괄 적용하는 방식이 아니라
오류 분석으로 근거가 확인된 class에만
작은 positive weight를 적용하는 방식으로 결정하였다.

---

## 9. 다른 Class에 대한 개입 여부 분석

### 9.1 전신쇠약

전신쇠약은 FN 수가 적지 않았지만
low-confidence FN과 borderline FN을 확인한 결과,
"쓰러짐", "움직이지 못함", 의식저하 등
다른 원인으로도 나타날 수 있는 표현이 많이 포함되어 있었다.

즉 전신쇠약 label과 직접 연결하기 어려운
의미적 모호성이 상대적으로 컸다.

따라서 weight를 추가하면
False Positive 증가 위험이 있다고 판단하여
별도의 class weight를 추가하지 않았다.

### 9.2 어지러움

어지러움 역시 일부 borderline FN에는
"어지럽다"와 같은 직접 표현이 있었지만,
현재 모델의 Precision / Recall 균형이 이미 비교적 안정적이었다.

따라서 추가 weighting으로 얻을 수 있는 이득보다
False Positive 증가 위험이 더 크다고 판단하여
weight를 추가하지 않았다.

---

## 10. Experiment 3 - 구토 Positive Weight

### 10.1 실험 이유

구토 FN 분석에서는
"먹으면 토해 버리고", "토할 것 같다", "토 한번 하고"처럼
명시적 표현이 borderline FN에서 확인되었다.

따라서 구토 역시 작은 positive weight를 추가하면
recall이 개선될 가능성이 있다고 판단하였다.

### 10.2 설정

기존 best 설정에 구토 `pos_weight = 1.3`을 추가하였다.

즉:

- 오심 = 2.0
- 두통 = 1.5
- 구토 = 1.3

### 10.3 결과

**Macro-F1 = 0.631112**

기존 0.635403 대비:

**-0.004291**

구토 자체는 소폭 개선되었지만,
오심 등 다른 class 성능이 함께 하락하였다.

### 10.4 결론

특정 class의 성능 개선이
전체 multi-label 모델의 Macro-F1 개선으로 이어지지는 않았다.

따라서 구토 weight는 최종 모델에서 제외하였다.

이 실험은
"낮은 class 성능을 개별적으로 모두 올리는 것"보다
class 간 trade-off를 포함한 전체 Macro-F1을 기준으로 판단해야 한다는
근거가 되었다.

---

## 11. Experiment 4 - Head+Tail Truncation

### 11.1 실험 이유

일부 긴 transcript에서
증상 표현이 512 token 이후에 존재할 가능성을 고려하였다.

Baseline에서는 tokenizer의 기본 truncation을 사용하므로
512 token을 초과하는 경우 뒤쪽 정보가 제거된다.

따라서 긴 대화의 앞부분과 뒷부분을 함께 보존하는
Head+Tail 전략을 실험하였다.

### 11.2 사전 관찰

token length 분석 결과
전체 데이터의 약 97%가 512 token 이내였으며,
오심/두통 FN 중에서도 대부분은 512 token 이하에 위치하였다.

따라서 truncation이 주된 병목일 가능성은 높지 않았지만,
독립적인 입력 전략으로 실제 효과를 확인하였다.

### 11.3 결과

- Basic truncation: **0.635403**
- Head+Tail: **0.633951**

변화:

**-0.001452**

일부 class에서는 개선이 있었지만
구토, 두통, 열상 등의 성능이 하락하여
전체 Macro-F1은 소폭 감소하였다.

### 11.4 결론

긴 입력의 뒷부분을 보존하는 것보다
기존 앞부분 문맥을 유지하는 것이 현재 데이터에서는 더 안정적이었다.

또한 대부분의 sample이 이미 512 token 이내였기 때문에
truncation은 현재 모델의 핵심 병목이 아니라고 판단하였다.

따라서 최종 모델은 Basic truncation을 유지하였다.

---

## 12. Seed 안정성 검증

최종 설정을 결정한 뒤
특정 random seed에서만 우연히 높은 성능이 나온 것은 아닌지 확인하기 위해
동일한 데이터, 모델, loss, threshold 조건에서 seed만 변경하여
3회 학습하였다.

| Run | Macro-F1 |
|---|---:|
| Seed Run 1 | 0.635403 |
| Seed Run 2 | **0.638604** |
| Seed Run 3 | 0.623516 |

평균 Macro-F1은 약 **0.6325**였으며,
최고와 최저 차이는 약 **0.0151**이었다.

seed에 따른 변동은 존재했지만
세 실험 모두 0.62~0.64 범위에 위치하였다.

최종 제출 checkpoint는
Validation Macro-F1이 가장 높은 **0.638604** 모델을 선택하였다.

---

## 13. 최종 모델 설정

- Model: `klue/roberta-base`
- Input: 신고자 + 119대원 transcript
- Max Length: 512
- Input Strategy: Basic truncation
- Loss: BCE with class-focused positive weighting
- 오심 `pos_weight = 2.0`
- 두통 `pos_weight = 1.5`
- 나머지 class `pos_weight = 1.0`
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Effective Batch Size: 16
- Maximum Epochs: 8
- Early Stopping Patience: 2
- Decision Threshold: 0.5
- Metric: Macro-F1

최종 best Validation Macro-F1:

**0.638604**

---

## 14. 최종 실험 요약

| 단계 | 관찰 / 가설 | 실험 | Macro-F1 | 판단 |
|---|---|---|---:|---|
| Baseline | 기본 성능 및 병목 확인 | BCE | 0.611252 | 기준 |
| 오심 분석 | 심한 under-prediction | 오심 weight 2.0 | 0.627050 | 채택 |
| 두통 FN 분석 | 명시적 두통 표현도 일부 miss | 두통 weight 1.5 추가 | 0.635403 | 채택 |
| 전신쇠약 분석 | 원인 모호성 및 FP 위험 | weight 추가 안 함 | - | 유지 |
| 구토 FN 분석 | 명시적 borderline FN 존재 | 구토 weight 1.3 | 0.631112 | 기각 |
| 어지러움 분석 | 현재 P/R 균형이 비교적 안정 | weight 추가 안 함 | - | 유지 |
| 길이 분석 | 일부 긴 transcript 존재 | Head+Tail | 0.633951 | 기각 |
| 안정성 검증 | seed 민감도 확인 | 3 seeds | best 0.638604 | 최종 선택 |

---

## 15. 최종 결론

이번 실험에서 가장 효과적이었던 접근은
모델 구조를 복잡하게 변경하는 것이 아니라,
class별 오류 패턴을 분석하고
필요한 class에만 제한적인 loss weighting을 적용하는 것이었다.

특히 오심은 baseline에서 매우 낮은 recall을 보였고,
positive weight 적용 후 큰 폭으로 개선되었다.
두통 역시 borderline FN 분석에서 명시적인 증상 표현 miss가 확인되어
추가 weighting의 근거가 있었다.

반면 전신쇠약과 어지러움은
오류 사례의 의미적 모호성 또는 기존 Precision/Recall 균형을 고려하여
의도적으로 추가 weighting을 적용하지 않았다.

또한 구토 weighting과 Head+Tail truncation처럼
가설에 근거한 실험을 수행했지만
전체 Macro-F1이 감소한 경우에는 최종 설정에서 제외하였다.

따라서 최종 모델은
단순히 가장 높은 점수를 낸 조합을 선택한 것이 아니라,
Validation error analysis를 바탕으로
각 변경의 필요성과 trade-off를 확인하며 구성하였다.
