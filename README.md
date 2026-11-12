# Data Creative Camp

119 신고 데이터를 활용해 음성 및 대화 전사 정보를 분석한 팀 프로젝트입니다.

프로젝트는 세 가지 Mission으로 구성됩니다.

- **Mission 1:** 신고자 음성 기반 성별 분류
- **Mission 2:** 음성 구간별 신고자·119대원 분류
- **Mission 3:** 신고 대화 전사 기반 증상 다중 분류

각 Mission은 서로 다른 입력과 모델을 사용하지만, 공통적으로 119 신고 데이터를 구조화하고 사전학습 모델을 활용해 분류 문제를 해결합니다.

> 이 저장소는 팀 프로젝트의 코드와 실험 기록을 공개하기 위한 저장소입니다.  
> 원천 데이터와 대용량 학습 가중치는 포함되어 있지 않으므로, 저장소만 복제해서 전체 추론을 바로 실행할 수는 없습니다.

---

## Project Overview

| Mission | 입력 | 목표 | 모델 | 평가 지표 |
|---|---|---|---|---|
| Mission 1 | 신고자 음성 | 성별 `M` / `F` 분류 | `facebook/wav2vec2-base` | Accuracy |
| Mission 2 | 개별 발화 음성 구간 | 신고자 / 119대원 분류 | `microsoft/wavlm-base-plus` | Accuracy |
| Mission 3 | 신고 대화 전사 | 9개 증상 다중 분류 | `klue/roberta-base` | Macro-F1 |

<img width="326" height="241" alt="KakaoTalk_20261113_045023389" src="https://github.com/user-attachments/assets/2ed5a612-5c25-4d3c-905d-718ae795bae0" />


## Missions

### Mission 1 — Caller Gender Classification

신고자의 음성 구간을 추출하고 성별을 `M` 또는 `F`로 분류합니다.

JSON의 `_id`와 WAV 파일명을 연결한 뒤, `utterances`에서 `speaker=1`인 발화의 `startAt`과 `endAt` 구간만 선택합니다. 선택된 구간은 하나의 음성으로 연결되어 Wav2Vec2 분류 모델에 입력됩니다.

#### Preprocessing

- 스테레오 음성을 모노로 변환
- 신고자(`speaker=1`) 발화 구간만 연결
- 16kHz 리샘플링
- 최대 8초로 절단
- 8초보다 짧은 입력은 zero-padding
- Training과 Validation에 동일한 전처리 적용

#### Model and Training

- Base model: `facebook/wav2vec2-base`
- Model: `Wav2Vec2ForSequenceClassification`
- Number of labels: 2
- Feature encoder 동결
- Optimizer: AdamW
- 기본 learning rate: `1e-5`
- 기본 epoch: 2
- 기본 batch size: 4

`inference.py`는 압축 해제된 폴더와 ZIP 입력을 모두 지원하며, 예측 결과를 `id, gender` 형식의 CSV로 저장합니다.

저장된 실험 문서에는 Training 29,200건과 Validation 3,640건을 사용했으며, Validation Accuracy `0.9763736264`가 기록되어 있습니다. 이 값은 저장소에 포함된 기록을 옮긴 것이며 현재 공개 저장소에서 재실행해 검증한 결과는 아닙니다.

[Mission 1 자세히 보기](./Mission1/README.md)

---

### Mission 2 — Caller / Dispatcher Classification

JSON에 정의된 개별 음성 구간을 듣고 각 발화를 신고자 또는 119대원으로 분류합니다.

추론 과정에서는 각 발화의 `startAt`과 `endAt`만 사용하며, `speaker`, `text`, `gender`, `symptom` 등의 값은 모델 입력에 사용하지 않습니다.

#### Audio Pipeline

```text
발화 구간
→ 8kHz 음성을 16kHz로 리샘플링
→ 최대 6초 처리
→ WavLM-base-plus의 앞 4개 Transformer Layer
→ Layer별 평균·표준편차 계산
→ 학습 가능한 Layer 가중 결합
→ Binary Classifier
→ threshold 0.5
```

#### Training Strategy

- 통화 단위로 Training과 내부 Development 데이터 분리
- 통화당 발화 일부를 고정 seed로 샘플링
- 1단계: WavLM을 동결하고 분류기 학습
- 2단계: 상위 Transformer 2개 층과 분류기 미세조정
- 내부 Development Accuracy를 기준으로 체크포인트 선택
- 중단된 학습을 이어갈 수 있는 상태 저장 로직 제공

저장된 실험 기록에는 다음 결과가 기록되어 있습니다.

- Internal dev Accuracy: `0.9209`
- Validation Accuracy: `0.9173`
- Validation utterances: 111,947개
- Total parameters: 38,070,742
- 기록된 Validation 처리 시간: 발화당 약 1.15ms

수치는 `logs/`에 포함된 기존 실험 기록을 기준으로 하며 현재 환경에서 재측정한 결과는 아닙니다.

[Mission 2 자세히 보기](./mission2/README.md)

---

### Mission 3 — Symptom Multi-label Classification

119 신고 대화의 전사 텍스트를 입력으로 받아 다음 9개 증상을 다중 분류합니다.

- 고열
- 구토
- 두통
- 복통
- 어지러움
- 열상
- 오심
- 전신쇠약
- 호흡곤란

`utterances[].text`만 모델 입력으로 사용합니다. 빈 발화는 제외하고 나머지 텍스트를 공백으로 연결한 뒤 최대 512토큰으로 제한합니다.

#### Model and Training

- Base model: `klue/roberta-base`
- Task: Multi-label sequence classification
- Loss: `BCEWithLogitsLoss`
- Maximum length: 512 tokens
- Optimizer: AdamW
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Gradient accumulation: 2 steps
- 최대 8 epochs
- Early stopping patience: 2
- 모든 클래스의 판정 threshold: `0.5`

코드에는 오심과 두통의 False Negative 분석을 바탕으로 다음 positive weight가 적용되어 있습니다.

- 오심: `2.0`
- 두통: `1.5`
- 나머지 클래스: `1.0`

실험 문서에는 최고 Validation Macro-F1 `0.638604`가 기록되어 있습니다. 다만 현재 가중치와 해당 최고 실험 로그의 일대일 대응 및 공식 데이터 구성 여부가 완전히 검증되지 않았다는 제한도 함께 기록되어 있습니다. 따라서 이 수치는 독립 Test 성능으로 해석하지 않습니다.

[Mission 3 자세히 보기](./Mission3/README.md)

---

## Tech Stack

### Language and Framework

- Python
- PyTorch
- Hugging Face Transformers
- Torch DataLoader

### Audio Processing

- Wav2Vec2
- WavLM
- librosa
- SoundFile
- SciPy
- NumPy

### NLP and Evaluation

- KLUE RoBERTa
- Hugging Face Tokenizers
- scikit-learn
- pandas

### Experiment Environment

코드와 기록에서 Google Colab, NVIDIA Tesla T4, NVIDIA A100 및 CUDA 환경의 사용이 확인됩니다. Mission 2와 Mission 3의 공개 requirements는 Python 3.11 실행 환경을 기준으로 정리되어 있습니다.

---

## My Contributions

이 프로젝트는 **팀 프로젝트**입니다.

저는 세 가지 Mission 중 **Mission 1에 기여했습니다.**

Mission 1 코드에서 확인되는 주요 구현 범위는 다음과 같습니다.

- JSON과 WAV 파일 ID 매칭
- `speaker=1` 신고자 발화 구간 추출
- 발화 구간 연결 및 음성 전처리
- 16kHz 리샘플링
- 8초 고정 길이 padding 및 truncation
- Wav2Vec2 기반 성별 분류 학습 코드
- 폴더 및 ZIP 입력을 지원하는 추론 파이프라인
- CSV 예측 결과 생성

다만 현재 저장소의 코드와 문서만으로는 위 항목별 개인 담당자를 독립적으로 확인할 수 없습니다. 따라서 특정 클래스나 기능 전체를 개인 작업으로 임의 귀속하지 않으며, 확인된 개인 기여 범위는 **Mission 1 참여**로 한정합니다.

---

## Project Structure

```text
Data_Creatative_Camp/
├── Mission1/
│   ├── dcc_mission1.ipynb
│   ├── train.py
│   ├── inference.py
│   ├── Mission1_실험정리.md
│   ├── README.md
│   ├── requirements.txt
│   └── model/
│       ├── config.json
│       ├── processor_config.json
│       ├── tokenizer_config.json
│       └── vocab.json
│
├── mission2/
│   ├── DCC_Mission2.ipynb
│   ├── prepare_data.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── 실험정리_mission2.md
│   ├── README.md
│   ├── requirements.txt
│   ├── logs/
│   └── records/
│
└── Mission3/
    ├── training.ipynb
    ├── model.py
    ├── train.py
    ├── inference.py
    ├── README.md
    ├── requirements.txt
    ├── model/
    └── records/
```

---

## Repository Limitations

이 저장소에는 저작권, 개인정보, 저장소 용량 및 GitHub의 단일 파일 크기 제한을 고려하여 다음 항목이 포함되어 있지 않습니다.

- 대회 원천 WAV 및 JSON 데이터
- Mission 1의 학습된 `model.safetensors`
- Mission 2의 `checkpoints/mission2.ckpt`
- Mission 3의 `model/model.pt`
- 학습 과정에서 생성된 캐시 및 중간 산출물

따라서 추론을 실행하려면 각 Mission에 대응하는 원천 데이터와 학습 가중치를 별도로 준비해야 합니다. 저장소에 포함된 설정 파일만으로는 학습된 모델을 복원할 수 없습니다.

---

## Results Summary

| Mission | 저장된 실험 기록 | 주의사항 |
|---|---:|---|
| Mission 1 | Validation Accuracy `0.9764` | 공개 저장소에서 재실행하지 않음 |
| Mission 2 | Validation Accuracy `0.9173` | 기존 로그 기준 |
| Mission 3 | Best documented Macro-F1 `0.638604` | 최고 기록과 현재 가중치의 일대일 대응이 완전히 검증되지 않음 |

위 결과는 저장소에 포함된 기존 실험 문서와 로그를 요약한 것입니다. 독립 Test 데이터에 대한 성능을 의미하지 않습니다.

---

## License and Data Notice

이 저장소에는 별도의 LICENSE 파일이 없습니다. 따라서 코드 또는 문서의 재사용 조건은 프로젝트 소유자와 팀의 확인이 필요합니다.

원천 119 신고 데이터는 이 저장소에 포함되어 있지 않으며, 데이터 사용 시 해당 데이터 제공기관의 이용 조건과 개인정보 보호 규정을 준수해야 합니다.
