# Mission 1: 신고자 성별 분류

## 모델

`facebook/wav2vec2-base`를 기반으로 성별 분류층을 fine-tuning했다. 제출물의 `model/` 폴더에서 모델을 로컬로 불러오므로 추론 시 인터넷 다운로드가 필요하지 않다.

## 전처리

- JSON의 `startAt`, `endAt`, `speaker`만 사용해 `speaker=1` 신고자 구간 추출
- 원본 WAV에서 신고자 구간만 연결
- 16 kHz로 변환
- 최대 8초로 자르고 짧은 음성은 zero-padding
- 모든 샘플에 같은 전처리 적용

## 실행

```bash
pip install -r requirements.txt
python inference.py --wav_dir "Validation/1.원천데이터/VS_서울_구급" --json_dir "Validation/2.라벨링데이터/VL_서울_구급" --model_dir "model" --output "mission1_result.csv"
```

동작 확인은 `--max_samples 100`을 추가한다. 최종 실행에서는 해당 옵션을 생략한다.

## 실험 결과

- Training: 29,200개
- Validation: 3,640개
- Validation Accuracy: 0.9763736 (97.64%)
- 학습 환경: Google Colab NVIDIA Tesla T4 GPU
- 학습: 2 epochs, batch size 4
- Validation 추론 batch size: 1
- Total parameters: 94,569,090
- Active parameters: 94,569,090
- 전체 Validation 추론 시간: 1,399.24초
- 샘플당 평균 추론 시간: 0.3844초

## 규정 준수

Validation 데이터는 역전파 학습에 사용하지 않았다. `gender`는 정답 라벨로만 사용하고, 입력 전처리에는 `startAt`, `endAt`, `speaker`만 사용했다.
