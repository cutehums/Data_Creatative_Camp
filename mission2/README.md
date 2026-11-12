# Mission 2 신고자 / 119대원 발화 분류

발화 조각 하나(라벨 json의 `startAt`~`endAt` 구간)를 듣고 신고자(1)인지 119대원(0)인지 맞히는 문제. 평가 지표는 Accuracy.

## 실행

```bash
pip install -r requirements.txt
python inference.py --audio_dir {wav folder} --label_dir {json folder} \
    --ckpt_path checkpoints/mission2.ckpt --output ./outputs/mission2.csv
```

- 위 명령 하나로 발화 목록 생성, 추론, CSV 저장까지 돌아감
- 출력 열: `audio file name, startAt, endAt, speaker` (입력 발화 1개당 1행)
- 라벨 json에서 읽는 값은 `startAt`, `endAt` 두 개뿐. `speaker`, `text`, 발화 순서는 추론에 안 씀
- 확률 0.5 이상이면 신고자(1). threshold는 0.5 고정
- WavLM 설정과 가중치를 체크포인트에 같이 넣어 둬서 인터넷 없이 실행 가능. GPU 있으면 자동으로 사용하고, CPU로 돌리려면 `--device cpu`
- 구간이 wav 길이보다 길면 wav 끝에서 자름. wav가 없거나 구간을 못 읽는 발화는 신고자(1)로 채워서 행 수를 맞춤
- 선택 인자: `--batch_size 32`, `--num_workers 0`, `--report report.json` (추론 시간, 파라미터 수, 환경 기록)

## 모델

```text
발화 구간 (8kHz → 16kHz, 최대 6초, 0.25초 단위 zero padding)
  → WavLM-base-plus CNN + Transformer 앞 4층
  → 층별(임베딩 + 4층) 평균, 표준편차 1536차원 (패딩 구간 제외)
  → 층별 표준화 → softmax 층 가중합 → LayerNorm → Linear 256 → GELU → Dropout → Linear 1 → sigmoid
```

- 신고자는 휴대폰이나 현장 소음이 섞인 채널, 119대원은 상황실 헤드셋이라 채널과 음색 차이가 큼. 이런 정보는 WavLM 아래층에 주로 남아 있어서 12층 중 앞 4층만 사용
- 6초 넘는 발화는 가운데 6초만 사용
- padding 길이를 그 발화 길이로만 정하기 때문에 batch를 어떻게 묶든 결과가 같음

## 학습 (`train.py`)

1. Training 통화를 통화 단위로 나눠서 내부 dev를 따로 뺌. 학습용은 통화마다 발화를 몇 개씩만 뽑음(label 안 보고 랜덤)
2. 1단계, WavLM 동결: 발화별 층 통계를 한 번만 계산해 저장해 두고 분류기만 학습
3. 2단계, 미세조정: 1단계 모델에서 이어서 Transformer 위쪽 2층 + 분류기 학습. 그 아래 층과 CNN은 동결
4. 내부 dev Accuracy(@0.5)가 제일 높은 모델을 체크포인트로 저장

Validation 폴더는 학습이나 모델 선택에 안 쓰고 `inference.py`로 평가만 했음. Training은 서울 데이터만 사용.

```bash
python prepare_data.py --zip_root {대회 제공 zip} --out_dir {D} --train_calls 29200 --val_calls 3640
python train.py --train_audio_dir {D}/train/wav --train_label_dir {D}/train/json \
    --out_ckpt checkpoints/mission2.ckpt --work_dir work \
    --dev_calls 1000 --utts_per_call 12 --dev_utts_per_call 1000000 --finetune_epochs 4
```

실제로 돌린 명령은 `logs/prepare_log.txt`, `logs/train_log.txt`에서 `$`로 시작하는 줄에 있음.

## 파일

| 파일 | 내용 |
|---|---|
| `inference.py` | 제출용 추론 (명령 1회로 CSV 생성) |
| `model.py` | 발화 목록 생성, 음성 구간 처리, 모델, 예측, 체크포인트 저장/로드 |
| `train.py` | 학습 (분류기 학습 → 미세조정 → 내부 dev 기준 선택) |
| `prepare_data.py` | 대회 제공 zip에서 학습/평가 통화 추출 |
| `checkpoints/mission2.ckpt` | 학습된 모델 |
| `logs/` | 학습 로그, 추론 로그, 효율 기록(`efficiency_report.md`) |
| `requirements.txt` | 패키지 목록 |

## 현재 제출 경로 점검

2026-10-12, Conda Python3.11.16 / torch2.8.0+cu128 / transformers4.57.1 / RTX5070에서 공식 CLI를 확인했다.
인터넷 연결을 차단하고 비어 있는 Hugging Face 캐시에서 합성 음성3개·발화6개를 출력했다.
speaker·text·gender·symptom을 바꿔도 추론 출력이 같았다. 결과는 `records/submission_smoke.json`에 기록했다.
이 확인은 실행 형식·오프라인 로딩 점검이며, 원본 Validation 재평가 수치로 해석하지 않는다.
기존 Validation 성능·학습 환경·추론 시간은 `logs/efficiency_report.md`의 원본 기록을 유지했다.
