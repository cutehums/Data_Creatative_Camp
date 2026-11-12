# Mission 3 — 대화 전사 기반 증상 분류

## 공식 실행

Python3.11 환경에서 `pip install -r requirements.txt` 후 이 폴더에서 실행한다.
GPU용 PyTorch를 설치하면 CUDA를 자동 사용하며 CPU도 지원한다.

```bash
python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path model/model.pt --output ./outputs/mission3.csv
```

명령 한 번으로 JSON 읽기, 동일 전처리, 모델 추론, 고정 기준 적용, CSV 저장까지 완료한다.
`--audio_dir`는 공통 형식에 맞춰 받으며 음성을 읽지 않는다.
선택 인자: `--device cpu`, `--batch_size 16`, `--report ./outputs/report.json`.
`--ckpt_path model` 디렉터리도 지원하지만 공식 실행에는 위 .pt 파일 경로를 권장한다.
가중치 옆 config/tokenizer 파일을 함께 유지한다. 추론은 local_files_only와 offline 모드로 실행하며 인터넷 다운로드가 없다.

## 입력과 출력

입력은 `utterances[].text`뿐이다. None/빈 발화는 제외하고 발화별 앞뒤 공백을 제거한 뒤 한 칸으로 연결한다. 빈 대화의 JSON도 예측 결과 행을 출력한다.
모든 입력에 동일한 Basic truncation(최대512토큰, 오른쪽 절단)과 배치별 padding을 적용한다.
단일 모델의 sigmoid 확률에 모든 클래스 동일하게 `>=0.5`를 적용한다. 라벨별 threshold나 threshold 탐색은 사용하지 않는다.
출력 열은 `label file name, symptom`이며 실제 JSON 파일 이름과 문자열 증상 목록을 저장한다. 예: `["두통", "복통"]`, `[]`.
라벨 순서: 고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란.

## 선택한 모델

공개 `klue/roberta-base` 기반 9-label sequence classifier 단일 모델이다.
발표 자료의 최종 후보 설정은 오심2.0·두통1.5·나머지1.0의 BCE 양성 가중치와 Basic512 입력이다.
이 패키지는 받은 모델의 가중치를 텐서 변경 없이 .pt로 변환했다. 새 학습을 실행하지 않았으며 이전 역빈도 재추출 3개 앙상블을 포함하지 않는다.
실험 문서의 최고 Macro F1은0.638604이고, 받은 가중치를 기존 양성 통화3,640건 토큰 캐시에서 재확인한 값은 **0.6385532655**다.
서로 다른 평가 범위의 점수를 성능 순위로 직접 비교하지 않는다. 독립 Test 성능은 확인되지 않았다.

## 실제 기록과 확인 범위

`training.ipynb`는 학습 코드와 실제 원본 노트북에서 남은 epoch 출력, 실험 요약, 새 제출 코드의 검증 결과를 포함한다.
`records/original_training_log.txt`의 원본 저장 run은 최고0.623516이다. 원본 코드의 seed123 표기와 출력2026이 다르며, 문서 최고0.638604의 run과 현재 가중치·학습 로그의 일대일 대응은 완전히 확인되지 않았다. 기록을 수정하거나 없는 출력 값을 만들지 않았다.
문서의 학습29,200/검증3,640건은 기존 CSV 기반이며 빈 타깃0건으로 기록돼 있다. **그 CSV가 주최 측 제공 파일만으로 구성됐는지 원본 배포 목록과 대조하지 못했다.** 따라서 이 패키지의 파일·실행 준비와 데이터 규정 검증 완료를 구분한다.
최종 데이터 범위는 주최 측 제공 파일 목록으로 확인해야 한다. 로컬 파일 개수 또는 라벨 유무만으로 범위를 정하지 않는다.

## 학습 코드

참고 구현은 전사만 모델 입력으로 사용하고 target9개 이외의 라벨만 무시한다. 통화 자체를 target 유무로 제거하지 않는다.
AdamW(lr2e-5, weight decay0.01), batch8·누적2, warmup10%, 최대8epoch·개선 없는2epoch 종료, 선택적 BCE 양성 가중치, CUDA FP16/GradScaler를 사용한다.
Validation은 no_grad 평가와 best checkpoint 선정에만 사용하며 역전파에 넣지 않는다.

```bash
python train.py --train_data {verified supplied Seoul JSON folder or CSV} --validation_data {verified supplied Validation JSON folder or CSV} --pretrained_dir {local fresh klue-roberta-base} --output_dir ./new_training_run --seed {actual seed}
```

이 참고 코드는 보고된29,200/3,640건 조건을 확인한다. 개수를 맞추려고 증상 없는 통화를 삭제하면 안 된다. 재학습에는 제출된 fine-tuned 모델이 아닌 공개 원본 사전학습 모델을 별도로 사용한다.

## 효율과 실행 검증

- Total / Active: **110,625,033 / 110,625,033**, 단일 dense 모델.
- 현재 검증 환경: Conda Python3.11.16, torch2.8.0+cu128, transformers4.57.1, NVIDIA GeForce RTX5070 및 CPU.
- Validation batch size16. CUDA에서는 원본 추론과 동일한 FP16 autocast/sigmoid를 유지한다.
- 인터넷 TCP 연결 차단·빈 HF캐시에서 GPU/CPU 공식 CLI를 확인한다. 금지 metadata 변경 불변성, 빈/긴 전사, 하위 폴더 입력, CSV 열/라벨도 확인한다.
- 전체 실제 제공 Validation 폴더의 end-to-end 시간을 측정한 것은 아니다. 기존 로컬7,748건 토큰 캐시의 forward 시간과 3,640건 양성 subset 점수는 `records/final_validation_verification.json`에 별도 범위로 기록했다. JSON 읽기·모델 로딩·CSV 저장은 그 forward 시간에 포함되지 않는다.

## 제출 파일

- `inference.py`: 공식 명령과 CSV 출력
- `model.py`: 로컬 모델 로딩·전사 전처리·확률 예측
- `train.py`: 선택적 양성 BCE 학습 참고 구현
- `training.ipynb`: 학습 코드·실제 원본 epoch 기록·실험·검증 결과
- `model/`: .pt 가중치, config, tokenizer
- `requirements.txt`: 현재 확인 환경의 패키지 버전
- `records/`: 실험·원본 로그·가중치 출처·검증 기록
- `CHECKSUMS.sha256`: 제출 파일 무결성 확인용

원천 데이터, 토큰 캐시, optimizer/resume 파일, 이전 앙상블, 실행 출력 CSV는 포함하지 않는다.
