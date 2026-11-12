"""
inference.py — Mission 2 추론 (신고자 = 1 / 119대원 = 0)

    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path checkpoints/mission2.ckpt --output ./outputs/mission2.csv

- 라벨 json 에서는 utterances 의 startAt, endAt 만 읽는다 (speaker, text, 발화 순서는 쓰지 않음)
- 발화마다 독립적으로 예측하고, 확률 >= 0.5 이면 신고자(1)
- 입력 발화마다 1행을 출력한다. 구간이 wav 길이를 넘으면 wav 끝에 맞춰 자르고,
  wav 가 없거나 구간을 읽을 수 없는 발화도 빼지 않고 신고자(1, 다수 클래스)로 채운다
- 출력 CSV 열: audio file name, startAt, endAt, speaker (startAt/endAt 은 입력 값 그대로)
- WavLM 설정·가중치가 체크포인트 안에 있어 인터넷 없이 실행된다
"""
import argparse
import csv
import json
import os
import platform
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")           # 인터넷 접속 시도 자체를 막음
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

import model as M


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="wav 폴더 (하위 폴더 포함)")
    ap.add_argument("--label_dir", required=True, help="라벨 json 폴더 (하위 폴더 포함)")
    ap.add_argument("--ckpt_path", default=os.path.join(here, "checkpoints", "mission2.ckpt"))
    ap.add_argument("--output", default="./outputs/mission2.csv")
    ap.add_argument("--batch_size", type=int, default=32, help="forward 1회에 넣는 발화 수")
    ap.add_argument("--num_workers", type=int, default=0, help="wav 읽기 프로세스 수")
    ap.add_argument("--device", default=None, help="cuda / cpu (기본: 자동)")
    ap.add_argument("--report", default=None, help="(선택) 추론 시간·파라미터 수·환경을 기록할 json 경로")
    args = ap.parse_args()
    started = time.time()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = M.load_checkpoint(args.ckpt_path, device)
    rows = M.build_manifest(args.label_dir, args.audio_dir, with_label=False)     # startAt, endAt 만 읽음

    labels = np.ones(len(rows), dtype=int)               # 읽을 수 없는 발화는 신고자(1)로 채움
    usable = [i for i, r in enumerate(rows) if r["ok"]]
    infer_started, model_seconds = time.time(), 0.0
    if usable:
        probabilities, model_seconds = M.predict(model, [rows[i] for i in usable], args.batch_size, device, args.num_workers)
        labels[usable] = (probabilities >= M.THRESHOLD).astype(int)
    infer_seconds = time.time() - infer_started

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio file name", "startAt", "endAt", "speaker"])
        for row, label in zip(rows, labels):
            writer.writerow([row["wav_name"], row["start"], row["end"], int(label)])

    n = max(len(usable), 1)
    M.log(f"저장: {args.output} ({len(rows):,}행, 읽을 수 없어 1로 채운 발화 {len(rows) - len(usable):,}개)")
    M.log(f"추론 시간: {infer_seconds:.1f}s (발화당 {infer_seconds / n * 1000:.2f}ms, 그중 모델 forward {model_seconds / n * 1000:.2f}ms), "
          f"batch size {args.batch_size}, device {device}, 모델 로드 포함 전체 {time.time() - started:.1f}s")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"utterances": len(rows), "unreadable_utterances": len(rows) - len(usable),
                       "batch_size": args.batch_size, "num_workers": args.num_workers, "device": device,
                       "inference_seconds": round(infer_seconds, 2), "ms_per_utterance": round(infer_seconds / n * 1000, 3),
                       "model_seconds": round(model_seconds, 2), "model_ms_per_utterance": round(model_seconds / n * 1000, 3),
                       "total_seconds": round(time.time() - started, 2), "parameters": M.count_parameters(model),
                       "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                       "cpu_count": os.cpu_count(), "python": platform.python_version(), "torch": torch.__version__,
                       "checkpoint_meta": meta}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
