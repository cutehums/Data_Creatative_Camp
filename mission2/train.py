"""
train.py — Mission 2 학습

    python train.py --train_audio_dir {wav 폴더} --train_label_dir {json 폴더} \
        --out_ckpt checkpoints/mission2.ckpt --work_dir {중간 결과 폴더}

절차 (Validation 폴더는 여기서 쓰지 않는다. 평가는 inference.py 로 따로 한다)
  1) 발화 목록 → 통화 단위로 내부 dev 를 떼고, 통화마다 발화 몇 개를 label 과 무관하게 고른다
     (한 통화의 발화는 같은 두 사람의 목소리라 서로 비슷하다. 통화당 몇 개만 쓰고 통화 수를 늘리는 편이 효율적)
  2) 1단계 — WavLM 동결: 발화마다 층별 통계를 한 번만 계산해 저장하고, 그 통계로 분류기만 학습
  3) 2단계 — 미세조정: 1단계 모델에서 시작해 WavLM Transformer 위쪽 2층과 분류기를 학습 (그 아래 층과 CNN 특징 추출기는 동결)
  4) 내부 dev Accuracy@0.5 가 가장 높은 모델(1단계 또는 2단계의 어느 epoch)을 체크포인트로 저장
모든 선택은 내부 dev 로만 하고, 판정 기준은 항상 확률 >= 0.5 다.
"""
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import model as M

HEAD = dict(epochs=40, batch_size=256, lr=1e-3, weight_decay=0.01, patience=6)                 # 1단계: 분류기
TUNE = dict(encoder_lr=1e-5, head_lr=1e-4, weight_decay=0.01, warmup_ratio=0.1, grad_clip=1.0)  # 2단계: 미세조정


# ------------------------------------------------------------
# 데이터: 통화 단위 분할, 통화당 발화 추출
# ------------------------------------------------------------
def split_calls(rows, dev_calls, seed):
    """통화를 무작위 순서로 섞어 앞의 dev_calls 개를 내부 dev 로 (같은 통화의 발화가 양쪽에 섞이지 않게)."""
    calls = sorted({r["call"] for r in rows})
    order = [calls[i] for i in np.random.default_rng(seed).permutation(len(calls))]
    dev = set(order[:dev_calls])
    return [r for r in rows if r["call"] not in dev], [r for r in rows if r["call"] in dev]


def sample_per_call(rows, per_call, seed):
    """통화마다 발화를 최대 per_call 개 고른다. 통화 이름으로 정한 난수를 써서 label·실행 순서와 무관하다."""
    by_call = {}
    for i, r in enumerate(rows):
        by_call.setdefault(r["call"], []).append(i)
    keep = []
    for call, index in by_call.items():
        rng = np.random.default_rng([seed, int(hashlib.sha256(call.encode()).hexdigest()[:8], 16)])
        index = sorted(index, key=lambda i: (rows[i]["a"], rows[i]["b"]))
        keep += [index[k] for k in rng.permutation(len(index))[:per_call]]
    return [rows[i] for i in sorted(keep)]


def scores(labels, probabilities):
    """Accuracy@0.5, log loss, 클래스별 Accuracy."""
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predicted = (probabilities >= M.THRESHOLD).astype(int)
    by_class = [round(float((predicted[labels == c] == c).mean()), 4) if (labels == c).any() else None for c in (0, 1)]
    return {"accuracy": float((predicted == labels).mean()),
            "loss": float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean()),
            "acc_119": by_class[0], "acc_caller": by_class[1]}


# ------------------------------------------------------------
# 1단계: 동결 WavLM 통계 → 분류기 학습
# ------------------------------------------------------------
def frozen_stats(model, rows, args, device, name):
    """발화마다 층별 통계를 한 번 계산 (float16 으로 저장, 같은 발화 목록이면 파일에서 다시 읽음)."""
    key = hashlib.md5("\n".join(f"{r['call']}|{r['a']}|{r['b']}" for r in rows).encode()).hexdigest()[:12]
    path = os.path.join(args.work_dir, f"stats_{name}_{model.wavlm.config.num_hidden_layers}layers_{key}.npy")
    if os.path.exists(path):
        M.log(f"저장된 통계 사용: {path}")
        return np.load(path)
    started, stats = time.time(), None
    loader = DataLoader(M.Segments(rows, model.do_normalize), batch_sampler=M.length_batches(rows, args.batch_size),
                        num_workers=args.num_workers, collate_fn=M.collate)
    model.eval()
    for k, (index, waveform, lengths) in enumerate(loader):
        with torch.inference_mode():
            s = model.layer_stats(waveform.to(device), lengths.to(device)).to(torch.float16).cpu().numpy()
        if stats is None:
            stats = np.zeros((len(rows), *s.shape[1:]), dtype=np.float16)
        stats[index] = s
        if (k + 1) % 200 == 0:
            M.log(f"  통계 계산 {name}: batch {k + 1}/{len(loader)} ({time.time() - started:.0f}s)")
    np.save(path, stats)
    M.log(f"통계 계산 {name}: 발화 {len(rows):,}개, {time.time() - started:.0f}s ({len(rows) / (time.time() - started):.0f} utt/s)")
    return stats


def fit_head(model, x_train, y_train, x_dev, y_dev, device, seed):
    """저장한 통계로 분류기만 학습. 매 epoch 내부 dev Accuracy@0.5(같으면 loss)가 가장 좋은 상태를 남긴다."""
    torch.manual_seed(seed)
    x = torch.from_numpy(x_train).to(device)
    y = torch.from_numpy(y_train.astype(np.float32)).to(device)
    xd = torch.from_numpy(x_dev).to(device)
    # 층별 표준화 통계 (학습 데이터로만 계산). 거의 변하지 않는 차원은 1e-4 로 하한을 둔다
    mean = torch.stack([x[:, k].float().mean(dim=0) for k in range(x.shape[1])])
    var = torch.stack([x[:, k].float().var(dim=0, unbiased=False) for k in range(x.shape[1])])
    model.stat_mean.copy_(mean)
    model.stat_scale.copy_(torch.sqrt(var + 1e-4))
    optimizer = torch.optim.AdamW(model.head_parameters(), lr=HEAD["lr"], weight_decay=HEAD["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    history, best = [], None
    for epoch in range(1, HEAD["epochs"] + 1):
        model.train()
        total = 0.0
        for index in torch.randperm(len(x), generator=generator).split(HEAD["batch_size"]):
            index = index.to(device)
            loss = loss_fn(model.classify(x[index]), y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        model.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(model.classify(xd[k:k + 4096])) for k in range(0, len(xd), 4096)]).cpu().numpy()
        record = {"epoch": epoch, "train_loss": total / len(x), **scores(y_dev, p.astype(np.float64))}
        history.append(record)
        if best is None or (record["accuracy"], -record["loss"]) > (best[0]["accuracy"], -best[0]["loss"]):
            best = (record, {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("wavlm.")})
        elif epoch - best[0]["epoch"] >= HEAD["patience"]:
            break                                # dev 가 patience epoch 동안 나아지지 않으면 중단
    model.load_state_dict(best[1], strict=False)             # 분류기만 되돌림 (WavLM 은 그대로)
    M.log(f"1단계 분류기: best epoch {best[0]['epoch']} / dev Accuracy {best[0]['accuracy']:.4f} (저장한 통계 기준)")
    return history


# ------------------------------------------------------------
# 2단계: 미세조정
# ------------------------------------------------------------
def evaluate(model, rows, labels, args, device):
    """내부 dev 를 실제 추론 경로(M.predict)로 평가."""
    probabilities, _ = M.predict(model, rows, args.batch_size, device, args.num_workers)
    return scores(labels, probabilities)


def finetune(model, train_rows, dev_rows, y_dev, args, device, state, resuming):
    """WavLM Transformer 위쪽 2층 + 분류기 학습. 좋아질 때마다 체크포인트를 저장하고, 끊기면 last.pt 에서 이어 간다."""
    model.set_finetune(True)
    encoder_parameters = [p for m in model.tuned_modules() for p in m.parameters()]
    optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": TUNE["encoder_lr"]},
                                   {"params": model.head_parameters(), "lr": TUNE["head_lr"]}],
                                  weight_decay=TUNE["weight_decay"])
    base_rates = [TUNE["encoder_lr"], TUNE["head_lr"]]
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    steps = len(M.length_batches(train_rows, args.batch_size))
    total, warmup = steps * args.finetune_epochs, int(steps * args.finetune_epochs * TUNE["warmup_ratio"])
    last = os.path.join(args.work_dir, "last.pt")
    if resuming:                                # 이어서 학습: 마지막으로 끝난 epoch 의 상태 복원
        resume = torch.load(last, map_location="cpu", weights_only=True)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        M.log(f"epoch {len(state['tune_history'])} 다음부터 이어서 학습")
    else:
        torch.save({"state": state, "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, last)
    labels = torch.tensor([r["speaker"] for r in train_rows], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(len(state["tune_history"]) + 1, args.finetune_epochs + 1):
        started = time.time()
        torch.manual_seed(args.seed * 1000 + epoch)
        rng = np.random.default_rng([args.seed, epoch])
        loader = DataLoader(M.Segments(train_rows, model.do_normalize, crop_seed=args.seed * 1000 + epoch),
                            batch_sampler=M.length_batches(train_rows, args.batch_size, rng),
                            num_workers=args.num_workers, collate_fn=M.collate)
        model.train()
        running, seen = 0.0, 0
        for k, (index, waveform, lengths) in enumerate(loader):
            step = (epoch - 1) * steps + k       # warmup 동안 선형 증가, 이후 끝까지 선형 감소
            factor = (step + 1) / max(1, warmup) if step < warmup else max(0.0, (total - step) / max(1, total - warmup))
            for group, rate in zip(optimizer.param_groups, base_rates):
                group["lr"] = rate * factor
            logits = model(waveform.to(device), lengths.to(device))
            loss = loss_fn(logits.float(), labels[index].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss 가 발산했습니다 (epoch {epoch}, step {k})")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder_parameters + model.head_parameters(), TUNE["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * len(index)
            seen += len(index)
            if (k + 1) % 200 == 0:
                M.log(f"  epoch {epoch} step {k + 1}/{steps} loss {running / seen:.4f} ({seen / (time.time() - started):.0f} utt/s)")
        record = {"epoch": epoch, "train_loss": running / seen, **evaluate(model, dev_rows, y_dev, args, device),
                  "seconds": round(time.time() - started, 1)}
        state["tune_history"].append(record)
        M.log(f"2단계 epoch {epoch}: train loss {record['train_loss']:.4f} / dev Accuracy {record['accuracy']:.4f} "
              f"(119대원 {record['acc_119']}, 신고자 {record['acc_caller']})")
        keep_best(model, record, f"미세조정 epoch {epoch}", args, state)
        # 재개용 파일: 학습 기록(state)도 함께 넣어 모델 상태와 항상 같은 epoch 를 가리키게 한다
        torch.save({"state": state, "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, last + ".tmp")
        os.replace(last + ".tmp", last)
        save_state(args, state)
    os.remove(last)


def keep_best(model, record, name, args, state):
    """내부 dev Accuracy@0.5 가 지금까지보다 높으면(같으면 loss 가 낮으면) 체크포인트로 저장."""
    best = state["best"]
    if best is None or (record["accuracy"], -record["loss"]) > (best["accuracy"], -best["loss"]):
        state["best"] = {"model": name, "accuracy": record["accuracy"], "loss": record["loss"]}
        M.save_checkpoint(args.out_ckpt, model, {**state["meta"], "selected": name, "dev_accuracy": record["accuracy"]})


def save_state(args, state):
    with open(os.path.join(args.work_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_audio_dir", required=True)
    ap.add_argument("--train_label_dir", required=True)
    ap.add_argument("--out_ckpt", default="checkpoints/mission2.ckpt")
    ap.add_argument("--work_dir", default="work", help="통계 캐시·재개 파일·학습 기록을 두는 폴더")
    ap.add_argument("--pretrained", default=M.PRETRAINED, help="Hugging Face 이름 또는 로컬 폴더")
    ap.add_argument("--layers", type=int, default=4, help="사용할 WavLM Transformer 층 수")
    ap.add_argument("--dev_calls", type=int, default=1000, help="내부 dev 로 떼는 통화 수")
    ap.add_argument("--utts_per_call", type=int, default=6, help="학습 통화당 발화 수")
    ap.add_argument("--dev_utts_per_call", type=int, default=12, help="dev 통화당 발화 수")
    ap.add_argument("--finetune_epochs", type=int, default=4, help="0 이면 1단계(동결)만")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrain", action="store_true", help="끝난 학습·끊긴 학습 기록을 무시하고 처음부터")
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()

    # 1) 데이터
    rows = M.build_manifest(args.train_label_dir, args.train_audio_dir, with_label=True)
    train_rows, dev_rows = split_calls(rows, args.dev_calls, args.seed)
    train_rows = sample_per_call(train_rows, args.utts_per_call, args.seed)
    dev_rows = sample_per_call(dev_rows, args.dev_utts_per_call, args.seed)
    if {r["call"] for r in train_rows} & {r["call"] for r in dev_rows} or not train_rows or not dev_rows:
        raise RuntimeError("train/dev 분할이 잘못되었습니다")
    y_train = np.array([r["speaker"] for r in train_rows])
    y_dev = np.array([r["speaker"] for r in dev_rows])
    for name, part, y in (("train", train_rows, y_train), ("dev", dev_rows, y_dev)):
        seconds = np.array([(r["b"] - r["a"]) / r["sr"] for r in part])
        if not 0.05 < np.median(seconds) < 30:           # 발화 길이가 비정상이면 startAt/endAt 단위(ms)가 다른 것
            raise RuntimeError(f"발화 길이 중앙값이 {np.median(seconds):.3f}초입니다. startAt/endAt 단위를 확인하세요")
        M.log(f"{name}: 통화 {len({r['call'] for r in part}):,}개 / 발화 {len(part):,}개 / 신고자 비율 {y.mean():.3f} / "
              f"길이 중앙값 {np.median(seconds):.2f}s / 6초 초과 {(seconds > 6).mean():.1%}")

    meta = {"pretrained": args.pretrained, "layers": args.layers, "seed": args.seed,
            "train_calls": len({r["call"] for r in train_rows}), "train_utterances": len(train_rows),
            "dev_calls": len({r["call"] for r in dev_rows}), "dev_utterances": len(dev_rows)}
    settings = {k: v for k, v in vars(args).items() if k not in ("retrain", "num_workers")}
    signature = hashlib.md5(json.dumps([meta, settings, HEAD, TUNE], sort_keys=True, default=str).encode()).hexdigest()
    log_path, last = os.path.join(args.work_dir, "train_log.json"), os.path.join(args.work_dir, "last.pt")
    if os.path.exists(log_path) and os.path.exists(args.out_ckpt) and not args.retrain:
        done = json.load(open(log_path, encoding="utf-8"))
        if done.get("signature") == signature and "train_minutes" in done:
            M.log(f"같은 설정의 학습이 이미 끝났습니다: {args.out_ckpt} (다시 학습하려면 --retrain)")
            return
    state = {"signature": signature, "meta": meta, "head": HEAD, "tune": TUNE, "head_history": [],
             "frozen": None, "tune_history": [], "best": None}
    resuming = False
    if os.path.exists(last) and not args.retrain:        # 미세조정 도중 끊긴 실행: 그 지점의 기록에서 이어 감
        saved = torch.load(last, map_location="cpu", weights_only=True)["state"]
        if saved["signature"] == signature:
            state, resuming = saved, True

    # 2) 1단계: 동결 WavLM 통계로 분류기 학습
    torch.manual_seed(args.seed)                 # 분류기 초기값까지 재현되도록
    model = M.RoleModel.from_pretrained(args.pretrained, args.layers).to(device)
    M.log(f"device {device} / WavLM {args.layers}층 / 파라미터 {M.count_parameters(model)['total']:,}")
    if not resuming:
        x_train = frozen_stats(model, train_rows, args, device, "train")
        x_dev = frozen_stats(model, dev_rows, args, device, "dev")
        state["head_history"] = fit_head(model, x_train, y_train, x_dev, y_dev, device, args.seed)
        del x_train, x_dev
        state["frozen"] = evaluate(model, dev_rows, y_dev, args, device)        # 실제 추론 경로로 다시 평가
        M.log(f"1단계(동결) dev Accuracy {state['frozen']['accuracy']:.4f} "
              f"(119대원 {state['frozen']['acc_119']}, 신고자 {state['frozen']['acc_caller']})")
        keep_best(model, state["frozen"], "동결", args, state)
        save_state(args, state)

    # 3) 2단계: 미세조정
    if args.finetune_epochs > 0:
        finetune(model, train_rows, dev_rows, y_dev, args, device, state, resuming)

    # 4) 요약
    best_model, _ = M.load_checkpoint(args.out_ckpt)
    state.update(parameters=M.count_parameters(best_model), train_minutes=round((time.time() - started) / 60, 1),
                 environment={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                              "cpu_count": os.cpu_count(), "python": platform.python_version(), "torch": torch.__version__},
                 checkpoint_mb=round(os.path.getsize(args.out_ckpt) / 1e6, 1))
    save_state(args, state)
    print("\n단계 | epoch | train loss | dev Accuracy | dev loss")
    print(f"1 동결 | - | - | {state['frozen']['accuracy']:.4f} | {state['frozen']['loss']:.4f}")
    for h in state["tune_history"]:
        print(f"2 미세조정 | {h['epoch']} | {h['train_loss']:.4f} | {h['accuracy']:.4f} | {h['loss']:.4f}")
    M.log(f"저장: {args.out_ckpt} ({state['checkpoint_mb']}MB) / 선택 {state['best']['model']} / "
          f"dev Accuracy {state['best']['accuracy']:.4f} / 전체 {state['train_minutes']}분")


if __name__ == "__main__":
    main()
