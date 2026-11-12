"""
model.py — Mission 2 (신고자 / 119대원 발화 분류) 모델과 공용 함수

입력 : 통화 wav 에서 라벨링데이터의 startAt~endAt(ms) 구간만 잘라낸 발화 조각 1개
출력 : 1 = 신고자, 0 = 119대원  (확률 >= 0.5 → 1, threshold 고정)
라벨 json 에서 읽는 값 : startAt, endAt (학습 때만 정답 speaker). text·발화 순서 등 다른 annotation 은 읽지 않음

모델
  발화 구간(16kHz, 최대 6초) → WavLM-base-plus 의 CNN + Transformer 앞 4층
    → 층마다(임베딩 + 4층 = 5개) 패딩을 뺀 평균·표준편차(1536차원)
    → 층별 표준화 → softmax 층 가중합 → LayerNorm → Linear 256 → GELU → Dropout → Linear 1 → sigmoid
  아래쪽 층만 쓰는 이유: 신고자(휴대폰·현장)와 119대원(상황실 헤드셋)을 가르는 채널·음색 정보가 아래 층에 남아 있다.

체크포인트 한 파일에 WavLM 설정·가중치와 분류기가 모두 들어 있어 추론 때 인터넷이 필요 없다.
"""
import glob
import json
import math
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask")

SR = 16000                      # WavLM 입력 샘플레이트 (원본 8kHz 를 변환)
WINDOW = 6 * SR                 # 모델 입력 최대 길이 6초 (더 긴 발화는 가운데 6초)
BUCKET = SR // 4                # 0.25초 단위로 zero padding → 같은 길이끼리 batch
MIN_SAMPLES = 400               # WavLM CNN 이 프레임 1개를 만드는 최소 길이
THRESHOLD = 0.5                 # 확률 >= 0.5 → 신고자(1). 조정하지 않는 고정값
PRETRAINED = "microsoft/wavlm-base-plus"
REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
FORMAT = "mission2-wavlm-layermix-v1"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 1. 발화 목록 (라벨 json ↔ 같은 이름의 wav)
# ============================================================
def _sample_index(value, sr):
    """startAt/endAt(ms) → 샘플 위치. 숫자가 아니면 None."""
    try:
        return int(float(value) * sr / 1000)
    except (TypeError, ValueError):
        return None


def build_manifest(label_dir, audio_dir, with_label):
    """라벨 json 의 utterances 를 발화 1개 = 1행으로 펼친다.

    - wav 매칭: json 과 같은 파일 이름 (다른 annotation 으로 찾지 않음)
    - 구간이 wav 길이를 넘으면 wav 끝에 맞춰 자른다
    - with_label=True (학습): 정답이 0/1 이 아니거나 구간을 읽을 수 없는 발화는 제외
    - with_label=False (추론): 어떤 발화도 빼지 않는다. 읽을 수 없는 발화는 ok=False 로 표시 → 입력 발화마다 CSV 1행
    행: call, wav, wav_name, start, end(입력 값 그대로), a, b(wav 안의 샘플 구간), sr, ok, [speaker]
    """
    wavs = {}
    for path in sorted(glob.glob(os.path.join(audio_dir, "**", "*"), recursive=True)):
        if path.lower().endswith(".wav"):
            wavs.setdefault(os.path.splitext(os.path.basename(path))[0], path)
    rows, dropped, clipped, unreadable = [], 0, 0, 0
    for label_path in sorted(glob.glob(os.path.join(label_dir, "**", "*"), recursive=True)):
        if not label_path.lower().endswith(".json"):
            continue
        call = os.path.splitext(os.path.basename(label_path))[0]
        try:
            with open(label_path, encoding="utf-8-sig") as f:
                utterances = json.load(f)["utterances"]
            assert isinstance(utterances, list)
        except Exception:
            unreadable += 1                      # 발화 목록을 읽을 수 없는 json
            continue
        wav, frames, sr = wavs.get(call), 0, SR
        if wav is not None:
            try:
                info = sf.info(wav)
                frames, sr = info.frames, info.samplerate
            except Exception:
                wav = None                       # 열 수 없는 wav
        for u in utterances:
            u = u if isinstance(u, dict) else {}
            a, b = _sample_index(u.get("startAt"), sr), _sample_index(u.get("endAt"), sr)
            ok = wav is not None and a is not None and b is not None
            if ok:
                clipped += b > frames
                a, b = max(a, 0), min(b, frames)
                ok = b > a
            row = dict(call=call, wav=wav, wav_name=os.path.basename(wav) if wav else call + ".wav",
                       start=u.get("startAt"), end=u.get("endAt"), a=a if ok else 0, b=b if ok else 0, sr=sr, ok=ok)
            if with_label:
                if not ok or str(u.get("speaker")) not in ("0", "1"):
                    dropped += 1
                    continue
                row["speaker"] = int(u["speaker"])
            rows.append(row)
    log(f"발화 목록: 통화 {len({r['call'] for r in rows}):,}개 / 발화 {len(rows):,}개"
        + (f" / 제외 {dropped:,}개" if dropped else "") + (f" / wav 끝에 맞춰 자른 발화 {clipped:,}개" if clipped else "")
        + (f" / 읽을 수 없는 json {unreadable:,}개" if unreadable else ""))
    return rows


# ============================================================
# 2. 발화 구간 음성 → 모델 입력
# ============================================================
def resampled_length(row):
    """발화 구간을 16kHz 로 바꿨을 때의 샘플 수."""
    g = math.gcd(row["sr"], SR)
    return -(-(row["b"] - row["a"]) * (SR // g) // (row["sr"] // g))


def padded_length(n):
    """padding 후 길이: 0.25초 배수로 올림, 최대 6초.

    WavLM-base 의 CNN 은 입력 전체 길이로 정규화하므로 padding 양이 결과에 영향을 준다.
    padding 후 길이를 발화 자신의 길이로만 정하면, 어떤 발화와 같은 batch 에 들어가든 결과가 같다.
    """
    return min(WINDOW, -(-max(n, MIN_SAMPLES) // BUCKET) * BUCKET)


def load_segment(row):
    """wav 에서 발화 구간만 읽어 mono·16kHz 로 변환 (통화의 다른 부분은 읽지 않음)."""
    with sf.SoundFile(row["wav"]) as f:
        f.seek(row["a"])
        x = f.read(row["b"] - row["a"], dtype="float32", always_2d=True).mean(axis=1)
    if row["sr"] != SR:
        g = math.gcd(row["sr"], SR)
        x = resample_poly(x, SR // g, row["sr"] // g).astype(np.float32)
    return np.nan_to_num(x)


class Segments(Dataset):
    """발화 1개 → (위치, zero padding 된 음성, 실제 길이). label 은 읽지 않는다.

    6초보다 긴 발화: 추론은 가운데 6초, 학습(crop_seed 지정)은 무작위 6초.
    """
    def __init__(self, rows, do_normalize, crop_seed=None):
        self.rows, self.do_normalize, self.crop_seed = rows, do_normalize, crop_seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = load_segment(self.rows[i])
        if len(x) > WINDOW:
            start = (len(x) - WINDOW) // 2 if self.crop_seed is None else \
                int(np.random.default_rng([self.crop_seed, i]).integers(0, len(x) - WINDOW + 1))
            x = x[start:start + WINDOW]
        if self.do_normalize:
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        padded = np.zeros(padded_length(len(x)), dtype=np.float32)
        padded[:len(x)] = x
        return i, padded, max(len(x), MIN_SAMPLES)


def length_batches(rows, batch_size, rng=None):
    """padding 후 길이가 같은 발화끼리 batch 로 묶는다. rng 를 주면(학습) 순서를 섞는다."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[padded_length(resampled_length(row))].append(i)
    batches = []
    for length in sorted(groups):
        order = groups[length] if rng is None else rng.permutation(groups[length]).tolist()
        batches += [order[k:k + batch_size] for k in range(0, len(order), batch_size)]
    return batches if rng is None else [batches[i] for i in rng.permutation(len(batches))]


def collate(items):
    width = max(len(item[1]) for item in items)
    waveform = np.zeros((len(items), width), dtype=np.float32)
    for k, item in enumerate(items):
        waveform[k, :len(item[1])] = item[1]
    return [item[0] for item in items], torch.from_numpy(waveform), torch.tensor([item[2] for item in items])


# ============================================================
# 3. 모델
# ============================================================
class RoleModel(nn.Module):
    """WavLM 앞 K층의 층별 통계 + 층 가중합 분류기. forward 출력은 logit (sigmoid 전)."""

    def __init__(self, config, do_normalize=False, use_attention_mask=True, head_dim=256, dropout=0.2):
        super().__init__()
        if config.add_adapter or config.do_stable_layer_norm:
            raise ValueError("WavLM-base 계열 구조만 지원합니다")
        config.apply_spec_augment = False                  # 무작위 마스킹·layerdrop 끔: 같은 입력 → 같은 출력
        config.mask_time_prob = config.mask_feature_prob = config.layerdrop = 0.0
        self.wavlm = WavLMModel(config)
        self.do_normalize, self.use_attention_mask = bool(do_normalize), bool(use_attention_mask)
        layers, dim = config.num_hidden_layers + 1, 2 * config.hidden_size
        # 층별 표준화 통계: 학습 데이터에서 한 번 계산해 고정 (학습 파라미터 아님)
        self.register_buffer("stat_mean", torch.zeros(layers, dim))
        self.register_buffer("stat_scale", torch.ones(layers, dim))
        self.layer_logits = nn.Parameter(torch.zeros(layers))             # softmax → 층 가중치
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(nn.Linear(dim, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, 1))
        self.finetuned = False
        self.set_finetune(False)

    @classmethod
    def from_pretrained(cls, source=PRETRAINED, layers=4):
        """사전학습 WavLM 에서 앞 layers 층만 가져온다 (학습할 때만 사용, 인터넷 또는 로컬 폴더)."""
        pin = {} if os.path.isdir(source) else {"revision": REVISION}
        full = WavLMModel.from_pretrained(source, **pin)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(source, **pin)
        config = WavLMConfig.from_dict(full.config.to_dict())
        config.num_hidden_layers = layers
        model = cls(config, processor.do_normalize, processor.return_attention_mask)
        # 앞 층의 출력은 뒤 층과 무관하므로, 뒤 층을 버려도 남은 층의 출력은 그대로다
        keep = {k: v for k, v in full.state_dict().items() if k in model.wavlm.state_dict()}
        model.wavlm.load_state_dict(keep, strict=True)
        return model

    def head_parameters(self):
        return [self.layer_logits, *self.norm.parameters(), *self.classifier.parameters()]

    def tuned_modules(self):
        """미세조정 대상: Transformer 위쪽 2층. 그 아래(CNN 특징 추출기·projection·아래 층)는 항상 동결."""
        return list(self.wavlm.encoder.layers[-2:])

    def set_finetune(self, enabled):
        self.finetune = bool(enabled)
        self.wavlm.requires_grad_(False)
        if self.finetune:
            self.finetuned = True
            for module in self.tuned_modules():
                module.requires_grad_(True)
        return self.train(self.training)

    def train(self, mode=True):
        """동결된 부분은 항상 eval(dropout 끔). 미세조정 중일 때만 미세조정 대상 층이 train 모드."""
        super().train(mode)
        self.wavlm.eval()
        if mode and self.finetune:
            for module in self.tuned_modules():
                module.train()
        return self

    def layer_stats(self, waveform, lengths):
        """[batch, 샘플] → 층별 평균·표준편차 [batch, 층 수, 2·hidden]. 동결이면 gradient 를 만들지 않는다."""
        with torch.set_grad_enabled(self.finetune and torch.is_grad_enabled()):
            mask = torch.arange(waveform.shape[1], device=waveform.device)[None, :] < lengths[:, None]
            with torch.autocast("cuda", dtype=torch.float16, enabled=waveform.is_cuda):     # GPU 에서만 반정밀도
                hidden_states = self.wavlm(waveform, attention_mask=mask.long() if self.use_attention_mask else None,
                                           output_hidden_states=True).hidden_states
            frames = lengths
            for kernel, stride in zip(self.wavlm.config.conv_kernel, self.wavlm.config.conv_stride):
                frames = torch.div(frames - kernel, stride, rounding_mode="floor") + 1          # 샘플 길이 → 프레임 길이
            valid = (torch.arange(hidden_states[0].shape[1], device=waveform.device)[None, :] < frames[:, None])
            weight = valid.unsqueeze(-1).float()
            count = weight.sum(dim=1).clamp_min(1)
            stats = []
            for hidden in hidden_states:
                hidden = hidden.float()                                      # 분산은 float32 로 계산
                mean = (hidden * weight).sum(dim=1) / count
                variance = ((hidden - mean[:, None, :]).square() * weight).sum(dim=1) / count
                stats.append(torch.cat([mean, torch.sqrt(variance + 1e-5)], dim=-1))
            return torch.stack(stats, dim=1)

    def classify(self, stats):
        """층별 통계 → logit. 저장해 둔 통계로 분류기만 학습할 때도 이 함수를 쓴다."""
        z = (stats.float() - self.stat_mean) / self.stat_scale
        pooled = (z * torch.softmax(self.layer_logits, dim=0)[None, :, None]).sum(dim=1)
        return self.classifier(self.norm(pooled)).squeeze(-1)

    def forward(self, waveform, lengths):
        return self.classify(self.layer_stats(waveform, lengths))


def predict(model, rows, batch_size=32, device="cpu", num_workers=0):
    """rows 의 발화마다 신고자 확률. 반환: (확률 배열, 모델 forward 에 쓴 시간(초))."""
    model.eval()
    probabilities, model_seconds = np.zeros(len(rows)), 0.0
    loader = DataLoader(Segments(rows, model.do_normalize), batch_sampler=length_batches(rows, batch_size),
                        num_workers=num_workers, collate_fn=collate)
    with torch.inference_mode():
        for index, waveform, lengths in loader:
            waveform, lengths = waveform.to(device), lengths.to(device)
            if waveform.is_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            p = torch.sigmoid(model(waveform, lengths)).float()
            if waveform.is_cuda:
                torch.cuda.synchronize()
            model_seconds += time.perf_counter() - started
            probabilities[index] = p.cpu().numpy()
    return probabilities, model_seconds


# ============================================================
# 4. 체크포인트 · 파라미터 수
# ============================================================
def save_checkpoint(path, model, meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"format": FORMAT, "threshold": THRESHOLD, "config": model.wavlm.config.to_dict(),
                "do_normalize": model.do_normalize, "use_attention_mask": model.use_attention_mask,
                "finetuned": model.finetuned, "meta": meta or {},
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path + ".tmp")
    os.replace(path + ".tmp", path)             # 쓰다가 끊겨도 반쯤 쓴 파일이 남지 않게


def load_checkpoint(path, device="cpu"):
    """체크포인트만으로 모델 복원 (다운로드 없음). 반환: (모델, meta)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)      # tensor 와 기본 자료형만 읽음
    if ckpt.get("format") != FORMAT or float(ckpt["threshold"]) != THRESHOLD:
        raise ValueError("지원하지 않는 체크포인트입니다")
    model = RoleModel(WavLMConfig.from_dict(ckpt["config"]), ckpt["do_normalize"], ckpt["use_attention_mask"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.finetuned = bool(ckpt["finetuned"])
    return model.to(device).eval(), ckpt["meta"]


def count_parameters(model):
    """Total / Active / 학습한 파라미터 수. dense 모델이라 forward 에 모든 파라미터가 쓰인다 (Active = Total)."""
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head_parameters())
    tuned = sum(p.numel() for m in model.tuned_modules() for p in m.parameters()) if model.finetuned else 0
    return {"total": total, "active": total, "trained": head + tuned, "classifier": head,
            "wavlm_layers": model.wavlm.config.num_hidden_layers, "wavlm_finetuned": model.finetuned}
