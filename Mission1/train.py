"""Reference training script for the folder-based dataset.

Example:
python train.py --train_wav_dir Training/1.원천데이터/TS_서울_구급 \
  --train_json_dir Training/2.라벨링데이터/TL_서울_구급 \
  --valid_wav_dir Validation/1.원천데이터/VS_서울_구급 \
  --valid_json_dir Validation/2.라벨링데이터/VL_서울_구급 \
  --output_dir Mission1_new_model
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

SAMPLE_RATE = 16000
MAX_LENGTH = SAMPLE_RATE * 8
LABELS = {"M": 0, "F": 1}


def build_wavs(directory):
    return {p.name.split("_")[0]: p for p in Path(directory).glob("*.wav")}


def read_samples(json_dir, wavs, limit=None):
    samples = []
    for path in sorted(Path(json_dir).glob("*.json")):
        with path.open("r", encoding="utf-8-sig") as f:
            item = json.load(f)
        sample_id = str(item.get("_id", ""))
        gender = str(item.get("gender", "")).upper()
        if sample_id not in wavs or gender not in LABELS:
            continue
        segments = []
        for u in item.get("utterances", []):
            if str(u.get("speaker")) != "1":
                continue
            start, end = u.get("startAt"), u.get("endAt")
            if start is not None and end is not None and float(end) > float(start):
                segments.append((float(start), float(end)))
        if segments:
            samples.append((wavs[sample_id], segments, LABELS[gender]))
        if limit and len(samples) >= limit:
            break
    return samples


def extract(path, segments):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    pieces = []
    for start_ms, end_ms in segments:
        a = max(0, int(start_ms * sr / 1000))
        b = min(len(audio), int(end_ms * sr / 1000))
        if b > a:
            pieces.append(audio[a:b])
    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = audio[:MAX_LENGTH]
    if len(audio) < MAX_LENGTH:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))
    return audio.astype("float32")


class AudioDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, segments, label = self.samples[i]
        audio = extract(path, segments)
        values = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.squeeze(0)
        return values, torch.tensor(label, dtype=torch.long)


def collate(batch):
    values, labels = zip(*batch)
    return {"input_values": torch.stack(values), "labels": torch.stack(labels)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_wav_dir", required=True)
    p.add_argument("--train_json_dir", required=True)
    p.add_argument("--valid_wav_dir")
    p.add_argument("--valid_json_dir")
    p.add_argument("--output_dir", default="Mission1_model")
    p.add_argument("--base_model", default="facebook/wav2vec2-base")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_train", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(args.base_model)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.base_model, num_labels=2).to(device)
    model.freeze_feature_encoder()

    train = read_samples(args.train_json_dir, build_wavs(args.train_wav_dir), args.max_train or None)
    loader = DataLoader(AudioDataset(train, processor), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    print(f"Device: {device}; training samples: {len(train)}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}, Step {step}, Loss {loss.item():.4f}")
        print(f"Epoch {epoch + 1} average loss: {total_loss / len(loader):.4f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved model: {args.output_dir}")


if __name__ == "__main__":
    main()
