import argparse
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

SAMPLE_RATE = 16_000
MAX_LENGTH = SAMPLE_RATE * 8


def wav_index(wav_dir):
    return {p.name.split("_")[0]: p for p in Path(wav_dir).rglob("*.wav")}


def read_samples(json_dir, wavs, caller_speaker="1", limit=None):
    rows = []
    for path in sorted(Path(json_dir).rglob("*.json")):
        with path.open("r", encoding="utf-8-sig") as f:
            item = json.load(f)
        sample_id = str(item.get("_id", ""))
        gender = str(item.get("gender", "")).upper()
        if sample_id not in wavs or gender not in {"M", "F"}:
            continue
        segments = []
        for u in item.get("utterances", []):
            if str(u.get("speaker")) != caller_speaker:
                continue
            start, end = u.get("startAt"), u.get("endAt")
            if start is not None and end is not None and float(end) > float(start):
                segments.append((float(start), float(end)))
        if segments:
            rows.append((sample_id, wavs[sample_id], segments, gender))
        if limit and len(rows) >= limit:
            break
    return rows


def extract_audio(wav_path, segments):
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    pieces = []
    for start_ms, end_ms in segments:
        a = max(0, int(start_ms * sr / 1000))
        b = min(len(audio), int(end_ms * sr / 1000))
        if b > a:
            pieces.append(audio[a:b])
    if not pieces:
        return None
    audio = np.concatenate(pieces)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = audio[:MAX_LENGTH]
    if len(audio) < MAX_LENGTH:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))
    return audio.astype("float32")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", default=None, help="압축 해제된 WAV 폴더")
    p.add_argument("--json_dir", default=None, help="압축 해제된 JSON 폴더")
    p.add_argument("--audio_zip", default=None, help="WAV ZIP 파일")
    p.add_argument("--json_zip", default=None, help="JSON ZIP 파일")
    p.add_argument("--model_dir", default="model")
    p.add_argument("--output", default="mission1_result.csv")
    p.add_argument("--caller_speaker", default="1")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    folder_mode = args.wav_dir and args.json_dir
    zip_mode = args.audio_zip and args.json_zip
    if not folder_mode and not zip_mode:
        p.error("--wav_dir/--json_dir 또는 --audio_zip/--json_zip 한 쌍을 지정하세요.")
    if folder_mode and zip_mode:
        p.error("폴더 방식과 ZIP 방식을 동시에 지정할 수 없습니다.")

    temp_root = None
    if zip_mode:
        temp_root = tempfile.mkdtemp(prefix="mission1_inference_")
        wav_root = Path(temp_root) / "wav"
        json_root = Path(temp_root) / "json"
        wav_root.mkdir()
        json_root.mkdir()
        print("ZIP 압축 해제 중...")
        with zipfile.ZipFile(args.audio_zip) as z:
            z.extractall(wav_root)
        with zipfile.ZipFile(args.json_zip) as z:
            z.extractall(json_root)
        wav_dir = wav_root
        json_dir = json_root
    else:
        wav_dir = Path(args.wav_dir)
        json_dir = Path(args.json_dir)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = Wav2Vec2Processor.from_pretrained(args.model_dir, local_files_only=True)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device)
        model.eval()

        total = sum(x.numel() for x in model.parameters())
        active = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print(f"Device: {device}")
        print(f"Total parameters: {total:,}")
        print(f"Active parameters: {active:,}")

        wavs = wav_index(wav_dir)
        rows = read_samples(json_dir, wavs, args.caller_speaker, args.max_samples)
        print(f"Samples: {len(rows)}")
        predictions = []
        start_time = time.perf_counter()
        with torch.no_grad():
            for i, (sample_id, wav_path, segments, true_gender) in enumerate(rows, 1):
                audio = extract_audio(wav_path, segments)
                if audio is None:
                    continue
                inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                pred = int(model(**inputs).logits.argmax(dim=1).item())
                predictions.append({"id": sample_id, "gender": "M" if pred == 0 else "F", "true_gender": true_gender})
                if i % 100 == 0:
                    print(f"{i}개 추론 완료")

        elapsed = time.perf_counter() - start_time
        result = pd.DataFrame(predictions)
        result[["id", "gender"]].to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"결과 저장 완료: {args.output}")
        print(f"추론 시간: {elapsed:.2f}초")
        if len(result):
            print(f"샘플당 평균 시간: {elapsed / len(result):.4f}초")
            print(f"Accuracy: {(result['gender'] == result['true_gender']).mean():.6f}")
    finally:
        if temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
