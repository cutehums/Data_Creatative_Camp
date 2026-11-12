"""
prepare_data.py — 대회 제공 zip 에서 학습·평가 통화의 wav·json 을 꺼낸다 (학습 준비 전용)

    python prepare_data.py --zip_root {zip 파일 또는 zip 이 있는 폴더} --out_dir {꺼낼 폴더} --train_calls 29200 --val_calls 3640

zip 안의 구조 (대학부_전체데이터_Training_Validation.zip)
    대학부 데이터/Training/1.원천데이터/TS_서울_구급/*.wav      대학부 데이터/Training/2.라벨링데이터/TL_서울_구급/*.json
    대학부 데이터/Validation/1.원천데이터/VS_서울_구급/*.wav    대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급/*.json

- Training: 경로에 '서울'이 있는 파일만 사용 (Training 은 서울 데이터만)
- Validation: 평가 전용이며 학습에는 쓰지 않는다
- 라벨 json 과 이름이 같은 wav 가 있는 통화만 대상. 요청한 수가 전체보다 적으면 무작위(seed 고정)로 고른다. 라벨 내용은 읽지 않는다
- 이미 꺼낸 파일은 건너뛴다
결과: {out_dir}/train/wav, train/json, validation/wav, validation/json, selection.json
"""
import argparse
import json
import os
import time
import unicodedata
import zipfile

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def nfc(text):
    """한글 이름이 자모가 분리된 형태(NFD)로 올 수 있어, 비교·출력 전에 NFC 로 맞춘다."""
    return unicodedata.normalize("NFC", text)


def entry_name(info):
    """zip 항목의 경로. Windows 에서 묶은 zip 은 한글 이름이 cp949 로 들어 있어 복원한다."""
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("cp949")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return nfc(name.replace("\\", "/"))


def split_of(folders):
    """경로의 폴더 이름으로 Training / Validation 을 구분한다."""
    for folder in folders:
        if folder == "Training" or folder.startswith(("TS_", "TL_")):
            return "train"
        if folder == "Validation" or folder.startswith(("VS_", "VL_")):
            return "validation"
    return None


def find_zips(zip_root):
    """zip 파일 하나를 직접 주거나, 폴더를 주면 그 아래의 zip 을 모두 찾는다."""
    if os.path.isfile(zip_root):
        return [zip_root]
    found = []
    for folder, subfolders, files in os.walk(zip_root):
        subfolders[:] = [d for d in subfolders if not d.startswith(".")]
        found += [os.path.join(folder, name) for name in files if name.lower().endswith(".zip")]
    return sorted(found)


def extract(archive, info, target):
    """항목 하나를 target 으로 꺼낸다. 같은 크기로 이미 있으면 건너뛴다."""
    if os.path.exists(target) and os.path.getsize(target) == info.file_size:
        return False
    with archive.open(info) as source, open(target + ".tmp", "wb") as out:
        while True:
            block = source.read(1 << 20)
            if not block:
                break
            out.write(block)
    os.replace(target + ".tmp", target)          # 중간에 끊겨도 반쯤 쓴 파일이 남지 않게
    return True


def prepare(zip_root, out_dir, train_calls, val_calls, seed=42):
    archives = [zipfile.ZipFile(path) for path in find_zips(zip_root)]
    items = {split: {".wav": {}, ".json": {}} for split in ("train", "validation")}
    not_seoul = 0
    for number, archive in enumerate(archives):
        for info in archive.infolist():
            name = entry_name(info)
            *folders, base = name.split("/")
            extension = os.path.splitext(base)[1].lower()
            split = split_of(folders)
            if split is None or extension not in (".wav", ".json") or base.startswith(".") or "__MACOSX" in folders:
                continue
            if split == "train" and "서울" not in name:
                not_seoul += 1
                continue
            items[split][extension].setdefault(base[:-len(extension)], (number, info))
    if not_seoul:
        log(f"[train] 서울이 아닌 파일 {not_seoul:,}개는 쓰지 않음")

    summary = {"seed": seed, "zips": [nfc(os.path.basename(a.filename)) for a in archives]}
    for number, (split, wanted) in enumerate((("train", train_calls), ("validation", val_calls))):
        wavs, labels = items[split][".wav"], items[split][".json"]
        names = sorted(set(wavs) & set(labels))                      # 같은 이름 1:1
        log(f"[{split}] 라벨 {len(labels):,} / wav {len(wavs):,} / 같은 이름 {len(names):,}")
        if not names:
            raise FileNotFoundError(f"{zip_root} 에서 {split} 의 wav·json 짝을 찾지 못했습니다")
        order = np.random.default_rng([seed, number]).permutation(len(names))
        chosen = sorted(names[i] for i in order[:wanted])
        log(f"[{split}] 통화 {len(names):,}개 중 {len(chosen):,}개 선택 (seed {seed})")
        wav_dir, json_dir = os.path.join(out_dir, split, "wav"), os.path.join(out_dir, split, "json")
        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        targets = [(wavs[c], os.path.join(wav_dir, c + ".wav")) for c in chosen] \
            + [(labels[c], os.path.join(json_dir, c + ".json")) for c in chosen]
        targets.sort(key=lambda t: (t[0][0], t[0][1].header_offset))  # zip 안에 저장된 순서대로 읽는다
        started, written = time.time(), 0
        for k, ((archive_number, info), target) in enumerate(targets):
            written += extract(archives[archive_number], info, target)
            if (k + 1) % 5000 == 0 or k + 1 == len(targets):
                elapsed = time.time() - started
                log(f"  [{split}] {k + 1:,}/{len(targets):,} 파일 ({elapsed:.0f}s, 남은 예상 {elapsed / (k + 1) * (len(targets) - k - 1):.0f}s)")
        summary[split] = {"available_calls": len(names), "selected_calls": len(chosen), "newly_extracted": written,
                          "calls": chosen}
    overlap = set(summary["train"]["calls"]) & set(summary["validation"]["calls"])
    log(f"Training 과 Validation 에 함께 있는 통화: {len(overlap):,}개")
    for archive in archives:
        archive.close()
    with open(os.path.join(out_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_root", required=True, help="대회 제공 zip 파일, 또는 zip 이 있는 폴더 (하위 폴더 포함)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_calls", type=int, default=29200, help="Training 에서 꺼낼 통화 수 (학습 + 내부 dev)")
    ap.add_argument("--val_calls", type=int, default=3640, help="Validation 에서 꺼낼 통화 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare(args.zip_root, args.out_dir, args.train_calls, args.val_calls, args.seed)
