#!/usr/bin/env python3
"""Prepare PhysioNet ecg-arrhythmia v1.0.0 for medtsllm2 classification.

The source headers can contain several SNOMED-CT diagnoses. This script extracts
one rhythm superclass per record and creates the four groups used by the
Chapman rhythm-classification protocol:

    SB
    AFIB = AFIB or atrial flutter
    GSVT = SVT, sinus tachycardia, AT, AVNRT, AVRT, or SAAWR/WAVN
    SR = sinus rhythm or sinus irregularity/arrhythmia

Full 5000-sample, 12-lead ECGs are linearly resampled to history_len (512 by
default), matching the whole-record resampling used by medtsllm2's PTB-XL
classification adapter.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


CANONICAL_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

CLASS_NAMES = ["SB", "AFIB", "GSVT", "SR"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

RHYTHM_CODE_GROUPS = {
    "SB": {
        "426177001",  # sinus bradycardia
    },
    "AFIB": {
        "164889003",  # atrial fibrillation
        "164890007",  # atrial flutter
    },
    "GSVT": {
        "426761007",  # supraventricular tachycardia
        "427084000",  # sinus tachycardia
        "713422000",  # atrial tachycardia
        "233896004",  # AVNRT
        "233897008",  # AVRT
        "195101003",  # SAAWR / wandering atrial pacemaker
    },
    "SR": {
        "426783006",  # sinus rhythm
        "427393009",  # sinus irregularity / sinus arrhythmia
    },
}

# Used only with --ambiguous-policy priority. The default is to drop records
# that genuinely contain rhythm codes from more than one superclass.
GROUP_PRIORITY = ["AFIB", "GSVT", "SB", "SR"]


def locate_dataset_root(root: Path) -> tuple[Path, Path]:
    root = root.expanduser().resolve()
    if (root / "WFDBRecords").is_dir():
        return root, root / "WFDBRecords"
    if root.name == "WFDBRecords" and root.is_dir():
        return root.parent, root
    raise FileNotFoundError(
        f"Could not find WFDBRecords under {root}. Pass either the "
        "ecg-arrhythmia-1.0.0 directory or its WFDBRecords directory."
    )


def comment_value(lines: list[str], key: str) -> str:
    prefix = f"#{key}:"
    for line in lines:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def matched_groups(dx_codes: set[str]) -> list[str]:
    return [
        group
        for group, codes in RHYTHM_CODE_GROUPS.items()
        if dx_codes.intersection(codes)
    ]


def choose_group(
    groups: list[str], ambiguous_policy: str
) -> str | None:
    if len(groups) == 1:
        return groups[0]
    if len(groups) == 0:
        return None
    if ambiguous_policy == "drop":
        return None
    for group in GROUP_PRIORITY:
        if group in groups:
            return group
    return None


def parse_header(path: Path, ambiguous_policy: str) -> tuple[dict | None, str]:
    lines = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8-sig", errors="replace"
        ).splitlines()
        if line.strip()
    ]
    if not lines:
        return None, "invalid_header"

    first = lines[0].split()
    if len(first) < 4:
        return None, "invalid_header"

    try:
        n_signals = int(first[1])
        sampling_rate = float(first[2])
        signal_length = int(first[3])
    except ValueError:
        return None, "invalid_header"

    if n_signals != 12 or len(lines) < 1 + n_signals:
        return None, "not_12_lead"

    leads = [lines[i + 1].split()[-1] for i in range(n_signals)]
    if set(leads) != set(CANONICAL_LEADS):
        return None, "noncanonical_leads"

    dx_raw = comment_value(lines, "Dx")
    dx_codes = {code.strip() for code in dx_raw.split(",") if code.strip()}
    groups = matched_groups(dx_codes)
    group = choose_group(groups, ambiguous_policy)
    if group is None:
        reason = "ambiguous_rhythm" if len(groups) > 1 else "no_target_rhythm"
        return None, reason

    age = comment_value(lines, "Age")
    sex = comment_value(lines, "Sex")
    age_text = age if age and age.lower() != "nan" else "unknown"
    sex_text = sex.lower() if sex else "unknown"
    description = f"Patient information: age {age_text}, sex {sex_text}."

    return {
        "record": str(path.with_suffix("")),
        "record_id": path.stem,
        "label_name": group,
        "label": CLASS_TO_INDEX[group],
        "age": age,
        "sex": sex,
        "dx_codes": ",".join(sorted(dx_codes)),
        "matched_groups": ",".join(groups),
        "description": description,
        "sampling_rate": sampling_rate,
        "signal_length": signal_length,
    }, "ok"


def resample_linear(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.ndim != 2:
        raise ValueError(f"Expected [T,F], got {signal.shape}")
    source_length, n_features = signal.shape
    if source_length == target_length:
        return signal.astype(np.float32, copy=False)

    source_x = np.linspace(0.0, 1.0, source_length, dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    output = np.empty((target_length, n_features), dtype=np.float32)
    for feature in range(n_features):
        output[:, feature] = np.interp(
            target_x, source_x, signal[:, feature]
        ).astype(np.float32)
    return output


def load_record(row: dict, target_length: int) -> np.ndarray:
    # Lazy import keeps repository imports working before wfdb is installed.
    import wfdb

    signal, fields = wfdb.rdsamp(row["record"])
    names = list(fields["sig_name"])
    missing = [lead for lead in CANONICAL_LEADS if lead not in names]
    if missing:
        raise ValueError(f"{row['record_id']} is missing leads: {missing}")

    order = [names.index(lead) for lead in CANONICAL_LEADS]
    signal = np.asarray(signal[:, order], dtype=np.float32)

    if not np.isfinite(signal).all():
        for lead_idx in range(signal.shape[1]):
            lead = signal[:, lead_idx]
            finite = np.isfinite(lead)
            if not finite.any():
                raise ValueError(
                    f"{row['record_id']} lead {CANONICAL_LEADS[lead_idx]} "
                    "contains no finite values."
                )
            lead[~finite] = np.median(lead[finite])

    return resample_linear(signal, target_length)


def make_splits(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    labels = [row["label"] for row in rows]
    train_rows, temporary_rows = train_test_split(
        rows,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    temporary_labels = [row["label"] for row in temporary_rows]
    val_rows, test_rows = train_test_split(
        temporary_rows,
        test_size=0.50,
        random_state=seed,
        stratify=temporary_labels,
    )
    return {"train": train_rows, "val": val_rows, "test": test_rows}


def write_metadata(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "record_id", "record", "label", "label_name", "age", "sex",
        "dx_codes", "matched_groups", "description", "sampling_rate",
        "signal_length",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def process_split(
    split: str,
    rows: list[dict],
    output_dir: Path,
    target_length: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    x_path = output_dir / f"{split}_x.npy"
    y_path = output_dir / f"{split}_y.npy"
    meta_path = output_dir / f"{split}_meta.csv"

    x_memmap = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), target_length, len(CANONICAL_LEADS)),
    )
    np.save(
        y_path,
        np.asarray([row["label"] for row in rows], dtype=np.int64),
    )
    write_metadata(meta_path, rows)

    running_sum = (
        np.zeros(len(CANONICAL_LEADS), dtype=np.float64)
        if split == "train" else None
    )
    running_sum_sq = (
        np.zeros(len(CANONICAL_LEADS), dtype=np.float64)
        if split == "train" else None
    )
    value_count = 0

    for idx, row in enumerate(tqdm(rows, desc=f"Preparing {split}")):
        signal = load_record(row, target_length)
        x_memmap[idx] = signal
        if split == "train":
            signal64 = signal.astype(np.float64)
            running_sum += signal64.sum(axis=0)
            running_sum_sq += np.square(signal64).sum(axis=0)
            value_count += signal.shape[0]

    x_memmap.flush()
    del x_memmap
    return running_sum, running_sum_sq, value_count


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["label_name"] for row in rows)
    return {name: counts.get(name, 0) for name in CLASS_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="ecg-arrhythmia-1.0.0 or its WFDBRecords directory",
    )
    parser.add_argument("--history-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Output directory name inside the dataset root",
    )
    parser.add_argument(
        "--ambiguous-policy",
        choices=["drop", "priority"],
        default="drop",
        help="How to handle records with rhythm codes from multiple groups",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.history_len <= 0:
        raise ValueError("--history-len must be positive")

    dataset_root, wfdb_root = locate_dataset_root(args.root)
    processed_name = (
        args.processed_dir or f"processed_4class_{args.history_len}"
    )
    output_dir = dataset_root / processed_name
    temporary_dir = dataset_root / f".{processed_name}.tmp"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    headers = sorted(wfdb_root.rglob("*.hea"))
    if not headers:
        raise FileNotFoundError(f"No .hea files found under {wfdb_root}")

    rows = []
    scan_counts = Counter()
    for header in tqdm(headers, desc="Scanning headers"):
        row, reason = parse_header(header, args.ambiguous_policy)
        scan_counts[reason] += 1
        if row is not None:
            rows.append(row)

    overall_counts = label_counts(rows)
    insufficient = {
        name: count for name, count in overall_counts.items() if count < 10
    }
    if insufficient:
        raise RuntimeError(
            "Insufficient records after four-class grouping: "
            f"{insufficient}. Full counts: {overall_counts}"
        )

    splits = make_splits(rows, args.seed)
    split_counts = {
        split: label_counts(items) for split, items in splits.items()
    }

    print(f"Headers scanned: {len(headers):,}")
    print(f"Eligible rhythm records: {len(rows):,}")
    print("Scan outcomes:", dict(scan_counts))
    print("Overall counts:", overall_counts)
    print("Split counts:", json.dumps(split_counts, indent=2))

    try:
        train_sum, train_sum_sq, train_count = process_split(
            "train", splits["train"], temporary_dir, args.history_len
        )
        process_split("val", splits["val"], temporary_dir, args.history_len)
        process_split("test", splits["test"], temporary_dir, args.history_len)

        mean = train_sum / train_count
        variance = train_sum_sq / train_count - np.square(mean)
        std = np.sqrt(np.maximum(variance, 1e-12))
        std = np.maximum(std, 1e-6)

        np.savez(
            temporary_dir / "normalization_stats.npz",
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            leads=np.asarray(CANONICAL_LEADS),
        )
        (temporary_dir / "label_names.json").write_text(
            json.dumps(
                {
                    "class_names": CLASS_NAMES,
                    "class_to_index": CLASS_TO_INDEX,
                    "rhythm_code_groups": {
                        key: sorted(value)
                        for key, value in RHYTHM_CODE_GROUPS.items()
                    },
                    "split_counts": split_counts,
                    "scan_counts": dict(scan_counts),
                    "seed": args.seed,
                    "history_len": args.history_len,
                    "ambiguous_policy": args.ambiguous_policy,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_dir.rename(output_dir)
    except Exception:
        print(f"Preparation failed; partial files remain in {temporary_dir}")
        raise

    print(f"Prepared files written to: {output_dir}")


if __name__ == "__main__":
    main()
