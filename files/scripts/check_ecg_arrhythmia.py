#!/usr/bin/env python3
"""Validate prepared ECG-Arrhythmia splits without constructing the LLM."""
from __future__ import annotations

import argparse
from collections import Counter

import toml
import torch

from datasets import get_dataset
from utils import dict_to_object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/datasets/ecg_arrhythmia_decoder.toml",
    )
    args = parser.parse_args()

    config = dict_to_object(toml.load(args.config))
    for split in ["train", "val", "test"]:
        dataset = get_dataset(config, split)
        counts_by_index = Counter(dataset.record_labels.tolist())
        counts = {
            dataset.class_names[idx]: counts_by_index.get(idx, 0)
            for idx in range(dataset.n_classes)
        }
        sample = dataset[0]

        print(
            f"{split:>5}: samples={len(dataset):6d}, "
            f"shape={tuple(dataset.records.shape)}, counts={counts}"
        )
        print(
            f"       x={tuple(sample['x_enc'].shape)}, "
            f"label={int(sample['labels'])}, "
            f"description={sample['descriptions']!r}"
        )

        assert sample["x_enc"].shape == (
            config.history_len,
            dataset.n_features,
        )
        assert sample["labels"].dtype == torch.long

    print("ECG-Arrhythmia dataset check passed.")


if __name__ == "__main__":
    main()
