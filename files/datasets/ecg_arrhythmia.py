from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .base import BaseDataset


RHYTHM_CLASSES = ["SB", "AFIB", "GSVT", "SR"]


class ECGArrhythmiaClassificationDataset(BaseDataset):
    """Four-class sequence classification for the Chapman/Shaoxing/Ningbo ECG set."""

    supported_tasks = ["classification"]
    description = (
        "The PhysioNet ECG-Arrhythmia database contains 10-second, 12-lead "
        "resting ECG recordings sampled at 500 Hz. Its rhythm diagnoses are "
        "grouped into sinus bradycardia (SB), atrial fibrillation/flutter "
        "(AFIB), general supraventricular tachycardia (GSVT), and sinus "
        "rhythm/irregularity (SR)."
    )
    task_description = (
        "Classify the complete 12-lead ECG into one of four rhythm groups: "
        "sinus bradycardia (SB), atrial fibrillation or flutter (AFIB), "
        "general supraventricular tachycardia (GSVT), or sinus rhythm (SR)."
    )

    n_classes = 4
    class_names = RHYTHM_CLASSES

    def load_data(self):
        dataset_root = Path(self.dataset_config.root).expanduser().resolve()
        processed_name = self.dataset_config.get(
            "processed_dir", f"processed_4class_{self.history_len}"
        )
        processed = dataset_root / processed_name

        x_path = processed / f"{self.split}_x.npy"
        y_path = processed / f"{self.split}_y.npy"
        metadata_path = processed / f"{self.split}_meta.csv"
        stats_path = processed / "normalization_stats.npz"

        required = [x_path, y_path, metadata_path, stats_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Prepared ECG-Arrhythmia files are missing:\n  "
                + "\n  ".join(missing)
                + "\nRun scripts/prepare_ecg_arrhythmia.py first."
            )

        self.records = np.load(x_path, mmap_mode="r")
        self.record_labels = torch.as_tensor(
            np.load(y_path), dtype=torch.long
        )
        self.metadata = pd.read_csv(metadata_path)

        stats = np.load(stats_path)
        self.norm_mean = torch.as_tensor(
            stats["mean"], dtype=torch.float32
        ).view(1, -1)
        self.norm_std = torch.as_tensor(
            stats["std"], dtype=torch.float32
        ).view(1, -1)

        if self.records.ndim != 3:
            raise ValueError(
                f"Expected prepared array [N,T,F], got {self.records.shape}"
            )
        if self.records.shape[1] != self.history_len:
            raise ValueError(
                f"Prepared T={self.records.shape[1]} but history_len="
                f"{self.history_len}. Re-run preparation with matching length."
            )
        if self.records.shape[2] != 12:
            raise ValueError(f"Expected 12 ECG leads, got {self.records.shape[2]}")
        if len(self.records) != len(self.record_labels):
            raise ValueError("Signal and label counts do not match.")
        if len(self.records) != len(self.metadata):
            raise ValueError("Signal and metadata counts do not match.")
        valid = (self.record_labels >= 0) & (
            self.record_labels < self.n_classes
        )
        if not torch.all(valid):
            raise ValueError("Prepared labels must be integers in [0, 3].")

    def get_data(self, split=None):
        raise RuntimeError(
            "This dataset uses prepared memory-mapped arrays; call the "
            "preparation script rather than BaseDataset.get_data()."
        )

    def __len__(self):
        return int(self.records.shape[0])

    def __getitem__(self, idx):
        # Copy one sample out of the read-only memory map.
        record = np.array(self.records[idx], dtype=np.float32, copy=True)
        x = torch.from_numpy(record)
        if self.config.data.normalize:
            x = (x - self.norm_mean) / self.norm_std

        row = self.metadata.iloc[idx]
        return {
            "x_enc": x,
            "labels": self.record_labels[idx],
            "descriptions": str(row.get("description", "")),
        }

    def inverse_index(self, idx):
        return idx

    @property
    def n_points(self):
        return int(self.records.shape[0] * self.records.shape[1])

    @property
    def n_features(self):
        return int(self.records.shape[2])

    @property
    def class_weights(self):
        counts = torch.bincount(
            self.record_labels, minlength=self.n_classes
        ).float()
        return counts.sum() / (counts.clamp(min=1) * self.n_classes)


ecg_arrhythmia_datasets = {
    "classification": ECGArrhythmiaClassificationDataset,
}
