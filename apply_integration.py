#!/usr/bin/env python3
"""Install ECG-Arrhythmia support into a kkianamm/medtsllm2 checkout.

Usage:
    python apply_integration.py /path/to/medtsllm2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"[already applied] {label}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Cannot safely apply '{label}' to {path}. "
            f"Expected one exact anchor, found {count}."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[patched] {label}")


def validate_repo(repo: Path) -> None:
    required = [
        repo / "models" / "medtsllm.py",
        repo / "datasets" / "__init__.py",
        repo / "tasks" / "__init__.py",
        repo / "requirements.txt",
        repo / "train.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The target is not a medtsllm2 checkout. Missing:\n  "
            + "\n  ".join(missing)
        )


def copy_files(bundle: Path, repo: Path) -> None:
    mapping = {
        bundle / "files" / "datasets" / "ecg_arrhythmia.py":
            repo / "datasets" / "ecg_arrhythmia.py",
        bundle / "files" / "tasks" / "classification.py":
            repo / "tasks" / "classification.py",
        bundle / "files" / "configs" / "datasets" / "ecg_arrhythmia_decoder.toml":
            repo / "configs" / "datasets" / "ecg_arrhythmia_decoder.toml",
        bundle / "files" / "scripts" / "prepare_ecg_arrhythmia.py":
            repo / "scripts" / "prepare_ecg_arrhythmia.py",
        bundle / "files" / "scripts" / "check_ecg_arrhythmia.py":
            repo / "scripts" / "check_ecg_arrhythmia.py",
    }
    for source, destination in mapping.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[copied] {destination.relative_to(repo)}")


def patch_model(repo: Path) -> None:
    path = repo / "models" / "medtsllm.py"

    replace_once(
        path,
        '    supported_tasks = ["forecasting", "reconstruction", "anomaly_detection", "semantic_segmentation", "segmentation", "pretraining"]',
        '    supported_tasks = ["forecasting", "reconstruction", "anomaly_detection", "semantic_segmentation", "segmentation", "classification", "pretraining"]',
        "register classification in MedTsLLM.supported_tasks",
    )

    old = '''        elif self.task == "segmentation":
            self.n_outputs_per_step = 1
            assert self.config.tasks.segmentation.mode in ["boundary-prediction", "steps-to-boundary"]
        else:
            raise ValueError(f"Task {self.task} is not supported.")
        self.n_outputs = self.n_outputs_per_step * self.pred_len
'''
    new = '''        elif self.task == "segmentation":
            self.n_outputs_per_step = 1
            assert self.config.tasks.segmentation.mode in ["boundary-prediction", "steps-to-boundary"]
        elif self.task == "classification":
            # Sequence-level prediction: one label for the whole window.
            self.n_outputs_per_step = self.n_classes
        else:
            raise ValueError(f"Task {self.task} is not supported.")
        self.n_outputs = self.n_outputs_per_step * self.pred_len
        if self.task == "classification":
            # Override: a single set of K logits per sequence (not per step).
            self.n_outputs = self.n_classes
'''
    replace_once(path, old, new, "set sequence-level classification outputs")

    old = '''            elif self.task == "segmentation":
                if self.config.tasks.segmentation.mode == "boundary-prediction":
                    pred = F.sigmoid(pred)

        return pred
'''
    new = '''            elif self.task == "segmentation":
                if self.config.tasks.segmentation.mode == "boundary-prediction":
                    pred = F.sigmoid(pred)
            elif self.task == "classification":
                pred = F.softmax(pred, dim=-1)

        return pred
'''
    replace_once(path, old, new, "apply softmax only in classification evaluation")

    old = '''        dec_out = self.output_projection(dec_out)       # [bs, pred_len * n_features]
        if self.covariate_mode == "independent":
'''
    new = '''        dec_out = self.output_projection(dec_out)       # [bs, pred_len * n_features]  (classification: [bs, n_classes])

        if self.task == "classification":
            if self.covariate_mode in ["independent", "merge-end"]:
                raise NotImplementedError(
                    "Classification currently supports the concat / interleave / weighted-average / add "
                    "covariate modes (those that preserve the batch dimension)."
                )
            return dec_out  # [bs, n_classes]

        if self.covariate_mode == "independent":
'''
    replace_once(path, old, new, "return flattened decoder logits for classification")

    old = '''        elif self.task == "segmentation":
            self.task_description = f"Identify the change points in the past {self.seq_len} steps of data to segment the sequence."
        else:
            raise ValueError(f"Task {self.task} is not supported.")
'''
    new = '''        elif self.task == "segmentation":
            self.task_description = f"Identify the change points in the past {self.seq_len} steps of data to segment the sequence."
        elif self.task == "classification":
            self.task_description = f"Classify the entire sequence of {self.seq_len} steps into one of the diagnostic classes using the following information."
        else:
            raise ValueError(f"Task {self.task} is not supported.")
'''
    replace_once(path, old, new, "add classification task description")


def patch_task_registry(repo: Path) -> None:
    path = repo / "tasks" / "__init__.py"
    replace_once(
        path,
        "from .semantic_segmentation import SemanticSegmentationTask\n",
        "from .semantic_segmentation import SemanticSegmentationTask\nfrom .classification import ClassificationTask\n",
        "import ClassificationTask",
    )
    replace_once(
        path,
        '    "semantic_segmentation": SemanticSegmentationTask,\n',
        '    "semantic_segmentation": SemanticSegmentationTask,\n    "classification": ClassificationTask,\n',
        "register ClassificationTask",
    )


def patch_dataset_registry(repo: Path) -> None:
    path = repo / "datasets" / "__init__.py"
    replace_once(
        path,
        "from .dreams import dreams_datasets\n",
        "from .dreams import dreams_datasets\nfrom .ecg_arrhythmia import ecg_arrhythmia_datasets\n",
        "import ECG-Arrhythmia dataset",
    )
    replace_once(
        path,
        '    "dreams": dreams_datasets,\n',
        '    "dreams": dreams_datasets,\n    "ECG-ARRHYTHMIA": ecg_arrhythmia_datasets,\n',
        "register ECG-Arrhythmia dataset",
    )


def patch_requirements(repo: Path) -> None:
    path = repo / "requirements.txt"
    text = path.read_text(encoding="utf-8")
    if any(line.strip().lower().startswith("wfdb") for line in text.splitlines()):
        print("[already present] wfdb requirement")
        return
    separator = "" if text.endswith("\n") else "\n"
    path.write_text(text + separator + "wfdb >= 4.1.0\n", encoding="utf-8")
    print("[patched] requirements.txt: added wfdb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path, help="Path to the medtsllm2 checkout")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    bundle = Path(__file__).resolve().parent
    validate_repo(repo)
    copy_files(bundle, repo)
    patch_model(repo)
    patch_task_registry(repo)
    patch_dataset_registry(repo)
    patch_requirements(repo)

    print("\nIntegration installed successfully.")
    print("Edit configs/datasets/ecg_arrhythmia_decoder.toml and set the dataset root.")


if __name__ == "__main__":
    main()
