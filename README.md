# MedTsLLM2 decoder-only integration for PhysioNet ECG-Arrhythmia

This bundle adapts `kkianamm/medtsllm2` to PhysioNet
`ecg-arrhythmia` version `1.0.0` while retaining the repository's exact
sequence-classification decoder path.

## Decoder retained exactly

The model still performs:

1. Whole-record resampling to 512 time steps.
2. Patch embedding with patch length 32 and stride 16.
3. Time-series-to-LLM reprogramming.
4. Decoder-only Llama-2 hidden-state extraction.
5. Selection of the final patch hidden states.
6. Linear hidden-size reduction to `d_ff=128`.
7. Flattening of all patch representations.
8. One `FlattenHead` linear projection to `K=4` logits.
9. Cross-entropy training and evaluation-only softmax.

No mean-pooling classifier, CNN, MLP replacement, or alternate decoder is
introduced.

## Four rhythm classes

The source headers are multi-diagnosis, while this MedTsLLM decoder is a
single-label softmax classifier. The preparation script extracts these four
published rhythm groups:

| Class | Included source rhythms |
|---|---|
| `SB` | sinus bradycardia |
| `AFIB` | atrial fibrillation, atrial flutter |
| `GSVT` | SVT, sinus tachycardia, AT, AVNRT, AVRT, SAAWR/WAVN |
| `SR` | sinus rhythm, sinus irregularity/arrhythmia |

Thus, `AVNRT` and `SAAWR` are not required as standalone classes. They are
members of `GSVT`, which prevents the previous zero-count failure.

By default, records containing rhythm codes from more than one group are
dropped rather than assigned arbitrarily. `--ambiguous-policy priority` is
available only when a deterministic priority assignment is specifically wanted.

## Install into the repository

```bash
python /path/to/medtsllm2_ecg_arrhythmia_decoder/apply_integration.py \
  /path/to/medtsllm2

cd /path/to/medtsllm2
pip install -r requirements.txt
```

The installer copies the dataset/task/config/scripts, applies the repository's
classification decoder changes, registers the task and dataset, and adds
`wfdb >= 4.1.0`.

## Expected dataset layout

```text
ecg-arrhythmia-1.0.0/
├── ConditionNames_SNOMED-CT.csv
├── RECORDS
└── WFDBRecords/
    ├── 01/
    │   └── 010/
    │       ├── JS00001.hea
    │       └── JS00001.mat
    └── ...
```

Set the absolute root in:

```toml
[datasets.ECG-ARRHYTHMIA]
root = "/absolute/path/to/ecg-arrhythmia-1.0.0"
```

## Prepare data

```bash
python scripts/prepare_ecg_arrhythmia.py \
  --root /absolute/path/to/ecg-arrhythmia-1.0.0 \
  --history-len 512 \
  --seed 0 \
  --overwrite
```

The script scans all `.hea` files, maps SNOMED rhythm codes, creates stratified
80/10/10 splits, reads the paired WFDB `.mat` signals, resamples each full
10-second ECG to `[512, 12]`, and writes memory-mapped NumPy arrays plus
training-set normalization statistics.

## Validate

```bash
python scripts/check_ecg_arrhythmia.py \
  configs/datasets/ecg_arrhythmia_decoder.toml
```

## Train and log every epoch

```bash
mkdir -p outputs/console
RUN_ID="ecg_arrhythmia_decoder_seed0"

python3 -u train.py \
  configs/datasets/ecg_arrhythmia_decoder.toml \
  "$RUN_ID" \
  2>&1 | tee "outputs/console/${RUN_ID}.log"
```

Validation and test accuracy, macro F1, macro precision, and macro recall are
computed and printed after every epoch by `tasks/classification.py`.

## Llama-2 access

The exact config uses `meta-llama/Llama-2-7b-hf`, a gated Hugging Face model.
Accept its license and authenticate on the training machine. Replacing the LLM
would no longer be the exact decoder-only configuration from `medtsllm2`.
