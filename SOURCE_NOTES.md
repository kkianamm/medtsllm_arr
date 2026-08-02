# Source correspondence

Constructed against the `main` branch of `kkianamm/medtsllm2`, specifically:

- `models/medtsllm.py`
- `medtsllm_classification.patch`
- `tasks/classification.py`
- `datasets/ptbxl.py`
- `configs/datasets/ptbxl_decoder.toml`

Target data:

- PhysioNet `ecg-arrhythmia`, version `1.0.0`
- 12 leads, 500 Hz, 5000 samples per 10-second record
- SNOMED-CT diagnoses in WFDB header `#Dx` comments

The four-class mapping is `SB`, `AFIB`, `GSVT`, and `SR`; sinus tachycardia is
included in `GSVT` together with SVT, AT, AVNRT, AVRT, and SAAWR/WAVN.
