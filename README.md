# AASIST Fine-Tuning for Speech Deepfake Detection

A reproducible Google Colab and Jupyter pipeline for fine-tuning the official AASIST audio anti-spoofing model on locally available ASVspoof 2021 Deepfake audio.

The notebook uses the upstream model directly from the official [clovaai/aasist](https://github.com/clovaai/aasist) repository, pinned to revision `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`.

## Repository contents

```text
.
├── notebooks/
│   └── aasist_deepfake_finetuning.ipynb
├── scripts/
│   └── validate_notebook.py
├── .github/workflows/
│   └── notebook-validation.yml
├── .gitignore
├── README.md
└── requirements.txt
```

## Corrections made during migration

The source notebook contained a working experiment history, but its final results were not valid evidence of classifier quality:

- it parsed 611,829 protocol rows while most sampled audio files were absent locally;
- failed audio reads could become zero waveforms instead of being excluded;
- it repeatedly rewrote `models/AASIST.py` to accept an incompatible custom configuration;
- it treated the model's 640-dimensional hidden representation as logits and added an adapter, even though AASIST already returns `embedding, logits`;
- its class mapping was reversed relative to the official pretrained checkpoint;
- the reported test accuracy was dominated by class imbalance while the confusion matrix showed zero correctly recognized bona-fide samples;
- it included a manually constructed illustrative confusion matrix.

This repository corrects those issues:

- indexes actual audio files before splitting;
- preserves the official AASIST configuration and two-class head;
- pins the upstream source revision and performs no source patching;
- preserves the official label order: spoof `0`, bona fide `1`;
- creates stratified, disjoint train/validation/test partitions;
- balances training samples;
- uses gradient accumulation and mixed precision;
- selects checkpoints using validation macro F1;
- reports measured balanced accuracy, macro F1, spoof F1, AUROC, EER, classification results, and confusion matrix.

## Data

The audio and protocol files are not committed. The default Colab paths reproduce the source experiment:

| Variable | Default purpose |
| --- | --- |
| `AASIST_DATA_ROOT` | Common `PFEmaster` data root |
| `ASVSPOOF_DF_AUDIO_ROOT` | ASVspoof 2021 DF audio directory |
| `ASVSPOOF_DF_PROTOCOL` | Released DF metadata/protocol file |
| `AASIST_OUTPUT_ROOT` | Checkpoints, figures, predictions, and metrics |
| `AASIST_SOURCE_ROOT` | Pinned upstream AASIST checkout |

ASVspoof 2021 data and metadata are available from the [official challenge site](https://www.asvspoof.org/index2021.html).

## Run in Colab

Open `notebooks/aasist_deepfake_finetuning.ipynb`, select a GPU runtime, and execute the cells in order. The notebook mounts Google Drive only when Colab is detected.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/aasist_deepfake_finetuning.ipynb
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Evaluation limits

ASVspoof 2021 DF is an evaluation corpus. Fine-tuning on its released labels is a local research adaptation experiment, not an official challenge submission. Official comparison requires the designated challenge protocol and metrics, along with controlled cross-condition or cross-corpus testing.

No performance metric is committed as a claimed result. Run the notebook against the available data to generate current measurements.

## Attribution

AASIST is the work of Jung et al. and is distributed by NAVER under the MIT License in its official repository. This project does not vendor or modify the upstream implementation; it checks out a pinned revision at runtime.

## Validation

```bash
python scripts/validate_notebook.py
```

The checker validates notebook structure, rejects committed outputs and execution counts, parses ordinary Python cells, and checks the migration safeguards.


## Audit and training status

The current notebook contains no committed training outputs, fabricated metric arrays, or fixed accuracy claims. Metrics and confusion matrices are computed from model predictions and labels at runtime. Regression-test expectations use small fixtures to verify the calculations; they are not model performance results. Seeds, model dimensions, thresholds, sample limits, and dataset row-count checks are configuration or validation constants, not claimed scores.

No completed training run or fine-tuned checkpoint is verified by this repository. Dataset names below describe the configured training inputs. Running the notebook writes split manifests and prediction files so a future result can be traced to the rows evaluated. Do not publish benchmark scores until a real dataset run has been completed and reviewed.

See [AUDIT.md](AUDIT.md) for findings, fixes, validation, and remaining limitations.

## Configured training datasets

| Stage | Dataset | Evidence / split |
| --- | --- | --- |
| Upstream pretrained AASIST | ASVspoof 2019 Logical Access | Official clovaai/aasist checkpoint and upstream documentation; pinned source revision |
| Configured local fine-tuning | ASVspoof 2021 Deepfake evaluation audio with released CM labels | Matched local files; stratified 80/10/10 train/validation/test |

The repository does not establish that local fine-tuning has completed. These local splits do not implement an official ASVspoof 2021 challenge experiment.

Regression checks: `python -m unittest discover -s tests -v` (install requirements first).
