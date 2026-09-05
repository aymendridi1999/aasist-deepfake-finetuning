# Notebook correctness audit

Audited 2026-09-05.

## Metric integrity and emoji review

No fabricated metric constants, manually populated confusion matrices, committed outputs, or emoji characters were present in the audited current notebook source. The original illustrative results had already been removed during migration. This audit fixes remaining code paths that could invalidate otherwise computed results. Legitimate hyperparameters and small regression-test expected values are retained.

## Fixes

- Supports official 2021 DF CM metadata as well as explicit compact protocols; the former parser only accepted the custom three-column layout. Rejects unknown/conflicting labels, empty protocols, ASV target/nontarget trials, and ambiguous audio IDs.
- Removes the remaining empty-waveform-to-silence and non-finite-to-zero substitution. Invalid audio now raises an error instead of becoming a training/evaluation example.
- Uses SoundFile decoding plus librosa resampling, removing the unused torchaudio dependency and its decoder-version coupling.
- Fixes the final partial gradient-accumulation window: its loss is divided by the number of batches actually in that window. Detects zero-batch training loaders.
- Saves train/validation/test manifests. Invalid metric inputs fail; single-class AUROC/EER are unavailable (JSON null).
- Documents the nearest-ROC-crossing EER as an empirical estimate, not the official challenge scorer.

## Training data and provenance

| Stage | Dataset | Evidence / split |
| --- | --- | --- |
| Upstream pretrained AASIST | ASVspoof 2019 Logical Access | Official clovaai/aasist checkpoint and upstream documentation; pinned source revision |
| Configured local fine-tuning | ASVspoof 2021 Deepfake evaluation audio with released CM labels | Matched local files; stratified 80/10/10 train/validation/test |

The repository does not establish that local fine-tuning has completed. These local splits do not implement an official ASVspoof 2021 challenge experiment.

## Validation

All 24 behavioral regressions passed across the four repositories (ML: 7, DL: 7, AASIST: 6, ARPGuard: 4). Each notebook also completed a CPU cell-integration smoke run on tiny fixtures, with training/sample limits reduced and the UNSW official row-count gate tested separately from the fixture run. The pinned upstream AASIST checkpoint loaded strictly and produced finite logits and gradients. These checks do not measure dataset accuracy.

Notebook schema and Python syntax (including transformed IPython magic cells), empty output checks, emoji checks, and behavioral regression tests. Tests cover metric sensitivity to actual predictions and the relevant parser, waveform, model, loss, cache, and inference fixes. Fixture data is used exclusively for software tests and is not reported as model performance.

## Remaining limits

Full dataset training and GPU execution were not performed. No benchmark score is claimed. Local random audio splits do not guarantee speaker, codec, attack-family, or source-recording independence; correlated variants can make local holdout scores optimistic. Validate those boundaries before drawing generalization conclusions.

## Primary references

- [ASVspoof 2021 CM metadata schemas](https://github.com/asvspoof-challenge/2021/tree/main/eval-package#on-meta-labels)
- [Official AASIST implementation and pretrained data](https://github.com/clovaai/aasist)
- [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [UNSW-NB15 publisher and partition counts](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
