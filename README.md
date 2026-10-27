# SpoofGeneralization

Course project: Generalization of Deep Audio Anti-Spoofing: Detecting Synthetic Speech from Unseen Generation Methods.

Synthetic speech detectors rarely fail on generation systems seen in training and do worse on unseen ones. The project measures this gap, checks whether detectors rely on silence, and tests ways to close the gap. Every variant is judged by quality and by compute cost.

## Layout

- `spoof/`: data protocols, sets by name, audio handling, metrics, models (SSL-AASIST, LFCC-LCNN, frozen XLS-R features).
- `scripts/score.py`: model and dataset to a score file. Runs on Kaggle GPU.
- `scripts/evaluate.py`: score files of all stages to result tables. Runs locally on CPU.
- `scripts/fetch_data.sh`: one-off download of stage 1 inputs that Kaggle lacks (In-the-Wild, SSL-AASIST checkpoints, LA19etrim marks). Runs once in a Kaggle notebook.
- `scripts/fetch_stage2.py`: one-off stage 2 inputs: RuASD sample, LibriSpeech, RuLS. Runs once in a Kaggle notebook without GPU.
- `scripts/silence_clf.py`: classifiers that see only simple properties of a recording: edge silence, loudness, duration, sample rate. Runs on CPU.
- `scripts/xlsr_features.py`, `scripts/frozen_lr.py`: frozen XLS-R 300M features and logistic regression on them.
- `scripts/train_lcnn.py`: LFCC-LCNN training.
- `scripts/kaggle_stage1.sh`, `scripts/kaggle_stage2.sh`: one stage in one Kaggle session.
- `docs/`: literature notes, course requirements, experiment log (in Russian).

## Setup

```
uv sync                  # data, metrics, tests
uv sync --group model    # adds CPU torch and transformers
uv run pytest
```

## Kaggle

Notebooks clone this repository into `/tmp` and use the preinstalled Python packages, nothing is installed.

Inputs, once. New notebook, internet on, no accelerator:
```
!git clone --depth 1 https://github.com/FinnymPargarut/SpoofGeneralization /tmp/repo
!bash /tmp/repo/scripts/fetch_data.sh
```
Save Version, then turn the notebook output into a private dataset.

Stage 1. New notebook, internet on, GPU T4 x2, inputs: the [ASVspoof 2019 mirror](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) and the private dataset above:
```
!git clone --depth 1 https://github.com/FinnymPargarut/SpoofGeneralization /tmp/repo
!bash /tmp/repo/scripts/kaggle_stage1.sh
```
Save Version with Save & Run All, so it runs without an open browser. Download the `stage1` folder from the version output into `outputs/stage1`, then locally:
```
uv run python scripts/evaluate.py outputs/stage1
```

Stage 2 inputs, once. New notebook, internet on, no accelerator:
```
!git clone --depth 1 https://github.com/FinnymPargarut/SpoofGeneralization /tmp/repo
!cd /tmp/repo && PYTHONPATH=/tmp/repo python scripts/fetch_stage2.py --out /kaggle/working
```
Save Version, then turn the notebook output into a private dataset.

Stage 2. New notebook, internet on, GPU T4 x2, inputs: the ASVspoof 2019 mirror and both private datasets:
```
!git clone --depth 1 https://github.com/FinnymPargarut/SpoofGeneralization /tmp/repo
!bash /tmp/repo/scripts/kaggle_stage2.sh all
```
`all` can be split into `frozen` and `lcnn` runs. Download the `stage2` folder without `features` into `outputs/stage2`, then locally:
```
uv run python scripts/evaluate.py outputs/stage1 outputs/stage2
```
