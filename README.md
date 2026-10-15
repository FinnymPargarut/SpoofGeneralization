# SpoofGeneralization

Course project: Generalization of Deep Audio Anti-Spoofing: Detecting Synthetic Speech from Unseen Generation Methods.

Synthetic speech detectors rarely fail on generation systems seen in training and do worse on unseen ones. The project measures this gap, checks whether detectors rely on silence, and tests ways to close the gap. Every variant is judged by quality and by compute cost.

## Layout

- `spoof/`: data protocols, audio handling, metrics, models.
- `scripts/score.py`: model and dataset to a score file. Runs on Kaggle GPU.
- `scripts/evaluate.py`: score files to result tables. Runs locally on CPU.
- `scripts/fetch_data.sh`: one-off download of inputs that Kaggle lacks (In-the-Wild, SSL-AASIST checkpoints, LA19etrim marks). Runs once in a Kaggle notebook.
- `scripts/silence_clf.py`: classifiers that see only edge silence and loudness. Runs on CPU.
- `scripts/kaggle_stage1.sh`: all of stage 1 in one Kaggle session.
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
