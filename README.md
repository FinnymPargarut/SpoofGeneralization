# SpoofGeneralization

Course project: Generalization of Deep Audio Anti-Spoofing: Detecting Synthetic Speech from Unseen Generation Methods.

Synthetic speech detectors rarely fail on generation systems seen in training and do worse on unseen ones. The project measures this gap, checks whether detectors rely on silence, and tests ways to close the gap. Every variant is judged by quality and by compute cost.

## Layout

- `spoof/`: data protocols, audio handling, metrics, models.
- `scripts/score.py`: model and dataset to a score file. Runs on Kaggle GPU.
- `scripts/evaluate.py`: score files to result tables. Runs locally on CPU.
- `docs/`: literature notes, course requirements, experiment log (in Russian).

## Setup

```
uv sync                  # data, metrics, tests
uv sync --group model    # adds CPU torch and transformers
uv run pytest
```
