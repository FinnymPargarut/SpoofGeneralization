"""Frozen XLS-R 300M features as in Pascu et al. 2024.

Each recording is normalized to zero mean and unit variance like the Hugging Face
feature extractor does, then every hidden state is averaged over the recording's
own frames. State 0 is the projected CNN output, state 24 the last layer after its
layer norm, the last_hidden_state that Pascu averages.
"""

import numpy as np
import torch
from transformers import Wav2Vec2Model

NAME = "facebook/wav2vec2-xls-r-300m"
STATES = 25


def length_batches(lengths, budget: int) -> list[list[int]]:
    """Indices in ascending length, each batch padded to its longest member fits the budget.

    A recording longer than the budget gets a batch of its own.
    """
    batches = []
    current = []
    for i in np.argsort(lengths, kind="stable"):
        if current and (len(current) + 1) * lengths[i] > budget:
            batches.append(current)
            current = []
        current.append(int(i))
    if current:
        batches.append(current)
    return batches


def normalized(wav: np.ndarray) -> np.ndarray:
    return (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)


def pooled_states(model, wavs: list[np.ndarray]) -> np.ndarray:
    """(batch, STATES, hidden) float32 means over each recording's frames."""
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    device = next(core.parameters()).device
    lengths = torch.tensor([w.size for w in wavs])
    batch = torch.zeros(len(wavs), int(lengths.max()))
    for i, wav in enumerate(wavs):
        batch[i, : wav.size] = torch.from_numpy(normalized(wav))
    mask = torch.arange(batch.shape[1]) < lengths[:, None]
    frames = core._get_feat_extract_output_lengths(lengths).to(device)
    with torch.inference_mode():
        # Tuples survive DataParallel gathering, Hugging Face output objects may not.
        _, _, states = model(
            batch.to(device),
            attention_mask=mask.long().to(device),
            output_hidden_states=True,
            return_dict=False,
        )
        stacked = torch.stack(states, dim=1)
        valid = torch.arange(stacked.shape[2], device=device) < frames[:, None]
        summed = (stacked * valid[:, None, :, None]).sum(2)
        return (summed / frames[:, None, None]).float().cpu().numpy()


def with_state_hooks(model: Wav2Vec2Model) -> Wav2Vec2Model:
    """Installs the transformers hidden state hooks once, before DataParallel copies the model.

    transformers installs them on first use and marks only the module it ran on.
    DataParallel makes fresh replicas on every call, each installs the hooks again into
    hook dicts shared with the model, and every layer is then recorded once per replica.
    """
    with torch.inference_mode():
        model(torch.zeros(1, 16_000), output_hidden_states=True)
    return model


def load_xlsr() -> Wav2Vec2Model:
    return with_state_hooks(Wav2Vec2Model.from_pretrained(NAME).eval())
