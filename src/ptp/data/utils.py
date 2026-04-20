from math import ceil

import torch
from torch.utils.data import Dataset
from typing import Sequence, TypeVar, List

from ptp.data.collate import IGNORE_INDEX

_T_co = TypeVar("_T_co", covariant=True)
class TupleAccessSubset(Dataset[_T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[_T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_metadata(self):
        all_meta = self.dataset.get_metadata()
        return [all_meta[i] for i in self.indices]


def predict_bin_edges(input_ids, model, input_mask=None, adapt_logits = None):
    outputs = model(
        input_ids=input_ids,
        attention_mask=input_mask,
    )

    # batch_size, completion_seq_len, vocab_size
    logits = outputs.logits[:, :-1, :]
    if adapt_logits is not None:
        logits = adapt_logits(logits).float()
    probs = torch.softmax(logits, dim=-1)
    selector = (
        torch.arange(probs.shape[0], device=probs.device)[:, None],
        torch.arange(probs.shape[1], device=probs.device)[None, :],
        torch.where(
            input_ids[:, 1:] == IGNORE_INDEX,
            0,
            input_ids[:, 1:]
        )
    )
    cum_probs = probs.cumsum(-1)
    left_bin_edge = cum_probs[selector] - probs[selector]
    right_bin_edge = cum_probs[selector]

    return left_bin_edge, right_bin_edge, outputs


def duplicate_avoiding_randint(low: int, high: int, size: int, **kwargs) -> List[int]:
    """
    Sample `size` unique integers from [low, high), avoiding duplicates.
    """
    if size == 0:
        return []
    span = high - low
    assert span > 0
    num_tours = ceil(size / span)
    chunks: List[torch.Tensor] = []
    for _ in range(num_tours):
        samples = torch.randperm(span, **kwargs)[:size - len(chunks) * span] + low
        chunks.append(samples)
    all_samples = torch.cat(chunks).sort().values
    assert len(all_samples) == size
    return all_samples.tolist()
