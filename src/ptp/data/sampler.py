import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Iterator, Tuple

RANK_PRIME = 9973
SHIFT_PRIME = 2654435761


class CoordinatedCompletionSampler(Sampler):
    def __init__(self, dataset_len, batch_size: int, seed: int = 42, shuffle: bool = True):
        super().__init__()
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = None
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        """
        Generate random cutoffs synchronized across all GPUs
        """
        generator = torch.Generator()
        # assert self.epoch is not None, "Must set epoch before using sampler"
        # generator.manual_seed(self.seed + self.epoch)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator.manual_seed(seed)

        # Random order
        if self.shuffle:
            indices = torch.randperm(self.dataset_len, generator=generator).tolist()
        else:
            indices = range(self.dataset_len)

        for batch_start in range(0, len(indices), self.batch_size):
            batch_indices = indices[batch_start:batch_start + self.batch_size]

            if dist.is_initialized():
                # Generate unique completion indices per GPU
                rank = dist.get_rank()

                # Generate permutation of completion indices
                completion_idx = torch.randint(0, SHIFT_PRIME, (1,), generator=generator).item()
                completion_idx = (completion_idx + rank * RANK_PRIME) % SHIFT_PRIME
            else:
                # Single GPU case -- just random number (to be moduloed later)
                completion_idx = torch.randint(0, SHIFT_PRIME, (1,), generator=generator).item()

            # Yield (index, cutoff) pairs
            for idx in batch_indices:
                yield idx, completion_idx

    def __len__(self):
        return self.dataset_len

    def set_epoch(self, epoch: int):
        self.epoch = epoch
