import random
import warnings
from pathlib import Path
import os
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm

import torch
from datasets import Dataset
from lightning import LightningDataModule
from transformers import AutoTokenizer

from ptp.data.collate import collate_fn
from ptp.data.packing import PackingDataset, packed_collate_fn
from ptp.data.utils import TupleAccessSubset, duplicate_avoiding_randint
from ptp.data.sampler import CoordinatedCompletionSampler


def seq_len_before_eos(input_ids, eos_token_id):
    is_eos: torch.Tensor = input_ids == eos_token_id
    if is_eos.all():
        return 0
    flipped = is_eos.flip(0)
    eos_len = (~flipped).float().argmax().item()
    return input_ids.shape[0] - eos_len


class PregeneratedDataset(torch.utils.data.Dataset):
    """
    Wraps a pre-generated HuggingFace dataset of the form::

        {
            prompt_name:      (prompt_seq_len,),
            completions_name: (num_completions, completion_seq_len),
            'left_bin_edges':  (num_completions, completion_seq_len),   # optional
            'right_bin_edges': (num_completions, completion_seq_len),   # optional
        }

    Implements the packing interface:

    * ``get_metadata()`` — returns ``(doc_length, [(prompt_len, doc_length)])`` for
      every item, used by PackingDataset to build BFD groups.  Doc length is estimated
      as ``len(prompt) + max(len(c) for c in completions)``.  Result is cached.

    * ``__getitem__(idx)`` — returns a dict with ``input_ids`` (prompt + one randomly
      chosen completion) and optionally ``bin_edges_left`` / ``bin_edges_right`` as
      per-token float tensors (NaN for prompt positions).  Completion selection happens
      here; position sampling is delegated to PackingDataset.
    """

    def __init__(self, completion_dataset, eos_token_id: int,
                 prompt_name: str = "input", completions_name: str = "completions",
                 max_sequence_length: int | None = None,
                 experiment_dir: str | Path | None = None,
                 right_truncate_eos: bool = True, load_bin_edges: bool = True):
        super().__init__()
        self.completion_dataset  = completion_dataset
        self.prompt_name         = prompt_name
        self.completions_name    = completions_name
        self.eos_token_id        = eos_token_id
        self.right_truncate_eos  = right_truncate_eos
        self.max_sequence_length = max_sequence_length
        self.load_bin_edges      = load_bin_edges
        self.cache_dir           = Path(experiment_dir) / 'data_cache' if experiment_dir else Path('data_cache')

        # Filter items that exceed max_sequence_length (length mask cached to disk)
        self.useful_indices = self._build_index()

    # ------------------------------------------------------------------
    # Index / filtering
    # ------------------------------------------------------------------

    def _doc_length(self, item) -> int:
        return len(item[self.prompt_name]) + max(
            len(c) for c in item[self.completions_name]
        )

    def _build_index(self) -> np.ndarray:
        if self.max_sequence_length is None:
            return np.arange(len(self.completion_dataset), dtype=int)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ds_tag   = getattr(getattr(self.completion_dataset, '_info', None), 'dataset_name', 'pregenerated')
        mask_file = self.cache_dir / f'{ds_tag}_{self.max_sequence_length}_mask.npy'

        if mask_file.exists():
            print(f"Loading sequence length mask from {mask_file}")
            mask = np.load(mask_file)
        else:
            print(f"Computing sequence length mask → {mask_file}")
            mask = np.array([
                self._doc_length(item) <= self.max_sequence_length
                for item in tqdm(self.completion_dataset, desc="Filtering by length")
            ])
            np.save(mask_file, mask)
            raise ValueError("Length mask created — please rerun to load the filtered dataset.")

        idx = np.where(mask)[0].astype(int)
        print(f"Filtered: {len(idx)}/{len(self.completion_dataset)} items fit in {self.max_sequence_length} tokens")
        return idx

    # ------------------------------------------------------------------
    # Packing interface
    # ------------------------------------------------------------------

    def get_metadata(self) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        ``(doc_length, [(prompt_len, doc_length)])`` for every item in ``useful_indices``.

        Doc length = len(prompt) + max completion length.
        Completion region = [prompt_len, doc_length) (the completion portion).
        Cached to disk.
        """
        cache_file = self.cache_dir / 'pregenerated_metadata.pkl'
        if cache_file.exists():
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"Computing metadata for {len(self.useful_indices)} pregenerated items …")
        result = []
        for orig_idx in tqdm(self.useful_indices, desc="Metadata"):
            item     = self.completion_dataset[int(orig_idx)]
            p_len    = len(item[self.prompt_name])
            doc_len  = p_len + max(len(c) for c in item[self.completions_name])
            result.append((doc_len, [(p_len, doc_len)]))

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.useful_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Return ``{"input_ids": Tensor, "bin_edges_left": Tensor, "bin_edges_right": Tensor}``
        for item *idx*.  Bin-edge tensors use NaN for prompt positions.
        """
        orig_idx    = int(self.useful_indices[idx])
        entry       = self.completion_dataset[orig_idx]
        completions = entry[self.completions_name]

        comp_idx    = random.randint(0, len(completions) - 1)
        comp_ids    = completions[comp_idx]

        if self.right_truncate_eos:
            valid_len = max(seq_len_before_eos(comp_ids, self.eos_token_id) + 1, 1)
        else:
            valid_len = len(comp_ids)
        comp_ids = comp_ids[:valid_len]

        prompt_ids = entry[self.prompt_name]
        input_ids  = torch.cat([prompt_ids, comp_ids])

        out = {"input_ids": input_ids}

        if self.load_bin_edges and "left_bin_edges" in entry and "right_bin_edges" in entry:
            # (prompt_len - 1) NaN entries so that edge at index comp_start-1 is real
            # when comp_start == prompt_len (the earliest valid completion position).
            nan_prompt = torch.full((len(prompt_ids) - 1,), float('nan'), dtype=torch.float)
            out["bin_edges_left"]  = torch.cat([nan_prompt, entry["left_bin_edges"][comp_idx][:valid_len].float()])
            out["bin_edges_right"] = torch.cat([nan_prompt, entry["right_bin_edges"][comp_idx][:valid_len].float()])

        return out


class PregeneratedDataModule(LightningDataModule):
    def __init__(self, root_dir: str | Path, train_completion_len: int, num_completions: int,
                 tokenizer: str,
                 batch_size: int, max_sequence_length: int = None, experiment_dir: str | Path = None,
                 prompt_name: str = "input", completions_name: str = "completions",
                 right_truncate_eos: bool = True, load_bin_edges: bool = True,
                 fallback_val_split: int | float = 1000, **kwargs):
        super().__init__()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        self.root_dir             = root_dir
        self.load_bin_edges       = load_bin_edges
        self.train_completion_len = train_completion_len
        self.num_completions      = num_completions
        self.max_sequence_length  = max_sequence_length
        self.experiment_dir       = experiment_dir
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer         = tokenizer
        self.eos_token_id      = tokenizer.eos_token_id
        self.right_truncate_eos = right_truncate_eos
        self.prompt_name       = prompt_name
        self.completions_name  = completions_name
        self.datasets          = {}
        self.fallback_val_split = fallback_val_split
        self.batch_size        = batch_size
        self.kwargs            = kwargs

    def _wrap(self, base: PregeneratedDataset) -> PackingDataset:
        return PackingDataset(
            base,
            max_sequence_length=self.max_sequence_length or base._doc_length(
                base.completion_dataset[int(base.useful_indices[0])]
            ),
            num_completions=self.num_completions,
            completion_length=self.train_completion_len,
            cache_dir=str(base.cache_dir),
        )

    def setup(self, stage: str) -> None:
        datasets = {}
        for split in ["train", "val", "test"]:
            if (self.root_dir / split).exists():
                hf_ds = Dataset.load_from_disk(self.root_dir / split)
                hf_ds.set_format(type="torch")
                base = PregeneratedDataset(
                    hf_ds,
                    eos_token_id=self.eos_token_id,
                    prompt_name=self.prompt_name,
                    completions_name=self.completions_name,
                    max_sequence_length=self.max_sequence_length,
                    experiment_dir=self.experiment_dir,
                    right_truncate_eos=self.right_truncate_eos,
                    load_bin_edges=self.load_bin_edges,
                )
                datasets[split] = self._wrap(base)
            elif split == "train":
                raise FileNotFoundError(f"Training dataset not found in {self.root_dir / split}")

        if "val" not in datasets and "test" in datasets:
            datasets["val"] = datasets["test"]
            warnings.warn("No validation data found, using test data for validation")
        if "val" not in datasets:
            if self.fallback_val_split is not None:
                val_size = self.fallback_val_split
                if isinstance(val_size, float):
                    val_size = int(len(datasets["train"]) * val_size)
                datasets["val"]   = TupleAccessSubset(datasets["train"], range(val_size))
                datasets["train"] = TupleAccessSubset(datasets["train"], range(val_size, len(datasets["train"])))
                print(f"No validation data found; using {val_size} train items for validation")
            else:
                warnings.warn("No validation data found, not validating")
        self.datasets = datasets

    def _dataloader(self, dataset, shuffle: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=packed_collate_fn,
            **self.kwargs,
        )

    def train_dataloader(self):
        return self._dataloader(self.datasets["train"], shuffle=True)

    def val_dataloader(self):
        if "val" in self.datasets:
            return self._dataloader(self.datasets["val"], shuffle=False)
        return []

    def test_dataloader(self):
        if "test" in self.datasets:
            return self._dataloader(self.datasets["test"], shuffle=False)
        return []
