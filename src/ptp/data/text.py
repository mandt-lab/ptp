import math
import os
import numpy as np
import torch
from typing import Any, List, Optional, Union
from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

from ptp.data.collate import collate_fn as base_collate_fn
from ptp.data.packing import PackingDataset, packed_collate_fn
from ptp.data.utils import duplicate_avoiding_randint

CACHE_DIR = 'data_cache'


class TextDocumentDataset(torch.utils.data.Dataset):
    """
    Loads any HuggingFace text dataset and concatenates the specified text column
    into one long token stream, then serves fixed-length windows with random
    completion starting positions.

    Returns batches compatible with lit.py:
        input_ids:         (seq_len,)
        input_mask:        (seq_len,)  — all ones
        completion_starts: List[int]   — num_completions random positions
        completion_length: int

    The full window is presented as input; completion_starts are sampled uniformly
    from [0, seq_len - completion_length], so the model learns to predict any
    sub-sequence of a text chunk.
    """

    def __init__(self, dataset_name: str, split: str, tokenizer,
                 max_sequence_length: int, num_completions: int, train_completion_len: int,
                 text_column: str = "text", dataset_config: str = None,
                 cache_dir: str = CACHE_DIR):
        self.max_sequence_length = max_sequence_length
        self.num_completions = num_completions
        self.train_completion_len = train_completion_len

        token_cache_file = os.path.join(
            cache_dir,
            dataset_name.replace('/', '_') + f'_{split}_{text_column}_tokenized.npy'
        )
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(token_cache_file):
            print(f"Loading tokens from {token_cache_file}")
            self.token_array = np.load(token_cache_file)
        else:
            print(f"Tokenizing dataset and saving to {token_cache_file}")
            data = load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
            token_list = []
            for item in tqdm(data, desc=f"Tokenizing {dataset_name}/{split}"):
                text = item[text_column]
                if not text:
                    continue
                ids = tokenizer(text)['input_ids']
                # Drop BOS if the tokenizer adds it (avoid duplicates at chunk boundaries)
                if ids and ids[0] == tokenizer.bos_token_id:
                    ids = ids[1:]
                token_list.append(np.array(ids, dtype=np.int32))
            self.token_array = np.concatenate(token_list)
            print(f"Dataset contains {len(self.token_array):,} tokens.")
            np.save(token_cache_file, self.token_array)

        # Golden-ratio stride permutation for near-uniform coverage without shuffling
        self._n = len(self.token_array) - self.max_sequence_length + 1
        self._perm_c = math.floor(0.6180339887 * self._n)
        while math.gcd(self._perm_c, self._n) != 1:
            self._perm_c += 1

    def __len__(self):
        return self._n

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return token IDs for the chunk at *idx*."""
        pidx = (self._perm_c * idx) % self._n
        return torch.tensor(
            self.token_array[pidx : pidx + self.max_sequence_length].astype(np.int64),
            dtype=torch.long,
        )

    def get_metadata(self) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Every chunk has a fixed length and the full sequence is the completion region.
        Returns ``[(max_sequence_length, [(0, max_sequence_length)])] * len(self)``.
        """
        region = [(0, self.max_sequence_length)]
        return [(self.max_sequence_length, region)] * len(self)


class TextDocumentDataModule(LightningDataModule):
    """
    DataModule that wraps any HuggingFace text dataset for direct distillation
    training (path b, text variant).

    Batch format matches lit.py expectations:
        input_ids, input_mask, completion_starts, completion_length.
    """

    def __init__(self, dataset_name: str, tokenizer: Union[str, PreTrainedTokenizerBase],
                 max_sequence_length: int, num_completions: int, train_completion_len: int,
                 text_column: str = "text", dataset_config: str = None,
                 splits: List[str] | None = None,
                 train_shuffle: bool = True,
                 cache_dir: str = CACHE_DIR,
                 pack_documents: bool = False,
                 max_docs: int | None = None,
                 **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.num_completions = num_completions
        self.train_completion_len = train_completion_len
        self.text_column = text_column
        self.dataset_config = dataset_config
        self.splits = splits if splits is not None else ['train', 'valid', 'test']
        self.train_shuffle = train_shuffle
        self.cache_dir = cache_dir
        self.pack_documents = pack_documents
        self.max_docs = max_docs
        self.kwargs = kwargs

    def _make_dataset(self, split: str):
        ds = TextDocumentDataset(
            dataset_name=self.dataset_name,
            split=split,
            tokenizer=self.tokenizer,
            max_sequence_length=self.max_sequence_length,
            num_completions=self.num_completions,
            train_completion_len=self.train_completion_len,
            text_column=self.text_column,
            dataset_config=self.dataset_config,
            cache_dir=self.cache_dir,
        )
        return PackingDataset(
            ds,
            max_sequence_length=self.max_sequence_length,
            num_completions=self.num_completions,
            completion_length=self.train_completion_len,
            cache_dir=self.cache_dir,
        )

    def setup(self, stage: Any = None) -> None:
        datasets = {}
        for split in self.splits:
            try:
                datasets[split] = self._make_dataset(split)
            except Exception as e:
                print(f"Warning: could not load split '{split}': {e}")

        self.train_dataset = datasets.get("train")
        self.val_dataset = datasets.get("valid")
        self.test_dataset = datasets.get("test")

        if self.val_dataset is None and self.test_dataset is not None:
            self.val_dataset = self.test_dataset
            print("No validation split found, using test split for validation.")
        if self.val_dataset is None and self.train_dataset is not None:
            self.val_dataset = self.train_dataset
            print("No validation split found, using train split for validation.")

    def _dataloader(self, dataset, shuffle: bool) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            collate_fn=packed_collate_fn,
            **self.kwargs,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset not available.")
        return self._dataloader(self.train_dataset, shuffle=self.train_shuffle)

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.val_dataset is None:
            return None
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.test_dataset is None:
            return None
        return self._dataloader(self.test_dataset, shuffle=False)
