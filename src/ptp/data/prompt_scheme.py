import torch
import numpy as np
from typing import Any, List
from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from tqdm import tqdm
import os

CACHE_DIR = 'data_cache'


class MaskCreationError(Exception):
    pass


class PromptSchemeDataset(torch.utils.data.Dataset):
    """
    Extracts problem descriptions from a dataset and pairs them with a given prompt scheme.
    """

    def __init__(self, dataset_name, split, cache_dir, prompt_scheme: str,
                 tokenizer, max_sequence_length: int):
        self.data = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        self.prompt_scheme = prompt_scheme

        mask_cache_file = os.path.join(
            cache_dir,
            dataset_name.replace('/', '_') + f'_{split}_{max_sequence_length}_mask.npy'
        )
        if os.path.exists(mask_cache_file):
            print(f"Loading mask from {mask_cache_file}")
            boolean_mask = np.load(mask_cache_file)
        else:
            print(f"Creating mask and saving to {mask_cache_file}")
            boolean_mask = np.array([
                len(tokenizer(prompt_scheme.format(**item))['input_ids']) <= max_sequence_length
                for item in tqdm(self.data, desc="Filtering dataset")
            ])
            np.save(mask_cache_file, boolean_mask)
            raise MaskCreationError("Dataset mask file was just created, please rerun to load it.")
        self.useful_indices = np.where(boolean_mask)[0]

    def __len__(self):
        return len(self.useful_indices)

    def __getitem__(self, index) -> Any:
        return self.prompt_scheme.format(**self.data[int(self.useful_indices[index])])


class PromptSchemeDataModule(LightningDataModule):
    """
    DataLoader wrapper for PromptScheme.
    """

    def __init__(self, dataset_name: str, prompt_scheme: str,
                 tokenizer, max_sequence_length: int, train_shuffle: bool = True,
                 splits: List[str] | None = None, padding: bool = True,
                 cache_dir: str = CACHE_DIR, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        if splits is None:
            splits = ['train', 'valid', 'test']
        self.splits = splits
        self.kwargs = kwargs
        self.prompt_scheme = prompt_scheme
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_sequence_length = max_sequence_length
        self.cache_dir = cache_dir
        self.train_shuffle = train_shuffle

    def collate_fn(self, batch):
        tokenizer_out = self.tokenizer(batch, return_tensors='pt', padding=self.padding)
        return {
            'input_ids': tokenizer_out['input_ids'],
            'input_mask': tokenizer_out['attention_mask'],
        }

    def setup(self, stage: Any = None) -> None:
        datasets = {}
        any_raised = False
        for split in self.splits:
            try:
                datasets[split] = PromptSchemeDataset(
                    self.dataset_name, split=split, cache_dir=self.cache_dir,
                    prompt_scheme=self.prompt_scheme,
                    tokenizer=self.tokenizer,
                    max_sequence_length=self.max_sequence_length,
                )
            except MaskCreationError as e:
                any_raised = e
        if any_raised is not False:
            raise any_raised
        self.train_dataset = datasets["train"]
        self.val_dataset = datasets.get("valid")
        self.test_dataset = datasets.get("test")
        
        # If no validation data exists but test data does, use test data for validation
        if self.val_dataset is None and self.test_dataset is not None:
            self.val_dataset = self.test_dataset
            print("No validation data found, using test data for validation")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset, shuffle=self.train_shuffle, collate_fn=self.collate_fn, **self.kwargs
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.val_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset, shuffle=False, collate_fn=self.collate_fn, **self.kwargs
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        if self.test_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.test_dataset, shuffle=False, collate_fn=self.collate_fn, **self.kwargs
        )
