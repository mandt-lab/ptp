import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutput

from ptp.data.collate import collate_fn
from ptp.data.utils import duplicate_avoiding_randint


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, completion_length, completions_per_batch, data):
        super().__init__()
        self.completion_length = completion_length
        self.completions_per_batch = completions_per_batch
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence_ids = self.data[index]
        unpadded = sequence_ids[sequence_ids >= 0]
        if len(unpadded) <= 1 and self.completions_per_batch > 1:
            raise ValueError("Sequence too short to sample multiple completions from.")
        del sequence_ids
        start_idx = duplicate_avoiding_randint(
            1, len(unpadded),
            self.completions_per_batch,
        )

        return {
            "input_ids": unpadded,
            "input_mask": torch.ones_like(unpadded, dtype=torch.bool),
            "completion_starts": start_idx,
            "completion_length": self.completion_length,
        }


class IncreasingSequenceDataModule(LightningDataModule):
    def __init__(self, seq_len=12, vocab_size=4, num_samples=1024,
                 completion_length=6, completions_per_batch=4,
                 mixed_length: bool = False, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples

        self.completion_length = completion_length
        self.completions_per_batch = completions_per_batch
        self.mixed_length = mixed_length

        self.train_data = None
        self.val_data = None

    def make_data(self, size):
        switch_prob = torch.rand(size, self.seq_len)
        # The first token is 0
        switch_prob[:, 0] = 0
        data = (switch_prob > 0.5).cumsum(dim=1).clamp(0, self.vocab_size - 1)
        if self.mixed_length:
            # Randomly truncate sequences
            lengths = torch.randint(
                2, self.seq_len + 1, (size,)
            )
            for i in range(size):
                data[i, lengths[i]:] = -1  # pad with invalid token
        return data

    def setup(self, stage=None):
        self.train_data = TokenDataset(
            self.completion_length, self.completions_per_batch, self.make_data(self.num_samples)
        )
        self.val_data = TokenDataset(
            self.completion_length, self.completions_per_batch, self.make_data(self.num_samples // 10)
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=collate_fn, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_data, collate_fn=collate_fn, **self.kwargs)


class IncreasingSequenceTokenizer:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
        self.pad_token_id = None

    def decode(self, token_ids, skip_special_tokens=False):
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids)
        return " ".join(
            str(t.item())
            for t in token_ids
            if t.item() != self.eos_token_id or not skip_special_tokens
        )


class IncreasingSequenceConfig(PretrainedConfig):
    model_type = "increasing-sequence"
    def __init__(self, vocab_size=100, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size


class IncreasingSequenceTeacherModel(PreTrainedModel, GenerationMixin):
    config_class = IncreasingSequenceConfig

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.tokenizer = IncreasingSequenceTokenizer(config.vocab_size - 1)

    def forward(self, input_ids, labels=None, attention_mask=None, return_dict=True):
        batch_size, seq_len = input_ids.shape
        probs = torch.full((batch_size, seq_len, self.vocab_size), 0.0, device=input_ids.device)
        for i in range(batch_size):
            for j in range(seq_len):
                if attention_mask is not None:
                    if attention_mask[i, j] == 0:
                        probs[i, j, :] = 0.0
                        continue
                current_token = input_ids[i, j].item()
                probs[i, j, current_token:current_token + 2] = 1.0  # same or next value
        probs /= probs.sum(dim=-1, keepdim=True)  # normalize to get probabilities
        logits = probs.log()

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    @property
    def device(self):
        return torch.device("cpu")
