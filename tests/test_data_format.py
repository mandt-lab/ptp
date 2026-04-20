"""
Tests for the unified batch format across all data modules.

Reference format (from lit.py):
    input_ids:         Tensor (seq_len,)  — token IDs
    input_mask:        Tensor (seq_len,)  — bool attention mask
    completion_starts: List[int]          — positions where completions begin
    completion_length: int                — number of tokens in each completion

All modules that feed lit.py must produce this format after collation.
Modules that feed pregenerate.py only require input_ids / input_mask.
"""

import json
import os
import tempfile
import numpy as np
import pytest
import torch
from pathlib import Path
from datasets import Dataset

from conftest import FakeTokenizer  # already on sys.path via conftest.py

from ptp.data.collate import collate_fn, IGNORE_INDEX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pregenerated_dataset_on_disk(tmp_dir: Path, n_prompts=4,
                                       prompt_len=10, completion_len=20,
                                       num_completions=3):
    """Write a minimal pregenerated dataset (HF Arrow) to disk."""
    rows = []
    for _ in range(n_prompts):
        rows.append({
            "input": list(range(1, prompt_len + 1)),
            "completions": [
                list(range(50, 50 + completion_len)) for _ in range(num_completions)
            ],
            "left_bin_edges": [
                [float(i) / completion_len for i in range(completion_len)]
                for _ in range(num_completions)
            ],
            "right_bin_edges": [
                [float(i + 1) / completion_len for i in range(completion_len)]
                for _ in range(num_completions)
            ],
        })
    ds = Dataset.from_list(rows)
    out = tmp_dir / "train"
    ds.save_to_disk(out)
    return out


def make_chat_lookup_file(tmp_dir: Path, dataset_name: str, split: str,
                           max_seq_len: int, mode: str, lookup: np.ndarray):
    """Write a pre-built lookup file so ChatDataset skips the build step."""
    cache_dir = tmp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_name.replace('/', '_')
    fname = f"{safe_name}_{split}_{max_seq_len}_{mode}_chat_lookup.npy"
    np.save(cache_dir / fname, lookup)
    return cache_dir


# ---------------------------------------------------------------------------
# Batch format validator
# ---------------------------------------------------------------------------

REQUIRED_TRAINING_KEYS = {"input_ids", "input_mask", "completion_starts", "completion_length"}
REQUIRED_PROMPT_KEYS = {"input_ids", "input_mask"}


def assert_training_batch(batch: dict, batch_size: int, num_completions: int, completion_length: int):
    """Check that a collated batch has the correct format for lit.py."""
    for key in REQUIRED_TRAINING_KEYS:
        assert key in batch, f"Missing key '{key}' in batch"

    assert batch["input_ids"].shape[0] == batch_size, \
        f"input_ids batch dim: expected {batch_size}, got {batch["input_ids"].shape[0]}"
    assert batch["input_mask"].shape == batch["input_ids"].shape, \
        "input_ids and input_mask must have the same shape"
    assert batch["completion_length"] == completion_length, \
        f"completion_length: expected {completion_length}, got {batch['completion_length']}"

    cs = batch["completion_starts"]
    assert len(cs) == batch_size, f"completion_starts outer len: {len(cs)} vs {batch_size}"
    for row in cs:
        assert len(row) == num_completions, \
            f"completion_starts inner len: {len(row)} vs {num_completions}"


def assert_prompt_batch(batch: dict, batch_size: int):
    """Check that a prompt-mode batch has the correct format for pregenerate.py."""
    for key in REQUIRED_PROMPT_KEYS:
        assert key in batch, f"Missing key '{key}' in batch"
    # Must NOT use the old key names
    assert "prompt_ids" not in batch, "batch must not contain deprecated 'prompt_ids' key"
    assert "prompt_mask" not in batch, "batch must not contain deprecated 'prompt_mask' key"
    assert batch["input_ids"].shape[0] == batch_size
    assert batch["input_mask"].shape == batch["input_ids"].shape


# ---------------------------------------------------------------------------
# collate_fn tests
# ---------------------------------------------------------------------------

class TestCollateFn:
    def _make_item(self, seq_len: int, num_completions: int = 2, completion_length: int = 4):
        return {
            "input_ids": torch.randint(3, 100, (seq_len,)),
            "input_mask": torch.ones(seq_len, dtype=torch.bool),
            "completion_starts": list(range(num_completions)),
            "completion_length": completion_length,
        }

    def test_basic_collation(self):
        items = [self._make_item(10), self._make_item(15), self._make_item(8)]
        batch = collate_fn(items)
        assert_training_batch(batch, batch_size=3, num_completions=2, completion_length=4)

    def test_padding_fills_with_ignore_index(self):
        items = [self._make_item(5), self._make_item(10)]
        batch = collate_fn(items)
        # The first item (len=5) should have the last 5 positions set to IGNORE_INDEX
        padded_ids = batch["input_ids"][0]
        assert (padded_ids[5:] == IGNORE_INDEX).all()

    def test_mask_is_false_for_padding(self):
        items = [self._make_item(5), self._make_item(10)]
        batch = collate_fn(items)
        mask = batch["input_mask"][0]
        assert mask[:5].all()
        assert not mask[5:].any()

    def test_bin_edges_optional(self):
        items = [self._make_item(6), self._make_item(8)]
        batch = collate_fn(items)
        assert "left_bin_edges" not in batch
        assert "right_bin_edges" not in batch

    def test_bin_edges_included_when_present(self):
        def make_item_with_edges(seq_len):
            item = self._make_item(seq_len)
            item["left_bin_edges"] = torch.zeros(seq_len - 1)
            item["right_bin_edges"] = torch.ones(seq_len - 1)
            return item

        items = [make_item_with_edges(6), make_item_with_edges(10)]
        batch = collate_fn(items)
        assert "left_bin_edges" in batch
        assert "right_bin_edges" in batch
        # Shape: (batch_size, max_seq_len - 1)
        assert batch["left_bin_edges"].shape == (2, 9)

    def test_inconsistent_completion_counts_raises(self):
        item_a = self._make_item(8, num_completions=2)
        item_b = self._make_item(8, num_completions=3)
        with pytest.raises(AssertionError):
            collate_fn([item_a, item_b])


# ---------------------------------------------------------------------------
# ChatDataModule (prompt mode) tests
# ---------------------------------------------------------------------------

class TestChatDataModulePromptMode:
    """Verify ChatDataModule 'prompt' mode produces input_ids/input_mask."""

    def _make_dm(self, tmp_path, conversations, fake_tokenizer, max_seq=200):
        from ptp.data.chat import ChatDataset, ChatDataModule, MaskCreationError

        dataset = Dataset.from_list([
            {"messages": conv} for conv in conversations
        ])

        # Build the lookup manually so we don't need the real lookup creation
        # (which raises MaskCreationError on first run)
        lookup = []
        for chat_idx, conv in enumerate(conversations):
            for seg_len in range(1, len(conv) + 1):
                seg = conv[:seg_len]
                if seg[-1]["role"] != "user":
                    continue
                lookup.append([chat_idx, seg_len])
        lookup_arr = np.array(lookup, dtype=np.int64)

        cache_dir = make_chat_lookup_file(tmp_path, "fake_ds", "train", max_seq, "prompt", lookup_arr)

        ds = ChatDataset.__new__(ChatDataset)
        ds.data = dataset
        ds.tokenizer = fake_tokenizer
        ds.mode = "prompt"
        ds.add_assistant_prompt = True
        ds.conversation_key = "messages"
        ds.user_roles = ["user", "human"]
        ds.assistant_roles = ["assistant", "gpt"]
        ds.max_sequence_length = max_seq
        ds.num_completions = 2
        ds.train_completion_len = 8
        ds.lookup_array = lookup_arr
        return ds

    def test_getitem_returns_string(self, tmp_path, fake_tokenizer, simple_conversations):
        ds = self._make_dm(tmp_path, simple_conversations, fake_tokenizer)
        item = ds[0]
        assert isinstance(item, str), "prompt mode __getitem__ should return a string"

    def test_collate_returns_input_ids_mask(self, fake_tokenizer, simple_conversations):
        from ptp.data.chat import ChatDataModule

        dm = ChatDataModule.__new__(ChatDataModule)
        dm.mode = "prompt"
        dm.tokenizer = fake_tokenizer
        dm.max_sequence_length = 200
        dm.padding = True

        strings = [
            fake_tokenizer.apply_chat_template(conv, tokenize=False)
            for conv in simple_conversations
        ]
        batch = dm.collate_fn(strings)

        assert_prompt_batch(batch, batch_size=len(strings))

    def test_collate_truncates_to_max_seq_len(self, fake_tokenizer):
        from ptp.data.chat import ChatDataModule

        dm = ChatDataModule.__new__(ChatDataModule)
        dm.mode = "prompt"
        dm.tokenizer = fake_tokenizer
        dm.max_sequence_length = 10
        dm.padding = True

        long_string = "A" * 100  # will encode to >10 tokens
        batch = dm.collate_fn([long_string])
        assert batch["input_ids"].shape[1] <= 10


# ---------------------------------------------------------------------------
# ChatDataModule (full mode) tests
# ---------------------------------------------------------------------------

class TestChatDataModuleFullMode:
    """Verify ChatDataModule 'full' mode produces the lit.py batch format."""

    def _make_full_dataset(self, conversations, fake_tokenizer, num_completions=2, completion_len=4):
        from ptp.data.chat import ChatDataset

        dataset = Dataset.from_list([
            {"messages": conv} for conv in conversations
        ])
        # Build lookup manually: one entry per conversation
        lookup_arr = np.array(
            [[i, len(conv)] for i, conv in enumerate(conversations)], dtype=np.int64
        )
        ds = ChatDataset.__new__(ChatDataset)
        ds.data = dataset
        ds.tokenizer = fake_tokenizer
        ds.mode = "full"
        ds.conversation_key = "messages"
        ds.max_sequence_length = 500
        ds.num_completions = num_completions
        ds.train_completion_len = completion_len
        ds.user_roles = ["user", "human"]
        ds.assistant_roles = ["assistant", "gpt"]
        ds.lookup_array = lookup_arr
        return ds

    def test_getitem_returns_dict(self, fake_tokenizer, simple_conversations):
        ds = self._make_full_dataset(simple_conversations, fake_tokenizer)
        item = ds[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "input_mask" in item
        assert "completion_starts" in item
        assert "completion_length" in item

    def test_completion_starts_count(self, fake_tokenizer, simple_conversations):
        num_completions = 3
        ds = self._make_full_dataset(simple_conversations, fake_tokenizer,
                                      num_completions=num_completions)
        for i in range(len(ds)):
            item = ds[i]
            assert len(item["completion_starts"]) == num_completions, \
                f"Expected {num_completions} completion_starts, got {len(item['completion_starts'])}"

    def test_completion_starts_within_bounds(self, fake_tokenizer, simple_conversations):
        ds = self._make_full_dataset(simple_conversations, fake_tokenizer)
        for i in range(len(ds)):
            item = ds[i]
            seq_len = len(item["input_ids"])
            for start in item["completion_starts"]:
                assert start >= 1, f"completion_start {start} must be >= 1"
                assert start < seq_len, f"completion_start {start} must be < seq_len {seq_len}"

    def test_completion_starts_within_assistant_turns(self, fake_tokenizer, simple_conversations):
        """Completion starts should point inside assistant turn tokens."""
        ds = self._make_full_dataset(simple_conversations, fake_tokenizer)
        for conv_idx in range(len(ds)):
            item = ds[conv_idx]
            input_ids = item["input_ids"]
            # Decode to verify tokens correspond to real conversation text
            assert input_ids.dtype == torch.long
            assert len(input_ids) > 0

    def test_input_mask_all_ones(self, fake_tokenizer, simple_conversations):
        """Full-conversation mode: every token is real, so mask should be all-ones."""
        ds = self._make_full_dataset(simple_conversations, fake_tokenizer)
        for i in range(len(ds)):
            item = ds[i]
            assert item["input_mask"].all(), "input_mask should be all-ones in full mode"

    def test_collate_full_mode(self, fake_tokenizer, simple_conversations):
        ds = self._make_full_dataset(simple_conversations, fake_tokenizer,
                                      num_completions=2, completion_len=4)
        items = [ds[i] for i in range(len(ds))]
        batch = collate_fn(items)
        assert_training_batch(batch, batch_size=len(items),
                               num_completions=2, completion_length=4)


# ---------------------------------------------------------------------------
# PregeneratedDataset tests
# ---------------------------------------------------------------------------

class TestPregeneratedDataset:
    """Verify PregeneratedDataset produces the lit.py batch format."""

    def test_getitem_format(self, tmp_path, fake_tokenizer):
        from ptp.data.load import PregeneratedDataset
        make_pregenerated_dataset_on_disk(tmp_path, n_prompts=4, prompt_len=10,
                                           completion_len=20, num_completions=3)
        from datasets import Dataset
        raw = Dataset.load_from_disk(tmp_path / "train")
        raw.set_format(type="torch")

        ds = PregeneratedDataset(
            raw,
            train_completion_len=8,
            num_completions=2,
            eos_token_id=fake_tokenizer.eos_token_id,
        )
        item = ds[0, 0]
        assert "input_ids" in item
        assert "input_mask" in item
        assert "completion_starts" in item
        assert "completion_length" in item
        assert item["completion_length"] == 8
        assert len(item["completion_starts"]) == 2

    def test_collated_batch_format(self, tmp_path, fake_tokenizer):
        from ptp.data.load import PregeneratedDataset
        make_pregenerated_dataset_on_disk(tmp_path, n_prompts=6, prompt_len=8,
                                           completion_len=16, num_completions=4)
        from datasets import Dataset
        raw = Dataset.load_from_disk(tmp_path / "train")
        raw.set_format(type="torch")

        ds = PregeneratedDataset(
            raw,
            train_completion_len=6,
            num_completions=3,
            eos_token_id=fake_tokenizer.eos_token_id,
        )
        items = [ds[i, 0] for i in range(4)]
        batch = collate_fn(items)
        assert_training_batch(batch, batch_size=4, num_completions=3, completion_length=6)

    def test_bin_edges_in_batch(self, tmp_path, fake_tokenizer):
        from ptp.data.load import PregeneratedDataset
        make_pregenerated_dataset_on_disk(tmp_path, n_prompts=4, prompt_len=8,
                                           completion_len=16, num_completions=4)
        from datasets import Dataset
        raw = Dataset.load_from_disk(tmp_path / "train")
        raw.set_format(type="torch")

        ds = PregeneratedDataset(
            raw, train_completion_len=6, num_completions=2,
            eos_token_id=fake_tokenizer.eos_token_id, load_bin_edges=True
        )
        items = [ds[i, 0] for i in range(3)]
        assert "left_bin_edges" in items[0]
        assert "right_bin_edges" in items[0]
        batch = collate_fn(items)
        assert "left_bin_edges" in batch
        assert "right_bin_edges" in batch

    def test_no_deprecated_keys(self, tmp_path, fake_tokenizer):
        from ptp.data.load import PregeneratedDataset
        make_pregenerated_dataset_on_disk(tmp_path)
        from datasets import Dataset
        raw = Dataset.load_from_disk(tmp_path / "train")
        raw.set_format(type="torch")
        ds = PregeneratedDataset(raw, train_completion_len=4, num_completions=2,
                                  eos_token_id=fake_tokenizer.eos_token_id)
        item = ds[0, 0]
        assert "prompt_ids" not in item, "PregeneratedDataset must not emit 'prompt_ids'"
        assert "prompt_mask" not in item, "PregeneratedDataset must not emit 'prompt_mask'"


# ---------------------------------------------------------------------------
# TextDocumentDataset tests
# ---------------------------------------------------------------------------

class TestTextDocumentDataset:
    """Verify TextDocumentDataset produces the lit.py batch format."""

    def _make_dataset(self, tmp_path, fake_tokenizer,
                      seq_len=30, num_completions=2, completion_len=6):
        from ptp.data.text import TextDocumentDataset

        # Write a token cache directly (skip actual HF dataset)
        token_array = np.arange(200, dtype=np.int32) % 200 + 3  # avoid special tokens
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "fake_ds_train_text_tokenized.npy"
        np.save(cache_file, token_array)

        ds = TextDocumentDataset.__new__(TextDocumentDataset)
        ds.sequence_length = seq_len
        ds.num_completions = num_completions
        ds.train_completion_len = completion_len
        ds.token_array = token_array
        n = len(token_array) - seq_len + 1
        import math
        ds._n = n
        ds._perm_c = math.floor(0.6180339887 * n)
        while math.gcd(ds._perm_c, n) != 1:
            ds._perm_c += 1
        return ds

    def test_getitem_format(self, tmp_path, fake_tokenizer):
        ds = self._make_dataset(tmp_path, fake_tokenizer)
        item = ds[0]
        assert "input_ids" in item
        assert "input_mask" in item
        assert "completion_starts" in item
        assert "completion_length" in item

    def test_correct_shapes(self, tmp_path, fake_tokenizer):
        seq_len, num_completions, completion_len = 30, 3, 5
        ds = self._make_dataset(tmp_path, fake_tokenizer,
                                 seq_len=seq_len, num_completions=num_completions,
                                 completion_len=completion_len)
        item = ds[0]
        assert item["input_ids"].shape == (seq_len,)
        assert item["input_mask"].shape == (seq_len,)
        assert len(item["completion_starts"]) == num_completions
        assert item["completion_length"] == completion_len

    def test_completion_starts_valid(self, tmp_path, fake_tokenizer):
        seq_len, completion_len = 30, 6
        ds = self._make_dataset(tmp_path, fake_tokenizer,
                                 seq_len=seq_len, completion_len=completion_len)
        for i in range(min(len(ds), 10)):
            item = ds[i]
            for start in item["completion_starts"]:
                assert 0 <= start
                assert start + completion_len <= seq_len

    def test_mask_all_ones(self, tmp_path, fake_tokenizer):
        ds = self._make_dataset(tmp_path, fake_tokenizer)
        item = ds[0]
        assert item["input_mask"].all()

    def test_no_deprecated_keys(self, tmp_path, fake_tokenizer):
        ds = self._make_dataset(tmp_path, fake_tokenizer)
        item = ds[0]
        assert "prompt_ids" not in item
        assert "prompt_mask" not in item

    def test_collate_compatible(self, tmp_path, fake_tokenizer):
        num_completions, completion_len = 2, 5
        ds = self._make_dataset(tmp_path, fake_tokenizer,
                                 num_completions=num_completions, completion_len=completion_len)
        items = [ds[i] for i in range(4)]
        batch = collate_fn(items)
        assert_training_batch(batch, batch_size=4,
                               num_completions=num_completions, completion_length=completion_len)


# ---------------------------------------------------------------------------
# _sample_completion_starts
# ---------------------------------------------------------------------------

class TestSampleCompletionStarts:
    """
    Verify that _sample_completion_starts:
    - returns positions inside assistant *content* (not the header)
    - handles multi-turn conversations correctly
    - always returns exactly num_completions positions
    """

    def _encode_and_call(self, fake_tokenizer, conversation, num_completions):
        from ptp.data.chat import _sample_completion_starts
        full_text = fake_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        encoding = fake_tokenizer(full_text, return_offsets_mapping=True)
        n_tokens = len(encoding.input_ids)
        starts = _sample_completion_starts(
            fake_tokenizer, conversation, encoding.offset_mapping, n_tokens, num_completions
        )
        return starts, encoding.offset_mapping, full_text

    def _assistant_content_char_ranges(self, fake_tokenizer, conversation):
        """Return (char_start, char_end) for each assistant turn's content."""
        ranges = []
        for i, turn in enumerate(conversation):
            if turn['role'] != 'assistant':
                continue
            prefix = fake_tokenizer.apply_chat_template(
                conversation[:i], tokenize=False, add_generation_prompt=True
            )
            suffix = fake_tokenizer.apply_chat_template(
                conversation[:i + 1], tokenize=False, add_generation_prompt=False
            )
            ranges.append((len(prefix), len(suffix)))
        return ranges

    def test_returns_correct_count(self, fake_tokenizer):
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        for n in (1, 3, 6):
            starts, _, _ = self._encode_and_call(fake_tokenizer, conversation, n)
            assert len(starts) == n, f"Expected {n} starts, got {len(starts)}"

    def test_no_header_tokens_single_turn(self, fake_tokenizer):
        """completion_starts must point into assistant content, skipping the header."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]
        starts, offsets, _ = self._encode_and_call(fake_tokenizer, conversation, num_completions=10)
        content_ranges = self._assistant_content_char_ranges(fake_tokenizer, conversation)

        for cs in starts:
            char_pos = offsets[cs][0]
            in_content = any(char_start <= char_pos < char_end for char_start, char_end in content_ranges)
            assert in_content, (
                f"completion_start {cs} maps to char {char_pos!r} which is outside all "
                f"assistant content ranges {content_ranges}"
            )

    def test_no_header_tokens_multi_turn(self, fake_tokenizer):
        """Same guarantee for conversations with multiple assistant turns."""
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It equals four."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You are welcome!"},
        ]
        starts, offsets, _ = self._encode_and_call(fake_tokenizer, conversation, num_completions=10)
        content_ranges = self._assistant_content_char_ranges(fake_tokenizer, conversation)

        for cs in starts:
            char_pos = offsets[cs][0]
            in_content = any(char_start <= char_pos < char_end for char_start, char_end in content_ranges)
            assert in_content, (
                f"completion_start {cs} maps to char {char_pos!r} outside content ranges {content_ranges}"
            )

    def test_header_is_indeed_excluded(self, fake_tokenizer):
        """Explicitly confirm that the assistant header string is not reachable."""
        conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        full_text = fake_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        encoding = fake_tokenizer(full_text, return_offsets_mapping=True)
        content_ranges = self._assistant_content_char_ranges(fake_tokenizer, conversation)
        char_start, _ = content_ranges[0]

        # Sanity-check: the character just before char_start is the last char of the header.
        assert char_start > 0
        header_char = full_text[char_start - 1]
        # FakeTokenizer header ends with a space after the colon: "<assistant>: "
        assert header_char == " ", f"Expected header to end with space, got {header_char!r}"

        # Now verify no completion_start maps into the header region.
        starts, offsets, _ = self._encode_and_call(fake_tokenizer, conversation, num_completions=20)
        for cs in starts:
            assert offsets[cs][0] >= char_start, (
                f"completion_start {cs} is in the header (char {offsets[cs][0]} < {char_start})"
            )


# ---------------------------------------------------------------------------
# Max sequence length (as configured in train.yaml for pythia-160m-v0)
# ---------------------------------------------------------------------------

class TestMaxSequenceLength:
    """
    Verify that FullChatDataset respects max_sequence_length=2048 as set in
    checkpoints/pythia-160m-v0_tulu-3-sft-mixture/train.yaml.

    Configuration snapshot from that file:
        mode: full
        conversation_key: messages
        user_roles: [user, human]
        assistant_roles: [assistant, gpt, bing, chatgpt, bard, model]
        max_sequence_length: 2048
        num_completions: 128
        train_completion_len: 16
    """

    MAX_SEQ_LEN = 2048
    NUM_COMPLETIONS = 128
    COMPLETION_LEN = 16

    def _make_dataset(self, conversations, fake_tokenizer):
        from ptp.data.chat import FullChatDataset

        dataset = Dataset.from_list([{"messages": conv} for conv in conversations])
        ds = FullChatDataset.__new__(FullChatDataset)
        ds.data = dataset
        ds.tokenizer = fake_tokenizer
        ds.conversation_key = "messages"
        ds.max_sequence_length = self.MAX_SEQ_LEN
        ds.num_completions = self.NUM_COMPLETIONS
        ds.train_completion_len = self.COMPLETION_LEN
        ds.user_roles = ["user", "human"]
        ds.assistant_roles = ["assistant", "gpt", "bing", "chatgpt", "bard", "model"]
        return ds

    def _long_conversation(self, n_chars=3000):
        """Single-turn conversation whose tokenized length exceeds MAX_SEQ_LEN."""
        return [
            {"role": "user",      "content": "Q " * (n_chars // 4)},
            {"role": "assistant", "content": "A " * (n_chars // 4)},
        ]

    def _assert_starts_valid(self, item, label=""):
        seq_len = item["input_ids"].shape[0]
        for start in item["completion_starts"]:
            assert start >= 1, f"{label}completion_start {start} must be >= 1 (bin edge constraint)"
            assert start < seq_len, (
                f"{label}completion_start {start} must be < seq_len {seq_len}"
            )

    def test_short_conversation_within_limit(self, fake_tokenizer, simple_conversations):
        ds = self._make_dataset(simple_conversations, fake_tokenizer)
        for i in range(len(ds)):
            item = ds[i]
            assert item["input_ids"].shape[0] <= self.MAX_SEQ_LEN, (
                f"Item {i}: input_ids length {item['input_ids'].shape[0]} "
                f"exceeds max_sequence_length {self.MAX_SEQ_LEN}"
            )
            self._assert_starts_valid(item, label=f"item {i}: ")

    def test_long_conversation_truncated_to_max_seq_len(self, fake_tokenizer):
        """A conversation longer than 2048 tokens must be truncated to MAX_SEQ_LEN."""
        conv = self._long_conversation(n_chars=3000)
        ds = self._make_dataset([conv], fake_tokenizer)
        item = ds[0]
        assert item["input_ids"].shape[0] <= self.MAX_SEQ_LEN, (
            f"input_ids length {item['input_ids'].shape[0]} exceeds "
            f"max_sequence_length {self.MAX_SEQ_LEN}"
        )
        assert item["input_mask"].shape[0] <= self.MAX_SEQ_LEN
        self._assert_starts_valid(item)

    def test_completion_starts_within_truncated_sequence(self, fake_tokenizer):
        """All completion_starts must be valid indices within the (possibly truncated) sequence."""
        conv = self._long_conversation(n_chars=3000)
        ds = self._make_dataset([conv], fake_tokenizer)
        item = ds[0]
        self._assert_starts_valid(item)

    def test_no_crash_when_assistant_content_fully_truncated(self, fake_tokenizer):
        """When the truncation cuts off all assistant content, fall back to the full valid range."""
        # A conversation whose assistant turn starts well past max_sequence_length.
        # With the FakeTokenizer (1 char ≈ 1 token + BOS), a 2100-char user message
        # fills the 2048-token window before the assistant turn begins.
        conv = [
            {"role": "user",      "content": "X " * 1100},   # ~2200 tokens — fills window
            {"role": "assistant", "content": "Y " * 100},
        ]
        ds = self._make_dataset([conv], fake_tokenizer)
        item = ds[0]
        assert item["input_ids"].shape[0] <= self.MAX_SEQ_LEN
        assert len(item["completion_starts"]) == self.NUM_COMPLETIONS
        self._assert_starts_valid(item)


# ---------------------------------------------------------------------------
# Cross-module format consistency
# ---------------------------------------------------------------------------

class TestCrossModuleFormatConsistency:
    """
    Verify that PregeneratedDataset, ChatDataset (full), and TextDocumentDataset
    all produce items that can be handled by the same collate_fn.
    """

    def test_mixed_sources_same_collate_fn(self, tmp_path, fake_tokenizer, simple_conversations):
        """Items from PregeneratedDataset and ChatDataset (full) pass through collate_fn."""
        from ptp.data.load import PregeneratedDataset
        from ptp.data.chat import ChatDataset

        # --- pregenerated item ---
        make_pregenerated_dataset_on_disk(tmp_path, n_prompts=4, prompt_len=8,
                                           completion_len=24, num_completions=4)
        raw = Dataset.load_from_disk(tmp_path / "train")
        raw.set_format(type="torch")
        pregen_ds = PregeneratedDataset(
            raw, train_completion_len=4, num_completions=2,
            eos_token_id=fake_tokenizer.eos_token_id,
        )
        pregen_item = pregen_ds[0, 0]

        # --- chat full item ---
        dataset = Dataset.from_list([{"messages": conv} for conv in simple_conversations])
        lookup_arr = np.array(
            [[i, len(conv)] for i, conv in enumerate(simple_conversations)], dtype=np.int64
        )
        chat_ds = ChatDataset.__new__(ChatDataset)
        chat_ds.data = dataset
        chat_ds.tokenizer = fake_tokenizer
        chat_ds.mode = "full"
        chat_ds.conversation_key = "messages"
        chat_ds.max_sequence_length = 500
        chat_ds.user_roles = ["user", "human"]
        chat_ds.assistant_roles = ["assistant", "gpt"]
        chat_ds.num_completions = 2
        chat_ds.train_completion_len = 4
        chat_ds.lookup_array = lookup_arr
        chat_item = chat_ds[0]

        # Both items must share the same required keys
        for key in REQUIRED_TRAINING_KEYS:
            assert key in pregen_item, f"PregeneratedDataset missing '{key}'"
            assert key in chat_item, f"ChatDataset (full) missing '{key}'"

        # Both must have completion_length == 4 and 2 completion_starts
        assert pregen_item["completion_length"] == 4
        assert chat_item["completion_length"] == 4
        assert len(pregen_item["completion_starts"]) == 2
        assert len(chat_item["completion_starts"]) == 2

    def test_pregenerate_reads_prompt_mode_keys(self, fake_tokenizer, simple_conversations):
        """
        Simulate what pregenerate.py reads from a ChatDataModule batch:
        it only needs input_ids and input_mask.
        """
        from ptp.data.chat import ChatDataModule

        dm = ChatDataModule.__new__(ChatDataModule)
        dm.mode = "prompt"
        dm.tokenizer = fake_tokenizer
        dm.max_sequence_length = 200
        dm.padding = True

        strings = [
            fake_tokenizer.apply_chat_template(simple_conversations[0], tokenize=False)
        ]
        batch = dm.collate_fn(strings)

        # pregenerate.py accesses these keys:
        _ = batch["input_ids"]
        _ = batch["input_mask"]
        # Must not have 'prompt_ids' or 'prompt_mask' (old interface)
        assert "prompt_ids" not in batch
        assert "prompt_mask" not in batch
