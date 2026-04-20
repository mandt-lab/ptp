import hashlib
import random
import torch
import numpy as np
from typing import Any, List, Literal, Optional, Union
from lightning.pytorch import LightningDataModule
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm
import os

from ptp.data.collate import collate_fn as base_collate_fn
from ptp.data.packing import PackingDataset, packed_collate_fn
from ptp.data.utils import duplicate_avoiding_randint

CACHE_DIR = 'data_cache'

# Role names that map to "user" / "assistant" in the standard chat format.
# Datasets use wildly different conventions; these defaults cover the most common ones.
DEFAULT_USER_ROLES: List[str] = ["user", "human"]
DEFAULT_ASSISTANT_ROLES: List[str] = ["assistant", "gpt", "bing", "chatgpt", "bard", "model"]


def _roles_cache_tag(user_roles: List[str], assistant_roles: List[str]) -> str:
    """Short hash so different role configs don't share lookup caches."""
    key = f"{sorted(user_roles)}|{sorted(assistant_roles)}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


class MaskCreationError(Exception):
    pass


class ChatDataset(torch.utils.data.Dataset):
    """Base class for chat datasets. Handles data loading, tokenizer setup, and role normalisation."""

    def __init__(self, dataset_name, split, cache_dir, tokenizer, max_sequence_length: int,
                 conversation_keys: Union[str, List[str]] = 'data',
                 user_roles: List[str] | None = None,
                 assistant_roles: List[str] | None = None):
        if dataset_name.endswith(".json"):
            self.data = load_dataset("json", data_files=dataset_name, split=split, cache_dir=cache_dir)
        else:
            self.data = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.conversation_keys: List[str] = [conversation_keys] if isinstance(conversation_keys, str) else list(conversation_keys)
        if not self.conversation_keys:
            raise ValueError("conversation_keys must be a non-empty string or list of strings")
        self._resolved_key: Optional[str] = None
        self.max_sequence_length = max_sequence_length
        self.user_roles: List[str] = user_roles if user_roles is not None else DEFAULT_USER_ROLES
        self.assistant_roles: List[str] = assistant_roles if assistant_roles is not None else DEFAULT_ASSISTANT_ROLES

    def _resolve_conversation_keys(self, check_consistency: bool = False) -> str:
        """
        Resolve the dataset conversation field name.

        If multiple candidate keys are configured, exactly one key must be used
        consistently across entries. When ``check_consistency`` is True, this is
        enforced by scanning the dataset and raising on mixed usage.
        """
        if self._resolved_key is not None:
            return self._resolved_key

        observed_key: Optional[str] = None
        missing_count = 0

        for idx, item in enumerate(self.data):
            present = [k for k in self.conversation_keys if k in item and item[k] is not None]
            if len(present) > 1:
                raise ValueError(
                    f"Entry {idx} contains multiple conversation keys {present}. "
                    f"Expected only one of {self.conversation_keys}."
                )
            if len(present) == 0:
                missing_count += 1
                continue

            key = present[0]
            if observed_key is None:
                observed_key = key
            elif check_consistency and key != observed_key:
                raise ValueError(
                    f"Inconsistent conversation key usage detected: saw '{observed_key}' and '{key}'. "
                    f"Expected exactly one key from {self.conversation_keys} across the dataset."
                )

        if observed_key is None:
            raise ValueError(
                f"None of the configured conversation keys {self.conversation_keys} were found in dataset entries."
            )

        if check_consistency and missing_count > 0:
            raise ValueError(
                f"Found {missing_count} entries without any conversation key from {self.conversation_keys}."
            )

        self._resolved_key = observed_key
        return observed_key

    def _convert_to_chat_format(self, conversation_data: Any) -> List[dict]:
        """
        Normalise a conversation from any common dataset format into a list of
        {"role": "user"|"assistant", "content": ...} dicts.

        Role name resolution order:
          1. "from" key (ShareGPT / Alpaca style)
          2. "role" key (OpenAI / HuggingFace style)
        Role name mapping uses self.user_roles and self.assistant_roles.
        System messages are silently skipped.
        Alternating string lists are treated as user/assistant turns by index parity.
        """
        conversation = []
        for i, message in enumerate(conversation_data):
            if isinstance(message, dict):
                raw_role = message.get('from') or message.get('role', '')
                content = message.get('value') or message.get('content', '')

                if raw_role in self.assistant_roles:
                    role = "assistant"
                elif raw_role in self.user_roles:
                    role = "user"
                elif raw_role == "system":
                    continue  # skip system messages
                else:
                    raise ValueError(
                        f"Unknown role '{raw_role}' in conversation data. "
                        f"Expected one of user_roles={self.user_roles} or "
                        f"assistant_roles={self.assistant_roles}."
                    )
                conversation.append({"role": role, "content": content})
            else:
                # Alternating string list: even index → user, odd index → assistant
                role = "user" if i % 2 == 0 else "assistant"
                conversation.append({"role": role, "content": message})
        return conversation


class PromptChatDataset(ChatDataset):
    """
    Returns the conversation text up to (but not including) the next assistant response,
    optionally with the assistant-turn header appended. Used to feed prompts to a teacher
    for pregeneration. ``__getitem__`` returns a formatted string.
    """

    lookup_array: np.ndarray

    def __init__(self, dataset_name, split, cache_dir, tokenizer, max_sequence_length: int,
                 add_assistant_prompt: bool = False, conversation_keys: Union[str, List[str]] = 'data',
                 rebuild_cache: bool = False, one_user_message_per_chat: bool = False,
                 user_roles: List[str] | None = None,
                 assistant_roles: List[str] | None = None):
        self.add_assistant_prompt = add_assistant_prompt
        self.one_user_message_per_chat = one_user_message_per_chat
        super().__init__(
            dataset_name, split, cache_dir, tokenizer, max_sequence_length,
            conversation_keys=conversation_keys,
            user_roles=user_roles, assistant_roles=assistant_roles,
        )

        roles_tag = _roles_cache_tag(self.user_roles, self.assistant_roles)
        lookup_cache_file = os.path.join(
            cache_dir,
            dataset_name.replace('/', '_')
            + f'_{split}_{max_sequence_length}_prompt_{roles_tag}_chat_lookup.npy'
        )
        if os.path.exists(lookup_cache_file) and not rebuild_cache:
            print(f"Loading chat lookup from {lookup_cache_file}")
            self.lookup_array = np.load(lookup_cache_file)
        else:
            print(f"Creating chat lookup and saving to {lookup_cache_file}")
            os.makedirs(cache_dir, exist_ok=True)
            self.lookup_array = np.array(self._build_lookup())
            np.save(lookup_cache_file, self.lookup_array)
            raise MaskCreationError("Chat dataset lookup file was just created, please rerun to load it.")

    def _build_lookup(self) -> list:
        """One entry per prefix ending with a user message."""
        conversation_key = self._resolve_conversation_keys(check_consistency=True)
        # Phase 1: apply chat template to all candidates (no tokenization yet)
        entries = []  # (chat_idx, segment_length, formatted_text)
        for chat_idx, item in enumerate(tqdm(self.data, desc="Applying chat template (prompt mode)")):
            conversation = self._convert_to_chat_format(item[conversation_key])
            for segment_length in range(1, len(conversation) + 1):
                chat_until_here = conversation[:segment_length]
                if chat_until_here[-1]['role'] != 'user':
                    continue
                if self.add_assistant_prompt:
                    chat_until_here = chat_until_here + [{"role": "assistant", "content": ""}]
                formatted = self.tokenizer.apply_chat_template(
                    chat_until_here, tokenize=False, add_generation_prompt=False
                )
                entries.append((chat_idx, segment_length, formatted))
                if self.one_user_message_per_chat:
                    break

        # Phase 2: batch tokenize all candidates at once
        lengths = [len(ids) for ids in self.tokenizer([e[2] for e in entries])['input_ids']]

        # Phase 3: apply length filter
        if self.one_user_message_per_chat:
            return [
                [chat_idx, seg_len]
                for (chat_idx, seg_len, _), length in zip(entries, lengths)
                if length <= self.max_sequence_length
            ]

        lookup_list = []
        current_chat = -1
        over_limit = False
        for (chat_idx, seg_len, _), length in zip(entries, lengths):
            if chat_idx != current_chat:
                current_chat = chat_idx
                over_limit = False
            if over_limit:
                continue
            lookup_list.append([chat_idx, seg_len])
            if length > self.max_sequence_length:
                over_limit = True
        return lookup_list

    def __len__(self):
        return len(self.lookup_array)

    def __getitem__(self, index):
        chat_idx, segment_length = self.lookup_array[index]
        item = self.data[int(chat_idx)]
        conversation_key = self._resolve_conversation_keys()
        conversation = self._convert_to_chat_format(item[conversation_key])
        segment = conversation[:segment_length]
        if self.add_assistant_prompt:
            segment = segment + [{"role": "assistant", "content": ""}]
        return self.tokenizer.apply_chat_template(
            segment, tokenize=False, add_generation_prompt=False
        )


def _find_completion_spans(
    tokenizer,
    conversation: List[dict],
    offsets: list,
    n_tokens: int,
) -> List[tuple]:
    """
    Return ``[(token_start, token_end), ...]`` for every assistant turn in *conversation*
    that falls within the (possibly truncated) token sequence.

    ``offsets`` is the ``offset_mapping`` from a tokenizer call with
    ``return_offsets_mapping=True``.  Position 0 is excluded (no left context).
    Falls back to ``[(1, n_tokens)]`` when all assistant content is truncated.
    """
    import bisect
    # Build a sorted array of char-starts for fast binary search
    char_starts = [cs for cs, _ in offsets]

    spans = []
    for i, turn in enumerate(conversation):
        if turn['role'] != 'assistant':
            continue
        prefix_text = tokenizer.apply_chat_template(
            conversation[:i], tokenize=False, add_generation_prompt=True
        )
        suffix_text = tokenizer.apply_chat_template(
            conversation[:i + 1], tokenize=False, add_generation_prompt=False
        )
        char_start = len(prefix_text)
        char_end   = len(suffix_text)
        idx_start = bisect.bisect_left(char_starts, char_start)
        if idx_start >= n_tokens:
            continue  # entire turn is beyond the truncated sequence
        token_start = max(idx_start, 1)  # position 0 has no left bin edge
        idx_end = bisect.bisect_left(char_starts, char_end)
        token_end = idx_end if idx_end < n_tokens else n_tokens
        if token_start >= token_end:
            continue
        spans.append((token_start, token_end))
    if not spans:
        spans = [(1, n_tokens)]
    return spans


class FullChatDataset(ChatDataset):
    """
    Tokenizes conversations and exposes assistant-turn spans for packing.

    ``get_metadata()`` returns ``(doc_length, [(span_start, span_end), ...])`` for
    every document and is cached to disk.  ``__getitem__`` returns only the token-ID
    tensor; completion sampling is delegated to PackingDataset.
    """

    def __init__(self, dataset_name, split, cache_dir, tokenizer, max_sequence_length: int,
                 conversation_keys: Union[str, List[str]] = 'data',
                 user_roles: List[str] | None = None,
                 assistant_roles: List[str] | None = None,
                 **_kwargs):
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            conversation_keys=conversation_keys,
            user_roles=user_roles,
            assistant_roles=assistant_roles,
        )
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return token IDs for document *index* (truncated to max_sequence_length)."""
        item = self.data[index]
        conversation_keys = self._resolve_conversation_keys()
        conversation = self._convert_to_chat_format(item[conversation_keys])
        full_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        ids = self.tokenizer(full_text)['input_ids']
        return torch.tensor(ids[:self.max_sequence_length], dtype=torch.long)

    def get_metadata(self) -> List[tuple]:
        """
        Return ``[(doc_length, [(span_start, span_end), ...]), ...]`` for all documents.

        Result is cached to ``cache_dir`` keyed by dataset, split, tokenizer name,
        chat template, and conversation keys — but NOT max_sequence_length, so the
        same cache is reusable across different sequence length settings.
        Truncation to max_sequence_length is applied after loading.
        """
        conversation_key = self._resolve_conversation_keys(check_consistency=True)
        import hashlib, os, pickle
        tag = hashlib.md5("|".join([
            self.dataset_name, self.split,
            self.tokenizer.name_or_path,
            str(self.tokenizer.chat_template or ""),
            ",".join(self.conversation_keys),
        ]).encode()).hexdigest()[:12]
        cache_file = os.path.join(self.cache_dir, f"chat_metadata_{tag}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                raw = pickle.load(f)
        else:
            print(f"Scanning completion spans ({len(self.data)} docs) …")
            from tqdm import tqdm

            # Single batched loop: Phase 1 (template rendering) and Phase 3 (span mapping)
            # run per-document; Phase 2 (tokenization) is batched to amortise Rust overhead.
            import bisect
            BATCH_SIZE = 512
            raw = []
            data_iter = iter(self.data)
            with tqdm(total=len(self.data), desc="Metadata", smoothing=0.1) as pbar:
                while True:
                    # --- Phase 1: collect a batch (template rendering + char spans) ---
                    batch_texts      = []
                    batch_char_spans = []
                    for item in data_iter:
                        conv = self._convert_to_chat_format(item[conversation_key])
                        batch_texts.append(self.tokenizer.apply_chat_template(
                            conv, tokenize=False, add_generation_prompt=False
                        ))
                        char_spans = []
                        for i, turn in enumerate(conv):
                            if turn['role'] != 'assistant':
                                continue
                            prefix = self.tokenizer.apply_chat_template(
                                conv[:i], tokenize=False, add_generation_prompt=True
                            )
                            suffix = self.tokenizer.apply_chat_template(
                                conv[:i + 1], tokenize=False, add_generation_prompt=False
                            )
                            char_spans.append((len(prefix), len(suffix)))
                        batch_char_spans.append(char_spans)
                        if len(batch_texts) == BATCH_SIZE:
                            break

                    if not batch_texts:
                        break

                    # --- Phase 2: batch tokenize ---
                    encodings = self.tokenizer(batch_texts, return_offsets_mapping=True)

                    # --- Phase 3: map char boundaries → token spans via bisect ---
                    for char_spans, ids, offsets in zip(
                        batch_char_spans, encodings['input_ids'], encodings['offset_mapping']
                    ):
                        n_tokens = len(ids)
                        tok_starts = [cs for cs, _ in offsets]
                        spans = []
                        for char_start, char_end in char_spans:
                            idx_start = bisect.bisect_left(tok_starts, char_start)
                            if idx_start >= n_tokens:
                                continue
                            token_start = max(idx_start, 1)
                            idx_end = bisect.bisect_left(tok_starts, char_end)
                            token_end = idx_end if idx_end < n_tokens else n_tokens
                            if token_start >= token_end:
                                continue
                            spans.append((token_start, token_end))
                        if not spans:
                            spans = [(1, n_tokens)]
                        raw.append((n_tokens, spans))

                    pbar.update(len(batch_texts))

            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(raw, f)

        # Apply max_sequence_length truncation on the way out
        result = []
        for n_tokens, spans in raw:
            n = min(n_tokens, self.max_sequence_length)
            truncated_spans = [(s, min(e, n)) for s, e in spans if s < n]
            if not truncated_spans:
                truncated_spans = [(1, n)]
            result.append((n, truncated_spans))
        return result


class ChatDataModule(LightningDataModule):
    """
    DataLoader wrapper for chat datasets.

    mode="prompt":
        Uses PromptChatDataset. Each batch item is tokenized text up to (optionally
        including) the assistant turn header. Batch keys: input_ids, input_mask.
        Intended for feeding to the teacher during pregeneration.

    mode="full":
        Uses FullChatDataset. Each batch item is the full tokenized conversation with
        completion_starts pointing to token positions within assistant turns.
        Batch keys: input_ids, input_mask, completion_starts, completion_length.
        Intended for direct training via lit.py (no pregeneration step).
    """

    def __init__(self, dataset_name: str, tokenizer: Union[str, PreTrainedTokenizerBase], max_sequence_length: int,
                 mode: Literal["prompt", "full"] = "prompt",
                 one_user_message_per_chat: bool = False,
                 train_shuffle: bool = True, splits: List[str] | None = None,
                 chat_template: Optional[str] = None,
                 add_assistant_prompt: bool = False,
                 conversation_keys: Union[str, List[str]] = 'data',
                 padding: bool = True, cache_dir: str = CACHE_DIR, rebuild_cache: bool = False,
                 num_completions: int = 4, train_completion_len: int = 16,
                 user_roles: List[str] | None = None,
                 assistant_roles: List[str] | None = None,
                 fallback_val_split: int | float = 1000,
                 **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        if splits is None:
            splits = ['train', 'valid', 'test']
        self.splits = splits
        self.kwargs = kwargs
        tok: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.tokenizer = tok
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"Tokenizer for '{getattr(tokenizer, 'name_or_path', tokenizer)}' has no "
                f"chat_template. Add a `chat_template` field under `data` in your config. "
                f"See https://huggingface.co/docs/transformers/chat_templating"
            )
        self.mode = mode
        self.padding = padding
        self.max_sequence_length = max_sequence_length
        self.one_user_message_per_chat = one_user_message_per_chat
        self.add_assistant_prompt = add_assistant_prompt
        self.conversation_keys = conversation_keys
        self.train_shuffle = train_shuffle
        self.cache_dir = cache_dir
        self.rebuild_cache = rebuild_cache
        self.num_completions = num_completions
        self.train_completion_len = train_completion_len
        self.user_roles = user_roles if user_roles is not None else DEFAULT_USER_ROLES
        self.assistant_roles = assistant_roles if assistant_roles is not None else DEFAULT_ASSISTANT_ROLES
        self.fallback_val_split = fallback_val_split

    def collate_fn(self, batch):
        if self.mode == "full":
            return packed_collate_fn(batch)
        tokenizer_out = self.tokenizer(batch, return_tensors='pt', padding=self.padding)
        return {
            'input_ids': tokenizer_out['input_ids'][..., :self.max_sequence_length],
            'input_mask': tokenizer_out['attention_mask'][..., :self.max_sequence_length],
        }

    def _make_raw_dataset(self, split: str) -> ChatDataset:
        shared = dict(
            dataset_name=self.dataset_name,
            split=split,
            cache_dir=self.cache_dir,
            tokenizer=self.tokenizer,
            max_sequence_length=self.max_sequence_length,
            conversation_keys=self.conversation_keys,
            user_roles=self.user_roles,
            assistant_roles=self.assistant_roles,
        )
        if self.mode == "prompt":
            return PromptChatDataset(
                **shared,
                rebuild_cache=self.rebuild_cache,
                add_assistant_prompt=self.add_assistant_prompt,
                one_user_message_per_chat=self.one_user_message_per_chat,
            )
        return FullChatDataset(**shared)

    def _pack(self, raw_ds: ChatDataset):
        return PackingDataset(
            raw_ds,
            max_sequence_length=self.max_sequence_length,
            num_completions=self.num_completions,
            completion_length=self.train_completion_len,
            cache_dir=self.cache_dir,
        )

    def _make_dataset(self, split: str) -> ChatDataset:
        raw = self._make_raw_dataset(split)
        if self.mode == "full":
            return self._pack(raw)
        return raw

    def setup(self, stage: Any = None) -> None:
        datasets = {}
        any_raised = False

        for split in self.splits:
            try:
                datasets[split] = self._make_dataset(split)
            except MaskCreationError as e:
                any_raised = e
            except ValueError as e:
                print(f"Warning: failed to load split '{split}': {e}")
                datasets[split] = None

        if any_raised is not False:
            raise any_raised

        self.train_dataset = datasets.get("train")
        self.val_dataset = datasets.get("valid")
        self.test_dataset = datasets.get("test")

        if self.val_dataset is None and self.test_dataset is not None:
            self.val_dataset = self.test_dataset
            print("No validation data found, using test data for validation")
        if self.val_dataset is None and self.train_dataset is not None:
            val_size = self.fallback_val_split
            raw_train = self._make_raw_dataset("train")
            n = len(raw_train)
            if isinstance(val_size, float):
                val_size = int(n * val_size)
            val_size = min(val_size, n)
            from ptp.data.utils import TupleAccessSubset
            random.seed(42)
            val_indices = sorted(random.sample(range(n), val_size))
            val_indices_set = set(val_indices)
            train_indices = [i for i in range(n) if i not in val_indices_set]
            val_raw = TupleAccessSubset(raw_train, val_indices)
            train_raw = TupleAccessSubset(raw_train, train_indices)
            if self.mode == "full":
                self.val_dataset = self._pack(val_raw)
                self.train_dataset = self._pack(train_raw)
            else:
                self.val_dataset = val_raw
                self.train_dataset = train_raw
            print(f"No validation data found, reserving random {val_size} train examples for validation")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset not available")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.val_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.test_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )
