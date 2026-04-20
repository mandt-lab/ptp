"""Shared fixtures for data-format tests."""
import sys
from pathlib import Path
import pytest

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class _Encoding(dict):
    """Dict subclass with attribute access, matching HuggingFace BatchEncoding."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class FakeTokenizer:
    """
    Minimal tokenizer that maps characters to integer IDs.
    Each character → its ASCII ordinal (capped at vocab_size-1).
    Supports apply_chat_template with a simple Jinja-less format.
    """

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    vocab_size = 256

    def __call__(self, text_or_list, return_tensors=None, padding=False,
                 return_offsets_mapping=False, **kwargs):
        if isinstance(text_or_list, list):
            all_ids = [self._encode(t) for t in text_or_list]
            if padding:
                max_len = max(len(ids) for ids in all_ids)
                attention_masks = []
                padded = []
                for ids in all_ids:
                    pad_len = max_len - len(ids)
                    attention_masks.append([1] * len(ids) + [0] * pad_len)
                    padded.append(ids + [self.pad_token_id] * pad_len)
                all_ids = padded
            else:
                attention_masks = [[1] * len(ids) for ids in all_ids]
            if return_tensors == "pt":
                import torch
                return _Encoding({
                    "input_ids": torch.tensor(all_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                })
            return _Encoding({"input_ids": all_ids, "attention_mask": attention_masks})
        # single string
        ids = self._encode(text_or_list)
        enc = _Encoding({"input_ids": ids, "attention_mask": [1] * len(ids)})
        if return_offsets_mapping:
            # BOS token has no character span; token k (k>=1) covers character k-1.
            enc["offset_mapping"] = [(0, 0)] + [(i, i + 1) for i in range(len(text_or_list))]
        if return_tensors == "pt":
            import torch
            enc["input_ids"] = torch.tensor([ids], dtype=torch.long)
            enc["attention_mask"] = torch.tensor([[1] * len(ids)], dtype=torch.long)
        return enc

    def _encode(self, text: str):
        # Simple character-level encoding; include BOS
        return [self.bos_token_id] + [min(ord(c), self.vocab_size - 1) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        chars = []
        for i in ids:
            if skip_special_tokens and i in (self.bos_token_id, self.eos_token_id, self.pad_token_id):
                continue
            chars.append(chr(i) if 32 <= i < 128 else "?")
        return "".join(chars)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        """Minimal chat template: <role>: content\n for each message."""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                parts.append(f"<{role}>: {content}\n")
            else:
                # Empty content (assistant header)
                parts.append(f"<{role}>: ")
        if add_generation_prompt:
            parts.append("<assistant>: ")
        text = "".join(parts)
        if tokenize:
            return self._encode(text)
        return text

    @property
    def chat_template(self):
        return None

    @chat_template.setter
    def chat_template(self, value):
        pass  # Ignored in tests


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def simple_conversations():
    """A few short conversations for testing."""
    return [
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help you?"},
        ],
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It equals four."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You are welcome!"},
        ],
        [
            {"role": "user", "content": "Tell me a story"},
            {"role": "assistant", "content": "Once upon a time in a distant land there lived a hero."},
        ],
    ]
