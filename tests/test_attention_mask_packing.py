"""
Integration test: packing pipeline → attention mask.

Production path under test:
  MockBaseDataset          (token IDs + get_metadata with spans)
      ↓  PackingDataset    (BFD grouping, proportional completion sampling)
      ↓  packed_collate_fn (stack into batch)
      ↓  make_completion_mask_mod  (flex_attention mask_mod)
      ↓  materialise               (dense bool tensor for comparison)

Compared against a naive reference that builds the mask entry-by-entry from
first principles with no shared code.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from torch.utils.data import Dataset

from ptp.attention import make_completion_mask_mod
from ptp.data.packing import PackingDataset, packed_collate_fn


# ---------------------------------------------------------------------------
# Naive reference mask  (no dependency on production code)
# ---------------------------------------------------------------------------

def naive_mask(
    doc_ids:            torch.Tensor,   # (B, S)
    doc_starts:         torch.Tensor,   # (B, D)
    doc_lengths:        torch.Tensor,   # (B, D)
    completion_starts:  torch.Tensor,   # (B, N)
    completion_doc_ids: torch.Tensor,   # (B, N)
    seq_len:            int,
    completion_length:  int,
) -> torch.Tensor:
    """Build (B, 1, N*L, S+N*L) bool mask token-by-token."""
    B, N = completion_starts.shape
    L    = completion_length
    S    = seq_len
    NL   = N * L

    mask = torch.zeros(B, 1, NL, S + NL, dtype=torch.bool)
    for b in range(B):
        for q in range(NL):
            comp_n  = q // L
            q_pos   = q % L
            q_doc   = completion_doc_ids[b, comp_n].item()
            q_start = completion_starts[b, comp_n].item()
            doc_end = doc_starts[b, q_doc].item() + doc_lengths[b, q_doc].item()
            if doc_lengths[b, q_doc].item() == 0:
                continue
            if q_start + q_pos >= doc_end:
                continue

            for kv in range(S + NL):
                if kv < S:
                    if doc_ids[b, kv].item() == q_doc and kv < q_start:
                        mask[b, 0, q, kv] = True
                else:
                    flat   = kv - S
                    kv_n   = flat // L
                    kv_pos = flat % L
                    if kv_n != comp_n:
                        continue
                    mask[b, 0, q, kv] = kv_pos <= q_pos
    return mask


# ---------------------------------------------------------------------------
# Materialise helper
# ---------------------------------------------------------------------------

def materialise(mask_mod, B, Q, KV, device='cpu'):
    b_  = torch.arange(B,  device=device)[:, None, None].expand(B, Q, KV).reshape(-1)
    q_  = torch.arange(Q,  device=device)[None, :, None].expand(B, Q, KV).reshape(-1)
    kv_ = torch.arange(KV, device=device)[None, None, :].expand(B, Q, KV).reshape(-1)
    h_  = torch.zeros_like(b_)
    return mask_mod(b_, h_, q_, kv_).reshape(B, 1, Q, KV)


# ---------------------------------------------------------------------------
# Mock base dataset
# ---------------------------------------------------------------------------

class MockBaseDataset(Dataset):
    """
    Minimal dataset: token IDs of specified lengths and full-sequence spans.
    Implements get_metadata() so PackingDataset can use it.
    """

    def __init__(self, doc_lengths: list[int], seed: int = 0):
        self.doc_lengths = doc_lengths
        self.seed        = seed

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        torch.manual_seed(self.seed + idx)
        return torch.randint(3, 1000, (self.doc_lengths[idx],))

    def get_metadata(self) -> list[tuple[int, list[tuple[int, int]]]]:
        # Completion region = entire document (minus position 0)
        return [(l, [(1, l)]) for l in self.doc_lengths]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    # (B, N, L, doc_lengths, total_length)
    (1, 4, 3, [10, 12, 8, 14],  30),
    (2, 6, 3, [10, 12, 14, 16, 9, 11, 13, 8], 30),
    (2, 8, 4, [16, 18, 12, 20, 14, 22, 10, 16], 40),
    (3, 6, 3, [15, 12, 18, 14, 10, 16, 11, 13, 9], 35),
])
def pack_cfg(request, tmp_path):
    B, N, L, doc_lens, total_length = request.param
    base   = MockBaseDataset(doc_lens)
    packed = PackingDataset(
        base,
        max_sequence_length=total_length,
        num_completions=N,
        completion_length=L,
        cache_dir=str(tmp_path),
    )
    # Use first B groups (or fewer if dataset is smaller)
    n_items = min(B, len(packed))
    batch   = packed_collate_fn([packed[i] for i in range(n_items)])
    return n_items, N, L, total_length, packed, batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mask_matches_naive(pack_cfg):
    """Production mask_mod materialised to dense must match the naive reference."""
    B, N, L, S, packed, batch = pack_cfg
    NL = N * L

    mask_mod = make_completion_mask_mod(
        completion_starts=batch["completion_starts"],
        completion_doc_ids=batch["completion_doc_ids"],
        doc_ids=batch["doc_ids"],
        doc_starts=batch["doc_starts"],
        doc_lengths=batch["doc_lengths"],
        seq_len=S,
        completion_length=L,
    )
    prod = materialise(mask_mod, B, NL, S + NL)
    ref  = naive_mask(
        doc_ids=batch["doc_ids"],
        doc_starts=batch["doc_starts"],
        doc_lengths=batch["doc_lengths"],
        completion_starts=batch["completion_starts"],
        completion_doc_ids=batch["completion_doc_ids"],
        seq_len=S,
        completion_length=L,
    )

    diff = (prod != ref).sum().item()
    assert diff == 0, (
        f"{diff} entries differ\n"
        f"prod[0,0]:\n{prod[0,0].int()}\nref[0,0]:\n{ref[0,0].int()}"
    )


def test_doc_ids_coverage(pack_cfg):
    """Real tokens → valid doc_id in [0, D); padding → sentinel D."""
    B, N, L, S, packed, batch = pack_cfg
    D    = batch["doc_starts"].shape[1]
    ids  = batch["doc_ids"]     # (B, S)
    mask = batch["input_mask"]  # (B, S)
    assert (ids[mask]  <  D).all(), "Real token has out-of-range doc_id"
    assert (ids[~mask] == D).all(), "Padding token does not have sentinel doc_id"


def test_comp_starts_within_docs(pack_cfg):
    """Every comp_start must lie within its document's token range."""
    B, N, L, S, packed, batch = pack_cfg
    for b in range(B):
        for n in range(N):
            d      = batch["completion_doc_ids"][b, n].item()
            start  = batch["completion_starts"][b, n].item()
            ds     = batch["doc_starts"][b, d].item()
            dl     = batch["doc_lengths"][b, d].item()
            assert ds <= start < ds + dl, (
                f"b={b} n={n}: comp_start={start} not in doc [{ds}, {ds+dl})"
            )


def test_no_cross_doc_prompt_attention(pack_cfg):
    """A completion may not attend to prompt tokens from a different document."""
    B, N, L, S, packed, batch = pack_cfg
    D  = batch["doc_starts"].shape[1]
    NL = N * L
    if D < 2:
        pytest.skip("Need D >= 2")

    mask_mod = make_completion_mask_mod(
        completion_starts=batch["completion_starts"],
        completion_doc_ids=batch["completion_doc_ids"],
        doc_ids=batch["doc_ids"],
        doc_starts=batch["doc_starts"],
        doc_lengths=batch["doc_lengths"],
        seq_len=S,
        completion_length=L,
    )
    mask = materialise(mask_mod, B, NL, S + NL)[:, 0]  # (B, NL, S+NL)

    for b in range(B):
        for n in range(N):
            q_doc  = batch["completion_doc_ids"][b, n].item()
            q_rows = slice(n * L, (n + 1) * L)
            for other_doc in range(D):
                if other_doc == q_doc:
                    continue
                other_mask = (batch["doc_ids"][b] == other_doc)  # (S,)
                if other_mask.any():
                    prompt_cols = mask[b, q_rows, :S]  # (L, S)
                    assert not prompt_cols[:, other_mask].any(), (
                        f"Cross-doc prompt attention b={b} n={n} q_doc={q_doc} kv_doc={other_doc}"
                    )


def test_causal_within_completion(pack_cfg):
    """With shape='causal', completion KV can only attend to earlier positions."""
    B, N, L, S, packed, batch = pack_cfg
    NL = N * L

    mask_mod = make_completion_mask_mod(
        completion_starts=batch["completion_starts"],
        completion_doc_ids=batch["completion_doc_ids"],
        doc_ids=batch["doc_ids"],
        doc_starts=batch["doc_starts"],
        doc_lengths=batch["doc_lengths"],
        seq_len=S,
        completion_length=L,
    )
    mask = materialise(mask_mod, B, NL, S + NL)[:, 0]  # (B, NL, S+NL)
    comp_part = mask[:, :, S:]   # (B, NL, NL)

    for n in range(N):
        q_rows = slice(n * L, (n + 1) * L)
        kv_same_block = slice(S + n * L, S + (n + 1) * L)
        block = mask[:, q_rows, kv_same_block]   # (B, L, L)
        for q_pos in range(L):
            for kv_pos in range(L):
                if kv_pos > q_pos:
                    assert not block[:, q_pos, kv_pos].any(), (
                        f"n={n} causal violated: q_pos={q_pos} attends kv_pos={kv_pos}"
                    )


def test_bfd_packs_tightly(tmp_path):
    """BFD should fit more documents than naive sequential packing for short docs."""
    # All docs are length 10, total_length 30 → exactly 3 per sequence
    doc_lens = [10] * 9   # 3 perfect groups
    base     = MockBaseDataset(doc_lens)
    packed   = PackingDataset(base, max_sequence_length=30, num_completions=2,
                              completion_length=3, cache_dir=str(tmp_path))
    assert len(packed) == 3
    assert packed.max_docs == 3
