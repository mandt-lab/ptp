"""
Single-document flex attention mask.

Provides the reference helpers imported by test_flex_attention_mask_multi_doc.py:
  - DEVICE
  - materialise_mask_mod
  - build_reference_mask

Also tests the production make_completion_mask_mod against build_reference_mask
for the single-document case (D=1).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from ptp.attention import make_completion_mask_mod


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Shared helpers (imported by test_flex_attention_mask_multi_doc.py)
# ---------------------------------------------------------------------------

def materialise_mask_mod(
    mask_mod,
    batch_size: int,
    q_len: int,
    kv_len: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Evaluate mask_mod at every (b, q, kv) index → (B, 1, Q, KV) bool tensor."""
    B, Q, KV = batch_size, q_len, kv_len
    b_  = torch.arange(B,  device=device)[:, None, None].expand(B, Q, KV).reshape(-1)
    q_  = torch.arange(Q,  device=device)[None, :, None].expand(B, Q, KV).reshape(-1)
    kv_ = torch.arange(KV, device=device)[None, None, :].expand(B, Q, KV).reshape(-1)
    h_  = torch.zeros_like(b_)
    return mask_mod(b_, h_, q_, kv_).reshape(B, 1, Q, KV)


def build_reference_mask(
    input_ids:          torch.Tensor,  # (B, S) – token ids (unused in mask logic)
    input_mask:         torch.Tensor,  # (B, S) bool – True for real tokens
    completion_starts:  list,          # (B, C) list[list[int]], positions in [0, S)
    completion_length:  int,           # L
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive single-document reference mask.

    Layout
    ------
    Queries : C completions of L tokens each (C*L total query positions)
    KV      : S prompt tokens, then C*L completion tokens

    Mask rules
    ----------
    Prompt branch    : input_mask[b, kv] AND kv < completion_start[b][c]
    Completion branch: same completion block, causal (kv_pos <= q_pos)
    Row validity     : input_mask[b, q_abs]  – padding rows stay all-False

    Returns
    -------
    mask       : (B, 1, C*L, S + C*L) bool
    valid_mask : (B, C, L) bool
    """
    B, S = input_mask.shape
    C    = len(completion_starts[0])
    L    = completion_length
    CL   = C * L
    device = input_mask.device

    mask  = torch.zeros(B, 1, CL, S + CL, dtype=torch.bool, device=device)
    valid = torch.zeros(B, C, L,          dtype=torch.bool, device=device)

    for b in range(B):
        for c in range(C):
            cs = completion_starts[b][c]
            for p in range(L):
                q_row = c * L + p
                q_abs = cs + p
                if q_abs >= S or not input_mask[b, q_abs]:
                    continue
                valid[b, c, p] = True
                # Prompt: real tokens strictly before the completion start
                for kv in range(S):
                    if input_mask[b, kv] and kv < cs:
                        mask[b, 0, q_row, kv] = True
                # Completion: same block, causal
                for kv_p in range(p + 1):
                    mask[b, 0, q_row, S + c * L + kv_p] = True

    return mask, valid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    # (B, S, C, L, doc_lengths)
    (1, 16, 2, 3, [16]),
    (2, 24, 3, 4, [24, 20]),
    (2, 32, 4, 4, [30, 32]),
    (3, 20, 2, 3, [18, 20, 15]),
])
def single_doc_cfg(request):
    B, S, C, L, doc_lengths = request.param
    assert len(doc_lengths) == B
    return B, S, C, L, doc_lengths


def _make_single_doc_batch(B, S, C, L, doc_lengths, device="cpu", seed=0):
    """Build tensors for the production make_completion_mask_mod (D=1 case)."""
    torch.manual_seed(seed)

    # input_ids and input_mask
    input_ids  = torch.randint(3, 1000, (B, S), device=device)
    input_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    for b in range(B):
        input_mask[b, :doc_lengths[b]] = True

    # doc_ids: 0 for real tokens, 1 (=D sentinel) for padding
    doc_ids = torch.ones(B, S, dtype=torch.long, device=device)  # sentinel = 1
    for b in range(B):
        doc_ids[b, :doc_lengths[b]] = 0

    doc_starts  = torch.zeros(B, 1, dtype=torch.long, device=device)
    doc_lengths_t = torch.tensor([[dl] for dl in doc_lengths],
                                 dtype=torch.long, device=device)  # (B, 1)

    # Completion starts: evenly spaced in [1, doc_length - L]
    comp_starts = torch.zeros(B, C, dtype=torch.long, device=device)
    for b in range(B):
        max_start = doc_lengths[b] - L
        assert max_start > 0, "doc too short for completion_length"
        step = max(1, max_start // (C + 1))
        for c in range(C):
            comp_starts[b, c] = min(max(1, 1 + c * step), max_start)

    comp_doc_ids = torch.zeros(B, C, dtype=torch.long, device=device)

    return input_ids, input_mask, doc_ids, doc_starts, doc_lengths_t, comp_starts, comp_doc_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_production_matches_reference(single_doc_cfg):
    """Production make_completion_mask_mod (D=1) must match build_reference_mask."""
    B, S, C, L, doc_lengths = single_doc_cfg
    device = DEVICE

    (input_ids, input_mask, doc_ids, doc_starts,
     doc_lengths_t, comp_starts, comp_doc_ids) = _make_single_doc_batch(
        B, S, C, L, doc_lengths, device=device
    )

    mask_mod = make_completion_mask_mod(
        completion_starts=comp_starts,
        completion_doc_ids=comp_doc_ids,
        doc_ids=doc_ids,
        doc_starts=doc_starts,
        doc_lengths=doc_lengths_t,
        seq_len=S,
        completion_length=L,
    )
    prod = materialise_mask_mod(mask_mod, B, C * L, S + C * L, device)

    starts_list = comp_starts.tolist()
    ref, valid = build_reference_mask(input_ids, input_mask, starts_list, L)

    diff = (prod != ref).sum().item()
    assert diff == 0, (
        f"{diff} entries differ\n"
        f"prod[0,0]:\n{prod[0,0].int()}\nref[0,0]:\n{ref[0,0].int()}"
    )


def test_causal_upper_triangle_empty(single_doc_cfg):
    """With shape='causal', no completion token may attend to a later position."""
    B, S, C, L, doc_lengths = single_doc_cfg
    device = DEVICE

    (_, _, doc_ids, doc_starts, doc_lengths_t,
     comp_starts, comp_doc_ids) = _make_single_doc_batch(
        B, S, C, L, doc_lengths, device=device
    )
    mask_mod = make_completion_mask_mod(
        completion_starts=comp_starts,
        completion_doc_ids=comp_doc_ids,
        doc_ids=doc_ids,
        doc_starts=doc_starts,
        doc_lengths=doc_lengths_t,
        seq_len=S,
        completion_length=L,
    )
    mask = materialise_mask_mod(mask_mod, B, C * L, S + C * L, device)[:, 0]  # (B, CL, S+CL)
    comp = mask[:, :, S:]  # (B, CL, CL)

    for c in range(C):
        block = comp[:, c * L:(c + 1) * L, c * L:(c + 1) * L]  # (B, L, L)
        for q_pos in range(L):
            for kv_pos in range(q_pos + 1, L):
                assert not block[:, q_pos, kv_pos].any(), (
                    f"causal violated: comp={c} q_pos={q_pos} attends kv_pos={kv_pos}"
                )


def test_no_prompt_cross_document_attention(single_doc_cfg):
    """Padding tokens must never appear as KV for any completion query."""
    B, S, C, L, doc_lengths = single_doc_cfg
    device = DEVICE

    (_, input_mask, doc_ids, doc_starts, doc_lengths_t,
     comp_starts, comp_doc_ids) = _make_single_doc_batch(
        B, S, C, L, doc_lengths, device=device
    )
    mask_mod = make_completion_mask_mod(
        completion_starts=comp_starts,
        completion_doc_ids=comp_doc_ids,
        doc_ids=doc_ids,
        doc_starts=doc_starts,
        doc_lengths=doc_lengths_t,
        seq_len=S,
        completion_length=L,
    )
    mask = materialise_mask_mod(mask_mod, B, C * L, S + C * L, device)[:, 0]  # (B, CL, S+CL)
    prompt_part = mask[:, :, :S]  # (B, CL, S)

    for b in range(B):
        pad_cols = ~input_mask[b]  # True where token is padding
        if pad_cols.any():
            assert not prompt_part[b, :, pad_cols].any(), (
                f"b={b}: completion attends to padding tokens"
            )

