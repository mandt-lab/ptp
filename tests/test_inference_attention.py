"""
Unit tests for the speculative-decoding attention mask used in generate().

We verify that make_generate_mask_mod produces the same (Q_LEN × KV_LEN) bool
mask as the dense construction in lit.py generate(), then test structural
invariants and BlockMask creation on CUDA.

Notation
--------
K        : kv_cache.get_seq_length()   – tokens already in cache
P        : prompt_ids.shape[1]          – total prompt length (incl. verify window)
pos      : P - K                        – frontier / verify query tokens
n_verify : number of tokens being verified this step
n_props  : list of proposal counts per depth d
Q_LEN    : pos + sum(n_props)          – new query tokens this step
KV_LEN   : P + sum(n_props)            – total KV (cached + new)

Query layout
------------
  [verify tokens 0..pos-1  |  group-0 proposals  |  group-1 proposals  | …]

KV layout (absolute positions)
-------------------------------
  [0 .. K-1  |  K .. P-1 (verify window)  |  P .. P+sum(n_props)-1 (proposals)]

Mask rules (same as dense construction in lit.py)
-------------------------------------------------
Verify token q (0 ≤ q < pos):
  attend iff  kv ≤ K + q   (standard causal)

Proposal group d, position j within group  (q = pos + sum(n_props[:d]) + j):
  attend iff  kv < P − n_verify + d          ("prefix": up to d accepted verifies)
          OR  (kv ≥ K + midx[d]  AND  kv ≤ K + q)   (same group, causal)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from ptp.attention import make_generate_mask_mod


# ---------------------------------------------------------------------------
# Reference: dense mask as built in lit.py generate()
# ---------------------------------------------------------------------------

def build_generate_dense_mask(
    K: int,
    P: int,
    n_verify: int,
    n_props: list[int],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Reproduce the (Q_LEN, KV_LEN) boolean attention mask from generate().

    Mirrors the logic:
        mask = tril(ones(Q_LEN, KV_LEN), diagonal=K)
        for d, n_prop in enumerate(n_props):
            mask[midx:midx+n_prop, P-n_verify+d : K+midx] = False
            midx += n_prop
    """
    pos        = P - K
    total      = sum(n_props)
    KV_LEN     = P + total
    Q_LEN      = pos + total

    mask = torch.tril(
        torch.ones(Q_LEN, KV_LEN, dtype=torch.bool, device=device),
        diagonal=K,
    )

    midx = pos
    for d, n_prop in enumerate(n_props):
        if n_prop > 0:
            col_start = P - n_verify + d   # first blocked column (inclusive)
            col_end   = K + midx           # last blocked column (exclusive)
            if col_start < col_end:
                mask[midx : midx + n_prop, col_start : col_end] = False
        midx += n_prop

    return mask  # (Q_LEN, KV_LEN) bool


# ---------------------------------------------------------------------------
# Helper: materialise mask_mod → dense bool tensor
# ---------------------------------------------------------------------------

def materialise_generate_mask(
    mask_mod,
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Evaluate mask_mod at every (q, kv) → (Q_LEN, KV_LEN) bool tensor."""
    q_   = torch.arange(Q_LEN,  device=device)[:, None].expand(Q_LEN, KV_LEN).reshape(-1)
    kv_  = torch.arange(KV_LEN, device=device)[None, :].expand(Q_LEN, KV_LEN).reshape(-1)
    zero = torch.zeros_like(q_)
    return mask_mod(zero, zero, q_, kv_).reshape(Q_LEN, KV_LEN)


# ---------------------------------------------------------------------------
# Parametrised correctness: flex mask_mod must match the dense reference
# ---------------------------------------------------------------------------

CONFIGS = [
    # (K,  P,  n_verify,  n_props)
    # First speculative step: nothing cached, no verify tokens yet
    (0,  5,  0, [3, 2, 1, 0]),
    # Warm cache, first speculative step (n_verify = 0)
    (10, 10, 0, [4, 3, 2, 1]),
    # Mid-generation: 3 verify tokens, mixed proposal counts
    (10, 13, 3, [3, 2, 1]),
    # Mid-generation: 5 verify tokens, 4 proposal groups (one empty)
    (20, 25, 5, [4, 3, 2, 1, 0]),
    # Single proposal group
    ( 5,  8, 3, [5]),
    # No proposals, verify-only (pure causal over frontier tokens)
    ( 5, 10, 5, []),
    # Groups with zero proposal counts scattered
    ( 0,  5, 3, [4, 0, 2]),
    # pos = 0 (all prompt tokens already cached, only proposals)
    (10, 10, 3, [3, 2, 1]),
    # Larger sequence
    (50, 55, 4, [5, 4, 3, 2, 1]),
    # n_verify = 0, pos > 0
    ( 3,  8, 0, [2, 2]),
]


@pytest.mark.parametrize("K,P,n_verify,n_props", CONFIGS)
def test_flex_mask_matches_dense(K, P, n_verify, n_props):
    """make_generate_mask_mod must produce the same mask as the dense reference."""
    device = "cpu"
    pos    = P - K
    Q_LEN  = pos + sum(n_props)
    KV_LEN = P + sum(n_props)

    if Q_LEN == 0:
        pytest.skip("empty query sequence – nothing to test")

    dense   = build_generate_dense_mask(K, P, n_verify, n_props, device)
    mask_mod = make_generate_mask_mod(K, P, n_verify, n_props, Q_LEN, KV_LEN, device)
    flex    = materialise_generate_mask(mask_mod, Q_LEN, KV_LEN, device)

    diff = (dense != flex).sum().item()
    assert diff == 0, (
        f"K={K} P={P} n_verify={n_verify} n_props={n_props}: "
        f"{diff} entries differ\n"
        f"dense:\n{dense.int()}\nflex:\n{flex.int()}"
    )


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

def test_verify_tokens_are_strictly_causal():
    """Verify tokens must use pure causal attention (no future peeking)."""
    K, P, n_verify, n_props = 5, 10, 5, [3, 2]
    device = "cpu"
    pos    = P - K
    Q_LEN  = pos + sum(n_props)
    KV_LEN = P + sum(n_props)

    mask_mod = make_generate_mask_mod(K, P, n_verify, n_props, Q_LEN, KV_LEN, device)
    mask     = materialise_generate_mask(mask_mod, Q_LEN, KV_LEN, device)

    for q in range(pos):
        # Must attend to exactly kv = 0 .. K+q
        expected = torch.arange(KV_LEN, device=device) <= (K + q)
        assert (mask[q] == expected).all(), (
            f"Verify token q={q}: expected causal mask up to kv={K+q}, "
            f"got {mask[q].int().tolist()}"
        )


def test_proposal_group_cannot_see_unverified_tokens():
    """
    Group d proposals must NOT attend to verify tokens at depths d..n_verify-1
    (absolute KV positions P-n_verify+d .. P-1).
    """
    K, P, n_verify, n_props = 10, 14, 4, [3, 2, 1]
    device = "cpu"
    pos    = P - K
    Q_LEN  = pos + sum(n_props)
    KV_LEN = P + sum(n_props)

    mask_mod = make_generate_mask_mod(K, P, n_verify, n_props, Q_LEN, KV_LEN, device)
    mask     = materialise_generate_mask(mask_mod, Q_LEN, KV_LEN, device)

    midx = pos
    for d, n_prop in enumerate(n_props):
        if n_prop > 0:
            kv_start = P - n_verify + d  # first unverified verify token for hypothesis d
            kv_end   = P                 # end of verify window
            if kv_start < kv_end:
                blocked = mask[midx : midx + n_prop, kv_start : kv_end]
                assert not blocked.any(), (
                    f"Group {d} attends to unverified tokens [{kv_start}:{kv_end}]: "
                    f"{blocked.int()}"
                )
        midx += n_prop


def test_proposal_group_cannot_see_other_groups():
    """
    Group d proposals must NOT attend to tokens from any other group d' ≠ d.
    (No cross-group attention.)
    """
    K, P, n_verify, n_props = 10, 14, 4, [3, 2, 1]
    device = "cpu"
    pos    = P - K
    Q_LEN  = pos + sum(n_props)
    KV_LEN = P + sum(n_props)

    mask_mod = make_generate_mask_mod(K, P, n_verify, n_props, Q_LEN, KV_LEN, device)
    mask     = materialise_generate_mask(mask_mod, Q_LEN, KV_LEN, device)

    # Build list of (q_start, q_end, kv_start, kv_end) per group
    groups = []
    midx = pos
    for n_prop in n_props:
        groups.append((midx, midx + n_prop, K + midx, K + midx + n_prop))
        midx += n_prop

    for d, (q0, q1, kv0, kv1) in enumerate(groups):
        if q0 == q1:
            continue
        for d2, (_, _, kv0_other, kv1_other) in enumerate(groups):
            if d2 == d or kv0_other == kv1_other:
                continue
            cross = mask[q0:q1, kv0_other:kv1_other]
            assert not cross.any(), (
                f"Group {d} (q={q0}:{q1}) attends to group {d2} "
                f"KV [{kv0_other}:{kv1_other}]:\n{cross.int()}"
            )


def test_proposal_group_is_causal_within_group():
    """Within each group the attention must be causal (no future token peeking)."""
    K, P, n_verify, n_props = 10, 14, 4, [3, 2, 1]
    device = "cpu"
    pos    = P - K
    Q_LEN  = pos + sum(n_props)
    KV_LEN = P + sum(n_props)

    mask_mod = make_generate_mask_mod(K, P, n_verify, n_props, Q_LEN, KV_LEN, device)
    mask     = materialise_generate_mask(mask_mod, Q_LEN, KV_LEN, device)

    midx = pos
    for d, n_prop in enumerate(n_props):
        if n_prop > 1:
            # Extract the (n_prop × n_prop) intra-group block
            block = mask[midx : midx + n_prop, K + midx : K + midx + n_prop]
            for q_pos in range(n_prop):
                for kv_pos in range(q_pos + 1, n_prop):
                    assert not block[q_pos, kv_pos], (
                        f"Group {d}: q_pos={q_pos} attends to future kv_pos={kv_pos}"
                    )
        midx += n_prop


# ---------------------------------------------------------------------------
# CUDA: BlockMask creation and sparsity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="BlockMask requires CUDA")
def test_block_mask_creation_on_cuda():
    """create_block_mask should succeed with make_generate_mask_mod on CUDA."""
    from torch.nn.attention.flex_attention import create_block_mask

    K, P, n_verify, n_props = 10, 14, 4, [3, 2, 1]
    device = "cuda"
    pos    = P - K
    Q_LEN  = pos + sum(n_props)
    KV_LEN = P + sum(n_props)

    mask_mod   = make_generate_mask_mod(K, P, n_verify, n_props, Q_LEN, KV_LEN, device)
    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )
    assert block_mask is not None
    # BlockMask.shape is (B, H, Q_LEN, KV_LEN); check the sequence dims
    assert block_mask.shape[-2] == Q_LEN
    assert block_mask.shape[-1] == KV_LEN
