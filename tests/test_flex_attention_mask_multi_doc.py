"""
Multi-document packing: flex attention mask extension.

Extends the single-document mask (test_flex_attention_mask.py) to handle
D documents concatenated per sequence, each with C completions of length L.

Data pipeline (mirrors the proposed production architecture)
------------------------------------------------------------
1. Dataset  →  make_mock_single_doc_item()
       One item per document.  completion_starts are LOCAL (0-indexed within
       the document).

2. PackingDataset  →  pack_documents(items, doc_slot)
       Concatenates D items into one packed item.  Pads each document to the
       fixed slot size and shifts completion_starts to GLOBAL positions
       (local + d * doc_slot).  Also records doc_lengths.

3. collate_fn  →  collate_packed(packed_items)
       Stacks B packed items into batch tensors.  Computes doc_offsets as
       d * doc_slot (constant across all batch items → torch.compile safe).
       Returns (doc_lengths, doc_offsets, completion_starts) tensors.

Layout
------
Prompt (KV positions 0 .. S-1):
    doc_0 tokens | doc_1 tokens | ... | doc_{D-1} tokens   (total S, padded)
    Document d occupies positions [doc_offsets[b,d], doc_offsets[b,d]+doc_lengths[b,d]).
    doc_offsets[b, d] = d * doc_slot  where  doc_slot = S // D  (fixed for compile).

Query / Completion (Q positions 0 .. D*C*L-1, KV positions S .. S+D*C*L-1):
    [doc_0_comp_0 | doc_0_comp_1 | ... | doc_0_comp_{C-1} |
     doc_1_comp_0 | ...                                    |
     doc_{D-1}_comp_{C-1}]

Mask rules (prepare_nested_batch role)
--------------------------------------
1. Row validity    : q_abs < doc_offsets[b, q_doc] + doc_lengths[b, q_doc]
2. Prompt branch   : kv is in the SAME document AND kv < completion_start
3. Completion branch : same document, same completion block, causal (kv_pos <= q_pos)

Three mask implementations are provided and tested for mutual agreement:

  build_reference_mask_multi_doc_vanilla()
      Constructs the large (B, 1, D*C*L, S+D*C*L) mask by calling the
      existing single-doc build_reference_mask() for each document and
      patching the result into a zero-initialised matrix.  Correctness
      follows directly from the already-tested per-doc implementation.

  build_reference_mask_multi_doc_direct()
      Vectorised, single-pass implementation – same result, faster.

  build_flex_mask_multi_doc()
      Uses make_multi_doc_completion_mask_mod() with flex_attention's
      materialise_mask_mod helper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from test_flex_attention_mask import (
    build_reference_mask,
    materialise_mask_mod,
    DEVICE,
)

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Pipeline layer 1 – Dataset
# ---------------------------------------------------------------------------

def make_mock_single_doc_item(
    doc_length: int,         # actual number of tokens in this document
    num_completions: int,    # C
    completion_length: int,  # L
    seed: int = 42,
) -> dict:
    """
    Simulate one dataset item for a single document.

    completion_starts are LOCAL positions within [1, doc_length - L], exactly
    as a real dataset would produce them.  The collate / packing layer is
    responsible for shifting these to global positions.
    """
    torch.manual_seed(seed)
    max_start = doc_length - completion_length
    assert max_start > 0, (
        f"doc_length={doc_length} too short for completion_length={completion_length}"
    )
    step   = max(1, max_start // (num_completions + 1))
    starts = [max(1, 1 + c * step) for c in range(num_completions)]
    starts = [min(s, max_start) for s in starts]

    return {
        "input_ids":        torch.randint(3, 1000, (doc_length,)),
        "input_mask":       torch.ones(doc_length, dtype=torch.bool),
        "completion_starts": starts,   # LOCAL positions, List[int] length C
        "completion_length": completion_length,
    }


# ---------------------------------------------------------------------------
# Pipeline layer 2 – PackingDataset
# ---------------------------------------------------------------------------

def pack_documents(items: list[dict], doc_slot: int) -> dict:
    """
    Simulate ``PackingDataset.__getitem__``: concatenate D single-doc items
    into one packed item ready for collation.

    Each document is padded to the fixed ``doc_slot`` size and its
    completion_starts are shifted from local to global coordinates:
        global_start = local_start + d * doc_slot

    This is the single place in the pipeline where the offset is applied.
    ``prepare_nested_batch`` (lit.py) then receives globally consistent starts
    and never needs to know the document index.

    Returns a dict with:
        doc_lengths       List[int] length D  – actual token counts per doc
        completion_starts List[List[int]] (D, C)  – GLOBAL positions
        completion_length int
    """
    assert len(set(item["completion_length"] for item in items)) == 1, \
        "All documents in a packed item must share the same completion_length"

    all_starts:  list[list[int]] = []
    all_lengths: list[int]       = []

    for d, item in enumerate(items):
        offset = d * doc_slot
        dl     = len(item["input_ids"])
        assert dl <= doc_slot, (
            f"doc {d}: actual_len={dl} exceeds doc_slot={doc_slot}"
        )
        all_starts.append([s + offset for s in item["completion_starts"]])
        all_lengths.append(dl)

    return {
        "doc_lengths":      all_lengths,   # List[int] length D
        "completion_starts": all_starts,   # List[List[int]] (D, C), GLOBAL
        "completion_length": items[0]["completion_length"],
    }


# ---------------------------------------------------------------------------
# Pipeline layer 3 – collate_fn
# ---------------------------------------------------------------------------

def collate_packed(
    packed_items: list[dict],
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate the updated ``collate_fn`` for packed batches.

    Stacks B packed items produced by ``pack_documents`` into batch tensors.
    doc_offsets are computed as ``d * doc_slot`` – constant for all batch
    items, which is the property that allows ``torch.compile`` to avoid
    retracing when doc_lengths vary.

    Returns
    -------
    doc_lengths       : (B, D)    actual token counts per document
    doc_offsets       : (B, D)    fixed slot starts  d * doc_slot
    completion_starts : (B, D, C) global start positions
    """
    B  = len(packed_items)
    D  = len(packed_items[0]["doc_lengths"])
    C  = len(packed_items[0]["completion_starts"][0])
    # All packed items share the same doc_slot (enforced by PackingDataset config)
    # Infer it from the first item: max doc_length rounded up, but since we don't
    # store it directly we require callers to ensure consistency.  In tests the
    # fixture always passes a consistent doc_slot.
    doc_lengths = torch.tensor(
        [item["doc_lengths"] for item in packed_items],
        dtype=torch.long, device=device,
    )  # (B, D)

    # doc_offsets are constant across the batch: only depends on D and doc_slot.
    # We recover doc_slot from the first item's completion_starts: the offset
    # of document d is d*doc_slot, so max_global_start ≤ D*doc_slot = S.
    # In tests doc_slot is passed explicitly; here we reconstruct it from the
    # completion_starts structure as:  doc_slot = (global_start_doc1 - local_start_doc0).
    # Rather than that fragility, collate simply stores it and computes offsets.
    # For test purposes we infer doc_slot from completion_starts[0][1][0] - completion_starts[0][0][0]
    # is unreliable, so we derive it from doc_lengths.  The max possible position
    # for document d's last token is (d+1)*doc_slot - 1.  Since we padded to
    # doc_slot in pack_documents, doc_slot = ceil(max(doc_lengths[b,d])) is not
    # right either.  Cleanest: derive from the first global start of doc 1.
    # In production, PackingDataset would pass doc_slot to the collator directly.
    # Here we store it on the packed dict to keep collate simple.
    if "doc_slot" in packed_items[0]:
        doc_slot = packed_items[0]["doc_slot"]
    else:
        # Fallback: infer from the offset jump between doc 0 and doc 1 starts.
        # global_start[d=1][c=0] - local_start[d=1][c=0]  == 1 * doc_slot
        # We don't have local starts here, so we rely on the convention that
        # global_starts[d][c] >= d * doc_slot and < (d+1) * doc_slot.
        # The minimum start per doc is off + 1, so doc_slot >= global_starts[1][0].
        # In practice production code passes doc_slot explicitly; this fallback
        # is for tests that already set it consistently via the fixture.
        raise ValueError(
            "packed items must include a 'doc_slot' key; "
            "set it in pack_documents or pass it explicitly."
        )

    doc_offsets = (
        torch.arange(D, dtype=torch.long, device=device) * doc_slot
    ).unsqueeze(0).expand(B, -1)  # (B, D) – same for every batch item

    completion_starts = torch.tensor(
        [item["completion_starts"] for item in packed_items],
        dtype=torch.long, device=device,
    )  # (B, D, C)

    return doc_lengths, doc_offsets, completion_starts


# ---------------------------------------------------------------------------
# Batch factory used by all tests (composes the three pipeline layers)
# ---------------------------------------------------------------------------

def make_mock_batch_multi_doc(
    batch_size: int,
    seq_len: int,            # S  (fixed total length; must be divisible by num_docs)
    num_docs: int,           # D
    num_completions: int,    # C  completions per document
    completion_length: int,  # L
    doc_actual_lengths: list[list[int]] | None = None,  # (B, D); ≤ doc_slot each
    device: str = "cpu",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full pipeline in miniature:
        make_mock_single_doc_item  ×  D  →  pack_documents  →  collate_packed

    Returns ``(doc_lengths, doc_offsets, completion_starts)`` tensors exactly
    as the real collate_fn would hand to ``prepare_nested_batch``.
    """
    assert seq_len % num_docs == 0, "seq_len must be divisible by num_docs"
    doc_slot = seq_len // num_docs

    if doc_actual_lengths is None:
        doc_actual_lengths = [[doc_slot] * num_docs for _ in range(batch_size)]

    packed_items = []
    for b in range(batch_size):
        items = [
            make_mock_single_doc_item(
                doc_actual_lengths[b][d], num_completions, completion_length,
                seed=seed + b * num_docs + d,
            )
            for d in range(num_docs)
        ]
        packed = pack_documents(items, doc_slot)
        packed["doc_slot"] = doc_slot   # carry forward for collate_packed
        packed_items.append(packed)

    return collate_packed(packed_items, device=device)


# ---------------------------------------------------------------------------
# Reference implementation 1 – vanilla patching
# ---------------------------------------------------------------------------

def build_reference_mask_multi_doc_vanilla(
    doc_lengths: torch.Tensor,        # (B, D)  real token counts
    doc_offsets: torch.Tensor,        # (B, D)  must equal d * doc_slot
    completion_starts: torch.Tensor,  # (B, D, C)  absolute positions
    seq_len: int,                     # S
    num_completions: int,             # C
    completion_length: int,           # L
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the multi-doc mask by calling ``build_reference_mask`` for each
    document and stitching the results into a zero matrix.

    Returns
    -------
    attention_mask : (B, 1, D*C*L, S + D*C*L)  bool
    valid_mask     : (B, D, C, L)               bool
    """
    device = doc_lengths.device
    B, D = doc_lengths.shape
    C, L = num_completions, completion_length
    DCL      = D * C * L
    doc_slot = seq_len // D

    big_mask  = torch.zeros(B, 1, DCL, seq_len + DCL, dtype=torch.bool, device=device)
    valid_all = torch.zeros(B, D, C, L,               dtype=torch.bool, device=device)

    for d in range(D):
        # Per-document input_mask: True for real tokens within the slot
        input_mask_d = torch.zeros(B, doc_slot, dtype=torch.bool, device=device)
        for b in range(B):
            input_mask_d[b, : doc_lengths[b, d]] = True

        # Completion starts relative to the document's own coordinate system
        rel_starts = (
            completion_starts[:, d, :] - doc_offsets[:, d : d + 1]
        ).tolist()  # List[List[int]] shape (B, C)

        dummy_ids = torch.zeros(B, doc_slot, dtype=torch.long, device=device)

        # Single-doc mask (always causal): (B, 1, C*L, doc_slot + C*L)
        doc_mask, doc_valid = build_reference_mask(dummy_ids, input_mask_d, rel_starts, L)

        comp_part = doc_mask[:, :, :, doc_slot:]           # (B, 1, C*L, C*L)

        # Embed prompt and completion blocks into the large mask
        q_rows         = slice(d * C * L,           (d + 1) * C * L)
        kv_prompt_cols = slice(d * doc_slot,         (d + 1) * doc_slot)
        kv_comp_cols   = slice(seq_len + d * C * L,  seq_len + (d + 1) * C * L)

        big_mask[:, :, q_rows, kv_prompt_cols] = doc_mask[:, :, :, :doc_slot]
        big_mask[:, :, q_rows, kv_comp_cols]   = comp_part
        valid_all[:, d, :, :]                  = doc_valid   # (B, C, L)

    return big_mask, valid_all


# ---------------------------------------------------------------------------
# Reference implementation 2 – direct vectorised
# ---------------------------------------------------------------------------

def build_reference_mask_multi_doc_direct(
    doc_lengths: torch.Tensor,        # (B, D)
    doc_offsets: torch.Tensor,        # (B, D)
    completion_starts: torch.Tensor,  # (B, D, C)
    seq_len: int,                     # S
    num_completions: int,             # C
    completion_length: int,           # L
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorised single-pass reference.  Produces the same result as the
    vanilla patcher but without the Python loop over documents.

    Returns
    -------
    attention_mask : (B, 1, D*C*L, S + D*C*L)  bool
    valid_mask     : (B, D, C, L)               bool
    """
    device = doc_lengths.device
    B, D = doc_lengths.shape
    C, L = num_completions, completion_length
    S    = seq_len
    DCL  = D * C * L

    # ---- Row descriptors (one entry per Q position) ----
    rows      = torch.arange(DCL, device=device)          # (DCL,)
    row_doc   = rows // (C * L)                           # which document
    row_comp  = (rows % (C * L)) // L                     # which completion in doc
    row_pos   = rows % L                                  # position in completion

    # Gather per-row completion start, doc offset, doc end  → (B, DCL)
    dc_idx         = row_doc * C + row_comp               # index into flattened (D,C)
    starts_flat    = completion_starts.reshape(B, D * C)  # (B, D*C)
    starts_for_row = starts_flat[:, dc_idx]               # (B, DCL)

    doc_off_for_row = doc_offsets[:, row_doc]             # (B, DCL)
    doc_end_for_row = doc_off_for_row + doc_lengths[:, row_doc]  # (B, DCL)

    q_abs   = starts_for_row + row_pos[None, :]           # (B, DCL) absolute position
    q_valid = q_abs < doc_end_for_row                     # (B, DCL)

    # ---- Left (prompt) part : (B, DCL, S) ----
    kv_p      = torch.arange(S, device=device)            # (S,)
    left_mask = (
        (kv_p[None, None, :] >= doc_off_for_row[:, :, None]) &
        (kv_p[None, None, :] <  doc_end_for_row[:, :, None]) &
        (kv_p[None, None, :] <  starts_for_row[:, :, None])
    )                                                      # (B, DCL, S)

    # ---- Right (completion) part : (DCL, DCL) broadcast to (B, DCL, DCL) ----
    kvs      = torch.arange(DCL, device=device)           # (DCL,)
    kv_doc   = kvs // (C * L)
    kv_block = (kvs % (C * L)) // L
    kv_pos_  = kvs % L

    same_doc   = row_doc[:, None]  == kv_doc[None, :]    # (DCL, DCL)
    same_block = row_comp[:, None] == kv_block[None, :]  # (DCL, DCL)
    causal     = kv_pos_[None, :]  <= row_pos[:, None]   # (DCL, DCL)
    right_mask = (same_doc & same_block & causal)[None, :, :].expand(B, -1, -1)

    # ---- Assemble and apply row validity ----
    attention_mask = torch.cat([left_mask, right_mask], dim=-1)   # (B, DCL, S+DCL)
    attention_mask = attention_mask[:, None, :, :]                 # (B, 1, DCL, S+DCL)
    attention_mask = attention_mask & q_valid[:, None, :, None]

    valid_mask = q_valid.reshape(B, D, C, L)
    return attention_mask, valid_mask


# ---------------------------------------------------------------------------
# Flex attention mask_mod
# ---------------------------------------------------------------------------

def make_multi_doc_completion_mask_mod(
    completion_starts: torch.Tensor,  # (B, D, C)  absolute start positions
    doc_offsets: torch.Tensor,        # (B, D)
    doc_lengths: torch.Tensor,        # (B, D)
    seq_len: int,                     # S
    num_completions: int,             # C
    completion_length: int,           # L
):
    """
    Return a ``mask_mod`` for use with ``create_block_mask`` / ``flex_attention``.

    Q   layout : [doc_0_comp_0, ..., doc_{D-1}_comp_{C-1}]  (D*C*L queries)
    KV  layout : [prompt tokens (S), completion tokens (D*C*L)]
    """
    C = num_completions
    L = completion_length
    S = seq_len

    def mask_mod(b, _h, q_idx, kv_idx):
        # ---- Decode query ----
        q_doc   = q_idx // (C * L)
        q_comp  = (q_idx % (C * L)) // L
        q_pos   = q_idx % L
        q_start = completion_starts[b, q_doc, q_comp]
        q_abs   = q_start + q_pos
        q_valid = q_abs < doc_offsets[b, q_doc] + doc_lengths[b, q_doc]

        # ---- Prompt branch (kv_idx < S) ----
        is_prompt = kv_idx < S
        safe_kv   = kv_idx.clamp(max=S - 1)   # safe index even when kv_idx ≥ S
        kv_in_doc = (
            (safe_kv >= doc_offsets[b, q_doc]) &
            (safe_kv <  doc_offsets[b, q_doc] + doc_lengths[b, q_doc])
        )
        prompt_ok = is_prompt & kv_in_doc & (kv_idx < q_start)

        # ---- Completion branch (kv_idx ≥ S) ----
        kv_flat  = (kv_idx - S).clamp(min=0)  # safe offset into completion space
        kv_doc   = kv_flat // (C * L)
        kv_block = (kv_flat % (C * L)) // L
        kv_pos_  = kv_flat % L
        same_slot = (~is_prompt) & (kv_doc == q_doc) & (kv_block == q_comp)
        comp_ok   = same_slot & (kv_pos_ <= q_pos)

        return q_valid & (prompt_ok | comp_ok)

    return mask_mod


def build_flex_mask_multi_doc(
    doc_lengths: torch.Tensor,        # (B, D)
    doc_offsets: torch.Tensor,        # (B, D)
    completion_starts: torch.Tensor,  # (B, D, C)
    seq_len: int,
    num_completions: int,
    completion_length: int,
) -> torch.Tensor:
    """Materialise the flex mask_mod to a dense (B, 1, D*C*L, S+D*C*L) tensor."""
    device = doc_lengths.device
    B = doc_lengths.shape[0]
    D = doc_lengths.shape[1]
    C, L = num_completions, completion_length
    DCL    = D * C * L
    kv_len = seq_len + DCL

    mask_mod = make_multi_doc_completion_mask_mod(
        completion_starts, doc_offsets, doc_lengths, seq_len, C, L,
    )
    return materialise_mask_mod(mask_mod, B, DCL, kv_len, device)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    # (B, S, D, C, L, doc_actual_lengths_or_None)
    # S must be divisible by D; each doc slot = S // D
    (1, 16, 2, 2, 2, None),
    (2, 24, 2, 2, 3, None),
    (2, 24, 2, 2, 3, [[9, 10], [11, 12]]),
    (3, 36, 3, 2, 3, None),
    (2, 48, 3, 2, 4, None),
    (2, 64, 4, 2, 4, None),
    (2, 64, 4, 3, 4, [[12, 10, 14, 16], [16, 16, 16, 16]]),
    (3, 48, 2, 3, 4, [[ 9, 20], [15, 18], [24, 24]]),
])
def multi_doc_cfg(request):
    B, S, D, C, L, raw = request.param
    doc_slot = S // D
    actual   = raw if raw is not None else [[doc_slot] * D for _ in range(B)]
    return B, S, D, C, L, actual


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_vanilla_matches_direct(multi_doc_cfg):
    """
    Vanilla patcher and direct vectorised reference must agree exactly.
    This confirms the direct implementation is correct before we use it
    to validate the flex mask.
    """
    B, S, D, C, L, doc_actual = multi_doc_cfg
    device = DEVICE

    doc_lengths, doc_offsets, completion_starts = make_mock_batch_multi_doc(
        B, S, D, C, L, doc_actual, device=device
    )

    van_mask, van_valid = build_reference_mask_multi_doc_vanilla(
        doc_lengths, doc_offsets, completion_starts, S, C, L
    )
    dir_mask, dir_valid = build_reference_mask_multi_doc_direct(
        doc_lengths, doc_offsets, completion_starts, S, C, L
    )

    assert van_mask.shape == dir_mask.shape, (
        f"Shape mismatch: vanilla={van_mask.shape}, direct={dir_mask.shape}"
    )
    mismatches = (van_mask != dir_mask).sum().item()
    assert mismatches == 0, (
        f"vanilla vs direct: {mismatches} entries differ.\n"
        f"Config: B={B} S={S} D={D} C={C} L={L}\n"
        f"vanilla[0,0]:\n{van_mask[0,0].int()}\n"
        f"direct [0,0]:\n{dir_mask[0,0].int()}"
    )
    assert (van_valid != dir_valid).sum().item() == 0


def test_flex_matches_vanilla(multi_doc_cfg):
    """
    Flex mask_mod (materialised to dense) must exactly match the vanilla
    patched reference mask.
    """
    B, S, D, C, L, doc_actual = multi_doc_cfg
    device = DEVICE

    doc_lengths, doc_offsets, completion_starts = make_mock_batch_multi_doc(
        B, S, D, C, L, doc_actual, device=device
    )

    van_mask, _ = build_reference_mask_multi_doc_vanilla(
        doc_lengths, doc_offsets, completion_starts, S, C, L
    )
    flex_mask = build_flex_mask_multi_doc(
        doc_lengths, doc_offsets, completion_starts, S, C, L
    )

    assert van_mask.shape == flex_mask.shape, (
        f"Shape mismatch: vanilla={van_mask.shape}, flex={flex_mask.shape}"
    )
    mismatches = (van_mask != flex_mask).sum().item()
    assert mismatches == 0, (
        f"vanilla vs flex: {mismatches} entries differ.\n"
        f"Config: B={B} S={S} D={D} C={C} L={L}\n"
        f"vanilla[0,0]:\n{van_mask[0,0].int()}\n"
        f"flex   [0,0]:\n{flex_mask[0,0].int()}"
    )


def test_no_cross_document_completion_attention(multi_doc_cfg):
    """Completion tokens must never attend to completion tokens of another document."""
    B, S, D, C, L, doc_actual = multi_doc_cfg
    if D < 2:
        pytest.skip("Need at least 2 documents")
    device = DEVICE

    doc_lengths, doc_offsets, completion_starts = make_mock_batch_multi_doc(
        B, S, D, C, L, doc_actual, device=device
    )
    mask, _ = build_reference_mask_multi_doc_vanilla(
        doc_lengths, doc_offsets, completion_starts, S, C, L
    )
    comp_part = mask[:, 0, :, S:]  # (B, D*C*L, D*C*L)

    for b in range(B):
        for q_doc in range(D):
            for kv_doc in range(D):
                if q_doc == kv_doc:
                    continue
                q_rows  = slice( q_doc * C * L, ( q_doc + 1) * C * L)
                kv_cols = slice(kv_doc * C * L, (kv_doc + 1) * C * L)
                cross   = comp_part[b, q_rows, kv_cols]
                assert not cross.any().item(), (
                    f"Cross-doc completion attention: "
                    f"batch={b} q_doc={q_doc} kv_doc={kv_doc}\n{cross.int()}"
                )


def test_no_cross_document_prompt_attention(multi_doc_cfg):
    """Completion tokens must never attend to prompt tokens of another document."""
    B, S, D, C, L, doc_actual = multi_doc_cfg
    if D < 2:
        pytest.skip("Need at least 2 documents")
    device = DEVICE

    doc_lengths, doc_offsets, completion_starts = make_mock_batch_multi_doc(
        B, S, D, C, L, doc_actual, device=device
    )
    mask, _ = build_reference_mask_multi_doc_vanilla(
        doc_lengths, doc_offsets, completion_starts, S, C, L
    )
    prompt_part = mask[:, 0, :, :S]  # (B, D*C*L, S)

    for b in range(B):
        for q_doc in range(D):
            q_rows = slice(q_doc * C * L, (q_doc + 1) * C * L)
            for kv_doc in range(D):
                if kv_doc == q_doc:
                    continue
                off  = int(doc_offsets[b, kv_doc].item())
                dl   = int(doc_lengths[b, kv_doc].item())
                cols = torch.arange(off, off + dl, device=device)
                if cols.numel() == 0:
                    continue
                cross = prompt_part[b, q_rows, :][:, cols]
                assert not cross.any().item(), (
                    f"Cross-doc prompt attention: "
                    f"batch={b} q_doc={q_doc} kv_doc={kv_doc}"
                )


def test_causal_within_completion_block(multi_doc_cfg):
    """Within each completion block, the mask must be lower-triangular."""
    B, S, D, C, L, doc_actual = multi_doc_cfg
    device = DEVICE

    doc_lengths, doc_offsets, completion_starts = make_mock_batch_multi_doc(
        B, S, D, C, L, doc_actual, device=device
    )
    mask, valid_mask = build_reference_mask_multi_doc_vanilla(
        doc_lengths, doc_offsets, completion_starts, S, C, L
    )
    comp_part = mask[:, 0, :, S:]  # (B, D*C*L, D*C*L)

    for b in range(B):
        for d in range(D):
            for c in range(C):
                r = (d * C + c) * L
                block = comp_part[b, r : r + L, r : r + L]  # (L, L)
                for i in range(L):
                    for j in range(i + 1, L):
                        if valid_mask[b, d, c, i]:
                            assert not block[i, j].item(), (
                                f"Anti-causal at batch={b} doc={d} comp={c} "
                                f"row={i} col={j}"
                            )


def test_no_cross_completion_attention_within_doc(multi_doc_cfg):
    """Within a document, completions must not attend to each other's blocks."""
    B, S, D, C, L, doc_actual = multi_doc_cfg
    if C < 2:
        pytest.skip("Need at least 2 completions per document")
    device = DEVICE

    doc_lengths, doc_offsets, completion_starts = make_mock_batch_multi_doc(
        B, S, D, C, L, doc_actual, device=device
    )
    mask, _ = build_reference_mask_multi_doc_vanilla(
        doc_lengths, doc_offsets, completion_starts, S, C, L
    )
    comp_part = mask[:, 0, :, S:]  # (B, D*C*L, D*C*L)

    for b in range(B):
        for d in range(D):
            for qc in range(C):
                for kc in range(C):
                    if qc == kc:
                        continue
                    qr = slice((d * C + qc) * L, (d * C + qc + 1) * L)
                    kc_ = slice((d * C + kc) * L, (d * C + kc + 1) * L)
                    cross = comp_part[b, qr, kc_]
                    assert not cross.any().item(), (
                        f"Cross-completion within doc: "
                        f"batch={b} doc={d} q_comp={qc} k_comp={kc}"
                    )


def test_no_padding_prompt_attention(multi_doc_cfg):
    """Completions must not attend to padding tokens (past doc_lengths) in any slot."""
    B, S, D, C, L, doc_actual = multi_doc_cfg
    device = DEVICE
    doc_slot = S // D

    doc_lengths, doc_offsets, completion_starts = make_mock_batch_multi_doc(
        B, S, D, C, L, doc_actual, device=device
    )
    mask, _ = build_reference_mask_multi_doc_vanilla(
        doc_lengths, doc_offsets, completion_starts, S, C, L
    )
    prompt_part = mask[:, 0, :, :S]  # (B, D*C*L, S)

    for b in range(B):
        for d in range(D):
            off = int(doc_offsets[b, d].item())
            dl  = int(doc_lengths[b, d].item())
            # Padding positions within the slot: [off+dl, off+doc_slot)
            pad_start = off + dl
            pad_end   = off + doc_slot
            if pad_start >= pad_end:
                continue
            pad_cols = torch.arange(pad_start, pad_end, device=device)
            attended  = prompt_part[b, :, pad_cols]
            assert not attended.any().item(), (
                f"Attention to padding: batch={b} doc={d} "
                f"pad_cols=[{pad_start},{pad_end})"
            )


def test_packing_globalises_starts(multi_doc_cfg):
    """
    pack_documents must shift every completion start from local (0-indexed
    within the document) to global (d * doc_slot + local).

    This is the single arithmetic invariant the PackingDataset must maintain.
    prepare_nested_batch receives only global starts and never reconstructs
    the per-document offset itself.
    """
    B, S, D, C, L, doc_actual = multi_doc_cfg
    doc_slot = S // D

    for b in range(B):
        items = [
            make_mock_single_doc_item(
                doc_actual[b][d], C, L, seed=b * D + d
            )
            for d in range(D)
        ]
        local_starts  = [item["completion_starts"] for item in items]  # (D, C)
        packed        = pack_documents(items, doc_slot)
        global_starts = packed["completion_starts"]                     # (D, C)

        for d in range(D):
            offset = d * doc_slot
            for c in range(C):
                expected = local_starts[d][c] + offset
                actual   = global_starts[d][c]
                assert actual == expected, (
                    f"batch={b} doc={d} comp={c}: "
                    f"global={actual} != local={local_starts[d][c]} + offset={offset}"
                )


def test_batch_shapes_consistent():
    """
    Two batches with different doc_actual_lengths must produce masks of the
    same shape – the torch.compile requirement.
    """
    S, D, C, L = 32, 2, 2, 3
    device = DEVICE

    dl_a, do_a, cs_a = make_mock_batch_multi_doc(
        2, S, D, C, L, [[8, 12], [16, 14]], device=device, seed=0
    )
    dl_b, do_b, cs_b = make_mock_batch_multi_doc(
        2, S, D, C, L, [[16, 16], [10, 14]], device=device, seed=1
    )

    m_a, _ = build_reference_mask_multi_doc_vanilla(dl_a, do_a, cs_a, S, C, L)
    m_b, _ = build_reference_mask_multi_doc_vanilla(dl_b, do_b, cs_b, S, C, L)
    assert m_a.shape == m_b.shape, f"Shape changed: {m_a.shape} vs {m_b.shape}"

    f_a = build_flex_mask_multi_doc(dl_a, do_a, cs_a, S, C, L)
    f_b = build_flex_mask_multi_doc(dl_b, do_b, cs_b, S, C, L)
    assert f_a.shape == f_b.shape, f"Flex shape changed: {f_a.shape} vs {f_b.shape}"


def test_create_block_mask_runs():
    """Smoke test: create_block_mask accepts the multi-doc mask_mod."""
    B, S, D, C, L = 2, 32, 2, 2, 3
    device = DEVICE

    doc_lengths, doc_offsets, completion_starts = make_mock_batch_multi_doc(
        B, S, D, C, L, device=device
    )
    DCL      = D * C * L
    mask_mod = make_multi_doc_completion_mask_mod(
        completion_starts, doc_offsets, doc_lengths, S, C, L
    )
    block_mask = create_block_mask(
        mask_mod, B=B, H=None, Q_LEN=DCL, KV_LEN=S + DCL, device=device
    )
    assert block_mask.shape == (B, 1, DCL, S + DCL), block_mask.shape


# ---------------------------------------------------------------------------
# Standalone demo  –  python tests/test_flex_attention_mask_multi_doc.py
# ---------------------------------------------------------------------------

def _demo():
    B, S, D, C, L = 2, 24, 2, 2, 3
    doc_actual = [[8, 10], [12, 12]]
    doc_slot   = S // D
    device     = DEVICE

    print(f"\n=== Multi-doc demo: B={B} S={S} D={D} C={C} L={L} doc_slot={doc_slot} ===")
    print(f"doc_actual_lengths = {doc_actual}")

    # ---- Layer 1: Dataset ------------------------------------------------
    print("\n--- Layer 1: Dataset items (local completion_starts) ---")
    all_items = []
    for b in range(B):
        batch_items = [
            make_mock_single_doc_item(doc_actual[b][d], C, L, seed=b * D + d)
            for d in range(D)
        ]
        all_items.append(batch_items)
        print(f"  batch {b}: local_starts = "
              f"{[item['completion_starts'] for item in batch_items]}")

    # ---- Layer 2: PackingDataset -----------------------------------------
    print("\n--- Layer 2: pack_documents (local → global starts) ---")
    packed_items = []
    for b in range(B):
        packed = pack_documents(all_items[b], doc_slot)
        packed["doc_slot"] = doc_slot
        packed_items.append(packed)
        print(f"  batch {b}: global_starts = {packed['completion_starts']}, "
              f"doc_lengths = {packed['doc_lengths']}")

    # ---- Layer 3: collate_fn --------------------------------------------
    print("\n--- Layer 3: collate_packed ---")
    doc_lengths, doc_offsets, cs = collate_packed(packed_items, device=device)
    print(f"  doc_lengths:\n{doc_lengths}")
    print(f"  doc_offsets:\n{doc_offsets}")
    print(f"  completion_starts:\n{cs}")

    # ---- Mask building (prepare_nested_batch role) ----------------------
    van_mask, van_valid = build_reference_mask_multi_doc_vanilla(
        doc_lengths, doc_offsets, cs, S, C, L
    )
    dir_mask, _  = build_reference_mask_multi_doc_direct(
        doc_lengths, doc_offsets, cs, S, C, L
    )
    flex_mask = build_flex_mask_multi_doc(doc_lengths, doc_offsets, cs, S, C, L)

    def _row_label(q):
        d    = q // (C * L)
        comp = (q % (C * L)) // L
        pos  = q % L
        return f"Q{d}.c{comp}.{pos}"

    def _print_mask(m, S, D, C, L):
        """Print mask with X/space, doc-index header, and │ at the prompt/completion boundary."""
        DCL      = D * C * L
        doc_slot = S // D
        # Header: one digit per column = document index
        header = ""
        for kv in range(S + DCL):
            if kv < S:
                header += str(kv // doc_slot)
            else:
                header += str((kv - S) // (C * L))
        label_w = 11
        print(" " * label_w + header)
        print(" " * label_w + "─" * S + "┬" + "─" * DCL)
        for q in range(DCL):
            row = ""
            for kv in range(S + DCL):
                row += "X" if m[q, kv].item() else " "
            print(f"{_row_label(q):<{label_w}}{row[:S]}│{row[S:]}")

    print(f"\n--- Masks ({D*C*L} rows × {S + D*C*L} cols) ---")
    print("    Columns: prompt KV (left of │) | completion KV (right of │)")
    print("    Digit = document index,  X = attended,  space = masked")
    for b in range(B):
        print(f"\n  batch {b}  valid_mask: {van_valid[b].int().tolist()}")
        print("  [causal]")
        _print_mask(van_mask[b, 0], S, D, C, L)
        print(f"  direct match: {(van_mask[b,0]==dir_mask[b,0]).all().item()}  "
              f"flex match: {(van_mask[b,0]==flex_mask[b,0]).all().item()}")

    total = (van_mask != flex_mask).sum().item()
    print(f"\nTotal mismatches (vanilla vs flex): {total}")
    assert total == 0, "Demo FAILED"
    print("Demo PASSED.")


if __name__ == "__main__":
    _demo()
