"""
flex_attention mask_mod for parallel-sampling (PTP) training.

Terminology
-----------
B  : batch size
D  : max documents packed per sequence
N  : completions per packed sequence  (= num_completions, fixed for torch.compile)
L  : completion length (fixed)
S  : packed sequence length (fixed)

Query layout  (N * L tokens, flat):
    [comp_0_pos_0 | … | comp_0_pos_{L-1} | comp_1_pos_0 | … | comp_{N-1}_pos_{L-1}]

KV layout  (S + N*L tokens):
    [prompt tokens (S)] [completion tokens (N*L)]

Inputs
------
completion_starts   (B, N)  global prompt position where each completion begins
completion_doc_ids  (B, N)  which packed document each completion belongs to
doc_ids       (B, S)  per-prompt-token document index; D for padding tokens
doc_starts    (B, D)  start position of each document in the packed sequence
doc_lengths   (B, D)  real token count per document (0 for padding slots)
"""

import torch


def make_completion_mask_mod(
    completion_starts:   torch.Tensor,           # (B, N)
    completion_doc_ids:  torch.Tensor,           # (B, N)
    doc_ids:       torch.Tensor,           # (B, S)
    doc_starts:    torch.Tensor,           # (B, D)
    doc_lengths:   torch.Tensor,           # (B, D)
    seq_len:       int,                    # S
    completion_length: int,                # L
):
    """
    Return a mask_mod callable for torch.nn.attention.flex_attention.
    """
    L       = completion_length
    S       = seq_len

    def mask_mod(b, _h, q_idx, kv_idx):
        # ---- Decode query ----
        comp_n  = q_idx // L                    # which completion (0 … N-1)
        q_pos   = q_idx % L                     # position within window (0 … L-1)
        q_doc   = completion_doc_ids[b, comp_n]       # which document
        q_start = completion_starts[b, comp_n]        # global prompt position
        doc_end = doc_starts[b, q_doc] + doc_lengths[b, q_doc]
        q_valid = (q_start + q_pos) < doc_end   # False for padding doc-slots

        # ---- Prompt branch  (kv_idx in [0, S)) ----
        is_prompt = kv_idx < S
        safe_kv   = kv_idx.clamp(max=S - 1)    # avoid OOB when kv_idx >= S
        kv_in_doc = doc_ids[b, safe_kv] == q_doc
        prompt_ok = is_prompt & kv_in_doc & (kv_idx < q_start)

        # ---- Completion branch  (kv_idx in [S, S + N*L)) ----
        kv_flat   = (kv_idx - S).clamp(min=0)
        kv_n      = kv_flat // L
        kv_pos    = kv_flat % L
        same_slot = (~is_prompt) & (kv_n == comp_n)
        comp_ok   = same_slot & (kv_pos <= q_pos)

        return q_valid & (prompt_ok | comp_ok)

    return mask_mod


def fill_generate_mask_buffers(
    q_cutoff:      torch.Tensor,   # (Q_LEN,) long – filled in-place
    q_group_start: torch.Tensor,   # (Q_LEN,) long – filled in-place
    q_group_end:   torch.Tensor,   # (Q_LEN,) long – filled in-place
    K:       int,                  # kv_cache.get_seq_length()
    P:       int,                  # prompt_ids.shape[1]
    n_verify: int,                 # tokens_to_verify - tokens_to_fill
    n_props:  list,                # proposal counts per depth d
    kv_len:   int,                 # KV_LEN = P + sum(n_props)
    device,
) -> None:
    """
    Fill *in-place* the three per-query lookup tensors used by the
    speculative-decoding mask_mod.

    Separate from make_generate_mask_mod so that callers can pre-allocate
    the buffers once and reuse the same mask_mod closure across loop
    iterations.  Reusing the same Python function object prevents
    torch.compile (WrappedFlexAttention) from recompiling the Triton kernel
    every step.

    The caller is responsible for pre-allocating tensors of length Q_LEN =
    (P - K) + sum(n_props) before the generate loop.  In fixed_tokens mode
    this equals 2 * tokens_per_student_call throughout.
    """
    pos = P - K

    if pos > 0:
        arange_pos = torch.arange(pos, dtype=torch.long, device=device)
        q_cutoff[:pos]      = K + arange_pos   # K+0, K+1, …, K+pos-1
        q_group_start[:pos] = kv_len           # sentinel: in_group is always False
        q_group_end[:pos]   = K + arange_pos   # causal upper bound = cutoff

    midx = pos
    for d, n_prop in enumerate(n_props):
        if n_prop > 0:
            arange_prop = torch.arange(midx, midx + n_prop, dtype=torch.long, device=device)
            q_cutoff     [midx: midx + n_prop] = P - n_verify + d - 1  # inclusive; may be -1
            q_group_start[midx: midx + n_prop] = K + midx
            q_group_end  [midx: midx + n_prop] = K + arange_prop       # K + q_idx
        midx += n_prop


def make_generate_mask_mod(
    K: int,                  # kv_cache.get_seq_length()
    P: int,                  # prompt_ids.shape[1]
    n_verify: int,           # tokens_to_verify - tokens_to_fill
    n_props:  list,          # proposal counts per depth d
    q_len:    int,           # Q_LEN = (P - K) + sum(n_props)
    kv_len:   int,           # KV_LEN = P + sum(n_props)
    device,
):
    """
    Allocate per-query lookup tensors, fill them, and return a mask_mod
    closure for the speculative-decoding step in generate().

    Query layout
    ------------
      q ∈ [0, pos)           – verify / frontier prompt tokens  (pure causal)
      q ∈ [pos, Q_LEN)       – proposal groups d=0,1,…

    KV layout (absolute positions 0..KV_LEN-1)
    -------------------------------------------
      [0 .. K-1]              – tokens already in the KV cache
      [K .. P-1]              – verify window  (P - K = pos tokens)
      [P .. KV_LEN-1]         – new proposal tokens

    Attention rule for proposal group d at query q = midx + j
    ----------------------------------------------------------
      Attend iff
        kv  <  P - n_verify + d          ("prefix": prompt up to d accepted verifies)
        OR
        kv ≥ K + midx  AND  kv ≤ K + q  (same group, causal)

    This precisely mirrors the dense mask construction in generate():
        mask = tril(ones(Q_LEN, KV_LEN), diagonal=K)
        for d, n_prop in enumerate(n_props):
            mask[midx:midx+n_prop, P-n_verify+d : K+midx] = False
            midx += n_prop

    Implementation
    --------------
    We precompute two per-query tensors to avoid any Python-level branching
    inside the mask_mod (which must be torch.compile-friendly):

      q_cutoff[q]      – inclusive upper bound for the "prefix" region
                         verify token q: K + q  (pure causal)
                         proposal group d: P - n_verify + d - 1
      q_group_start[q] – absolute KV start of this query's proposal group
                         verify token q: kv_len  (sentinel → group never matches)
                         proposal group d: K + midx_d

    Note: for repeated calls with the same Q_LEN, prefer pre-allocating the
    buffers once and using fill_generate_mask_buffers to update them in-place,
    then reusing the same closure.  See GenerateSession for the recommended
    pattern.
    """
    q_cutoff      = torch.empty(q_len, dtype=torch.long, device=device)
    q_group_start = torch.empty(q_len, dtype=torch.long, device=device)
    q_group_end   = torch.empty(q_len, dtype=torch.long, device=device)

    fill_generate_mask_buffers(
        q_cutoff, q_group_start, q_group_end,
        K, P, n_verify, n_props, kv_len, device,
    )

    def mask_mod(_b, _h, q_idx, kv_idx):
        # Prefix region: kv up to the accepted-verify cutoff
        before_cutoff = kv_idx <= q_cutoff[q_idx]
        # Same-group causal region
        in_group = (kv_idx >= q_group_start[q_idx]) & (kv_idx <= q_group_end[q_idx])
        return before_cutoff | in_group

    return mask_mod


def make_ar_mask_mod(doc_ids: torch.Tensor):
    """
    Causal mask with document isolation for the packed AR forward pass.

    doc_ids[b, i] is the document index of prompt token i.
    The padding sentinel (max_docs) is never equal to any real document index,
    so padding tokens are automatically blocked from attending to real tokens.
    """
    def mask_mod(b, _h, q_idx, kv_idx):
        same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
        causal   = kv_idx <= q_idx
        return same_doc & causal
    return mask_mod
