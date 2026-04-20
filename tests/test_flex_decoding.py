"""
Isolation test for the flex_decoding kernel failure observed during generate().

Background
----------
error.log shows that `p_generate checkpoints/qwen3-8b_tulu-3-sft-mixture/`
crashes on the first user prompt with:

    NoValidChoicesError: No choices exist for backend.
    target: flex_attention
    ...create_flex_decoding_kernel...

`ptp_train` over the same checkpoint succeeds.  The two code paths differ in:

  Training  – AR forward: full sequence, no KV cache, Q_LEN == KV_LEN.
              `flex_attention` (not `flex_decoding`) is selected by inductor
              because Q_LEN is not small relative to KV_LEN.
  Inference – generate() speculative step: Q_LEN << KV_LEN (KV cache present).
              Inductor selects `flex_decoding` because Q_LEN < BLOCK_M (64).
              `flex_decoding` fails with `NoValidChoicesError`.

Execution path in inference
---------------------------
  generate() → model.inference_forward()
    → self.model(inputs_embeds, attention_mask=block_mask, past_key_values=cache)
    → Qwen3Attention.forward()
    → flex_attention_forward() [transformers]
    → compile_friendly_flex_attention()  ← WrappedFlexAttention singleton
         = torch.compile(flex_attention)  # dynamic=None (PyTorch 2.7+)
    → flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True,
                     scale=..., return_lse=True)
    → inductor lowers to create_flex_decoding_kernel  (Q_LEN < BLOCK_M=64)
    → NoValidChoicesError

Why does training NOT fail?
---------------------------
In the AR forward pass, Q_LEN == KV_LEN (full packed sequence, no cache).
With Q_LEN = KV_LEN ≥ BLOCK_M, inductor selects the standard flex_attention
kernel (not flex_decoding), which does not exhibit this bug.

Reproducing without loading Qwen3-8B
-------------------------------------
This test calls `compile_friendly_flex_attention` (the exact singleton used by
transformers) directly with tensors that match the shapes and strides produced
by the Qwen3-8B model.  The Qwen3-8B checkpoint is not required.

Config derived from
    checkpoints/qwen3-8b_tulu-3-sft-mixture/train.yaml

    model_id  : Qwen/Qwen3-8B  (32 query heads, 8 KV heads, head_dim=128)
    precision : bf16-mixed

Running
-------
    pytest tests/test_flex_decoding.py -v        # expect 1 pass, 2 xfail
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers.integrations.flex_attention import (
    WrappedFlexAttention,
    compile_friendly_flex_attention,
)

from ptp.attention import make_ar_mask_mod, make_generate_mask_mod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="flex_attention requires CUDA (bfloat16 + Triton kernels)",
)


@pytest.fixture(autouse=True)
def reset_flex_attention_singleton():
    """
    Reset the WrappedFlexAttention singleton and dynamo compilation cache
    before every test so each test compiles flex_attention independently.
    Without this, later tests may reuse a kernel compiled for an earlier
    test's shapes, masking failures.
    """
    torch._dynamo.reset()
    WrappedFlexAttention._instance = None
    WrappedFlexAttention._is_flex_compiled = False
    WrappedFlexAttention._compiled_flex_attention = None
    yield
    torch._dynamo.reset()
    WrappedFlexAttention._instance = None
    WrappedFlexAttention._is_flex_compiled = False
    WrappedFlexAttention._compiled_flex_attention = None

# ---------------------------------------------------------------------------
# Qwen3-8B GQA dimensions (read directly off the shapes in error.log)
# ---------------------------------------------------------------------------
N_Q_HEADS  = 32          # num_attention_heads
N_KV_HEADS = 8           # num_key_value_heads  (GQA ratio = 4)
HEAD_DIM   = 128         # head_dim
DEVICE     = torch.device("cuda")
DTYPE      = torch.bfloat16    # "bf16-mixed" from train.yaml
SCALE      = HEAD_DIM ** -0.5  # default attention scale


def _make_qkv_like_model(q_len: int, kv_len: int):
    """
    Build Q, K, V with the memory layout produced by Qwen3's QKV projection.

    The projection computes hidden → [B, S, H, D] then transposes to [B, H, S, D].
    This gives Q a non-standard stride pattern:
        stride[head] = HEAD_DIM  (NOT q_len * HEAD_DIM)
        stride[seq]  = N_Q_HEADS * HEAD_DIM

    This layout is what torch inductor sees in the generate() call and is
    visible in error.log:
        FixedLayout stride = [4096*s37, 128, 4096, 1]

    K/V come from torch.cat([past_kv, new_kv]) inside DynamicCache, producing
    a contiguous tensor.
    """
    # Q: [B, S, H, D] in memory, viewed as [B, H, S, D]
    q_bshd = torch.randn(1, q_len, N_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    q = q_bshd.transpose(1, 2)   # view: [1, N_Q_HEADS, q_len, HEAD_DIM]
    # Confirm the non-standard strides matching error.log FixedLayout:
    assert q.stride() == (q_len * N_Q_HEADS * HEAD_DIM,
                          HEAD_DIM,
                          N_Q_HEADS * HEAD_DIM,
                          1)

    # K/V: contiguous [B, N_KV_HEADS, kv_len, HEAD_DIM] (DynamicCache concat result)
    k = torch.randn(1, N_KV_HEADS, kv_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(1, N_KV_HEADS, kv_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    return q, k, v


# ---------------------------------------------------------------------------
# Test 1 – Training path (AR forward): Q_LEN == KV_LEN, make_ar_mask_mod
#           Inductor selects flex_attention kernel (not flex_decoding).
#           Must PASS – this is the training path that works.
# ---------------------------------------------------------------------------

def test_training_ar_flex_attention_succeeds():
    """
    Replicate the AR forward pass used during training.

    In training, the AR pass processes the full packed sequence with no KV
    cache (Q_LEN == KV_LEN).  With Q_LEN == KV_LEN ≥ BLOCK_M, inductor
    selects the standard flex_attention Triton kernel.

    Calls compile_friendly_flex_attention (training=True) to match the exact
    execution path of the transformer model during training.
    """
    seq_len = 128   # abbreviated from train.yaml's max_sequence_length=2048
    B = 1

    doc_ids  = torch.zeros(B, seq_len, dtype=torch.int32, device=DEVICE)
    mask_mod = make_ar_mask_mod(doc_ids)
    block_mask = create_block_mask(
        mask_mod, B=B, H=None,
        Q_LEN=seq_len, KV_LEN=seq_len,
        device=DEVICE,
    )

    q, k, v = _make_qkv_like_model(seq_len, seq_len)

    with torch.no_grad():
        # training=True matches the AR forward pass in training
        out, _lse = compile_friendly_flex_attention(
            q, k, v,
            block_mask=block_mask,
            enable_gqa=True,
            scale=SCALE,
            return_lse=True,
            training=True,
        )

    assert out.shape == (B, N_Q_HEADS, seq_len, HEAD_DIM)


# ---------------------------------------------------------------------------
# Test 2 – Inference path: first speculative step after prefill.
#           Q_LEN=15 < BLOCK_M=64 → inductor selects flex_decoding.
#           Expected to FAIL (xfail) – this is the bug from error.log.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "flex_decoding kernel fails for inference speculative step. "
        "Q_LEN=37 matches the s37 symbolic dimension in error.log and triggers "
        "create_flex_decoding_kernel, which raises NoValidChoicesError. "
        "Training succeeds because Q_LEN == KV_LEN keeps inductor on the "
        "flex_attention path."
    ),
)
def test_inference_flex_decoding_first_step():
    """
    Replicate a speculative step in generate(), directly matching error.log.

    From error.log, the failing call has Q symbolic shape s37 ≈ 37.
    We choose K=5, P=5, n_verify=0, n_props=[10,9,8,7,3]:
        Q_LEN  = 0 + 37 = 37   (= s37 from error.log)
        KV_LEN = 5 + 37 = 42

    Q_LEN=37 triggers create_flex_decoding_kernel (37 < BLOCK_M=64).
    The flex_decoding kernel raises:

        NoValidChoicesError: No choices exist for backend.
    """
    K, P, n_verify, n_props = 5, 5, 0, [10, 9, 8, 7, 3]
    pos    = P - K          # = 0
    Q_LEN  = pos + sum(n_props)   # = 37
    KV_LEN = P + sum(n_props)     # = 42

    mask_mod = make_generate_mask_mod(K, P, n_verify, n_props, Q_LEN, KV_LEN, DEVICE)
    with torch.inference_mode(False):
        block_mask = create_block_mask(
            mask_mod, B=None, H=None,
            Q_LEN=Q_LEN, KV_LEN=KV_LEN,
            device=DEVICE,
        )

    q, k, v = _make_qkv_like_model(Q_LEN, KV_LEN)

    with torch.no_grad():
        # training=False matches the generate() inference path.
        # compile_friendly_flex_attention → WrappedFlexAttention(training=False)
        # → torch.compile(flex_attention)   [PyTorch 2.7+ default path]
        # → inductor lowers flex_attention  → selects flex_decoding (Q_LEN < 64)
        # → NoValidChoicesError
        out, _lse = compile_friendly_flex_attention(
            q, k, v,
            block_mask=block_mask,
            enable_gqa=True,
            scale=SCALE,
            return_lse=True,
            training=False,
        )

    assert out.shape == (1, N_Q_HEADS, Q_LEN, HEAD_DIM)


# ---------------------------------------------------------------------------
# Test 3 – Inference path: mid-generation step with verify tokens.
#           Shapes closer to error.log (Q_LEN ≈ 37, KV_LEN ≈ 71).
#           Q_LEN=47 < BLOCK_M=64 → flex_decoding.  Same failure.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Same flex_decoding failure as test_inference_flex_decoding_first_step "
        "but with verify tokens (n_verify>0) and shapes matching error.log: "
        "Q_LEN=47 ≈ s37, KV_LEN=77 ≈ s71."
    ),
)
def test_inference_flex_decoding_mid_generation():
    """
    Replicate a mid-generation speculative step matching error.log shapes.

    From error.log:
        args[0] (Q): [1, 32, s37, 128]   → Q_LEN ≈ 37
        block_mask:  Q_LEN=s37, KV_LEN=s71

    We use K=30, P=40, n_verify=10, n_props=[10,9,8,7,3]:
        Q_LEN  = (40-30) + 37 = 47   (close to s37; 47 < BLOCK_M=64)
        KV_LEN = 40 + 37 = 77        (close to s71)
    """
    K, P, n_verify, n_props = 30, 40, 10, [10, 9, 8, 7, 3]
    pos    = P - K
    Q_LEN  = pos + sum(n_props)   # = 47
    KV_LEN = P + sum(n_props)     # = 77

    mask_mod = make_generate_mask_mod(K, P, n_verify, n_props, Q_LEN, KV_LEN, DEVICE)
    with torch.inference_mode(False):
        block_mask = create_block_mask(
            mask_mod, B=None, H=None,
            Q_LEN=Q_LEN, KV_LEN=KV_LEN,
            device=DEVICE,
        )

    q, k, v = _make_qkv_like_model(Q_LEN, KV_LEN)

    with torch.no_grad():
        out, _lse = compile_friendly_flex_attention(
            q, k, v,
            block_mask=block_mask,
            enable_gqa=True,
            scale=SCALE,
            return_lse=True,
            training=False,
        )

    assert out.shape == (1, N_Q_HEADS, Q_LEN, HEAD_DIM)


# ---------------------------------------------------------------------------
# Test 4 – ATEN fallback: same inference shapes as test 2, but with
#           max_autotune_gemm_backends="TRITON,ATEN" so that autotune finds
#           at least one valid choice and the call succeeds.
#           Fix: wrap inference_forward() with this config patch.
# ---------------------------------------------------------------------------

def test_inference_triton_backend_succeeds():
    """
    Verify that kernel_options={'BACKEND': 'TRITON'} resolves the failure.

    The error message suggests 'ATEN into max_autotune_gemm_backends', but
    that is a generic hint from autotune_select_algorithm and does NOT fix
    flex_decoding.  The actual fix is to bypass flex_decoding altogether.

    Looking at torch/_inductor/kernel/flex/flex_attention.py:

        use_decode = (backend == "TRITON_DECODE") or (backend == "AUTO" and can_use_decode)

    With BACKEND='AUTO' (the default), can_use_decode=True for small Q_LEN
    → flex_decoding is selected → NoValidChoicesError.

    With BACKEND='TRITON', use_decode is False regardless of Q_LEN → the
    standard flex_attention Triton kernel is used → call succeeds.

    Valid _Backend values: ('AUTO', 'TRITON', 'FLASH', 'TRITON_DECODE').

    The fix for production code is to pass kernel_options={'BACKEND': 'TRITON'}
    through to flex_attention_forward in TransformerModel.inference_forward().
    """
    K, P, n_verify, n_props = 5, 5, 0, [10, 9, 8, 7, 3]
    pos    = P - K
    Q_LEN  = pos + sum(n_props)   # = 37  (same as test_inference_flex_decoding_first_step)
    KV_LEN = P + sum(n_props)     # = 42

    mask_mod = make_generate_mask_mod(K, P, n_verify, n_props, Q_LEN, KV_LEN, DEVICE)
    with torch.inference_mode(False):
        block_mask = create_block_mask(
            mask_mod, B=None, H=None,
            Q_LEN=Q_LEN, KV_LEN=KV_LEN,
            device=DEVICE,
        )

    q, k, v = _make_qkv_like_model(Q_LEN, KV_LEN)

    with torch.no_grad():
        out, _lse = compile_friendly_flex_attention(
            q, k, v,
            block_mask=block_mask,
            enable_gqa=True,
            scale=SCALE,
            return_lse=True,
            kernel_options={"BACKEND": "TRITON"},
            training=False,
        )

    assert out.shape == (1, N_Q_HEADS, Q_LEN, HEAD_DIM)