"""
End-to-end test: accepted tokens in generate() match the teacher (base model) distribution.

Algorithm
---------
1. Fix a random seed, call generate() → output_ids  (shape [1, P+N+extra]).
2. Reproduce z_rnd_all — the first torch.rand call inside generate().
3. Load the pretrained base model from scratch (no LoRA), using the same
   model_id stored in train.yaml next to the checkpoint.
4. Run the base model once on output_ids[:, :-1].
5. For each output position k in [0, max_new_tokens):
       logits_k  = base_logits[:, P-1+k, :]    # predicts token at position P+k
       p_k       = softmax(logits_k / temp)     # temp=None → no scaling
       tgt_p, tgt_idx = adapt_p(p_k)
       bin_edges = tgt_p.cumsum(dim=-1)
       bin_edges[..., -1] = 1.0
       sampled   = tgt_idx[(bin_edges > z_rnd_all[k]).max(dim=-1).indices]
       assert sampled == output_ids[0, P+k]

The invariant being tested
--------------------------
  output_ids[P+k] == teacher_sample(
      base_model, context=output_ids[:, :P+k], z=z_rnd_all[k]
  )

This holds whether token k was accepted via the "no-match" path (teacher sample
directly) or the "match" path (student prediction equalled the teacher sample):
in both cases the acceptance criterion ensures
  output_ids[P+k] == correct_tokens[step][match_position]
which is the inverse-CDF sample from the teacher at that position.

The base model context is correct because:
  - At step deciding token k, the KV cache contains exactly output_ids[:, :P+k].
  - Running the base model on output_ids[:, :-1] gives the same distribution
    at position P-1+k as the KV-cached teacher in generate().

Requires CUDA (bfloat16 + FlexAttention) and the local checkpoint.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ptp.utils import instantiate
from ptp.cli.generate import find_best_checkpoint, HIST_CACHE_NAME


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENT_DIR = REPO_ROOT / "checkpoints" / "pythia-160m-v0_tulu-3-sft-mixture"
TRAIN_YAML = EXPERIMENT_DIR / "train.yaml"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="generate() requires CUDA (bfloat16 + FlexAttention)",
)


# ---------------------------------------------------------------------------
# Fixtures  (module-scoped to avoid reloading models for every test)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    if not TRAIN_YAML.exists():
        pytest.skip(f"train.yaml not found at {TRAIN_YAML}")
    with open(TRAIN_YAML) as f:
        return DictConfig(yaml.safe_load(f))


@pytest.fixture(scope="module")
def lit_model(config):
    """PTP model in inference mode."""
    ckpt_dir = Path(config["training"].get("ckpt_dir", EXPERIMENT_DIR))
    ckpt_path = find_best_checkpoint(ckpt_dir)

    model = instantiate(config["model"])
    model.configure_model()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    model.top_k = 50
    model.top_p = 0.9
    model.temperature = None          # no temperature scaling
    model.tokens_per_student_call = 20

    hist_cache = ckpt_dir / HIST_CACHE_NAME
    if hist_cache.exists():
        model.hist_base = torch.load(hist_cache, map_location="cpu", weights_only=True)

    device = torch.device("cuda")
    model = model.to(device).eval()
    model.enter_inference_mode(gate_window=model.tokens_per_student_call)

    yield model

    model.exit_inference_mode()


@pytest.fixture(scope="module")
def base_model(config):
    """Plain pretrained model loaded from scratch — no LoRA adapters."""
    model_id = config["model"]["model"]["model_id"]
    device = torch.device("cuda")
    # Use the same attention implementation as the PTP model (TransformerModel
    # passes attn_implementation="flex_attention") to get bit-identical logits.
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flex_attention"
    )
    m = m.to(device).eval()
    yield m
    del m


@pytest.fixture(scope="module")
def tokenizer(lit_model, config):
    tok = lit_model.model.tokenizer
    if not getattr(tok, "chat_template", None):
        template = config.get("data", {}).get("chat_template", None)
        if template:
            tok.chat_template = template
    return tok


@pytest.fixture(scope="module")
def prompt_ids(tokenizer):
    device = torch.device("cuda")
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(formatted, return_tensors="pt").input_ids.to(device)


# ---------------------------------------------------------------------------
# Helper: reproduce the adapt_p + inverse-CDF sampling used in generate()
# ---------------------------------------------------------------------------

def sample_from_teacher(
    logits_k: torch.Tensor,    # (V,) float32  — raw logits at one position
    z: float,                   # scalar from z_rnd_all
    top_k: int,
    top_p: float,
    temperature: float | None,
) -> int:
    """
    Apply the same adapt_p + cumsum sampling as generate():
      1. Scale logits by temperature (if set).
      2. Softmax.
      3. Top-k.
      4. Mask tail beyond top-p cumulative probability.
      5. Renormalise.
      6. Sort by token index.
      7. CDF threshold: first bin exceeding z.
    Returns the sampled vocabulary index.
    """
    logits = logits_k.float()
    if temperature is not None and temperature != 1.0:
        logits = logits / temperature
    p = torch.softmax(logits, dim=-1)          # (V,)

    # --- top-k ---
    top_k_probs, top_k_indices = torch.topk(p, k=top_k, dim=-1)   # both (top_k,)

    # --- top-p mask (same logic as adapt_p) ---
    remove = (top_k_probs.cumsum(dim=-1) - top_k_probs) > top_p
    top_k_probs = top_k_probs.masked_fill(remove, 0.0)

    # --- renormalise ---
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

    # --- sort by token index (adapt_p sorts so CDF is in vocab order) ---
    sort_idx = top_k_indices.argsort(dim=-1)
    tgt_p   = top_k_probs.gather(-1, sort_idx)    # (top_k,) sorted by token id
    tgt_idx = top_k_indices.gather(-1, sort_idx)  # (top_k,) sorted token ids

    # --- inverse-CDF: first bin whose right edge exceeds z ---
    bin_edges = tgt_p.cumsum(dim=-1)
    bin_edges[-1] = 1.0    # guard against floating-point rounding

    # PyTorch max() on a boolean tensor returns the FIRST True occurrence
    bin_idx = (bin_edges > z).max(dim=-1).indices.item()
    return tgt_idx[bin_idx].item()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_teacher_sampling_reproduces_output(lit_model, base_model, prompt_ids):
    """
    For each of the first max_new_tokens output tokens, verify that sampling
    from the independently-loaded base model with z_rnd_all[k] gives the
    same token as generate() placed at output position P+k.
    """
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    max_new_tokens = 5
    tpsc = lit_model.tokens_per_student_call

    # -----------------------------------------------------------------------
    # Step 1: Run generate() with a fixed seed.
    # -----------------------------------------------------------------------
    SEED = 42
    torch.manual_seed(SEED)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = lit_model.generate(
            {"prompt_ids": prompt_ids},
            max_new_tokens=max_new_tokens,
        )
    # output_ids is (1, P + max_new_tokens + extra_speculative)

    # -----------------------------------------------------------------------
    # Step 2: Reproduce z_rnd_all.
    # generate() calls torch.rand([B, max_new_tokens + 2*tpsc], ...) as its
    # very first random operation after the seed is set.
    # -----------------------------------------------------------------------
    torch.manual_seed(SEED)
    z_rnd_all = torch.rand(
        [1, max_new_tokens + 2 * tpsc],
        device=device,
        dtype=torch.float32,
    )
    # Postcondition: z_rnd_all[0, k] is the z value used to sample output
    # token k for k in [0, max_new_tokens).

    # -----------------------------------------------------------------------
    # Step 3: Run the base model once on the full (prompt + output) sequence.
    # We exclude the last token because we need logits that predict it.
    # -----------------------------------------------------------------------
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base_out = base_model(output_ids[:, :-1], use_cache=False)
    # base_out.logits: (1, P+N-1, V)  where N = output_ids.shape[1] - P
    base_logits = base_out.logits[0].float()   # (P+N-1, V) on GPU, float32

    # -----------------------------------------------------------------------
    # Step 4: For each output position k, sample from teacher and compare.
    # -----------------------------------------------------------------------
    mismatches = []
    for k in range(max_new_tokens):
        # Logits at position P-1+k predict the token at position P+k.
        logits_k = base_logits[P - 1 + k]

        sampled = sample_from_teacher(
            logits_k,
            z=z_rnd_all[0, k].item(),
            top_k=lit_model.top_k,
            top_p=lit_model.top_p,
            temperature=lit_model.temperature,
        )
        expected = output_ids[0, P + k].item()

        if sampled != expected:
            mismatches.append(
                f"  k={k}: base_model sampled token {sampled!r}, "
                f"generate() produced {expected!r}"
            )

    assert not mismatches, (
        f"{len(mismatches)} / {max_new_tokens} output tokens differ "
        f"between the base model teacher and generate():\n"
        + "\n".join(mismatches)
        + f"\n\nFull output: {output_ids[0].tolist()}"
    )
