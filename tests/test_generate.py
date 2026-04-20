"""
Integration test for ParallelSamplingLightningModule.generate().

Loads the pythia-160m-v0 checkpoint from the local experiment directory,
enters inference mode, and generates a short completion to verify the
full generate() pipeline works end-to-end.

Requires CUDA: the model uses bfloat16 and FlexAttention, both of which
need a CUDA device (same as the CLI).
"""
import sys
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import DictConfig

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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lit_model():
    """Load the trained lit model from the best local checkpoint."""
    if not TRAIN_YAML.exists():
        pytest.skip(f"train.yaml not found at {TRAIN_YAML}")

    with open(TRAIN_YAML) as f:
        config = DictConfig(yaml.safe_load(f))

    ckpt_dir = Path(config["training"].get("ckpt_dir", EXPERIMENT_DIR))
    ckpt_path = find_best_checkpoint(ckpt_dir)

    model = instantiate(config["model"])
    model.configure_model()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    # Sampling parameters matching cli/generate.py defaults
    model.top_k = 50
    model.top_p = 0.9
    model.tokens_per_student_call = 20

    # Load the pre-computed token-acceptance histogram (needed by proposals())
    hist_cache = ckpt_dir / HIST_CACHE_NAME
    if hist_cache.exists():
        model.hist_base = torch.load(hist_cache, map_location="cpu", weights_only=True)

    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    model.enter_inference_mode(gate_window=20)

    yield model

    model.exit_inference_mode()


@pytest.fixture(scope="module")
def tokenizer(lit_model):
    tok = lit_model.model.tokenizer
    if not getattr(tok, "chat_template", None):
        with open(TRAIN_YAML) as f:
            config = DictConfig(yaml.safe_load(f))
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
# Tests
# ---------------------------------------------------------------------------

def test_generate_returns_more_tokens_than_prompt(lit_model, prompt_ids):
    """generate() should return at least one new token beyond the prompt."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = lit_model.generate(
            {"prompt_ids": prompt_ids},
            max_new_tokens=10,
        )
    assert output.shape[1] > prompt_ids.shape[1], (
        f"Expected output longer than prompt ({prompt_ids.shape[1]}), "
        f"got {output.shape[1]}"
    )


def test_generate_output_length_is_bounded(lit_model, prompt_ids):
    """
    With fixed_tokens=True the output includes max_new_tokens verified tokens
    plus up to tokens_per_student_call - 1 speculative proposals, so the total
    new tokens must lie in [max_new_tokens, max_new_tokens + tpsc].
    """
    max_new_tokens = 10
    tpsc = lit_model.tokens_per_student_call
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = lit_model.generate(
            {"prompt_ids": prompt_ids},
            max_new_tokens=max_new_tokens,
        )
    n_new = output.shape[1] - prompt_ids.shape[1]
    assert n_new >= max_new_tokens, (
        f"Expected at least {max_new_tokens} new tokens, got {n_new}"
    )
    assert n_new <= max_new_tokens + tpsc, (
        f"Expected at most {max_new_tokens + tpsc} new tokens "
        f"(max_new_tokens={max_new_tokens} + tpsc={tpsc}), got {n_new}"
    )


def test_generate_return_metrics(lit_model, prompt_ids):
    """return_metrics=True should include correctness statistics."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, metrics = lit_model.generate(
            {"prompt_ids": prompt_ids},
            max_new_tokens=10,
            return_metrics=True,
        )
    assert output.shape[1] > prompt_ids.shape[1]
    assert "correct_per_call" in metrics
    assert "correct_all" in metrics
    assert "num_calls" in metrics
    assert isinstance(metrics["correct_per_call"], float)
    assert metrics["num_calls"] > 0


def test_generate_callback_is_called(lit_model, prompt_ids):
    """The callback should be invoked at least once during generation."""
    calls = []

    def cb(current_completion, verified_until_idx):
        calls.append((current_completion.shape, verified_until_idx))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        lit_model.generate(
            {"prompt_ids": prompt_ids},
            max_new_tokens=10,
            callback=cb,
        )
    assert len(calls) > 0, "callback was never called"


def test_generate_eos_stops_early(lit_model, tokenizer, prompt_ids):
    """Passing the EOS token ID should be accepted and generation terminates cleanly."""
    eos = tokenizer.eos_token_id
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = lit_model.generate(
            {"prompt_ids": prompt_ids},
            max_new_tokens=10,
            eos=eos,
        )
    assert output.ndim == 2
    assert output.dtype == torch.long


def test_generate_return_past_key_values(lit_model, prompt_ids):
    """return_past_key_values=True should return a (completion, cache) tuple."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, past = lit_model.generate(
            {"prompt_ids": prompt_ids},
            max_new_tokens=10,
            return_past_key_values=True,
        )
    past_prompt_ids, kv_cache = past
    assert past_prompt_ids.shape == output.shape
    assert kv_cache is not None


def test_generate_output_is_decodable(lit_model, tokenizer, prompt_ids):
    """The generated token IDs must be decodable to a string without errors."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = lit_model.generate(
            {"prompt_ids": prompt_ids},
            max_new_tokens=10,
        )
    new_tokens = output[0, prompt_ids.shape[1]:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    assert isinstance(text, str)


def test_generate_demo(lit_model, tokenizer, prompt_ids, capsys):
    """Print prompt + completion for both the PTP generate and the raw model,
    so the two outputs can be compared side-by-side."""
    # --- PTP speculative generate ---
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ptp_output, metrics = lit_model.generate(
            {"prompt_ids": prompt_ids},
            max_new_tokens=10,
            return_metrics=True,
        )

    # --- Raw model generate (HF standard, adapters inactive) ---
    # lit_model.model.model is the PeftModelForCausalLM; calling its .generate()
    # runs the fused weights through normal autoregressive decoding without any
    # speculative proposals.
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        raw_output = lit_model.model.model.generate(
            prompt_ids,
            max_new_tokens=10,
            do_sample=True,
            top_k=lit_model.top_k,
            top_p=lit_model.top_p,
        )

    prompt_text = tokenizer.decode(prompt_ids[0].tolist(), skip_special_tokens=True)
    ptp_text = tokenizer.decode(
        ptp_output[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=True
    )
    raw_text = tokenizer.decode(
        raw_output[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=True
    )

    with capsys.disabled():
        print(f"\n{'=' * 60}")
        print(f"PROMPT     : {prompt_text!r}")
        print(f"PTP        : {ptp_text!r}  "
              f"(correct/call: {metrics['correct_per_call']:.2f}, "
              f"calls: {metrics['num_calls']})")
        print(f"RAW model  : {raw_text!r}")
        print("=" * 60)
