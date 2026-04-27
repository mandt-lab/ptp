"""
ptp_generate — interactive text generation from a trained checkpoint.

Loads the best checkpoint from a training directory, puts the model in
inference mode, and runs an interactive prompt loop.
"""
from argparse import ArgumentParser
import contextlib
import readline
from pathlib import Path
import sys
import time
import warnings

HIST_CACHE_NAME = "correct_count_hist.pt"
HIST_FILE_NAME = ".ptp_generate_history"


def _configure_readline_history(history_path: Path) -> None:
    """Enable line editing + persistent history for prompt input."""
    readline.parse_and_bind("set editing-mode emacs")
    readline.set_history_length(1000)
    if history_path.exists():
        try:
            readline.read_history_file(str(history_path))
        except OSError as exc:
            print(f"Warning: could not read history file {history_path}: {exc}")


def _save_readline_history(history_path: Path) -> None:
    try:
        readline.write_history_file(str(history_path))
    except OSError as exc:
        print(f"Warning: could not save history file {history_path}: {exc}")


def _load_or_compute_hist_base(lit_model, config, experiment_dir, ckpt_path, ckpt_dir, autocast_dtype, device):
    import torch
    from tqdm import tqdm
    from ptp.utils import instantiate
    # Patch datasets library to handle 'List' feature type
    try:
        from datasets.features.features import _FEATURE_TYPES
        _FEATURE_TYPES['List'] = _FEATURE_TYPES['Sequence']
    except Exception:
        pass

    cache_path = ckpt_dir / HIST_CACHE_NAME

    if cache_path.exists():
        if ckpt_path.stat().st_mtime <= cache_path.stat().st_mtime:
            print(f"Loading hist_base from cache: {cache_path}")
            lit_model.hist_base = torch.load(cache_path, map_location="cpu", weights_only=True)
            return
        print(f"Warning: {ckpt_path.name} is newer than cached {cache_path.name}.")
        answer = input("Recompute hist_base from validation data? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Using cached histogram.")
            lit_model.hist_base = torch.load(cache_path, map_location="cpu", weights_only=True)
            return

    if "data" not in config:
        print("No data config found; cannot compute hist_base.")
        return

    data_config = config["data"]
    if data_config.get("root_dir") and "EXP_DIR" in str(data_config["root_dir"]):
        data_config["root_dir"] = str(data_config["root_dir"]).replace("EXP_DIR", str(experiment_dir))

    print("Computing hist_base from validation data...")
    datamodule = instantiate(data_config)
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    if not val_loader:
        print("No validation data available; skipping hist_base computation.")
        return

    eval_steps = config.get("training", {}).get("eval_steps_to_run")
    hist = torch.zeros(21, dtype=torch.float64)
    with torch.no_grad(), torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating correct histograms", unit="batch")):
            if eval_steps is not None and i >= eval_steps:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            metrics = lit_model.forward(batch, eval=True)
            if "correct_counts" in metrics:
                counts = metrics["correct_counts"].cpu().clamp(max=20)
                hist += torch.bincount(counts.long(), minlength=21).double()
            print(f"\r  {i + 1} batches processed", end="", flush=True)
    print()

    if hist.sum() > 0:
        hist_base = hist / hist.sum()
        torch.save(hist_base, cache_path)
        print(f"Saved hist_base to {cache_path}")
        lit_model.hist_base = hist_base
    else:
        print("No correct counts collected; skipping hist_base computation.")


def _make_stream_callback(tokenizer, prompt_length, show_only_valid):
    """Return a generate() callback that only appends newly verified text."""
    last_printed_valid_tokens = 0

    def callback(current_completion, verified_until_idx):
        nonlocal last_printed_valid_tokens
        new_tokens = current_completion[prompt_length:].tolist()
        n_valid = max(0, verified_until_idx - prompt_length)
        # Only append tokens that became newly verified since the last callback.
        # If the verified frontier ever moves backward, we ignore that update to
        # keep output monotonic and avoid redraw/reset behavior.
        if n_valid <= last_printed_valid_tokens:
            return

        appended_tokens = new_tokens[last_printed_valid_tokens:n_valid]
        if not appended_tokens:
            return

        appended_text = tokenizer.decode(
            appended_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not appended_text:
            last_printed_valid_tokens = n_valid
            return

        sys.stdout.write(appended_text)
        sys.stdout.flush()
        last_printed_valid_tokens = n_valid

    return callback


def _measure_ar_ms_per_token(
    lit_model,
    device,
    autocast_dtype,
    prompt_length: int = 128,
    n_tokens: int = 50,
) -> float:
    """Timed autoregressive baseline using HF's standard generate (KV-cached, greedy).

    Called once after speculative warmup so Triton / torch.compile kernels for
    both paths are already compiled.  Returns ms per generated token.
    """
    import torch
    dummy = torch.zeros([1, prompt_length], dtype=torch.long, device=device)
    gen_kwargs = dict(max_new_tokens=n_tokens, do_sample=False, use_cache=True)

    def _run():
        autocast_ctx = (
            torch.autocast(device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            lit_model.model.generate(dummy, **gen_kwargs)

    # Warm-up run so HF's generate path is compiled/cached too.
    _run()

    if device.type == "cuda":
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        autocast_ctx = (
            torch.autocast(device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            t0.record()
            lit_model.model.generate(dummy, **gen_kwargs)
            t1.record()
        torch.cuda.synchronize()
        elapsed_ms = t0.elapsed_time(t1)
    else:
        wall_t0 = time.perf_counter()
        _run()
        elapsed_ms = (time.perf_counter() - wall_t0) * 1000.0

    return elapsed_ms / n_tokens


def _print_generation_stats(prompt_length: int, completion, metrics: dict | None, elapsed_ms: float | None = None, ar_ms_per_token: float | None = None) -> None:
    generated_tokens = max(0, completion.shape[1] - prompt_length)
    ms_per_token_str = ""
    if elapsed_ms is not None and generated_tokens > 0:
        ms_per_token_str = f", {elapsed_ms / generated_tokens:.1f} ms/token"

    if not metrics:
        print(f"Generated {generated_tokens} tokens{ms_per_token_str}.")
        return

    num_calls = metrics.get("num_calls", 0)
    if generated_tokens == 0:
        print(f"Generation stats: 0 tokens, {num_calls} model calls.")
        return

    tokens_per_call = generated_tokens / num_calls if num_calls else float("inf")

    ar_str = ""
    if ar_ms_per_token is not None and elapsed_ms is not None and generated_tokens > 0:
        spec_ms_per_token = elapsed_ms / generated_tokens
        speedup = ar_ms_per_token / spec_ms_per_token
        implementation_overhead = tokens_per_call / speedup - 1
        ar_str = f", AR baseline {ar_ms_per_token:.1f} ms/token, speedup {speedup:.2f}x (includes {implementation_overhead:.0%} implementation overhead)"

    print(
        "Generation stats: "
        f"{generated_tokens} tokens, "
        f"{num_calls} model calls, "
        f"{tokens_per_call:.4f} tokens/call"
        f"{ms_per_token_str}"
        f"{ar_str}."
    )


def _decode_assistant_reply(tokenizer, completion, prompt_length: int) -> str:
    """Decode only newly generated assistant text from completion."""
    generated = completion[0, prompt_length:]
    if generated.numel() == 0:
        return ""
    return tokenizer.decode(generated, skip_special_tokens=True)


def find_best_checkpoint(ckpt_dir: Path) -> Path:
    """
    Restore a ModelCheckpoint callback from last.ckpt and return the path
    of the checkpoint with the best monitored metric. Falls back to
    last.ckpt itself when no best_model_path has been recorded yet.
    """
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint

    last = ckpt_dir / "last.ckpt"
    if not last.exists():
        ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        print(f"No last.ckpt found; using: {ckpt_files[-1]}")
        return ckpt_files[-1]

    ckpt_data = torch.load(last, map_location="cpu", weights_only=False)
    cb = ModelCheckpoint()
    for key, state in ckpt_data.get("callbacks", {}).items():
        if "ModelCheckpoint" in str(key):
            cb.load_state_dict(state)
            break

    best = cb.best_model_path
    if best and Path(best).exists():
        print(f"Best checkpoint (score {cb.best_model_score}): {best}")
        return Path(best)

    print("No best_model_path recorded; using last.ckpt.")
    return last


class GenerateSession:
    """
    Stateful wrapper for one interactive generation session.

    Holds KV cache state across calls so that a repeated prompt prefix
    (system prompt + growing conversation) is not re-processed by the
    prefill step.

    Usage
    -----
        session = GenerateSession(lit_model, gate_window=20,
                                  device=device, autocast_dtype=torch.bfloat16)
        session.warmup()
        ...
        completion, metrics = session.generate(input_ids, max_new_tokens=512,
                                               return_metrics=True, callback=cb)
        # on /reset:
        session.reset()
    """

    def __init__(
        self,
        lit_model,
        gate_window: int,
        device,
        autocast_dtype,         # torch.bfloat16 / torch.float16 / None
        *,
        eos: int | None = None,
        fixed_tokens: bool = True,
        pad_token: int = 13,
    ):
        self.lit_model = lit_model
        self.gate_window = gate_window
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.eos = eos
        self.fixed_tokens = fixed_tokens
        self.pad_token = pad_token
        self._past_kv_cache: tuple | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Discard the cached KV state.  Call after /reset or a new topic."""
        self._past_kv_cache = None

    def warmup(self, warmup_prompt_lengths: tuple[int, ...] = (128,)) -> None:
        """
        Warm up torch.compile kernels for representative prompt lengths.

        Runs a dummy ``generate()`` for each length so that the first real
        user prompt doesn't pay the compilation cost.
        KV state accumulated during warmup is discarded via ``reset()``.
        """
        import torch
        max_new_tokens = self.gate_window + 1   # one speculative step is enough
        for length in warmup_prompt_lengths:
            dummy = torch.zeros([1, length], dtype=torch.long, device=self.device)
            self._call_model(dummy, max_new_tokens=max_new_tokens,
                             return_metrics=False, callback=None)
        self.reset()

    def generate(
        self,
        prompt_ids,
        max_new_tokens: int,
        callback=None,
        return_metrics: bool = False,
    ) -> tuple:
        """
        Run one speculative-decoding step, reusing KV and mask state.

        Returns ``completion`` tensor, or ``(completion, metrics)`` when
        ``return_metrics=True``.
        """
        return self._call_model(prompt_ids, max_new_tokens=max_new_tokens,
                                return_metrics=return_metrics, callback=callback)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _autocast(self):
        import torch
        if self.autocast_dtype is not None:
            return torch.autocast(self.device.type, dtype=self.autocast_dtype)
        return contextlib.nullcontext()

    def _call_model(self, prompt_ids, *, max_new_tokens, return_metrics, callback):
        with self._autocast():
            result = self.lit_model.generate(
                {"prompt_ids": prompt_ids},
                max_new_tokens=max_new_tokens,
                return_metrics=return_metrics,
                return_past_key_values=True,
                past_kv_cache=self._past_kv_cache,
                callback=callback,
                eos=self.eos,
                fixed_tokens=self.fixed_tokens,
                pad_token=self.pad_token,
            )
        if return_metrics:
            completion, self._past_kv_cache, metrics = result
            return completion, metrics
        completion, self._past_kv_cache = result
        return completion


def main(
    experiment_dir: Path,
    checkpoint: Path | None = None,
    variant_name: str = "",
    max_tokens_per_proposal: int = 20,
    total_token_budget: int | None = None,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float | None = None,
    max_new_tokens: int = 512,
    show_only_valid: bool = False,
    compile: bool = True,
):
    import torch
    import yaml
    from omegaconf import DictConfig, OmegaConf
    from ptp.utils import instantiate

    with open(experiment_dir / "train.yaml") as f:
        config = DictConfig(yaml.safe_load(f))
    if variant_name != "":
        with open(experiment_dir / f"train-{variant_name}.yaml") as f:
            variant_config = DictConfig(yaml.safe_load(f))
        config = OmegaConf.merge(config, variant_config)

    ckpt_dir = Path(config["training"].get("ckpt_dir", experiment_dir))
    if variant_name != "":
        ckpt_dir = ckpt_dir / variant_name
    ckpt_path = checkpoint or find_best_checkpoint(ckpt_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    # Use SDPA for generation: no per-shape Triton recompilation from variable KV_LEN.
    OmegaConf.update(config, "model.model.attn_implementation", "sdpa", merge=True)
    lit_model = instantiate(config["model"])
    lit_model.configure_model()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lit_model.load_state_dict(ckpt["state_dict"])

    # Apply inference sampling parameters
    if top_k is not None:
        lit_model.top_k = top_k
    if top_p is not None:
        lit_model.top_p = top_p
    if temperature is not None:
        lit_model.temperature = temperature
    if lit_model.top_k is None or lit_model.top_p is None or lit_model.temperature is None:
        warnings.warn(
            "Some sampling parameters are missing. Consider setting them in the config "
            "or via command-line arguments for consistent generation behavior."
        )
    total_budget = total_token_budget or max_tokens_per_proposal
    lit_model.tokens_per_student_call = max_tokens_per_proposal
    lit_model.total_token_budget = total_budget

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()

    # Mirror the autocast context that Lightning applies during training.
    precision = config["training"].get("precision", "32-true")
    if "bf16" in str(precision):
        autocast_dtype = torch.bfloat16
    elif "16" in str(precision):
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    _load_or_compute_hist_base(lit_model, config, experiment_dir, ckpt_path, ckpt_dir, autocast_dtype, device)
    lit_model.enter_inference_mode(gate_window=total_budget)

    # Compile after enter_inference_mode so torch.compile traces the fused
    # GatedLinearLoraMerged layers, not the original PEFT LoRA modules.
    # dynamic=True handles variable prompt lengths across turns.
    if compile:
        lit_model.model.model = torch.compile(lit_model.model.model, dynamic=True)

    history_path = ckpt_dir / HIST_FILE_NAME
    _configure_readline_history(history_path)

    tokenizer = lit_model.model.tokenizer

    # Set the custom chat template on the tokenizer if one is provided in the
    # data config and the tokenizer does not already have its own.
    custom_template = config.get("data", {}).get("chat_template", None)
    if custom_template and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = custom_template

    # Create a session that holds KV cache state between calls.
    session = GenerateSession(
        lit_model,
        gate_window=total_budget,
        device=device,
        autocast_dtype=autocast_dtype,
    )

    # Warm up torch.compile kernels for a typical prompt length.
    print("Warming up...", end="", flush=True)
    session.warmup()
    print(" done.")

    print("Measuring AR baseline...", end="", flush=True)
    ar_ms_per_token = _measure_ar_ms_per_token(lit_model, device, autocast_dtype)
    print(f" {ar_ms_per_token:.1f} ms/token (AR baseline).")

    print(f"\nModel ready on {device}. Type your prompt and press Enter.")
    help_string = "(Ctrl-D, 'quit', or '/reset' to clear chat history)\n"
    print(help_string)

    messages = []

    try:
        while True:
            try:
                prompt = input(">>> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print(f"\nPrompt ignored. {help_string}")
                continue

            if prompt.lower() in ("quit", "exit", "q"):
                break
            if prompt.lower() == "/reset":
                messages = []
                session.reset()
                print("Chat history cleared.")
                continue

            if not prompt:
                continue

            messages.append({"role": "user", "content": prompt})
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
            print("assistant: ", end="", flush=True)

            stream = _make_stream_callback(
                tokenizer,
                input_ids.shape[1],
                show_only_valid,
            )

            try:
                t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                wall_t0 = time.perf_counter()
                if t0 is not None:
                    t0.record()
                completion, metrics = session.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    return_metrics=True,
                    callback=stream,
                )
                if t1 is not None:
                    t1.record()
                    torch.cuda.synchronize()
                    elapsed_ms = t0.elapsed_time(t1)
                else:
                    elapsed_ms = (time.perf_counter() - wall_t0) * 1000.0
                assistant_reply = _decode_assistant_reply(tokenizer, completion, input_ids.shape[1])
                messages.append({"role": "assistant", "content": assistant_reply})
                print("\n")
                _print_generation_stats(input_ids.shape[1], completion, metrics, elapsed_ms, ar_ms_per_token)
            except KeyboardInterrupt:
                print("\n\nGeneration interrupted by user.")
                print(help_string)
                session.reset()
    finally:
        _save_readline_history(history_path)


def _parse_args():
    parser = ArgumentParser(
        description=(
            "Interactive generation from a trained PTP checkpoint. "
            "Reads train.yaml from the experiment directory to set up the model."
        )
    )
    parser.add_argument(
        "experiment_dir", type=Path,
        help="Experiment directory containing train.yaml and saved checkpoints.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help=(
            "Path to a specific .ckpt file. "
            "Defaults to the best checkpoint recorded in last.ckpt."
        ),
    )
    parser.add_argument(
        "-v", "--variant", type=str, default="",
        help=(
            "Optional variant name. If given, merges train-<variant>.yaml on top of train.yaml. "
            "Useful for quickly trying hyperparameter changes without editing the base config."
        ),
    )
    parser.add_argument(
        "--max-tokens-per-proposal", type=int, default=20,
        help="Maximum tokens the student proposes per speculative depth (default: 20).",
    )
    parser.add_argument(
        "--total-token-budget", type=int, default=None,
        help=(
            "Total proposal tokens across all depths per forward pass. "
            "Defaults to --max-tokens-per-proposal (single-depth mode). "
            "Set larger to enable multi-depth speculation."
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k for sampling (default: 50).",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p nucleus sampling (default: 0.9).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (default: no scaling).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Maximum tokens to generate per prompt (default: 512).",
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Don't use torch.compile to speed up generation (default: use compile).",
        default=False,
    )
    parser.add_argument(
        "--show-only-valid", action="store_true",
        help="Only display verified tokens; hide draft proposals (default: show drafts dimmed).",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.exists():
        print(f"Error: directory not found: {experiment_dir}")
        raise SystemExit(1)
    if not (experiment_dir / "train.yaml").exists():
        print(f"Error: train.yaml not found in {experiment_dir}")
        raise SystemExit(1)

    main(
        experiment_dir=experiment_dir,
        checkpoint=args.checkpoint,
        variant_name=args.variant,
        max_tokens_per_proposal=args.max_tokens_per_proposal,
        total_token_budget=args.total_token_budget,
        max_new_tokens=args.max_new_tokens,
        show_only_valid=args.show_only_valid,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        compile=not args.no_compile,
    )


if __name__ == "__main__":
    _parse_args()
