"""
ptp_distill — set up a distillation experiment directory.

Creates the experiment directory, fills config templates, writes a README.md
that tells the user exactly what to run next, then exits. The heavy steps
(pregeneration and training) are handled by the separate ptp_pregenerate and
ptp_train commands so each can be run, monitored, and resumed independently.
"""
import json
import re
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

MODES = {
    "prompt-distill": "Have the base model generate answers; PTP distills those responses.",
    "prompt-scheme-distill": "Format dataset items with a prompt template; PTP distills teacher responses.",
    "full-chat": "Train directly on complete conversations in the dataset.",
    "fixed": "Train directly on raw text in the dataset.",
}


def slugify(name: str) -> str:
    part = name.split("/")[-1]
    part = part.lower()
    part = re.sub(r"[^a-z0-9]+", "-", part)
    return part.strip("-")


def _yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(items) + "]"


def _yaml_str(s: str) -> str:
    return json.dumps(s)


def _parse_roles(raw: str) -> list[str]:
    return [r.strip() for r in raw.split(",") if r.strip()]


def fill_template(template: str, replacements: dict) -> str:
    result = template
    for key, value in replacements.items():
        result = result.replace(f"__{key}__", value)
    return result


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def _ask_roles(prompt: str, defaults: list[str]) -> list[str]:
    raw = input(f"{prompt} [{', '.join(defaults)}]: ").strip()
    return _parse_roles(raw) if raw else defaults


def _ask_chat_setup(mode: str) -> dict:
    """Ask the questions common to prompt-distill and full-chat modes."""
    from ptp.data.chat import DEFAULT_USER_ROLES, DEFAULT_ASSISTANT_ROLES
    print()
    conversation_keys = _ask(
        "Possible message field name(s), in priority order (comma-separated)",
        "messages,conversations,data",
    )
    print()
    user_roles = _ask_roles(
        "User role name(s) — separate with commas if there are several variants",
        DEFAULT_USER_ROLES,
    )
    assistant_roles = _ask_roles(
        "Assistant role name(s) — separate with commas if there are several variants",
        DEFAULT_ASSISTANT_ROLES,
    )
    return dict(mode=mode, conversation_keys=_parse_roles(conversation_keys),
                user_roles=user_roles, assistant_roles=assistant_roles,
                text_column=None, prompt_scheme=None)


def _ask_prompt_scheme_setup() -> dict:
    """Ask for a Python format string used to build each prompt."""
    print()
    print("Provide a Python format string used to build each prompt.")
    print("  Example: '<|user|>\\n{description}</s>\\n<|assistant|>\\n'")
    print("  Dataset fields are injected via {field_name} placeholders.")
    prompt_scheme = input("Prompt scheme: ").strip().replace('\\n', '\n')
    return dict(mode="prompt-scheme-distill", prompt_scheme=prompt_scheme,
                conversation_keys=None, user_roles=None, assistant_roles=None,
                text_column=None)


def _ask_common_settings() -> dict:
    """Ask settings that apply regardless of mode."""
    print()
    wandb_project = _ask(
        "W&B project name (enter 'offline' to skip logging)", "ptp-distill"
    )
    max_sequence_length = int(_ask(
        "Maximum sequence length in tokens", "2048"
    ))
    completion_length = int(_ask(
        "Number of tokens predicted in parallel at each position", "16"
    ))
    num_completions = int(_ask(
        "Number of completions per packed sequence",
        str(max_sequence_length // completion_length),
    ))
    lora_rank = int(_ask("LoRA rank", "256"))
    return dict(wandb_project=wandb_project, max_sequence_length=max_sequence_length,
                completion_length=completion_length, num_completions=num_completions,
                use_lora=True, lora_rank=lora_rank)


def ask_setup(model_name: str) -> dict:
    """
    Two-level interactive flow. Returns a dict with all fields needed to fill
    config templates: mode, conversation_keys, user_roles, assistant_roles,
    text_column, prompt_scheme.
    """
    # Level 1: how to use the training data
    print("How would you like to use the training data for distillation?\n")
    print("  [1] Interactive — have the base model respond to the prompts in the dataset.")
    print("                    PTP will distill those responses.")
    print("  [2] Fixed       — distill on the training data as-is, without base model completions.")
    print()
    while True:
        c = input("Enter 1 or 2: ").strip()
        if c in ("1", "2"):
            break
        print("  Please enter 1 or 2.")

    if c == "1":
        # Level 2: what format to use for the prompts
        print()
        print("What format is the dataset in?\n")
        print(f"  [1] Chat          — extract prompts from conversation fields.")
        print("  [2] Prompt scheme — build prompts from a Python format string.")
        print()
        while True:
            c2 = input("Enter 1 or 2: ").strip()
            if c2 in ("1", "2"):
                break
            print("  Please enter 1 or 2.")

        if c2 == "1":
            setup = _ask_chat_setup("prompt-distill")
        else:
            setup = _ask_prompt_scheme_setup()
        setup.update(_ask_common_settings())
        return setup

    # Level 2: what kind of dataset
    print()
    print("What kind of dataset is this?\n")
    print(f"  [1] Chat  — apply the chat template of {model_name} and tokenize.")
    print("  [2] Fixed — tokenize as-is.")
    print()
    while True:
        c2 = input("Enter 1 or 2: ").strip()
        if c2 in ("1", "2"):
            break
        print("  Please enter 1 or 2.")

    if c2 == "1":
        setup = _ask_chat_setup("full-chat")
        setup.update(_ask_common_settings())
        return setup

    print()
    text_column = _ask("Name of the text field in each dataset entry", "text")
    setup = dict(mode="fixed", conversation_keys=None,
                 user_roles=None, assistant_roles=None,
                 text_column=text_column, prompt_scheme=None)
    setup.update(_ask_common_settings())
    return setup


def _resolve_setup(args) -> dict:
    """Return the setup dict either from CLI flags or interactively."""
    from ptp.data.chat import DEFAULT_USER_ROLES, DEFAULT_ASSISTANT_ROLES
    if args.mode is None:
        return ask_setup(args.model_name)

    if args.conversation_keys:
        conversation_keys = _parse_roles(args.conversation_keys)
    elif args.conversation_key:
        conversation_keys = [args.conversation_key]
    elif args.mode == "fixed":
        conversation_keys = None
    elif args.mode == "prompt-scheme-distill":
        conversation_keys = None
    else:
        conversation_keys = ["messages", "conversations", "data"]

    return dict(
        mode=args.mode,
        conversation_keys=conversation_keys,
        text_column=args.text_column or ("text" if args.mode == "fixed" else None),
        user_roles=_parse_roles(args.user_roles) if args.user_roles else DEFAULT_USER_ROLES,
        assistant_roles=_parse_roles(args.assistant_roles) if args.assistant_roles else DEFAULT_ASSISTANT_ROLES,
        prompt_scheme=args.prompt_scheme or None,
        wandb_project=args.wandb_project,
        max_sequence_length=args.max_sequence_length,
        completion_length=args.completion_length,
        num_completions=args.num_completions or (args.max_sequence_length // args.completion_length),
        use_lora=True,
        lora_rank=args.lora_rank,
    )


# ---------------------------------------------------------------------------
# Chat template check
# ---------------------------------------------------------------------------

def _has_chat_template(model_name: str) -> bool:
    """Return True if the model's tokenizer has a chat_template, False otherwise."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        return tok.chat_template is not None
    except Exception:
        return True  # can't check — don't warn


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------

def make_readme(model_name: str, dataset_name: str, mode: str, exp_dir: Path,
                has_chat_template: bool = True) -> str:
    try:
        rel_dir = exp_dir.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        rel_dir = exp_dir.resolve()

    if mode in ("prompt-distill", "prompt-scheme-distill"):
        config_bullets = (
            f"   - `{rel_dir}/pregenerate.yaml`\n"
            f"   - `{rel_dir}/train.yaml`"
        )
        run_steps = (
            f"2. Generate completions:\n"
            f"   ```\n"
            f"   ptp_pregenerate {rel_dir}\n"
            f"   ```\n"
            f"3. Train:\n"
            f"   ```\n"
            f"   ptp_train {rel_dir}\n"
            f"   ```"
        )
    else:
        config_bullets = f"   - `{rel_dir}/train.yaml`"
        run_steps = (
            f"2. Train:\n"
            f"   ```\n"
            f"   ptp_train {rel_dir}\n"
            f"   ```"
        )

    config_file = "pregenerate.yaml" if mode in ("prompt-distill", "prompt-scheme-distill") else "train.yaml"
    chat_template_warning = "" if has_chat_template else (
        f"\n> [!WARNING]\n> This model's tokenizer has no `chat_template`. "
        f"Add a `chat_template:` field under `data:` in `{rel_dir}/{config_file}` "
        f"before proceeding. "
        f"See https://huggingface.co/docs/transformers/chat_templating\n"
    )

    return (
        f"# Distillation: {model_name.split('/')[-1]} on {dataset_name.split('/')[-1]}\n\n"
        f"Mode: `{mode}`\n"
        f"{chat_template_warning}\n"
        f"1. Review and edit the config(s) — adjust batch sizes, sequence lengths, etc.\n"
        f"{config_bullets}\n"
        f"{run_steps}\n"
    )


def _readme_for_cli(text: str) -> str:
    """Render README markdown for terminal output: strip backticks, show commands with >."""
    # Fenced code blocks → indented > command
    text = re.sub(r"   ```\n   (.+?)\n   ```", r"   > \1", text, flags=re.DOTALL)
    # Inline backticks → plain text
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


# ---------------------------------------------------------------------------
# Config writing
# ---------------------------------------------------------------------------

def _lora_config_block(lora_rank: int) -> str:
    """Return the indented YAML block for lora_config to splice into model config."""
    return (
        f"    lora_config:\n"
        f"      r: {lora_rank}\n"
        f"      target_modules: all-linear"
    )


def write_config(exp_dir: Path, template_name: str, replacements: dict,
                 dst_name: str | None = None):
    src = CONFIGS_DIR / template_name
    base = CONFIGS_DIR / "train_base.yaml"
    if template_name.startswith("train") and base.exists():
        content = base.read_text() + "\n" + src.read_text()
    else:
        content = src.read_text()
    dst = exp_dir / (dst_name or template_name)
    dst.write_text(fill_template(content, replacements))
    print(f"  {dst.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(
        description=(
            "Set up a PTP distillation experiment directory. "
            "Generates config files and a README, then exits. "
            "Run ptp_pregenerate and ptp_train next."
        )
    )
    parser.add_argument("model_name", help="HuggingFace model ID (e.g. EleutherAI/pythia-160m-v0)")
    parser.add_argument("dataset_name", help="HuggingFace dataset ID (e.g. allenai/tulu-3-sft-mixture)")
    parser.add_argument("--mode", choices=list(MODES), default=None,
                        help="Distillation mode (asked interactively if omitted).")
    parser.add_argument("--conversation-keys", default=None,
                        help="Comma-separated candidate dataset columns that may hold conversations.")
    parser.add_argument("--conversation-key", default=None,
                        help="Single dataset column holding conversations (legacy).")
    parser.add_argument("--text-column", default=None,
                        help="Dataset column holding text (fixed mode).")
    parser.add_argument("--prompt-scheme", default=None,
                        help="Python format string for prompt-scheme-distill mode "
                             "(e.g. '<|user|>\\n{description}</s>\\n<|assistant|>\\n').")
    parser.add_argument("--user-roles", default=None,
                        help="Comma-separated user role names (e.g. 'human,user').")
    parser.add_argument("--assistant-roles", default=None,
                        help="Comma-separated assistant role names (e.g. 'assistant,gpt').")
    parser.add_argument("--wandb-project", default="offline",
                        help="W&B project name. 'offline' disables logging (default: offline).")
    parser.add_argument("--max-sequence-length", type=int, default=2048,
                        help="Maximum sequence length in tokens (default: 2048).")
    parser.add_argument("--completion-length", type=int, default=16,
                        help="Number of tokens predicted in parallel at each position (default: 16).")
    parser.add_argument("--num-completions", type=int, default=None,
                        help="Completions per packed sequence (default: max_sequence_length // completion_length).")
    parser.add_argument("--lora-rank", type=int, default=256,
                        help="LoRA rank (default: 256).")
    args = parser.parse_args()

    exp_dir = CHECKPOINTS_DIR / f"{slugify(args.model_name)}_{slugify(args.dataset_name)}"

    # If the experiment already exists, just remind the user and exit.
    if exp_dir.exists():
        readme = exp_dir / "README.md"
        print(f"Experiment directory already exists: {exp_dir}\n")
        if readme.exists():
            print(readme.read_text())
        else:
            print("(No README.md found — delete the directory and re-run to regenerate.)")
        return

    setup = _resolve_setup(args)
    mode = setup["mode"]

    has_chat_template = True
    if mode in ("prompt-distill", "full-chat"):
        has_chat_template = _has_chat_template(args.model_name)

    exp_dir.mkdir(parents=True)
    print(f"\nCreated {exp_dir}/")
    print("Configs:")

    from ptp.data.chat import DEFAULT_USER_ROLES, DEFAULT_ASSISTANT_ROLES
    lora_block = _lora_config_block(setup["lora_rank"]) if setup["use_lora"] else ""
    replacements = {
        "MODEL_ID": args.model_name,
        "DATASET_NAME": args.dataset_name,
        "CONVERSATION_KEYS": _yaml_list(setup["conversation_keys"] or []),
        "USER_ROLES": _yaml_list(setup["user_roles"] or DEFAULT_USER_ROLES),
        "ASSISTANT_ROLES": _yaml_list(setup["assistant_roles"] or DEFAULT_ASSISTANT_ROLES),
        "TEXT_COLUMN": setup["text_column"] or "",
        "PROMPT_SCHEME": _yaml_str(setup.get("prompt_scheme") or ""),
        "WANDB_PROJECT": setup["wandb_project"],
        "MAX_SEQUENCE_LENGTH": str(setup["max_sequence_length"]),
        "COMPLETION_LENGTH": str(setup["completion_length"]),
        "NUM_COMPLETIONS": str(setup["num_completions"]),
        "LORA_CONFIG": lora_block,
    }

    if mode == "prompt-distill":
        write_config(exp_dir, "pregenerate.yaml", replacements)
        write_config(exp_dir, "train_prompt_distill.yaml", replacements, dst_name="train.yaml")
    elif mode == "prompt-scheme-distill":
        write_config(exp_dir, "pregenerate_prompt_scheme.yaml", replacements, dst_name="pregenerate.yaml")
        write_config(exp_dir, "train_prompt_distill.yaml", replacements, dst_name="train.yaml")
    elif mode == "full-chat":
        write_config(exp_dir, "train_full_chat.yaml", replacements, dst_name="train.yaml")
    elif mode == "fixed":
        write_config(exp_dir, "train_fixed.yaml", replacements, dst_name="train.yaml")

    readme_text = make_readme(args.model_name, args.dataset_name, mode, exp_dir,
                              has_chat_template=has_chat_template)
    (exp_dir / "README.md").write_text(readme_text)

    print("\n" + "─" * 60)
    print(_readme_for_cli(readme_text).rstrip())
    print("─" * 60)


if __name__ == "__main__":
    main()
