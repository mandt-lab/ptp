"""
Tests for the ptp_distill CLI (config generation + README).

These tests exercise the setup logic without hitting HuggingFace or GPU.
They work by calling main() directly with sys.argv patched, and by pointing
CHECKPOINTS_DIR / CONFIGS_DIR at temporary directories.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ptp.cli.distill as cli_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_distill(tmp_path, monkeypatch, argv: list[str],
                *, interactive_answers: list[str] | None = None):
    """
    Run ptp_distill with the given CLI args.

    If interactive_answers is provided, each call to input() returns the next
    value from the list (empty string = accept default).
    """
    monkeypatch.setattr(cli_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(cli_module, "CONFIGS_DIR", Path(__file__).parent.parent / "configs")
    monkeypatch.setattr(cli_module, "_has_chat_template", lambda _model: True)

    answers = list(interactive_answers or [])
    monkeypatch.setattr("builtins.input", lambda _prompt="": answers.pop(0))

    monkeypatch.setattr(sys, "argv", ["ptp_distill"] + argv)
    cli_module.main()


def exp_dir(tmp_path, model="EleutherAI/pythia-160m-v0", dataset="allenai/tulu-3-sft-mixture"):
    return tmp_path / "checkpoints" / f"{cli_module.slugify(model)}_{cli_module.slugify(dataset)}"


# ---------------------------------------------------------------------------
# ptp_distill — non-interactive (--mode flag) tests
# ---------------------------------------------------------------------------

class TestPtpDistillNonInteractive:

    def test_prompt_distill_creates_both_configs(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "prompt-distill", "--conversation-key", "messages"])
        d = exp_dir(tmp_path)
        assert (d / "pregenerate.yaml").exists()
        assert (d / "train.yaml").exists()
        assert (d / "README.md").exists()

    def test_full_chat_creates_only_train_yaml(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "full-chat", "--conversation-key", "messages"])
        d = exp_dir(tmp_path)
        assert not (d / "pregenerate.yaml").exists()
        assert (d / "train.yaml").exists()

    def test_fixed_mode_creates_only_train_yaml(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "wikipedia",
                     "--mode", "fixed", "--text-column", "text"])
        d = exp_dir(tmp_path, dataset="wikipedia")
        assert not (d / "pregenerate.yaml").exists()
        assert (d / "train.yaml").exists()

    def test_model_id_substituted(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "prompt-distill", "--conversation-key", "messages"])
        content = (exp_dir(tmp_path) / "train.yaml").read_text()
        assert "EleutherAI/pythia-160m-v0" in content
        assert "__MODEL_ID__" not in content

    def test_dataset_id_substituted(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "prompt-distill", "--conversation-key", "messages"])
        content = (exp_dir(tmp_path) / "pregenerate.yaml").read_text()
        assert "allenai/tulu-3-sft-mixture" in content
        assert "__DATASET_NAME__" not in content

    def test_conversation_key_substituted(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "prompt-distill", "--conversation-key", "conversations"])
        content = (exp_dir(tmp_path) / "pregenerate.yaml").read_text()
        assert "conversations" in content
        assert "__CONVERSATION_KEYS__" not in content

    def test_conversation_keys_substituted(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "full-chat", "--conversation-keys", "messages,conversations,data"])
        content = (exp_dir(tmp_path) / "train.yaml").read_text()
        assert "conversation_keys: [messages, conversations, data]" in content

    def test_chat_mode_defaults_to_common_conversation_keys(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "full-chat"])
        content = (exp_dir(tmp_path) / "train.yaml").read_text()
        assert "conversation_keys: [messages, conversations, data]" in content

    def test_default_user_roles_substituted(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "prompt-distill", "--conversation-key", "messages"])
        content = (exp_dir(tmp_path) / "pregenerate.yaml").read_text()
        assert "user_roles:" in content
        assert "__USER_ROLES__" not in content

    def test_custom_roles_substituted(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "full-chat", "--conversation-key", "messages",
                     "--user-roles", "human,user",
                     "--assistant-roles", "assistant,gpt"])
        content = (exp_dir(tmp_path) / "train.yaml").read_text()
        assert "human" in content
        assert "gpt" in content
        assert "__USER_ROLES__" not in content
        assert "__ASSISTANT_ROLES__" not in content

    def test_readme_mentions_ptp_pregenerate_for_prompt_distill(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "prompt-distill", "--conversation-key", "messages"])
        readme = (exp_dir(tmp_path) / "README.md").read_text()
        assert "ptp_pregenerate" in readme
        assert "ptp_train" in readme

    def test_readme_no_pregenerate_for_full_chat(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "full-chat", "--conversation-key", "messages"])
        readme = (exp_dir(tmp_path) / "README.md").read_text()
        import re
        assert not re.search(r"`ptp_pregenerate\s", readme)
        assert "ptp_train" in readme

    def test_readme_is_short(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "full-chat", "--conversation-key", "messages"])
        lines = (exp_dir(tmp_path) / "README.md").read_text().strip().splitlines()
        assert len(lines) <= 10, f"README has {len(lines)} lines (limit is 10)"

    def test_existing_dir_does_not_overwrite(self, tmp_path, monkeypatch, capsys):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "prompt-distill", "--conversation-key", "messages"])
        d = exp_dir(tmp_path)
        original_mtime = (d / "train.yaml").stat().st_mtime

        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture",
                     "--mode", "full-chat", "--conversation-key", "data"])
        assert (d / "train.yaml").stat().st_mtime == original_mtime
        assert "already exists" in capsys.readouterr().out

    def test_slugify(self):
        assert cli_module.slugify("EleutherAI/pythia-160m-v0") == "pythia-160m-v0"
        assert cli_module.slugify("allenai/tulu-3-sft-mixture") == "tulu-3-sft-mixture"
        assert cli_module.slugify("some/path/My Dataset") == "my-dataset"


# ---------------------------------------------------------------------------
# ptp_distill — interactive flow tests
# ---------------------------------------------------------------------------

class TestPtpDistillInteractive:
    """
    The two-level flow:
      Level 1: "1" = Interactive, "2" = Fixed
      Level 2 (if Interactive): "1" = Chat (prompt-distill), "2" = Prompt Scheme (prompt-scheme-distill)
      Level 2 (if Fixed): "1" = Chat (full-chat), "2" = Fixed text
      Then: field name, user roles, assistant roles (or text column / prompt scheme)
    """

    def _answers_prompt_distill(self, conv_key="messages",
                                 user_roles="", assistant_roles="",
                                 wandb_project="", max_seq_len="", completion_len="",
                                 num_completions="", use_lora="", lora_rank=""):
        # L1=1 (Interactive), L2=1 (Chat), conv_key, user_roles, assistant_roles,
        # wandb_project, max_seq_len, completion_len, num_completions, use_lora, lora_rank
        return ["1", "1", conv_key, user_roles, assistant_roles,
                wandb_project, max_seq_len, completion_len, num_completions, use_lora, lora_rank]

    def _answers_prompt_scheme_distill(self, prompt_scheme="<|user|>\n{text}</s>\n<|assistant|>\n",
                                        wandb_project="", max_seq_len="", completion_len="",
                                        num_completions="", use_lora="", lora_rank=""):
        # L1=1 (Interactive), L2=2 (Prompt Scheme), prompt_scheme,
        # wandb_project, max_seq_len, completion_len, num_completions, use_lora, lora_rank
        return ["1", "2", prompt_scheme,
                wandb_project, max_seq_len, completion_len, num_completions, use_lora, lora_rank]

    def _answers_full_chat(self, conv_key="messages",
                            user_roles="", assistant_roles="",
                            wandb_project="", max_seq_len="", completion_len="",
                            num_completions="", use_lora="", lora_rank=""):
        # L1=2 (Fixed), L2=1 (Chat), conv_key, user_roles, assistant_roles,
        # wandb_project, max_seq_len, completion_len, num_completions, use_lora, lora_rank
        return ["2", "1", conv_key, user_roles, assistant_roles,
                wandb_project, max_seq_len, completion_len, num_completions, use_lora, lora_rank]

    def _answers_fixed(self, text_col="text", wandb_project="", max_seq_len="", completion_len="",
                       num_completions="", use_lora="", lora_rank=""):
        # L1=2 (Fixed), L2=2 (Fixed text), text_col,
        # wandb_project, max_seq_len, completion_len, num_completions, use_lora, lora_rank
        return ["2", "2", text_col,
                wandb_project, max_seq_len, completion_len, num_completions, use_lora, lora_rank]

    def test_interactive_prompt_distill(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture"],
                    interactive_answers=self._answers_prompt_distill())
        d = exp_dir(tmp_path)
        assert (d / "pregenerate.yaml").exists()
        assert (d / "train.yaml").exists()

    def test_interactive_full_chat(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture"],
                    interactive_answers=self._answers_full_chat())
        d = exp_dir(tmp_path)
        assert not (d / "pregenerate.yaml").exists()
        assert (d / "train.yaml").exists()

    def test_interactive_fixed(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture"],
                    interactive_answers=self._answers_fixed())
        d = exp_dir(tmp_path)
        assert not (d / "pregenerate.yaml").exists()
        assert (d / "train.yaml").exists()

    def test_interactive_prompt_scheme_distill(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture"],
                    interactive_answers=self._answers_prompt_scheme_distill())
        d = exp_dir(tmp_path)
        assert (d / "pregenerate.yaml").exists()
        assert (d / "train.yaml").exists()
        content = (d / "pregenerate.yaml").read_text()
        assert "PromptSchemeDataModule" in content
        assert "__PROMPT_SCHEME__" not in content

    def test_interactive_custom_conv_key(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture"],
                    interactive_answers=self._answers_full_chat(conv_key="conversations"))
        content = (exp_dir(tmp_path) / "train.yaml").read_text()
        assert "conversations" in content

    def test_interactive_custom_roles(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture"],
                    interactive_answers=self._answers_full_chat(
                        user_roles="human, USER",
                        assistant_roles="assistant, model"))
        content = (exp_dir(tmp_path) / "train.yaml").read_text()
        assert "human" in content
        assert "model" in content

    def test_interactive_default_roles_on_empty_input(self, tmp_path, monkeypatch):
        """Pressing Enter on role prompts should use the defaults."""
        from ptp.data.chat import DEFAULT_USER_ROLES, DEFAULT_ASSISTANT_ROLES
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture"],
                    interactive_answers=self._answers_full_chat())  # "" = use default
        content = (exp_dir(tmp_path) / "train.yaml").read_text()
        for role in DEFAULT_USER_ROLES:
            assert role in content
        for role in DEFAULT_ASSISTANT_ROLES:
            assert role in content

    def test_interactive_text_column(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture"],
                    interactive_answers=self._answers_fixed(text_col="document"))
        content = (exp_dir(tmp_path) / "train.yaml").read_text()
        assert "document" in content

    def test_interactive_default_text_column(self, tmp_path, monkeypatch):
        run_distill(tmp_path, monkeypatch,
                    ["EleutherAI/pythia-160m-v0", "allenai/tulu-3-sft-mixture"],
                    interactive_answers=self._answers_fixed(text_col=""))  # "" = default "text"
        content = (exp_dir(tmp_path) / "train.yaml").read_text()
        assert "text_column" in content


# ---------------------------------------------------------------------------
# ptp_pregenerate guard tests
# ---------------------------------------------------------------------------

class TestPtpPregenerate:

    def test_missing_experiment_dir_exits(self, tmp_path, monkeypatch):
        import ptp.cli.pregenerate as cmd
        monkeypatch.setattr(sys, "argv", ["ptp_pregenerate", str(tmp_path / "nonexistent")])
        with pytest.raises(SystemExit) as exc:
            cmd.main()
        assert exc.value.code != 0

    def test_no_pregenerate_yaml_for_direct_mode(self, tmp_path, monkeypatch):
        import ptp.cli.pregenerate as cmd
        exp = tmp_path / "exp"
        exp.mkdir()
        (exp / "train.yaml").write_text("# stub\n")
        monkeypatch.setattr(sys, "argv", ["ptp_pregenerate", str(exp)])
        with pytest.raises(SystemExit) as exc:
            cmd.main()
        assert exc.value.code != 0

    def test_existing_data_dir_exits_cleanly(self, tmp_path, monkeypatch, capsys):
        import ptp.cli.pregenerate as cmd
        exp = tmp_path / "exp"
        exp.mkdir()
        (exp / "pregenerate.yaml").write_text("# stub\n")
        (exp / "data").mkdir()
        monkeypatch.setattr(sys, "argv", ["ptp_pregenerate", str(exp)])
        with pytest.raises(SystemExit) as exc:
            cmd.main()
        assert exc.value.code == 0
        assert "already exists" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ptp_train guard tests
# ---------------------------------------------------------------------------

class TestPtpTrain:

    def test_missing_experiment_dir_exits(self, tmp_path, monkeypatch):
        import ptp.cli.train as cmd
        monkeypatch.setattr(sys, "argv", ["ptp_train", str(tmp_path / "nonexistent")])
        with pytest.raises(SystemExit) as exc:
            cmd.main()
        assert exc.value.code != 0

    def test_missing_train_yaml_exits(self, tmp_path, monkeypatch):
        import ptp.cli.train as cmd
        exp = tmp_path / "exp"
        exp.mkdir()
        monkeypatch.setattr(sys, "argv", ["ptp_train", str(exp)])
        with pytest.raises(SystemExit) as exc:
            cmd.main()
        assert exc.value.code != 0

    def test_missing_data_dir_prints_warning(self, tmp_path, monkeypatch, capsys):
        import ptp.training as training_mod
        import ptp.cli.train as cmd
        exp = tmp_path / "exp"
        exp.mkdir()
        (exp / "train.yaml").write_text("# stub\n")
        (exp / "pregenerate.yaml").write_text("# stub\n")

        monkeypatch.setattr(training_mod, "main", lambda **_: None)

        monkeypatch.setattr(sys, "argv", ["ptp_train", str(exp)])
        cmd.main()
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "ptp_pregenerate" in captured.out


# ---------------------------------------------------------------------------
# Role handling in _convert_to_chat_format
# ---------------------------------------------------------------------------

class TestConvertToChatFormat:

    def _make_dataset(self, user_roles=None, assistant_roles=None):
        from ptp.data.chat import ChatDataset
        ds = ChatDataset.__new__(ChatDataset)
        ds.user_roles = user_roles or ["user", "human"]
        ds.assistant_roles = assistant_roles or ["assistant", "gpt"]
        return ds

    def test_standard_roles(self):
        ds = self._make_dataset()
        conv = ds._convert_to_chat_format([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])
        assert conv == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

    def test_sharegpt_roles(self):
        ds = self._make_dataset()
        conv = ds._convert_to_chat_format([
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi"},
        ])
        assert conv == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

    def test_system_message_skipped(self):
        ds = self._make_dataset()
        conv = ds._convert_to_chat_format([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])
        assert len(conv) == 2
        assert conv[0]["role"] == "user"

    def test_unknown_role_raises(self):
        ds = self._make_dataset()
        with pytest.raises(ValueError, match="Unknown role"):
            ds._convert_to_chat_format([
                {"role": "moderator", "content": "test"},
            ])

    def test_custom_user_roles(self):
        ds = self._make_dataset(user_roles=["USER", "player"])
        conv = ds._convert_to_chat_format([
            {"role": "player", "content": "move"},
            {"role": "assistant", "content": "ok"},
        ])
        assert conv[0]["role"] == "user"

    def test_custom_assistant_roles(self):
        ds = self._make_dataset(assistant_roles=["model", "bot"])
        conv = ds._convert_to_chat_format([
            {"role": "user", "content": "hi"},
            {"role": "model", "content": "hello"},
        ])
        assert conv[1]["role"] == "assistant"

    def test_alternating_string_list(self):
        ds = self._make_dataset()
        conv = ds._convert_to_chat_format(["Hello", "Hi", "How are you?", "Fine"])
        assert [t["role"] for t in conv] == ["user", "assistant", "user", "assistant"]
        assert conv[0]["content"] == "Hello"
        assert conv[1]["content"] == "Hi"

    def test_multiple_user_role_variants(self):
        """Datasets sometimes mix 'human' and 'user' in the same file."""
        ds = self._make_dataset(user_roles=["user", "human"])
        conv = ds._convert_to_chat_format([
            {"role": "human", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ])
        assert all(t["role"] in ("user", "assistant") for t in conv)
        assert conv[0]["role"] == "user"
        assert conv[2]["role"] == "user"
