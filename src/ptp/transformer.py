from contextlib import contextmanager
from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners import lora
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
from ptp.patch_norms import patch_llama_network

from ptp import auxiliary_embed


class CustomCheckpointWrapper(nn.Module):
    def __init__(self, layer, use_reentrant: bool = False, preserve_rng_state: bool = True):
        super().__init__()
        self.layer = layer
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state

    def _call(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # kwargs are supported in modern PyTorch. If your env is old, wrap inputs into a tuple-only signature.
        return checkpoint(
            self._call,
            *args,
            use_reentrant=self.use_reentrant,
            preserve_rng_state=self.preserve_rng_state,
            **kwargs,
        )

    def __getattr__(self, name):
        """Forward attribute access to the wrapped layer"""
        if name in ['layer', 'use_reentrant', 'preserve_rng_state']:
            return super().__getattr__(name)
        return getattr(self.layer, name)


# Example: wrap every 3rd layer in model.model.layers
def enable_custom_checkpointing(model, every_n: int = 3, start_index: int = 0,
                                use_reentrant: bool = False, preserve_rng_state: bool = True, verbose: bool = True):
    """
    Wraps every Nth block under model.model.layers with a checkpoint wrapper.
    Adjust the path if your model stores blocks elsewhere (e.g., model.transformer.h for GPT-2).
    Returns number of layers wrapped.
    """
    # Turn off use_cache during training when using checkpointing
    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    # You may need to change this path depending on the HF architecture you use.
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("Could not find 'model.model.layers'. Adjust enable_custom_checkpointing() for your model arch.")

    layers = model.model.layers
    wrapped = 0
    for i, layer in enumerate(layers):
        if i >= start_index and every_n > 0 and ((i - start_index) % every_n == 0):
            if verbose:
                print(f"[GC] Wrapping layer index {i} with checkpoint wrapper")
            layers[i] = CustomCheckpointWrapper(layer,
                                                use_reentrant=use_reentrant,
                                                preserve_rng_state=preserve_rng_state)
            wrapped += 1
    return wrapped


class GatedLinearLoraMerged(nn.Module):
    def __init__(self, inner: nn.Linear, lora_a: nn.Linear, lora_b: nn.Linear, scaling, gate_window: int) -> None:
        super().__init__()
        self.gate_window  = gate_window
        self.out_features = inner.out_features
        self.rank         = lora_a.out_features   # = LORA_RANK = 128

        # Precompute W_fused = W_inner + W_B @ W_A, then merge with W_A.
        # Shape: (out + rank, in).
        with torch.no_grad():
            W_fused = inner.weight.float() + scaling * (lora_b.weight.float() @ lora_a.weight.float())
            W_fused = W_fused.to(inner.weight.dtype)
            W_merged = torch.cat([W_fused, lora_a.weight], dim=0)
            bias = inner.bias.float() if inner.bias is not None else None

        self.register_buffer("W_merged", W_merged)   # (out + rank, in)
        if bias is not None:
            self.register_buffer("bias", bias)        # (out,)
        else:
            self.bias = None
        # lora_b stays as a module so it participates in .to(dtype) etc.
        self.lora_scaling = scaling
        self.lora_b = lora_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gw   = self.gate_window
        out  = self.out_features
        rank = self.rank

        # -----------------------------------------------------------------
        # 1. Single merged GEMM: x → (B, T, out + rank)
        #    y_fused already includes the LoRA contribution for all tokens.
        # -----------------------------------------------------------------
        merged = torch.functional.F.linear(x, self.W_merged)     # (B, T, out + rank)

        # -----------------------------------------------------------------
        # 2. Split the merged output (views, no copies).
        # -----------------------------------------------------------------
        y     = merged[..., :out]               # (B, T, out)  – fused result
        h_all = merged[..., out:]               # (B, T, rank) – lora_a for all tokens

        # Add bias to the base output (bias only applies to the first `out` features).
        if self.bias is not None:
            y = y + self.bias

        if gw >= x.shape[-2]:
            # All tokens gated — LoRA is applied to every token, no correction needed.
            return y

        # -----------------------------------------------------------------
        # 3. Subtract LoRA from non-gated tokens (the first T - gw).
        #    This is the small correction GEMM: only (T - gw) tokens.
        # -----------------------------------------------------------------
        non_gated = x.shape[-2] - gw
        h_non_gated = h_all[:, :non_gated, :]          # (B, T-gw, rank)
        y = y.clone()
        y[:, :non_gated, :] -= self.lora_scaling * self.lora_b(h_non_gated)

        return y

class TransformerModel(torch.nn.Module):
    def __init__(self, model_id, reset_parameters=False, dtype: torch.dtype | str = torch.float32,
                 use_gradient_checkpointing: bool = False, lora_config=None, patch_norm: bool = False,
                 gate_window=0, **kwargs):
        super().__init__()
        # Convert string dtype to torch dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        if isinstance(model_id, torch.nn.Module):
            self.tokenizer = None
            self.model = model_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Use torch_dtype parameter for Hugging Face compatibility
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype,
                attn_implementation="flex_attention",
                **kwargs
            )
            if reset_parameters:
                self.model = AutoModelForCausalLM.from_config(self.model.config)
        if patch_norm:
            patch_llama_network(self.model)
        self.norm_patched = patch_norm

        # Enable gradient checkpointing to save memory (trades compute for memory)
        if use_gradient_checkpointing:
            print("Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        from peft import get_peft_model, LoraConfig, TaskType
        if lora_config is not None:
            print("Applying LoRA adapters")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **lora_config
            )
            self.model = get_peft_model(self.model, peft_config)
        self.has_adapter_state = False
        self.inference_mode = False
        self._saved_lora_modules: dict[str, lora.Linear] = {}

        self.model.train()

    @contextmanager
    def enable_adapters(self, enabled: bool = True):
        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=enabled)
                else:
                    module.disable_adapters = not enabled
        self.has_adapter_state = True
        yield
        self.has_adapter_state = False

    def enter_inference_mode(self, gate_window: int):
        """Replace all lora.Linear layers with fused GatedLinearLoraMerged for inference."""
        assert not self.inference_mode, "Already in inference mode"
        self._saved_lora_modules = {}
        for name, module in list(self.model.named_modules()):
            if isinstance(module, lora.Linear):
                *path, attr = name.split('.')
                parent = self.model
                for part in path:
                    parent = getattr(parent, part)
                self._saved_lora_modules[name] = module
                setattr(parent, attr, GatedLinearLoraMerged(
                    module.base_layer,
                    module.lora_A['default'],
                    module.lora_B['default'],
                    module.scaling['default'],
                    gate_window=gate_window,
                ))
        self.inference_mode = True
        self.model.eval()

    def exit_inference_mode(self):
        """Restore lora layers for training."""
        assert self.inference_mode, "Not in inference mode"
        for name, module in self._saved_lora_modules.items():
            *path, attr = name.split('.')
            parent = self.model
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, attr, module)
        self._saved_lora_modules = {}
        self.inference_mode = False
        self.model.train()

    def set_gate_window(self, gate_window: int):
        """Update gate_window on all GatedLinearLoraMerged layers in-place."""
        for module in self.model.modules():
            if isinstance(module, GatedLinearLoraMerged):
                module.gate_window = gate_window

    def inference_forward(self, *args, **kwargs):
        """Forward pass in inference mode, bypassing the adapter-state check."""
        assert self.inference_mode, "Call enter_inference_mode() before inference_forward()"
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        if not self.has_adapter_state:
            raise RuntimeError(
                "Adapters state not set. Use 'with model.enable_adapters()' or 'with model.disable_adapters()' context manager.")
        return self.model(*args, **kwargs)


class MixedTransformerModel(TransformerModel):
    def __init__(self, shift_positions: bool = True, adapter_name="linear_interpolation",
                 adapter_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        if adapter_name == "sawtooth":
            adapter_class = getattr(auxiliary_embed, "SawtoothFloatEmbedding")
        elif adapter_name == "quarter_cos":
            adapter_class = getattr(auxiliary_embed, "QuarterCosEmbedding")
        elif adapter_name == "binary":
            adapter_class = getattr(auxiliary_embed, "BinaryFloatEmbedding")
        elif adapter_name == "linear_interpolation":
            adapter_class = getattr(auxiliary_embed, "LinearInterpolationEmbedding")
        elif adapter_name == "round":
            adapter_class = getattr(auxiliary_embed, "RoundingEmbedding")
        else:
            raise ValueError(f"Unknown adapter_name {adapter_name}")
        if adapter_kwargs is None:
            adapter_kwargs = {}
        self.u_embed = adapter_class(self.model.config.hidden_size, **adapter_kwargs)
        self.shift_positions = shift_positions

    def _get_padding_token_id(self) -> int:
        """Get the padding token ID from tokenizer, fallback to eos_token_id if not available."""
        if self.tokenizer is None:
            from ptp.data.collate import IGNORE_INDEX
            return IGNORE_INDEX
        
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id defined.")

    def _replace_ignore_index(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace all IGNORE_INDEX tokens with the tokenizer's padding token."""
        from ptp.data.collate import IGNORE_INDEX
        
        if IGNORE_INDEX not in input_ids:
            return input_ids
        
        padding_token_id = self._get_padding_token_id()
        input_ids = input_ids.clone()
        input_ids[input_ids == IGNORE_INDEX] = padding_token_id
        return input_ids

    def ar_forward(self, input_ids, attention_mask=None) -> Any:
        # Replace IGNORE_INDEX tokens with padding token before passing to model
        input_ids = self._replace_ignore_index(input_ids)
        
        with self.enable_adapters(enabled=False):
            teacher_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                auxiliaries=None
            )
        return teacher_outputs

    def forward(self, input_ids, auxiliaries, ar_outputs=None,
                input_mask=None, auxiliary_mask=None, auxiliary_position_ids=None):
        if ar_outputs is None:
            ar_outputs = self.ar_forward(input_ids, input_mask)
        kv_cache = ar_outputs.past_key_values
        with self.enable_adapters(enabled=True):
            if self.shift_positions:
                if auxiliary_position_ids is None:
                    if input_mask is None:
                        input_lengths = torch.full(
                            (auxiliaries.shape[0],), input_ids.shape[1],
                            device=auxiliaries.device
                        )
                    else:
                        if len(input_mask.shape) != 2:
                            raise ValueError(f"input_mask must be 2D, got {input_mask.shape}")
                        input_lengths = input_mask.sum(dim=1)
                    auxiliary_position_ids = torch.arange(-1, auxiliaries.shape[1] - 1, device=auxiliaries.device)
                    auxiliary_position_ids = auxiliary_position_ids[None].repeat(auxiliaries.shape[0], 1)
                    auxiliary_position_ids += input_lengths[:, None]
                else:
                    auxiliary_position_ids = auxiliary_position_ids - 1
            auxiliary_embeds = self.u_embed(auxiliaries)
            if self.norm_patched:
                kwargs = dict(condition=auxiliary_embeds)
            else:
                kwargs = dict()
            completion_outputs = self.model(
                inputs_embeds=auxiliary_embeds,
                position_ids=auxiliary_position_ids,
                attention_mask=auxiliary_mask,
                past_key_values=kv_cache,
                use_cache=False,
                **kwargs
            )
        return ar_outputs, completion_outputs

    def inference_forward(self, input_ids, auxiliaries=None, past_key_values=None,
                          use_cache=True, attention_mask=None, position_ids=None):
        """
        Single-pass inference forward combining AR tokens and auxiliary (proposed) tokens.

        Embeds input_ids with the model's token embeddings and auxiliaries with u_embed,
        concatenates them, and runs a single forward pass through the model with
        gate_window = len(auxiliaries) so that LoRA only applies to the proposed tokens.

        Args:
            input_ids:        (B, T) token IDs for the AR/verify portion
            auxiliaries:      (B, A) z-values for the proposed tokens, or None
            past_key_values:  KV cache from previous steps
            use_cache:        whether to return the updated cache
            attention_mask:   custom attention mask (e.g. causal mask with KV cache offset)
            position_ids:     (B, T+A) explicit position IDs for all tokens
        """
        embed_layer = self.model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)            # (B, T, H)

        if auxiliaries is not None and auxiliaries.shape[1] > 0:
            aux_embeds = self.u_embed(auxiliaries)        # (B, A, H)
            self.set_gate_window(auxiliaries.shape[1])
            inputs_embeds = torch.cat([inputs_embeds, aux_embeds], dim=1)  # (B, T+A, H)
        else:
            self.set_gate_window(0)

        return super().inference_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            kernel_options={"BACKEND": "TRITON"}
        )
