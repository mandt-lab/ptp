from typing import Optional, Unpack

import torch
from transformers import Cache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.utils import TransformersKwargs


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class PatchedDecoderLayer(LlamaDecoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        linear = torch.nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        linear.weight.data.fill_(0)
        linear.bias.data.fill_(0)
        self.adaLN_modulation = torch.nn.ModuleList([
            torch.nn.SiLU(),
            linear
        ])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            condition: Optional[torch.Tensor] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if condition is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        # Compute modulation parameters, like in DiT
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6,
                                                                                                                dim=-1)
        residual = hidden_states
        hidden_states = self.input_layernorm(modulate(hidden_states, shift_msa, scale_msa))
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + gate_msa * hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(modulate(hidden_states, shift_mlp, scale_mlp))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp * hidden_states
        return hidden_states


def patch_llama_network(network: torch.nn.Module) -> None:
    num_changed = 0
    for name, module in network.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            # Trace the module
            parent_module = network
            for part in name.split(".")[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name.split(".")[-1], PatchedDecoderLayer(module.config, int(name.split(".")[-1])))
            num_changed += 1
    if num_changed == 0:
        raise ValueError("No LlamaDecoderLayer found in the network to patch.")
