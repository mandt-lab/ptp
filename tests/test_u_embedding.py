import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from ptp.transformer import MixedTransformerModel


@pytest.mark.parametrize("embedding_type", [
    "sawtooth",
    "quarter_cos",
    "binary",
    "linear_interpolation",
    "round"
])
def test_u_embedding(embedding_type: str):
    config = torch.nn.Module()
    config.__dict__["hidden_size"] = 128
    fake_model = torch.nn.Module()
    fake_model.config = config

    model = MixedTransformerModel(adapter_name=embedding_type, model_id=fake_model)
    us = torch.linspace(0, 1, steps=100)[:-1, None]
    embeddings = model.u_embed(us)
    assert embeddings.shape == (us.shape[0], 1, config.hidden_size)

    # Check that embeddings are different for different u values
    diff = torch.abs(embeddings[1:] - embeddings[:-1])
    if embedding_type == "round":  # round embedding will have many repeated values
        assert torch.any(diff > 0), f"Embeddings are identical for embedding type {embedding_type}"
    else:
        assert torch.all(diff > 0), f"Embeddings are not varying for embedding type {embedding_type}"
