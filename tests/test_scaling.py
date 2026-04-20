import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from ptp.scaling import CONFIGS, make_scaling_llama


@pytest.mark.parametrize("config_name", CONFIGS.keys())
def test_scaling(config_name: str):
    model = make_scaling_llama(config_name, vocab_size=0, max_position_embeddings=0)
    expected_num_parameters = (
            float(config_name[:-1])
            * {
                "k": 1_000,
                "m": 1_000_000,
                "b": 1_000_000_000,
            }[config_name[-1].lower()]
    )
    total_params = sum(p.numel() for p in model.parameters())
    assert expected_num_parameters * 0.9 <= total_params <= expected_num_parameters * 1.15, \
        f"Model {config_name} has {total_params} parameters, expected around {expected_num_parameters}"
