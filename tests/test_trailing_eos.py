import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest
from ptp.data.pregenerate import fix_trailing_eos


def one_hot(vocab_size, idx, device):
    row = torch.zeros(vocab_size, device=device)
    row[idx] = 1.0
    return row

@pytest.mark.parametrize("output_ids,expected_masked", [
    # Case 1: No EOS → unchanged
    (torch.tensor([[1, 2, 3]]), []),

    # Case 2: One EOS at the end → unchanged
    (torch.tensor([[1, 2, 5]]), []),

    # Case 3: Two EOS at the end → last pos forced to EOS
    (torch.tensor([[1, 5, 5]]), [(0, 2)]),

    # Case 4: Three EOS at the end → last two pos forced to EOS
    (torch.tensor([[1, 5, 5, 5]]), [(0, 2), (0, 3)]),

    # Case 5: Multiple EOS but not at end → unchanged
    (torch.tensor([[5, 1, 5]]), []),

    # Case 6: Multiple sequences in batch
    (torch.tensor([[1, 2, 5],    # unchanged
                   [1, 5, 5],    # last pos forced to EOS
                   [5, 1, 5],    # unchanged
                   [1, 2, 3]]),  # unchanged
     [(1, 2)]),

    # Case 7: Edge case T=1 with EOS → unchanged
    (torch.tensor([[5]]), []),

    # Case 8: Several long sequences
    (torch.tensor([[1, 2, 3, 4, 5, 5],   # last two pos forced to EOS
                   [1, 2, 3, 4, 5, 1],   # unchanged
                   [5, 5, 5, 5, 5, 5],   # last two pos forced to EOS
                   [1, 2, 3, 4, 3, 2]]), # unchanged
     [(0, 5), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]),
])
def test_fix_trailing_eos(output_ids, expected_masked):
    eos_id = 5
    B, T = output_ids.shape
    V = 6

    # start with uniform probs
    probs = torch.full((B, T, V), 1.0 / V)
    orig_probs = probs.clone()

    fix_trailing_eos(output_ids.clone(), probs, eos_id)

    eos_one_hot = one_hot(V, eos_id, probs.device)

    # check expected masked positions are replaced
    for b, t in expected_masked:
        assert torch.allclose(probs[b, t], eos_one_hot)

    # check all other positions remain unchanged
    for b in range(B):
        for t in range(T):
            if (b, t) not in expected_masked:
                assert torch.allclose(probs[b, t], orig_probs[b, t])
