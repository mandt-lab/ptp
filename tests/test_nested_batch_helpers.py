"""
Tests for the helper methods extracted from prepare_nested_batch in lit.py:

  - _make_completion_positions
  - _gather_completion_ids   (including cross-document boundary logic)
  - _gather_bin_edges

And for the AR loss boundary masking in forward():
  - document-boundary positions are excluded from the AR loss
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from ptp.data.collate import IGNORE_INDEX


# ---------------------------------------------------------------------------
# Minimal stub so we can call instance methods without a real model
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    eos_token_id = 2


class _FakeModel:
    tokenizer = _FakeTokenizer()


class _Stub:
    """Exposes only what the helper methods actually read from `self`."""
    completion_attention_shape = "causal"
    model = _FakeModel()

    # Bind the real methods from lit.py onto this stub
    from ptp.lit import ParallelSamplingLightningModule as _M
    _make_completion_positions = _M._make_completion_positions
    _gather_completion_ids     = _M._gather_completion_ids
    _gather_bin_edges          = _M._gather_bin_edges


stub = _Stub()


# ===========================================================================
# _make_completion_positions
# ===========================================================================

class TestMakeCompletionPositions:
    def test_position_ids_are_start_plus_offsets(self):
        starts = torch.tensor([[2, 5]])       # B=1, N=2
        pos, valid, safe = stub._make_completion_positions(starts, completion_length=3,
                                                           seq_len=10, device="cpu")
        expected = torch.tensor([[[2, 3, 4], [5, 6, 7]]])
        assert pos.equal(expected)

    def test_valid_mask_true_within_seq_len(self):
        # starts=[7], length=4, seq_len=10  →  positions 7,8,9,10 → last is invalid
        starts = torch.tensor([[7]])
        _, valid, _ = stub._make_completion_positions(starts, completion_length=4,
                                                      seq_len=10, device="cpu")
        expected = torch.tensor([[[True, True, True, False]]])
        assert valid.equal(expected)

    def test_safe_positions_clamped_at_seq_len_minus_one(self):
        starts = torch.tensor([[8]])
        _, _, safe = stub._make_completion_positions(starts, completion_length=4,
                                                     seq_len=10, device="cpu")
        # positions 8,9,10,11 → clamped to 8,9,9,9
        expected = torch.tensor([[[8, 9, 9, 9]]])
        assert safe.equal(expected)

    def test_batch_and_multi_completion(self):
        starts = torch.tensor([[1, 3], [2, 4]])   # B=2, N=2
        pos, valid, safe = stub._make_completion_positions(starts, completion_length=2,
                                                           seq_len=6, device="cpu")
        assert pos.shape == (2, 2, 2)
        assert valid.shape == (2, 2, 2)
        assert safe.shape == (2, 2, 2)
        assert valid.all()   # all positions < 6


# ===========================================================================
# _gather_completion_ids  — basic gathering, no doc_ids
# ===========================================================================

class TestGatherCompletionIdsBasic:
    def _run(self, input_ids, safe_positions, valid_mask, num_completions,
             doc_ids=None, completion_doc_ids=None, starts=None):
        if starts is None:
            # derive starts from safe_positions[:, :, 0]
            starts = safe_positions[:, :, 0]
        return stub._gather_completion_ids(
            input_ids, safe_positions, valid_mask, num_completions,
            doc_ids, completion_doc_ids, starts)

    def test_gathers_correct_tokens(self):
        # Sequence: [10, 11, 12, 13, 14], one completion starting at 1, length 3
        input_ids     = torch.tensor([[10, 11, 12, 13, 14]])
        safe_positions = torch.tensor([[[1, 2, 3]]])
        valid_mask     = torch.ones(1, 1, 3, dtype=torch.bool)
        ids = self._run(input_ids, safe_positions, valid_mask, num_completions=1)
        assert ids.equal(torch.tensor([[[11, 12, 13]]]))

    def test_out_of_bounds_set_to_ignore(self):
        input_ids      = torch.tensor([[10, 11, 12]])
        safe_positions = torch.tensor([[[1, 2, 2]]])   # last position clamped
        valid_mask     = torch.tensor([[[True, True, False]]])
        ids = self._run(input_ids, safe_positions, valid_mask, num_completions=1)
        assert ids[0, 0, 2].item() == IGNORE_INDEX

    def test_all_valid_no_masking(self):
        input_ids      = torch.arange(20).unsqueeze(0)  # (1, 20)
        safe_positions = torch.tensor([[[3, 4, 5]]])
        valid_mask     = torch.ones(1, 1, 3, dtype=torch.bool)
        ids = self._run(input_ids, safe_positions, valid_mask, num_completions=1)
        assert ids.equal(torch.tensor([[[3, 4, 5]]]))


# ===========================================================================
# _gather_completion_ids  — cross-document boundary logic
# ===========================================================================

class TestGatherCompletionIdsDocBoundary:
    """
    Sequence layout used in most tests:
        positions: 0  1  2  3  4  5  6  7  8  9
        doc_ids:   0  0  0  0  1  1  1  2  2  2
        input_ids: 10 11 12 13 14 15 16 17 18 19

    A completion starting at position 2 (doc 0) of length 5 spans into doc 1.
    """

    SEQ_LEN = 10
    input_ids  = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    doc_ids    = torch.tensor([[0,  0,  0,  0,  1,  1,  1,  2,  2,  2]])

    def _gather(self, starts_val, length, completion_doc_ids=None):
        starts         = torch.tensor([[starts_val]])
        offsets        = torch.arange(length)
        positions      = starts[:, :, None] + offsets[None, None, :]     # (1,1,L)
        valid_mask     = positions < self.SEQ_LEN
        safe_positions = positions.clamp(max=self.SEQ_LEN - 1)
        return stub._gather_completion_ids(
            self.input_ids, safe_positions, valid_mask,
            num_completions=1,
            doc_ids=self.doc_ids,
            completion_doc_ids=completion_doc_ids,
            starts=starts,
        )

    def test_no_boundary_crossing_unchanged(self):
        # Completion at pos 1, length 3 → positions 1,2,3 all in doc 0
        ids = self._gather(starts_val=1, length=3)
        assert ids.equal(torch.tensor([[[11, 12, 13]]]))

    def test_first_cross_doc_gets_eos(self):
        # Completion at pos 2, length 5 → 2,3 in doc 0; 4 crosses into doc 1
        ids = self._gather(starts_val=2, length=5)
        # positions: 2→12, 3→13  (in-doc), 4→EOS, 5→IGNORE, 6→IGNORE
        assert ids[0, 0, 0].item() == 12
        assert ids[0, 0, 1].item() == 13
        assert ids[0, 0, 2].item() == _FakeTokenizer.eos_token_id
        assert ids[0, 0, 3].item() == IGNORE_INDEX
        assert ids[0, 0, 4].item() == IGNORE_INDEX

    def test_in_doc_tokens_before_boundary_unmasked(self):
        ids = self._gather(starts_val=1, length=5)
        # positions 1,2,3 in doc 0; 4 is boundary; 5 is ignored
        assert ids[0, 0, 0].item() == 11
        assert ids[0, 0, 1].item() == 12
        assert ids[0, 0, 2].item() == 13

    def test_completion_doc_ids_overrides_lookup(self):
        # Same layout but we explicitly say completion is in doc 1
        # so positions in doc 0 would be the "wrong" side
        completion_doc_ids = torch.tensor([[1]])   # (B=1, N=1)
        ids = self._gather(starts_val=4, length=4, completion_doc_ids=completion_doc_ids)
        # positions 4,5,6 in doc 1 (ok), 7 crosses into doc 2
        assert ids[0, 0, 0].item() == 14
        assert ids[0, 0, 1].item() == 15
        assert ids[0, 0, 2].item() == 16
        assert ids[0, 0, 3].item() == _FakeTokenizer.eos_token_id

    def test_start_exactly_at_doc_boundary_no_completion_doc_ids(self):
        # Completion starts at position 4, which is the first token of doc 1.
        # With completion_doc_ids=None we derive start_doc_ids from doc_ids[starts].
        # All target positions 4,5,6 are in doc 1 → nothing should be masked.
        ids = self._gather(starts_val=4, length=3)
        assert ids[0, 0, 0].item() == 14
        assert ids[0, 0, 1].item() == 15
        assert ids[0, 0, 2].item() == 16

    def test_boundary_is_last_completion_token(self):
        # Completion at pos 2, length 3 → positions 2,3,4.
        # Position 4 is the first cross-doc token AND the last slot → just EOS, no trailing IGNORE.
        ids = self._gather(starts_val=2, length=3)
        assert ids[0, 0, 0].item() == 12
        assert ids[0, 0, 1].item() == 13
        assert ids[0, 0, 2].item() == _FakeTokenizer.eos_token_id

    def test_no_eos_token_id_falls_back_to_ignore(self):
        """When tokenizer has no eos_token_id, the boundary position is also IGNORE_INDEX."""
        class _NoEosTokenizer:
            eos_token_id = None

        class _NoEosModel:
            tokenizer = _NoEosTokenizer()

        class _NoEosStub(_Stub):
            model = _NoEosModel()

        no_eos_stub = _NoEosStub()

        starts         = torch.tensor([[2]])
        positions      = starts[:, :, None] + torch.arange(4)[None, None, :]
        valid_mask     = positions < self.SEQ_LEN
        safe_positions = positions.clamp(max=self.SEQ_LEN - 1)
        ids = no_eos_stub._gather_completion_ids(
            self.input_ids, safe_positions, valid_mask,
            num_completions=1,
            doc_ids=self.doc_ids,
            completion_doc_ids=None,
            starts=starts,
        )
        # position 4 is first cross-doc — should be IGNORE_INDEX, not EOS
        assert ids[0, 0, 2].item() == IGNORE_INDEX
        assert ids[0, 0, 3].item() == IGNORE_INDEX


# ===========================================================================
# _gather_bin_edges
# ===========================================================================

class TestGatherBinEdges:
    def test_gathers_at_shifted_positions(self):
        # left_bin_edges[b, p] corresponds to position p+1 in input_ids
        # so for a completion token at position 3, we look at edge index 2 (= 3 - 1)
        B, S, N, L = 1, 8, 1, 3
        left  = torch.arange(S, dtype=torch.float).unsqueeze(0)   # [0,1,2,3,4,5,6,7]
        right = left + 0.5

        # completion at positions 3,4,5  → edge indices 2,3,4
        safe_positions = torch.tensor([[[3, 4, 5]]])
        valid_mask     = torch.ones(B, N, L, dtype=torch.bool)

        gl, gr = stub._gather_bin_edges(left, right, safe_positions, valid_mask,
                                        num_completions=N)
        assert gl.equal(torch.tensor([[[2.0, 3.0, 4.0]]]))
        assert gr.equal(torch.tensor([[[2.5, 3.5, 4.5]]]))

    def test_invalid_positions_zeroed(self):
        B, S, N, L = 1, 6, 1, 3
        left  = torch.ones(B, S)
        right = torch.ones(B, S) * 2.0

        safe_positions = torch.tensor([[[1, 2, 5]]])
        valid_mask     = torch.tensor([[[True, True, False]]])

        gl, gr = stub._gather_bin_edges(left, right, safe_positions, valid_mask,
                                        num_completions=N)
        assert gl[0, 0, 2].item() == 0.0
        assert gr[0, 0, 2].item() == 0.0

    def test_clamps_edge_positions_to_bin_edges_length(self):
        # bin_edges has length S-1, but safe_positions may reference position S-1
        # which would produce edge index S-2 after clamping — check no out-of-bounds
        B, S, N, L = 1, 5, 1, 2
        left  = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # length 4 = S-1
        right = left + 1.0

        # position 4 → edge 3 (= S-2, last valid)
        # position 5 would be out-of-bounds but clamped during safe_positions step
        safe_positions = torch.tensor([[[4, 4]]])
        valid_mask     = torch.ones(B, N, L, dtype=torch.bool)

        gl, gr = stub._gather_bin_edges(left, right, safe_positions, valid_mask,
                                        num_completions=N)
        assert gl[0, 0, 0].item() == 4.0   # edge at index 3
        assert gl[0, 0, 1].item() == 4.0

    def test_start_at_position_1_gives_edge_index_0(self):
        # Position 1 → edge index 0 (exercises the clamp(min=0) path)
        B, S, N, L = 1, 8, 1, 1
        left  = torch.arange(S, dtype=torch.float).unsqueeze(0)  # [0,1,2,...,7]
        right = left + 0.5
        safe_positions = torch.tensor([[[1]]])
        valid_mask     = torch.ones(B, N, L, dtype=torch.bool)
        gl, gr = stub._gather_bin_edges(left, right, safe_positions, valid_mask,
                                        num_completions=N)
        assert gl[0, 0, 0].item() == 0.0   # edge at index max(1-1,0)=0
        assert gr[0, 0, 0].item() == 0.5

    def test_multi_batch_and_completions(self):
        B, S, N, L = 2, 10, 2, 2
        left  = torch.arange(S, dtype=torch.float).unsqueeze(0).expand(B, -1)
        right = left + 0.5

        # B=2, N=2, L=2  → safe_positions (2, 2, 2)
        safe_positions = torch.tensor([[[2, 3], [5, 6]], [[1, 2], [4, 5]]])
        valid_mask     = torch.ones(B, N, L, dtype=torch.bool)

        gl, gr = stub._gather_bin_edges(left, right, safe_positions, valid_mask,
                                        num_completions=N)
        assert gl.shape == (B, N, L)
        # spot-check: batch 0, comp 0, pos 0 → edge at 2-1=1 → value 1.0
        assert gl[0, 0, 0].item() == 1.0
        # batch 1, comp 1, pos 1 → edge at 5-1=4 → value 4.0
        assert gl[1, 1, 1].item() == 4.0


# ===========================================================================
# AR loss: document-boundary masking (logic from forward())
# ===========================================================================

def _compute_ar_targets(input_ids, doc_ids):
    """Mirror of the boundary-masking logic in ParallelSamplingLightningModule.forward()."""
    ar_targets = input_ids[:, 1:].clone()
    if doc_ids is not None:
        doc_boundary = doc_ids[:, 1:] != doc_ids[:, :-1]
        ar_targets[doc_boundary] = IGNORE_INDEX
    return ar_targets


class TestArLossBoundaryMasking:
    """
    Sequence layout used throughout:
        positions: 0  1  2  3  4  5  6  7  8  9
        doc_ids:   0  0  0  0  1  1  1  2  2  2
        input_ids: 10 11 12 13 14 15 16 17 18 19

    Document boundaries are at positions 4 and 7 (first token of docs 1 and 2).
    In ar_targets (= input_ids[:, 1:]), those correspond to indices 3 and 6.
    """

    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    doc_ids   = torch.tensor([[0,  0,  0,  0,  1,  1,  1,  2,  2,  2]])

    def test_boundary_positions_set_to_ignore(self):
        ar_targets = _compute_ar_targets(self.input_ids, self.doc_ids)
        # index 3 in ar_targets predicts input_ids position 4 (first token of doc 1)
        assert ar_targets[0, 3].item() == IGNORE_INDEX
        # index 6 in ar_targets predicts input_ids position 7 (first token of doc 2)
        assert ar_targets[0, 6].item() == IGNORE_INDEX

    def test_non_boundary_positions_unchanged(self):
        ar_targets = _compute_ar_targets(self.input_ids, self.doc_ids)
        # All positions except the two boundaries should retain their token IDs
        expected = self.input_ids[0, 1:].clone()
        expected[3] = IGNORE_INDEX
        expected[6] = IGNORE_INDEX
        assert ar_targets[0].equal(expected)

    def test_no_doc_ids_leaves_targets_intact(self):
        ar_targets = _compute_ar_targets(self.input_ids, doc_ids=None)
        assert ar_targets[0].equal(self.input_ids[0, 1:])

    def test_boundary_at_ar_targets_index_0(self):
        # Doc boundary at position 1 → ar_targets[0] should be IGNORE_INDEX.
        input_ids = torch.tensor([[10, 11, 12, 13]])
        doc_ids   = torch.tensor([[0,  1,  1,  1]])   # position 1 starts doc 1
        ar_targets = _compute_ar_targets(input_ids, doc_ids)
        assert ar_targets[0, 0].item() == IGNORE_INDEX  # predicting input_ids[1] across boundary
        assert ar_targets[0, 1].item() == 12            # within doc 1, untouched
        assert ar_targets[0, 2].item() == 13

    def test_single_document_no_boundaries(self):
        doc_ids = torch.zeros(1, 10, dtype=torch.long)
        ar_targets = _compute_ar_targets(self.input_ids, doc_ids)
        assert ar_targets[0].equal(self.input_ids[0, 1:])

    def test_batch_with_different_boundaries(self):
        # Two sequences with boundaries at different positions
        input_ids = torch.tensor([
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
        ])
        doc_ids = torch.tensor([
            [0, 0, 0, 1, 1, 1],  # boundary at position 3 → ar_targets index 2
            [0, 0, 1, 1, 2, 2],  # boundaries at positions 2,4 → ar_targets indices 1,3
        ])
        ar_targets = _compute_ar_targets(input_ids, doc_ids)

        # Batch 0: index 2 masked, rest intact
        assert ar_targets[0, 2].item() == IGNORE_INDEX
        assert ar_targets[0, 0].item() == 11
        assert ar_targets[0, 1].item() == 12
        assert ar_targets[0, 3].item() == 14

        # Batch 1: indices 1 and 3 masked, rest intact
        assert ar_targets[1, 1].item() == IGNORE_INDEX
        assert ar_targets[1, 3].item() == IGNORE_INDEX
        assert ar_targets[1, 0].item() == 21
        assert ar_targets[1, 2].item() == 23
