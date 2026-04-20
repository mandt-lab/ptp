"""
Greedy document packing with proportional completion sampling.

Architecture
------------
Base datasets (FullChatDataset, TextDocumentDataset) expose two things:

  get_metadata() -> List[(doc_length, [(span_start, span_end), ...])]
      Called once at construction; result is cached to disk.
      Spans are LOCAL token indices (within the document).

  __getitem__(idx) -> Tensor  (token IDs, variable length)

PackingDataset
  - Uses Best-Fit Decreasing (BFD) to pack documents into fixed-length
    sequences, minimising padding.
  - Builds a static group index at construction time.
  - __getitem__ fetches token IDs, concatenates them, then samples
    `num_completions` start positions PROPORTIONALLY across all completion
    regions of all documents in the pack.  This gives more completions to
    longer documents, which is fairer than the previous per-document fixed-C.

packed_collate_fn
  - Stacks a batch of PackingDataset items.
  - All tensors are pre-padded to max_docs/num_completions in __getitem__,
    so collation is just torch.stack.

Output format (per batch)
  input_ids         (B, S)       packed token IDs; PAD_ID after last real token
  input_mask        (B, S)       True for real tokens
  doc_ids           (B, S)       per-token doc index; max_docs for padding
  doc_starts        (B, D)       start position of each doc in packed seq
  doc_lengths       (B, D)       real token count per doc (0 for padding slots)
  completion_starts       (B, N)       global position of each completion
  completion_doc_ids      (B, N)       which document each completion belongs to
  completion_length int          fixed L

Query layout for attention: N * L flat tokens (comp 0 tokens, comp 1 tokens …)
KV layout: [prompt S tokens | completion N*L tokens]
"""

import bisect
import os
import pickle

import torch
from torch.utils.data import Dataset

from ptp.data.utils import duplicate_avoiding_randint

PAD_ID = 0   # token id used for padding (never attended to)


# ---------------------------------------------------------------------------
# Best-Fit Decreasing bin packing
# ---------------------------------------------------------------------------

def _bfd_groups(lengths: list[int], total_length: int) -> list[list[int]]:
    """
    Assign base-dataset indices to groups using Best-Fit Decreasing.

    Documents are sorted longest-first; each is placed into the bin with the
    smallest remaining capacity that still fits it (tightest fit).  This is
    O(n log n) via a sorted list of (remaining_capacity, group_index) pairs.

    Returns a list of groups; each group is an ordered list of base indices
    (order = insertion order within the group, largest doc first).
    """
    order = sorted(range(len(lengths)), key=lambda i: -min(lengths[i], total_length))
    # bins: sorted list of (remaining_cap, group_idx) — ascending by remaining_cap
    bins:   list[tuple[int, int]] = []
    groups: list[list[int]]       = []

    for i in order:
        l = min(lengths[i], total_length)
        # bisect_left((l, -1)) finds the first bin with remaining_cap >= l
        pos = bisect.bisect_left(bins, (l, -1))
        if pos < len(bins):
            cap, gidx = bins.pop(pos)
            bisect.insort(bins, (cap - l, gidx))
            groups[gidx].append(i)
        else:
            gidx = len(groups)
            groups.append([i])
            bisect.insort(bins, (total_length - l, gidx))

    return groups


# ---------------------------------------------------------------------------
# PackingDataset
# ---------------------------------------------------------------------------

class PackingDataset(Dataset):
    """
    Wrap a base dataset with BFD packing and proportional completion sampling.

    Parameters
    ----------
    base               : dataset with get_metadata() and __getitem__ → Tensor
    max_sequence_length: S — fixed packed sequence length
    num_completions    : N — completions sampled per packed sequence
    completion_length  : L — tokens per completion window
    cache_dir          : where to store the metadata scan cache
    """

    def __init__(
        self,
        base: Dataset,
        max_sequence_length: int,
        num_completions: int,
        completion_length: int,
        cache_dir: str = "data_cache",
    ):
        self.base               = base
        self.max_sequence_length = max_sequence_length
        self.num_completions    = num_completions
        self.completion_length  = completion_length
        self.cache_dir          = cache_dir

        # Load or compute metadata
        self._metadata = self._load_metadata()

        # Build BFD groups
        lengths       = [m[0] for m in self._metadata]
        self._groups  = _bfd_groups(lengths, max_sequence_length)
        self.max_docs = max(len(g) for g in self._groups)

    # ------------------------------------------------------------------
    # Metadata caching
    # ------------------------------------------------------------------

    def _load_metadata(self) -> list[tuple[int, list[tuple[int, int]]]]:
        """Call base.get_metadata(), using a disk cache when available."""
        # Fast path: TextDocumentDataset returns a trivial list in O(1)
        # and doesn't need caching.
        if hasattr(self.base, 'get_metadata'):
            return self.base.get_metadata()
        raise TypeError(
            f"{type(self.base).__name__} must implement get_metadata() → "
            "List[(doc_length, [(span_start, span_end), ...])]"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, idx: int) -> dict:
        group               = self._groups[idx]
        max_sequence_length = self.max_sequence_length
        num_completions     = self.num_completions
        completion_length   = self.completion_length
        max_docs            = self.max_docs

        input_ids  = torch.full((max_sequence_length,), PAD_ID, dtype=torch.long)
        input_mask = torch.zeros(max_sequence_length,            dtype=torch.bool)
        doc_ids    = torch.full((max_sequence_length,), max_docs, dtype=torch.long)  # max_docs = padding sentinel

        doc_starts_out  = torch.zeros(max_docs, dtype=torch.long)
        doc_lengths_out = torch.zeros(max_docs, dtype=torch.long)

        # Extra per-token tensors from base (e.g. bin_edges); filled on first doc seen.
        extra_bufs: dict[str, torch.Tensor] = {}

        # Completion region pool: list of (global_start, region_length, doc_slot)
        region_pool: list[tuple[int, int, int]] = []

        pos = 0
        for doc_slot, base_idx in enumerate(group):
            if doc_slot >= max_docs:
                break

            # Base may return a plain Tensor or a dict {"input_ids": ..., ...}
            raw = self.base[base_idx]
            if isinstance(raw, torch.Tensor):
                tokens = raw
                extra  = {}
            else:
                tokens = raw["input_ids"]
                extra  = {k: v for k, v in raw.items()
                          if k != "input_ids" and isinstance(v, torch.Tensor)}

            dl = min(len(tokens), max_sequence_length - pos)
            if dl <= 0:
                break

            input_ids[pos : pos + dl]  = tokens[:dl]
            input_mask[pos : pos + dl] = True
            doc_ids[pos : pos + dl]    = doc_slot
            doc_starts_out[doc_slot]   = pos
            doc_lengths_out[doc_slot]  = dl

            # Initialise extra buffers on first doc (same dtype, shape (max_sequence_length, ...))
            for k, v in extra.items():
                if k not in extra_bufs:
                    tail = v.shape[1:]
                    extra_bufs[k] = torch.zeros((max_sequence_length, *tail), dtype=v.dtype)
                src_len = min(len(v), dl)
                extra_bufs[k][pos : pos + src_len] = v[:src_len]

            # Shift completion spans to global positions; enforce start >= pos+1
            _, spans = self._metadata[base_idx]
            for (r_start, r_end) in spans:
                g_start = pos + max(r_start, 1)
                g_end   = pos + min(r_end, dl)
                if g_start < g_end:
                    region_pool.append((g_start, g_end - g_start, doc_slot))

            pos += dl

        # ------------------------------------------------------------------
        # Sample N completions proportionally across the region pool
        # ------------------------------------------------------------------
        total_pool = sum(length for _, length, _ in region_pool)
        if total_pool == 0:
            region_pool = [(1, max(pos - 1, 1), 0)]
            total_pool  = region_pool[0][1]

        # Sample num_completions positions on a virtual "ruler" of length total_pool,
        # then sort so we can sweep regions once in a single O(regions + num_completions) pass.
        sampled_ruler_positions = duplicate_avoiding_randint(0, total_pool, num_completions)

        completion_starts_list  : list[int] = []
        completion_doc_ids_list : list[int] = []
        ruler_offset      = 0   # cumulative length of regions already passed
        sample_idx        = 0   # index into sampled_ruler_positions
        for (region_global_start, region_length, doc_slot) in region_pool:
            while sample_idx < len(sampled_ruler_positions) and \
                    sampled_ruler_positions[sample_idx] < ruler_offset + region_length:
                # Convert ruler position → global token index within this region
                offset_within_region = sampled_ruler_positions[sample_idx] - ruler_offset
                completion_starts_list.append(region_global_start + offset_within_region)
                completion_doc_ids_list.append(doc_slot)
                sample_idx += 1
            ruler_offset += region_length
            if sample_idx >= len(sampled_ruler_positions):
                break

        out = {
            "input_ids":         input_ids,
            "input_mask":        input_mask,
            "doc_ids":           doc_ids,
            "doc_starts":        doc_starts_out,
            "doc_lengths":       doc_lengths_out,
            "completion_starts":       torch.tensor(completion_starts_list,  dtype=torch.long),
            "completion_doc_ids":      torch.tensor(completion_doc_ids_list, dtype=torch.long),
            "completion_length": completion_length,
        }
        out.update(extra_bufs)
        return out


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def packed_collate_fn(batch: list[dict]) -> dict:
    """
    Stack a batch of PackingDataset items into batch tensors.

    All per-item tensors are already padded to (D,) and (N,) in __getitem__,
    so this is purely torch.stack.
    """
    return {
        "input_ids":         torch.stack([x["input_ids"]    for x in batch]),
        "input_mask":        torch.stack([x["input_mask"]   for x in batch]),
        "doc_ids":           torch.stack([x["doc_ids"]      for x in batch]),
        "doc_starts":        torch.stack([x["doc_starts"]   for x in batch]),
        "doc_lengths":       torch.stack([x["doc_lengths"]  for x in batch]),
        "completion_starts":       torch.stack([x["completion_starts"]  for x in batch]),
        "completion_doc_ids":      torch.stack([x["completion_doc_ids"] for x in batch]),
        "completion_length": batch[0]["completion_length"],
    }
