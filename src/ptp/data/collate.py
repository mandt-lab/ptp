import torch


IGNORE_INDEX = -100


def collate_fn(batch):
    """
    Batch is list of dicts:
    {
        "input_ids": (inconsistent_prompt_seq_len,),
        "input_mask": (inconsistent_prompt_seq_len,),
        "left_bin_edges": (inconsistent_prompt_seq_len,),
        "right_bin_edges": (inconsistent_prompt_seq_len,)
        "completion_starts": (consistent_completion_count,),
        "completion_length": int,
    }

    yields a single dict:

    {
        "input_ids": (batch_size, max_prompt_seq_len),
        "input_mask": (batch_size, max_prompt_seq_len),
        "input_lengths": (batch_size,) [List[int]],
        "left_bin_edges": (batch_size, consistent_completion_count, completion_length),
        "right_bin_edges": (batch_size, consistent_completion_count, completion_length),
        "completion_starts": (batch_size, consistent_completion_count,) [List[List[int]]],
        "completion_length": int,
    }

    Entries `left_bin_edges` and `right_bin_edges` are optional and skipped if not present.
    """
    batch_size = len(batch)

    # ---- prompt padding ----
    prompt_lens = [item["input_ids"].shape[0] for item in batch]
    max_prompt_len = max(prompt_lens)

    input_ids = torch.full((batch_size, max_prompt_len), IGNORE_INDEX, dtype=batch[0]["input_ids"].dtype)
    input_mask = torch.zeros(batch_size, max_prompt_len, dtype=torch.bool)
    input_lengths = []
    for i, item in enumerate(batch):
        L = item["input_ids"].shape[0]
        assert item["input_ids"].shape == item["input_mask"].shape, \
            f"input_ids and input_mask shapes do not match for batch item {i}: {item['input_ids'].shape} vs {item['input_mask'].shape}"
        input_ids[i, :L] = item["input_ids"]
        input_mask[i, :L] = item["input_mask"].bool()
        input_lengths.append(L)

    # ---- completions (no padding required) ----
    # Assumptions:
    #   - completion_count is consistent across batch
    #   - completion_length is consistent across batch
    completion_count = len(batch[0]["completion_starts"])
    completion_length = batch[0]["completion_length"]
    for item in batch:
        assert len(item["completion_starts"]) == completion_count, \
            f"Inconsistent completion counts in batch"
        assert item["completion_length"] == completion_length, \
            f"Inconsistent completion lengths in batch"
    collated = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "input_lengths": input_lengths,
        "completion_starts": [item["completion_starts"] for item in batch],
        "completion_length": completion_length,
    }

    if "left_bin_edges" in batch[0] and "right_bin_edges" in batch[0]:
        left_bin_edges = torch.full(
            (batch_size, max_prompt_len - 1),
            float('nan'), dtype=batch[0]["left_bin_edges"].dtype
        )
        right_bin_edges = torch.full(
            (batch_size, max_prompt_len - 1),
            float('nan'), dtype=batch[0]["left_bin_edges"].dtype
        )
        for i, item in enumerate(batch):
            L = item["left_bin_edges"].shape[0]
            expected_bin_edge_count = len(item["input_ids"]) - 1
            assert len(item["left_bin_edges"]) == expected_bin_edge_count, \
                f"left_bin_edges count does not match expected for batch item {i}: {len(item['left_bin_edges'])} vs {expected_bin_edge_count}"
            assert len(item["left_bin_edges"]) == expected_bin_edge_count, \
                f"left_bin_edges count does not match expected for batch item {i}: {len(item['left_bin_edges'])} vs {expected_bin_edge_count}"
            left_bin_edges[i, :L] = item["left_bin_edges"]
            right_bin_edges[i, :L] = item["right_bin_edges"]
        collated["left_bin_edges"] = left_bin_edges
        collated["right_bin_edges"] = right_bin_edges

    return collated
