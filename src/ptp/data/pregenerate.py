from contextlib import nullcontext
from ptp.transformer import TransformerModel
import yaml
from omegaconf import DictConfig
from pathlib import Path
import shutil
import os
import torch.distributed as dist

from argparse import ArgumentParser
from datasets import Dataset
import torch
from tqdm import tqdm
from time import sleep

from ptp.utils import instantiate


def fix_trailing_eos(output_ids: torch.Tensor, probs: torch.Tensor, eos_id):
    """
    Truncate EOS tokens at the end of sequences in output_ids by forcing
    the probabilities at those positions to be one-hot EOS.
    """
    T = output_ids.size(1)

    # sequences whose last two tokens are EOS
    at_least_two = (
        (output_ids[:, -2] == eos_id) & (output_ids[:, -1] == eos_id)
        if T >= 2 else
        torch.zeros(probs.size(0), dtype=torch.bool, device=probs.device)
    )

    # first EOS position (or T if none)
    eos_mask = output_ids.eq(eos_id)
    has_eos = eos_mask.any(dim=1)
    first_eos = torch.where(
        has_eos,
        eos_mask.float().cumsum(dim=1).clamp(0, 1).argmax(dim=1),
        torch.full((probs.size(0),), T, device=probs.device, dtype=torch.long)
    )

    # second-EOS position (first_eos+1) only for sequences with >=2 EOS-at-end, otherwise T
    second_eos_pos = torch.where(at_least_two, first_eos + 1, torch.full_like(first_eos, T))

    # mask positions >= second_eos_pos and replace those positions with EOS one-hot
    pos = torch.arange(T, device=probs.device).unsqueeze(0)  # (1, T)
    mask_after = pos >= second_eos_pos.unsqueeze(1)  # (B, T)
    eos_row = torch.zeros((probs.size(-1),), device=probs.device, dtype=probs.dtype)
    eos_row[eos_id] = 1.0
    probs[mask_after] = eos_row  # boolean-index assigns (num_masked, V) <- (V,) broadcast


def should_init_distributed():
    return (
            "RANK" in os.environ and
            "WORLD_SIZE" in os.environ and
            int(os.environ["WORLD_SIZE"]) > 1
    )


def init_distributed() -> tuple[int, int, int]:
    if should_init_distributed():
        dist.init_process_group(backend="gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Initialized distributed process group: rank {global_rank}/{world_size}")
    else:
        local_rank = global_rank = 0
        world_size = 1
        print("Running in single-process mode.")
    return local_rank, global_rank, world_size


@torch.inference_mode()
def main(experiment_dir: Path, precision: str, batch_size: int, store_interval: int):
    local_rank, global_rank, world_size = init_distributed()

    with open(experiment_dir / 'pregenerate.yaml', 'r') as f:
        config = DictConfig(yaml.safe_load(f))

    if torch.cuda.is_available():
        device = torch.device('cuda', local_rank % torch.cuda.device_count())
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    teacher = TransformerModel(**config['teacher']).eval().to(device)
    try:
        teacher = torch.compile(teacher, mode='max-autotune')
    except RuntimeError:
        pass
    # ddp, get local rank
    data_kwargs = config['data']
    data_kwargs['batch_size'] = 1
    data_kwargs['train_shuffle'] = False  # no shuffling when pregenerating
    cache_dir = experiment_dir / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    datamodule = instantiate(
        data_kwargs,
        tokenizer=teacher.tokenizer,
        cache_dir=cache_dir,
    )
    from ptp.data.prompt_scheme import MaskCreationError
    try:
        datamodule.setup(None)
    except (ValueError, MaskCreationError):
        # Dataset mask file was just created, rerun
        datamodule.setup(None)

    dataloaders = {
        'train': datamodule.train_dataloader(),
        'test': datamodule.test_dataloader(),
    }
    if datamodule.val_dataset is not None:
        dataloaders['valid'] = datamodule.val_dataloader()

    num_completions = config.generate.num_completions  # e.g., 32
    for split, dataloader in dataloaders.items():
        if dataloader is None:
            continue
        pregenerate_data(
            batch_size, config, dataloader, device, num_completions,
            experiment_dir / "data" / split, precision, split, teacher,
            store_interval, rank=global_rank, world_size=world_size,
        )


def pregenerate_data(
        batch_size: int, config: DictConfig, dataloader, device, num_completions, out_dir: Path,
        precision: str, split: str, teacher: TransformerModel, store_interval: int, use_cache: bool = True,
        rank: int = 0, world_size: int = 1):
    # Load existing dataset if it exists
    if (out_dir / "dataset_info.json").exists():
        try:
            existing_dataset = Dataset.load_from_disk(out_dir)
            all_data = existing_dataset.to_list()
            print(f"Loaded {len(all_data)} existing samples from {out_dir}")
        except Exception as e:
            print(f"Failed to load existing dataset: {e}")
            all_data = []
    else:
        all_data = []
        out_dir.mkdir(parents=True, exist_ok=True)
    offset_idx = len(all_data)
    collected_data = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating {split} completions")):
        if batch_idx % world_size != rank:
            # Skip batches not assigned to this rank
            continue
        if batch_idx < offset_idx:
            if batch_idx == 0:
                print("First generation (generated previously):")
                print_batch(all_data[0], teacher.tokenizer)
            # Skip already generated batches
            continue
        input_ids = batch['input_ids']
        assert input_ids.shape[0] == 1, "Dataloader batch size must be 1"
        attention_mask = batch['input_mask']
        max_length = config.generate.max_total_length
        # print(f"Generating {ceil(num_completions / batch_size)}x{batch_size}x{max_new_tokens} new tokens to input {input_ids.shape}.")
        input_ids_device = input_ids.to(device)
        attention_mask_device = attention_mask.to(device)
        # Use autocast on CUDA to run generation in mixed precision which
        # reduces memory and can improve throughput on tensor-core GPUs.
        if device.type != 'mps':
            autocast = torch.autocast(device_type=device.type, dtype=getattr(torch, precision))
        else:
            autocast = nullcontext()
        with autocast:
            completion_chunks = []
            left_edges = []
            right_edges = []
            for start in range(0, num_completions, batch_size):
                current_batch_size = min(batch_size, num_completions - start)
                outputs = teacher.generate(
                    input_ids=input_ids_device,
                    attention_mask=attention_mask_device,
                    max_length=max_length,
                    do_sample=True,
                    pad_token_id=teacher.tokenizer.eos_token_id,
                    temperature=config.generate.temperature,
                    top_p=config.generate.top_p,
                    top_k=config.generate.top_k,
                    output_scores=True,
                    # output_logits=True,
                    return_dict_in_generate=True,
                    num_return_sequences=current_batch_size,
                    use_cache=use_cache,
                )
                chunk_ids = outputs.sequences[:, input_ids.shape[1]:]
                scores = torch.stack(outputs.scores, dim=1)

                # confirm error correction etc works
                # from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper
                # processors = LogitsProcessorList([TemperatureLogitsWarper(config.generate.temperature), TopKLogitsWarper(config.generate.top_k),TopPLogitsWarper(config.generate.top_p)])
                # out = teacher(input_ids=outputs.sequences).logits
                # pout = torch.stack([processors(None, o) for o in out], dim=0)
                # assert torch.isclose(scores, pout[:, -max_new_tokens-1:-1]).all()

                del outputs
                probs = torch.softmax(scores, dim=-1)
                fix_trailing_eos(chunk_ids, probs, teacher.tokenizer.eos_token_id)
                selector = (
                    torch.arange(probs.shape[0], device=probs.device)[:, None],
                    torch.arange(probs.shape[1], device=probs.device)[None, :],
                    chunk_ids
                )
                assert (probs[selector] > 0).all(), "Some token has zero probability"
                cum_probs = probs.cumsum(-1)
                left_edge = cum_probs[selector] - probs[selector]
                right_edge = cum_probs[selector]

                pad_len = (max_length - input_ids.shape[1]) - chunk_ids.shape[1]
                if pad_len > 0:
                    eos_id = teacher.tokenizer.eos_token_id
                    bsz = chunk_ids.size(0)
                    # pad ids with EOS
                    chunk_ids = torch.cat(
                        [chunk_ids, torch.full((bsz, pad_len), eos_id, device=chunk_ids.device, dtype=chunk_ids.dtype)],
                        dim=1
                    )
                    # pad edges: left=0, right=1
                    left_edge = torch.cat(
                        [left_edge, torch.zeros((bsz, pad_len), device=left_edge.device, dtype=left_edge.dtype)],
                        dim=1
                    )
                    right_edge = torch.cat(
                        [right_edge, torch.ones((bsz, pad_len), device=right_edge.device, dtype=right_edge.dtype)],
                        dim=1
                    )

                completion_chunks.append(chunk_ids.to("cpu"))
                left_edges.append(left_edge.to("cpu"))
                right_edges.append(right_edge.to("cpu"))

        output_ids = torch.cat(completion_chunks, dim=0)
        left_bin_edge = torch.cat(left_edges, dim=0)
        right_bin_edge = torch.cat(right_edges, dim=0)

        completion_entry = {
            'input': input_ids[0][attention_mask[0].bool()].to("cpu").numpy().tolist(),
            'completions': output_ids.numpy().tolist(),
            'left_bin_edges': left_bin_edge.numpy().tolist(),
            'right_bin_edges': right_bin_edge.numpy().tolist(),
        }
        collected_data.append(completion_entry)
        if len(collected_data) % store_interval == 0:
            collect_and_save(collected_data, all_data, out_dir, rank, world_size)
            collected_data = []

        if len(all_data) == 1:
            print("First generation (generated now):")
            print_batch(all_data[0], teacher.tokenizer)

    collect_and_save(collected_data, all_data, out_dir, rank, world_size)


def collect_data(my_collected_data, rank: int, world_size: int):
    if world_size > 1:
        if rank == 0:
            object_gather_list = [None for _ in range(dist.get_world_size())]
            dist.gather_object(my_collected_data, object_gather_list, dst=0)
            new_collected_data = []
            for recvd in object_gather_list:
                new_collected_data.extend(recvd)
            return new_collected_data
        else:
            dist.gather_object(my_collected_data, dst=0)
            return None
    return my_collected_data


def save_all_data(all_data, out_dir: Path):
    # Remove existing backup directory
    backup_dir = out_dir.parent / (out_dir.name + "_backup")
    if backup_dir.exists():
        for _ in range(10):
            try:
                shutil.rmtree(backup_dir)
                break
            except Exception:
                pass
    # Use different backup name if needed
    if backup_dir.exists():
        backup_idx = 0
        while True:
            candidate = out_dir.parent / f"{out_dir.name}_backup_{backup_idx}"
            if candidate.exists():
                backup_idx += 1
            else:
                backup_dir = candidate
                break
    # Rename existing output directory to backup directory
    if out_dir.exists():
        out_dir.rename(backup_dir)
        # wait for filesystem to settle
        for _ in range(10):
            if not out_dir.exists():
                break
            sleep(0.1)

    dataset = Dataset.from_list(all_data)
    dataset.save_to_disk(out_dir)
    print(f"Saved {len(all_data)} samples to {out_dir}")
    if backup_dir.exists():
        try:
            shutil.rmtree(backup_dir)
        except Exception:
            print(f"Warning: failed to delete backup directory {backup_dir}")


def collect_and_save(my_collected_data, all_data, out_dir: Path, rank: int, world_size: int):
    data_to_be_saved = collect_data(my_collected_data, rank, world_size)
    if rank == 0:
        assert data_to_be_saved is not None
        all_data.extend(data_to_be_saved)
        save_all_data(all_data, out_dir)
    else:
        assert data_to_be_saved is None


def print_batch(batch, tokenizer, completion_idx: int = 0):
    print("Prompt:")
    print(tokenizer.decode(batch['input'], skip_special_tokens=False))
    print()
    print("First completion:")
    print(tokenizer.decode(batch['completions'][completion_idx], skip_special_tokens=False))
    print()


def _parse_args():
    parser = ArgumentParser(
        description=(
            "Generate teacher completions for a prompt-distill experiment. "
            "Reads pregenerate.yaml from the experiment directory and writes "
            "completions to <experiment_dir>/data/."
        )
    )
    parser.add_argument(
        "experiment_dir", type=Path,
        help="Path to the experiment directory created by ptp_distill.",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=8,
        help="Number of completions to generate in parallel per teacher forward pass (default: 8).",
    )
    parser.add_argument(
        "--precision", "-p", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Floating-point precision for generation (default: bfloat16).",
    )
    parser.add_argument(
        "--store-interval", "-s", type=int, default=200,
        help="Save progress to disk every N prompts (default: 200).",
    )
    args = parser.parse_args()

    experiment_dir: Path = args.experiment_dir.resolve()

    if not experiment_dir.exists():
        print(f"Error: experiment directory not found: {experiment_dir}")
        print("Run ptp_distill first to set up the experiment.")
        raise SystemExit(1)

    pregenerate_yaml = experiment_dir / "pregenerate.yaml"
    if not pregenerate_yaml.exists():
        train_yaml = experiment_dir / "train.yaml"
        if train_yaml.exists():
            print(
                f"No pregenerate.yaml found in {experiment_dir}.\n"
                "This experiment was set up for direct training (full-chat or text mode).\n"
                "Pregeneration is not needed — run ptp_train directly."
            )
        else:
            print(f"Error: {pregenerate_yaml} not found. Is this a valid experiment directory?")
        raise SystemExit(1)

    data_dir = experiment_dir / "data"
    print(f"Pregenerating into {data_dir} ...")
    main(
        experiment_dir=experiment_dir,
        precision=args.precision,
        batch_size=args.batch_size,
        store_interval=args.store_interval,
    )
    print(f"\nDone. Run ptp_train {experiment_dir} to start training.")


if __name__ == "__main__":
    _parse_args()
