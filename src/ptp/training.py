import os
from omegaconf import DictConfig, OmegaConf
import yaml
import warnings
import lightning.pytorch
from datetime import timedelta
from pathlib import Path
import torch
from argparse import ArgumentParser

# Patch datasets library to handle 'List' feature type
from datasets.features.features import _FEATURE_TYPES
try:
    _FEATURE_TYPES['List'] = _FEATURE_TYPES['Sequence']
except Exception:
    pass

from ptp.utils import instantiate


def main(experiment_dir: Path, offline: bool = False, wandb_postfix: str = '', variant_name: str = ''):
    with open(experiment_dir / 'train.yaml', 'r') as f:
        config = DictConfig(yaml.safe_load(f))
    if variant_name != '':
        with open(experiment_dir / f'train-{variant_name}.yaml', 'r') as f:
            variant_config = DictConfig(yaml.safe_load(f))
        config = OmegaConf.merge(config, variant_config)
    print(OmegaConf.to_yaml(config))

    torch.set_float32_matmul_precision('medium')
    print(f'='* 50)
    print(f'num nodes: {int(os.environ.get("NNODES", 1))}')
    print(f'='* 50)
    ckpt_dir = Path(config['training'].get("ckpt_dir", experiment_dir))
    if variant_name != '':
        ckpt_dir = ckpt_dir / variant_name
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving checkpoints to {ckpt_dir}")
    wandb_project = config['training'].get('wandb_project', 'ptp')
    trainer = lightning.pytorch.Trainer(
        max_steps=config['training'].get('n_steps', -1),
        max_epochs=config['training'].get('n_epochs'),
        logger=lightning.pytorch.loggers.WandbLogger(
            project=wandb_project,
            id=config['training'].get("run_id", experiment_dir.name) + variant_name + wandb_postfix,
            name=experiment_dir.name + variant_name + wandb_postfix,
            offline=offline or wandb_project == 'offline' or os.environ.get("WANDB_MODE") == "offline",
            config=OmegaConf.to_container(config, resolve=True),
        ),
        val_check_interval=config['training'].get('eval_every_steps'),
        limit_val_batches=config['training'].get('eval_steps_to_run'),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        callbacks=[
            lightning.pytorch.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                save_top_k=2,
                monitor="val/correct",
                mode="max",
                save_last=True,
                filename="tinyllama-code-{step:02d}",
                train_time_interval=timedelta(**config['training'].get('check_point_interval', {'minutes': 30})),
            ),
            lightning.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        precision=config["training"].get("precision", 'bf16-mixed'),
        num_nodes=int(os.environ.get("NNODES", 1)),
        strategy=config["training"].get("strategy", "auto"),
    )
    # logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)
    data_config = config['data']
    if data_config.get("root_dir", None) is not None and "EXP_DIR" in data_config.root_dir:
        data_config.root_dir = data_config.root_dir.replace("EXP_DIR", str(experiment_dir))

    datamodule = instantiate(data_config)
    lit_model = instantiate(config['model'])
    fit_kwargs=dict(
        model=lit_model,
        datamodule=datamodule,
    )
    try:
        trainer.fit(**fit_kwargs, ckpt_path=config.get("restore_ckpt", "last"))
    except FileNotFoundError:
        warnings.warn(
            "No checkpoint found, training will continue without checkpoint.",
            RuntimeWarning,
        )
        trainer.fit(**fit_kwargs)



def _parse_args():
    parser = ArgumentParser(
        description=(
            "Train a PTP distillation model. "
            "Reads train.yaml from the experiment directory. "
            "For prompt-distill mode, run ptp_pregenerate first."
        )
    )
    parser.add_argument(
        "experiment_dir", type=Path,
        help="Path to the experiment directory created by ptp_distill.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Disable wandb online logging.",
    )
    parser.add_argument(
        "-p", "--wandb-postfix", type=str, default="",
        help="Append a postfix to the wandb run name and ID.",
    )
    parser.add_argument(
        "-v", "--variant", type=str, default="",
        help=(
            "Optional variant name. If given, merges train-<variant>.yaml on top of train.yaml. "
            "Useful for quickly trying hyperparameter changes without editing the base config."
        ),
    )
    parser.add_argument(
        "--local-ckpt-tmp", action="store_true",
        help=(
            "Stage checkpoint temp files in the target directory instead of /tmp. "
            "Use this when /tmp is too small for atomic saves."
        ),
    )
    args = parser.parse_args()

    experiment_dir: Path = args.experiment_dir.resolve()

    if not experiment_dir.exists():
        print(f"Error: experiment directory not found: {experiment_dir}")
        print("Run ptp_distill first to set up the experiment.")
        raise SystemExit(1)

    train_yaml = experiment_dir / "train.yaml"
    if not train_yaml.exists():
        print(f"Error: {train_yaml} not found. Is this a valid experiment directory?")
        raise SystemExit(1)

    pregenerate_yaml = experiment_dir / "pregenerate.yaml"
    data_dir = experiment_dir / "data"
    if pregenerate_yaml.exists() and not data_dir.exists():
        print(
            f"Warning: pregenerate.yaml found but {data_dir} does not exist.\n"
            f"Did you run ptp_pregenerate {experiment_dir} first?\n"
            "Continuing anyway — training will fail if the data module requires pregenerated data.\n"
        )

    if args.local_ckpt_tmp:
        from ptp.atomic_fs import register
        register()

    main(
        experiment_dir=experiment_dir,
        offline=args.offline,
        wandb_postfix=args.wandb_postfix,
        variant_name=args.variant,
    )


if __name__ == "__main__":
    _parse_args()
