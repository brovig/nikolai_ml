# Piper __main__.py, patched for PyTorch Lightning 2.x (rtx5070ti).
#
# Changes vs upstream (piper commit 73c04d8):
#   * Trainer.add_argparse_args / Trainer.from_argparse_args were REMOVED in
#     Lightning 2.0. Replaced with an explicit argparse group + direct
#     Trainer(...) construction (only the flags the project actually uses).
#   * `--resume_from_checkpoint` is passed to trainer.fit() (2.x API).
# Everything else is byte-identical to upstream.

import argparse
import json
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


def _add_trainer_args(parser):
    """Trainer flags (subset; mirrors old Trainer.add_argparse_args needs)."""
    g = parser.add_argument_group("Trainer")
    g.add_argument(
        "--default_root_dir",
        default=None,
        help="Path to logs/checkpoints (default: dataset dir)",
    )
    g.add_argument(
        "--accelerator",
        default="gpu",
        help="Lightning accelerator (default: gpu)",
    )
    g.add_argument(
        "--devices",
        default=1,
        type=int,
        help="Number of devices to use (default: 1)",
    )
    g.add_argument(
        "--precision",
        default=32,
        type=int,
        help="Training precision: 32 or 16 (default: 32)",
    )
    g.add_argument(
        "--max_epochs",
        default=-1,
        type=int,
        help="Maximum number of epochs (-1 for infinite) (default: -1)",
    )
    g.add_argument(
        "--limit_train_batches",
        default=1.0,
        type=float,
        help="How much of training dataset to check (default: 1.0)",
    )
    g.add_argument(
        "--limit_val_batches",
        default=1.0,
        type=float,
        help="How much of validation dataset to check (default: 1.0)",
    )
    g.add_argument(
        "--num_sanity_val_steps",
        default=2,
        type=int,
        help="Number of val checks to do before the actual val stage (default: 2)",
    )
    g.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a Lightning checkpoint to resume training from",
    )


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    _add_trainer_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    # Lightning 2.x: Trainer.from_argparse_args is gone; build it explicitly.
    trainer = Trainer(
        default_root_dir=args.default_root_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
    )
    if args.checkpoint_epochs is not None:
        trainer.callbacks = [ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)]
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing in the single-speaker model)
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )

    # Lightning 2.x: resume is passed to fit(), not the Trainer ctor.
    trainer.fit(model, ckpt_path=args.resume_from_checkpoint)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        # Use saved value
        if k in saved_state_dict:
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
