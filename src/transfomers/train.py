from __future__ import annotations

import argparse
import json
import math
import random
import time

from pathlib import Path

import torch

from torch.optim import AdamW

from torch.optim.lr_scheduler import (
    LambdaLR
)

from torch.utils.data import (
    DataLoader
)

from src.transfomers.dataset import (
    QADataset,
    collate_qa,
    load_qa_file,
    split_items
)

from src.transfomers.model import (
    GPTConfig,
    MiniGPT
)

from src.transfomers.tokenizer import (
    ByteTokenizer
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train our GPT-style "
            "Q&A LLM from scratch"
        )
    )

    # Dataset
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="JSONL or CSV Q&A file"
    )

    # Output folder
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints"
    )

    # ------------------------------------------------------
    # MODEL ARCHITECTURE
    # ------------------------------------------------------

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=256
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        default=6
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        default=8
    )

    parser.add_argument(
        "--ff_mult",
        type=int,
        default=4
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    # ------------------------------------------------------
    # TRAINING PARAMETERS
    # ------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--train_on_prompt",
        action="store_true"
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use CUDA mixed precision"
        )
    )

    return parser.parse_args()


# ==========================================================
# DEVICE
# ==========================================================

def choose_device():

    # NVIDIA GPU
    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    # Apple Silicon
    if (
        hasattr(
            torch.backends,
            "mps"
        )
        and
        torch.backends.mps.is_available()
    ):

        return torch.device(
            "mps"
        )

    # CPU
    return torch.device(
        "cpu"
    )


# ==========================================================
# RANDOM SEEDS
# ==========================================================

def set_seed(
    seed: int
):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ==========================================================
# LEARNING-RATE SCHEDULER
# ==========================================================

def make_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float
):

    def lr_lambda(
        step: int
    ):

        # ------------------------------------------
        # Warmup
        # ------------------------------------------

        if total_steps <= 1:

            return 1.0

        if step < warmup_steps:

            return max(
                1e-8,

                step
                /
                max(
                    1,
                    warmup_steps
                )
            )

        # ------------------------------------------
        # Cosine decay
        # ------------------------------------------

        progress = (

            (
                step
                -
                warmup_steps
            )

            /

            max(
                1,
                total_steps
                -
                warmup_steps
            )
        )

        cosine = (

            0.5

            *

            (
                1.0
                +
                math.cos(

                    math.pi

                    *

                    min(
                        1.0,
                        progress
                    )
                )
            )
        )

        return (

            min_lr_ratio

            +

            (
                1.0
                -
                min_lr_ratio
            )

            *
            cosine
        )

    return LambdaLR(
        optimizer,
        lr_lambda
    )


# ==========================================================
# VALIDATION
# ==========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0

    batches = 0

    for batch in loader:

        input_ids = (
            batch[
                "input_ids"
            ].to(device)
        )

        targets = (
            batch[
                "targets"
            ].to(device)
        )

        _, loss = model(
            input_ids,
            targets
        )

        total_loss += float(
            loss.item()
        )

        batches += 1

    return (

        total_loss

        /

        max(
            1,
            batches
        )
    )


# ==========================================================
# SAVE CHECKPOINT
# ==========================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_val,
    config,
    args
):

    checkpoint = {

        "model":
            model.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "scheduler":
            scheduler.state_dict(),

        "epoch":
            epoch,

        "global_step":
            global_step,

        "best_val":
            best_val,

        "config":
            config.to_dict(),

        "train_args":
            vars(args),

        "tokenizer": {

            "type":
                "byte",

            "vocab_size":
                261,

            "byte_offset":
                5
        }
    }

    torch.save(
        checkpoint,
        path
    )


# ==========================================================
# MAIN TRAINING FUNCTION
# ==========================================================

def main():

    args = parse_args()

    # ------------------------------------------------------
    # SETUP
    # ------------------------------------------------------

    set_seed(
        args.seed
    )

    device = choose_device()

    print(
        f"Device: {device}"
    )

    # ------------------------------------------------------
    # TOKENIZER
    # ------------------------------------------------------

    tokenizer = ByteTokenizer()

    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )

    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------

    items = load_qa_file(
        args.data
    )

    train_items, val_items = (
        split_items(

            items,

            args.val_fraction,

            args.seed
        )
    )

    print(
        f"Examples: "
        f"total={len(items)}, "
        f"train={len(train_items)}, "
        f"validation={len(val_items)}"
    )

    # ------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------

    train_dataset = QADataset(

        train_items,

        tokenizer,

        args.max_seq_len,

        train_on_prompt=(
            args.train_on_prompt
        )
    )

    val_dataset = QADataset(

        val_items,

        tokenizer,

        args.max_seq_len,

        train_on_prompt=(
            args.train_on_prompt
        )
    )

    # ------------------------------------------------------
    # COLLATE FUNCTION
    # ------------------------------------------------------

    def collate_fn(batch):

        return collate_qa(

            batch,

            tokenizer.special.pad
        )

    # ------------------------------------------------------
    # DATA LOADERS
    # ------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=(
            args.num_workers
        ),

        collate_fn=(
            collate_fn
        )
    )

    if len(val_dataset):

        val_loader = DataLoader(

            val_dataset,

            batch_size=(
                args.batch_size
            ),

            shuffle=False,

            num_workers=(
                args.num_workers
            ),

            collate_fn=(
                collate_fn
            )
        )

    else:

        val_loader = None

    # ------------------------------------------------------
    # CREATE MODEL
    # ------------------------------------------------------

    config = GPTConfig(

        vocab_size=(
            tokenizer.vocab_size
        ),

        max_seq_len=(
            args.max_seq_len
        ),

        d_model=(
            args.d_model
        ),

        n_layers=(
            args.n_layers
        ),

        n_heads=(
            args.n_heads
        ),

        ff_mult=(
            args.ff_mult
        ),

        dropout=(
            args.dropout
        )
    )

    model = MiniGPT(
        config
    )

    model = model.to(
        device
    )

    print(
        "Parameters:",
        f"{model.parameter_count():,}"
    )

    # ------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------

    optimizer = AdamW(

        model.parameters(),

        lr=args.lr,

        weight_decay=(
            args.weight_decay
        )
    )

    # ------------------------------------------------------
    # CALCULATE TRAINING STEPS
    # ------------------------------------------------------

    updates_per_epoch = (
        math.ceil(

            len(train_loader)

            /

            max(
                1,
                args.grad_accum
            )
        )
    )

    total_steps = max(

        1,

        updates_per_epoch
        *
        args.epochs
    )

    warmup_steps = int(

        total_steps
        *
        args.warmup_ratio
    )

    scheduler = make_scheduler(

        optimizer,

        total_steps,

        warmup_steps,

        args.min_lr_ratio
    )

    # ------------------------------------------------------
    # MIXED PRECISION
    # ------------------------------------------------------

    use_amp = (

        args.amp

        and

        device.type
        ==
        "cuda"
    )

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=use_amp
    )

    # ------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(

        parents=True,

        exist_ok=True
    )

    # Save model config

    with (

        out_dir
        /
        "config.json"

    ).open(

        "w",
        encoding="utf-8"

    ) as file:

        json.dump(

            config.to_dict(),

            file,

            indent=2
        )

    # ------------------------------------------------------
    # TRAINING STATE
    # ------------------------------------------------------

    best_validation_loss = (
        float("inf")
    )

    global_step = 0

    # ======================================================
    # EPOCH LOOP
    # ======================================================

    for epoch in range(

        1,

        args.epochs + 1
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0

        start_time = time.time()

        # ==================================================
        # BATCH LOOP
        # ==================================================

        for batch_index, batch in enumerate(

            train_loader,

            start=1
        ):

            input_ids = (

                batch[
                    "input_ids"
                ]

                .to(device)
            )

            targets = (

                batch[
                    "targets"
                ]

                .to(device)
            )

            # ----------------------------------------------
            # CUDA MIXED PRECISION
            # ----------------------------------------------

            if use_amp:

                with torch.amp.autocast(

                    device_type="cuda",

                    dtype=torch.float16
                ):

                    _, loss = model(

                        input_ids,

                        targets
                    )

                    loss_for_backward = (

                        loss
                        /
                        args.grad_accum
                    )

                scaler.scale(

                    loss_for_backward

                ).backward()

            # ----------------------------------------------
            # NORMAL PRECISION
            # ----------------------------------------------

            else:

                _, loss = model(

                    input_ids,

                    targets
                )

                loss_for_backward = (

                    loss
                    /
                    args.grad_accum
                )

                loss_for_backward.backward()

            running_loss += float(
                loss.item()
            )

            # ----------------------------------------------
            # GRADIENT ACCUMULATION
            # ----------------------------------------------

            should_update = (

                (
                    batch_index
                    %
                    args.grad_accum
                    ==
                    0
                )

                or

                (
                    batch_index
                    ==
                    len(train_loader)
                )
            )

            if should_update:

                # Unscale before clipping
                if use_amp:

                    scaler.unscale_(
                        optimizer
                    )

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    args.grad_clip
                )

                # Optimizer update

                if use_amp:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                scheduler.step()

                global_step += 1

        # ==================================================
        # END OF EPOCH
        # ==================================================

        train_loss = (

            running_loss

            /

            max(
                1,
                len(train_loader)
            )
        )

        elapsed = (

            time.time()
            -
            start_time
        )

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        if val_loader is not None:

            validation_loss = evaluate(

                model,

                val_loader,

                device
            )

        else:

            validation_loss = (
                train_loss
            )

        # --------------------------------------------------
        # PERPLEXITY
        # --------------------------------------------------

        try:

            perplexity = math.exp(

                min(
                    20.0,
                    validation_loss
                )
            )

        except OverflowError:

            perplexity = (
                float("inf")
            )

        current_lr = (

            optimizer
            .param_groups[0]
            ["lr"]
        )

        # --------------------------------------------------
        # PRINT TRAINING RESULT
        # --------------------------------------------------

        print(

            f"Epoch "
            f"{epoch:03d}/"
            f"{args.epochs} | "

            f"train_loss="
            f"{train_loss:.4f} | "

            f"val_loss="
            f"{validation_loss:.4f} | "

            f"ppl="
            f"{perplexity:.2f} | "

            f"lr="
            f"{current_lr:.2e} | "

            f"time="
            f"{elapsed:.1f}s"
        )

        # --------------------------------------------------
        # SAVE LAST MODEL
        # --------------------------------------------------

        save_checkpoint(

            out_dir
            /
            "last.pt",

            model,

            optimizer,

            scheduler,

            epoch,

            global_step,

            best_validation_loss,

            config,

            args
        )

        # --------------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------------

        if (
            validation_loss
            <
            best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            save_checkpoint(

                out_dir
                /
                "best.pt",

                model,

                optimizer,

                scheduler,

                epoch,

                global_step,

                best_validation_loss,

                config,

                args
            )

            print(

                "Saved new best model:",
                out_dir / "best.pt"
            )

    # ======================================================
    # FINISHED
    # ======================================================

    print(
        "\nTraining complete."
    )

    print(
        "Best model:",
        out_dir / "best.pt"
    )


if __name__ == "__main__":

    main()