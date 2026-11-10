"""Train MiniGPT by pressing Run in your Python editor or Kaggle notebook.

No argparse and no command-line options. Edit TrainingSettings below and press Run.
The script automatically finds the included 24,000/3,000 Q&A train/validation files.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.transfomers.dataset import QADataset, collate_qa, load_qa_file, split_items
from src.transfomers.model import GPTConfig, MiniGPT
from src.transfomers.tokenizer import ByteTokenizer


# ============================================================
# 1. EDIT SETTINGS HERE, THEN PRESS RUN
# ============================================================

@dataclass
class TrainingSettings:
    # None means automatically find the correct file in the data/ folder.
    train_file: str | None = None
    validation_file: str | None = None
    output_dir: str = "checkpoints"

    # MiniGPT architecture (same 4.93-million-parameter model).
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1

    # Training settings.
    epochs: int = 12
    batch_size: int | None = None       # Auto: 32 on large GPUs, 2 on 4GB GPUs.
    grad_accum: int | None = None       # Auto: effective batch size about 32.
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    val_fraction: float = 0.1         # Used only if no validation file exists.
    seed: int = 1337
    num_workers: int = 0              # Compatible with Windows and Kaggle.
    train_on_prompt: bool = False     # False = loss only on answer and EOS.
    amp: bool | None = None           # Auto: enables AMP on newer CUDA GPUs.
    log_every_batches: int = 100


SETTINGS = TrainingSettings()
PROJECT_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. SETUP HELPERS
# ============================================================

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_batch_size(device: torch.device) -> int:
    if device.type != "cuda":
        return 2

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if gpu_memory_gb >= 12:
        return 32
    if gpu_memory_gb >= 6:
        return 8
    return 2


def select_amp(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    # T4/RTX cards benefit more from AMP than older Pascal cards.
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 7


def locate_file(explicit_path: str | None, candidates: list[str]) -> Path | None:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured dataset does not exist: {path}")
        return path

    for name in candidates:
        path = PROJECT_DIR / "data" / name
        if path.is_file():
            return path
        path = PROJECT_DIR / name
        if path.is_file():
            return path

    return None


def resolve_datasets(settings: TrainingSettings):
    train_path = locate_file(settings.train_file, [
        "minigpt_qa_train_24000.jsonl",
        "python_qa_train.jsonl",
        "qa_train.jsonl",
        "qa.jsonl",
        "sample.jsonl",
        "minigpt_qa_shuffled_all_30000.jsonl",
    ])
    if train_path is None:
        raise FileNotFoundError(
            "No Q&A dataset found. Put a JSONL file in the project's data/ folder, "
            "or edit SETTINGS.train_file at the top of train.py."
        )

    # Never use the held-out test data for training or validation.
    validation_candidates = []
    if train_path.name == "minigpt_qa_train_24000.jsonl":
        validation_candidates = ["minigpt_qa_validation_3000.jsonl"]
    elif train_path.name == "python_qa_train.jsonl":
        validation_candidates = ["python_qa_validation.jsonl"]

    validation_path = locate_file(settings.validation_file, validation_candidates)
    if validation_path is not None and validation_path.resolve() == train_path.resolve():
        raise ValueError("Training and validation files must be different.")

    return train_path, validation_path


def make_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if total_steps <= 1:
            return 1.0
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 3. VALIDATE MODEL
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        _, loss = model(input_ids, targets)
        valid_tokens = int((targets != -100).sum().item())
        loss_sum += float(loss.item()) * valid_tokens
        token_count += valid_tokens

    return loss_sum / max(1, token_count)


# ============================================================
# 4. SAVE MODEL CHECKPOINTS
# ============================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_val: float,
    config: GPTConfig,
    settings: TrainingSettings,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "config": config.to_dict(),
        "train_args": asdict(settings),
        "tokenizer": {"type": "byte", "vocab_size": 261, "byte_offset": 5},
    }
    torch.save(checkpoint, path)


# ============================================================
# 5. TRAINING — RUNS AUTOMATICALLY
# ============================================================

def main(settings: TrainingSettings | None = None) -> None:
    settings = settings or SETTINGS
    set_seed(settings.seed)
    device = choose_device()
    batch_size = settings.batch_size or select_batch_size(device)
    grad_accum = settings.grad_accum or max(1, math.ceil(32 / batch_size))
    use_amp = select_amp(device) if settings.amp is None else (settings.amp and device.type == "cuda")

    train_path, validation_path = resolve_datasets(settings)
    tokenizer = ByteTokenizer()
    items = load_qa_file(train_path)

    if validation_path is not None:
        train_items = items
        val_items = load_qa_file(validation_path)
    else:
        train_items, val_items = split_items(items, settings.val_fraction, settings.seed)

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Training dataset: {train_path} ({len(train_items):,} examples)", flush=True)
    print(f"Validation: {validation_path or 'internal split'} ({len(val_items):,} examples)", flush=True)
    print(f"Batch size: {batch_size} | Gradient accumulation: {grad_accum} | AMP: {use_amp}", flush=True)

    train_dataset = QADataset(
        train_items, tokenizer, settings.max_seq_len, train_on_prompt=settings.train_on_prompt
    )
    val_dataset = QADataset(
        val_items, tokenizer, settings.max_seq_len, train_on_prompt=settings.train_on_prompt
    )

    collate_fn = partial(collate_qa, pad_token_id=tokenizer.special.pad)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if len(val_dataset):
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=settings.max_seq_len,
        d_model=settings.d_model,
        n_layers=settings.n_layers,
        n_heads=settings.n_heads,
        ff_mult=settings.ff_mult,
        dropout=settings.dropout,
    )
    model = MiniGPT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)

    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max(1, updates_per_epoch * settings.epochs)
    warmup_steps = int(total_steps * settings.warmup_ratio)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps, settings.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(settings.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "training_settings.json").write_text(
        json.dumps(asdict(settings), indent=2), encoding="utf-8"
    )

    best_val = float("inf")
    global_step = 0
    print(f"Parameters: {model.parameter_count():,} | Epochs: {settings.epochs}", flush=True)

    for epoch in range(1, settings.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        started = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(input_ids, targets)

            # Divide by the true number of microbatches in this accumulation group.
            group_start = ((batch_idx - 1) // grad_accum) * grad_accum
            group_size = min(grad_accum, len(train_loader) - group_start)
            loss_for_backward = loss / group_size

            if use_amp:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            running_loss += float(loss.item())
            should_update = batch_idx % grad_accum == 0 or batch_idx == len(train_loader)

            if should_update:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)

                if use_amp:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    did_update = scaler.get_scale() >= scale_before
                else:
                    optimizer.step()
                    did_update = True

                optimizer.zero_grad(set_to_none=True)
                if did_update:
                    scheduler.step()
                    global_step += 1

            if settings.log_every_batches > 0 and batch_idx % settings.log_every_batches == 0:
                print(
                    f"  Epoch {epoch}/{settings.epochs} | "
                    f"batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}",
                    flush=True,
                )

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device) if val_loader is not None else train_loss
        perplexity = math.exp(min(20.0, val_loss))
        seconds = time.perf_counter() - started
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{settings.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | ppl={perplexity:.2f} | "
            f"lr={current_lr:.2e} | elapsed={seconds:.1f}s",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "best.pt", model, optimizer, scheduler,
                epoch, global_step, best_val, config, settings,
            )
            print(f"  Saved best checkpoint: {out_dir / 'best.pt'}", flush=True)

        save_checkpoint(
            out_dir / "last.pt", model, optimizer, scheduler,
            epoch, global_step, best_val, config, settings,
        )

    print(f"Training complete. Best checkpoint: {out_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
