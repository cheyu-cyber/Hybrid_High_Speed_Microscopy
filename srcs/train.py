"""
Full training loop for EventVFIModel (cycle-consistency, self-supervised).

Pipeline per step:
  [I0, I1, E01]             -> pred_05     (E01   = events centered at t0.5 ± half_window)
  [I1, I2, E12]             -> pred_15     (E12   = events centered at t1.5 ± half_window)
  [pred_05, pred_15, E0515] -> pred_1_cyc  (E0515 = events centered at t1   ± half_window)

All settings are read from config.json section "event_vfi_train".
Model architecture is read from the section named by model_section.

Run:
    python srcs/train.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.loss import SelfSupervisedVFILoss
from models.model import EventVFIModel
from utils.config import load_config
from utils.data_preparation import AugmentConfig, SpatialConfig
from utils.debug_visualization import save_cycle_debug_outputs
from utils.get_train_data import build_centered_window_dataset
from utils.metrics import summarize_reconstruction_metrics


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# DataLoader collation  (drop meta dict, keep only model-input tensors)
# ---------------------------------------------------------------------------

_TENSOR_KEYS = ("I0", "I1", "I2", "E01", "E12", "E0515")


def _collate_fn(batch):
    out = {}
    for k in _TENSOR_KEYS:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _parse_spatial(cfg) -> SpatialConfig:
    resize_hw = None
    if str(getattr(cfg, "resize", "") or "").strip():
        h, w = [int(x) for x in str(cfg.resize).split(",")]
        resize_hw = (h, w)

    crop_hw = None
    if str(getattr(cfg, "crop", "") or "").strip():
        h, w = [int(x) for x in str(cfg.crop).split(",")]
        crop_hw = (h, w)

    return SpatialConfig(
        resize_hw=resize_hw,
        crop_hw=crop_hw,
        random_crop=bool(getattr(cfg, "random_crop", False)),
    )


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_dataset(cfg, split: str, overfit_n: int = 0, seed: int = 42):
    if split == "train":
        frame_dir   = cfg.train_frame_dir
        frame_csv   = cfg.train_frame_csv
        events_path = cfg.train_events_path
        hflip_prob  = float(getattr(cfg, "horizontal_flip_prob", 0.5))
        augment     = AugmentConfig(horizontal_flip_prob=hflip_prob)
    else:
        frame_dir   = cfg.val_frame_dir
        frame_csv   = cfg.val_frame_csv
        events_path = cfg.val_events_path
        augment     = AugmentConfig()

    ds = build_centered_window_dataset(
        sequence_name=getattr(cfg, "sequence_name", split),
        frame_dir=frame_dir,
        frame_timestamps_csv=frame_csv,
        events_path=events_path,
        num_event_bins=int(cfg.num_event_bins),
        event_window_us=float(getattr(cfg, "event_window_us", 1000.0)),
        sample_step=int(getattr(cfg, "sample_step", 1)),
        spatial=_parse_spatial(cfg),
        augment=augment,
        strict_validation=bool(getattr(cfg, "strict_validation", True)),
        rng_seed=seed,
    )

    if overfit_n > 0:
        ds = Subset(ds, list(range(min(overfit_n, len(ds)))))
        log.info(f"[{split}] Overfitting to {len(ds)} samples")
    else:
        log.info(f"[{split}] Dataset: {len(ds)} samples")

    return ds


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    step: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    best_psnr: float,
) -> None:
    state: Dict[str, Any] = {
        "step":      step,
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_psnr": best_psnr,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)
    log.info(f"Checkpoint -> {path}")


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[Any],
    device: torch.device,
) -> Dict[str, Any]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    log.info(f"Resumed from {path}  (step={state.get('step', 0)})")
    return state


# ---------------------------------------------------------------------------
# Forward + loss (shared by train loop and validation)
# ---------------------------------------------------------------------------

def _forward_and_loss(model, criterion, batch):
    out = model(
        I0=batch["I0"], I1=batch["I1"], I2=batch["I2"],
        E01=batch["E01"], E12=batch["E12"], E0515=batch["E0515"],
        return_debug=False,
    )
    loss_dict = criterion(
        pred_center_cyc=out["pred_1_cyc"],
        gt_center=batch["I1"],
        motion_01=out["stage_01"]["motion"],
        motion_12=out["stage_12"]["motion"],
        motion_cyc=out["stage_cyc"]["motion"],
        img0=batch["I0"], img1=batch["I1"], img2=batch["I2"],
        pred05=out["pred_05"], pred15=out["pred_15"],
    )
    return out, loss_dict


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, float]:
    model.eval()
    total_psnr = total_ssim = 0.0
    n = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            I0=batch["I0"], I1=batch["I1"], I2=batch["I2"],
            E01=batch["E01"], E12=batch["E12"], E0515=batch["E0515"],
            return_debug=False,
        )
        m = summarize_reconstruction_metrics(out["pred_1_cyc"], batch["I1"])
        total_psnr += m["psnr"]
        total_ssim += m["ssim"]
        n += 1

    model.train()
    if n == 0:
        return {"val_psnr": 0.0, "val_ssim": 0.0}
    return {"val_psnr": total_psnr / n, "val_ssim": total_ssim / n}


# ---------------------------------------------------------------------------
# Visualize one sample (debug images)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _visualize(model, batch, vis_dir: Path, step: int, device: torch.device) -> None:
    model.eval()
    b1 = {k: v[:1].to(device) for k, v in batch.items()}
    out = model(
        I0=b1["I0"], I1=b1["I1"], I2=b1["I2"],
        E01=b1["E01"], E12=b1["E12"], E0515=b1["E0515"],
        return_debug=True,
    )
    save_cycle_debug_outputs(out_dir=vis_dir, step=step, sample=b1, outputs=out)
    model.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    cfg = load_config("event_vfi_train")

    torch.manual_seed(int(getattr(cfg, "seed", 42)))
    device = _resolve_device(str(getattr(cfg, "device", "auto")))
    log.info(f"Device: {device}")

    # ── Output directories ────────────────────────────────────────────────
    out_dir  = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    vis_dir  = out_dir / "vis"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    # ── Datasets ──────────────────────────────────────────────────────────
    overfit_n = int(getattr(cfg, "overfit_num_samples", 0))
    seed      = int(getattr(cfg, "seed", 42))
    train_ds  = _build_dataset(cfg, "train", overfit_n=overfit_n, seed=seed)
    val_ds    = _build_dataset(cfg, "val",   overfit_n=0,         seed=0)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=(overfit_n == 0),
        num_workers=int(getattr(cfg, "num_workers", 0)),
        collate_fn=_collate_fn,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(getattr(cfg, "val_batch_size", 2)),
        shuffle=False,
        num_workers=int(getattr(cfg, "num_workers", 0)),
        collate_fn=_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_section = str(getattr(cfg, "model_section", "event_vfi_model"))
    model = EventVFIModel.from_config(model_section).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model ({model_section}): {n_params:,} parameters")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = SelfSupervisedVFILoss(
        lambda_char=float(getattr(cfg, "lambda_char",   1.0)),
        lambda_ssim=float(getattr(cfg, "lambda_ssim",   0.2)),
        lambda_smooth=float(getattr(cfg, "lambda_smooth", 0.01)),
        lambda_perc=float(getattr(cfg, "lambda_perc",   0.0)),
        lambda_pseudo=float(getattr(cfg, "lambda_pseudo", 0.0)),
    ).to(device)

    # ── Optimizer & AMP ───────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(getattr(cfg, "weight_decay", 1e-4)),
    )
    use_amp   = bool(getattr(cfg, "amp", False)) and device.type == "cuda"
    scaler    = torch.cuda.amp.GradScaler() if use_amp else None
    grad_clip = float(getattr(cfg, "grad_clip", 1.0))

    # ── Resume ────────────────────────────────────────────────────────────
    step        = 0
    start_epoch = 0
    best_psnr   = 0.0
    resume      = str(getattr(cfg, "resume", "")).strip()
    if resume:
        state       = _load_checkpoint(Path(resume), model, optimizer, scaler, device)
        step        = state.get("step", 0)
        start_epoch = state.get("epoch", 0)
        best_psnr   = state.get("best_psnr", 0.0)

    max_steps  = int(getattr(cfg, "max_train_steps", 0))
    epochs     = int(getattr(cfg, "epochs", 30))
    log_every  = int(getattr(cfg, "log_every_steps",  10))
    val_every  = int(getattr(cfg, "val_every_steps",  100))
    save_every = int(getattr(cfg, "save_every_steps", 200))
    vis_every  = int(getattr(cfg, "vis_every_steps",  100))

    log.info(
        f"epochs={epochs}  max_steps={max_steps or 'inf'}"
        f"  lr={cfg.lr}  event_window_us={getattr(cfg, 'event_window_us', 1000)}"
        f"  amp={use_amp}"
    )

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        for batch in train_loader:
            step += 1
            if max_steps > 0 and step > max_steps:
                log.info(f"Reached max_train_steps={max_steps}. Stopping.")
                _save_checkpoint(ckpt_dir / "final.pt", step, epoch, model, optimizer, scaler, best_psnr)
                return

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    out, loss_dict = _forward_and_loss(model, criterion, batch)
                scaler.scale(loss_dict["loss_total"]).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                out, loss_dict = _forward_and_loss(model, criterion, batch)
                loss_dict["loss_total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # ── Log ────────────────────────────────────────────────────
            if step % log_every == 0:
                elapsed = time.time() - t0
                row = {
                    "step": step, "epoch": epoch, "elapsed_s": round(elapsed, 1),
                    **{k: round(float(v.detach()), 6) for k, v in loss_dict.items()},
                }
                log.info(
                    f"step {step:6d}  loss={row['loss_total']:.4f}"
                    f"  char={row['loss_char']:.4f}"
                    f"  ssim_l={row['loss_ssim']:.4f}"
                    f"  smooth={row['loss_smooth']:.4f}"
                )
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")

            # ── Visualise ──────────────────────────────────────────────
            if step % vis_every == 0:
                _visualize(model, batch, vis_dir, step, device)

            # ── Validate ───────────────────────────────────────────────
            if step % val_every == 0:
                val_m = _run_validation(model, val_loader, device)
                log.info(
                    f"[VAL] step {step:6d}"
                    f"  psnr={val_m['val_psnr']:.2f}"
                    f"  ssim={val_m['val_ssim']:.4f}"
                )
                with log_path.open("a") as f:
                    f.write(json.dumps({"step": step, **val_m}) + "\n")
                if val_m["val_psnr"] > best_psnr:
                    best_psnr = val_m["val_psnr"]
                    _save_checkpoint(
                        ckpt_dir / "best.pt",
                        step, epoch, model, optimizer, scaler, best_psnr,
                    )

            # ── Checkpoint ─────────────────────────────────────────────
            if step % save_every == 0:
                _save_checkpoint(
                    ckpt_dir / f"ckpt_step_{step:07d}.pt",
                    step, epoch, model, optimizer, scaler, best_psnr,
                )

        log.info(f"Epoch {epoch + 1}/{epochs} complete  (global step {step})")

    _save_checkpoint(ckpt_dir / "final.pt", step, epochs, model, optimizer, scaler, best_psnr)
    log.info("Training complete.")


if __name__ == "__main__":
    train()