"""
Tiny-subset overfitting experiment for EventVFIModel.

Loads a small number of real training samples and repeatedly trains on them
until the model clearly overfits, verifying that the loss and gradient flow
work correctly on real data before attempting full-scale training.

Success criteria (both must pass):
  - final loss  < initial loss  * 0.5   (>50% reduction)
  - final PSNR  > initial PSNR  + 3 dB

All settings are read from config.json section "event_vfi_overfit".

Run:
    python srcs/overfit_test.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List

# Force unbuffered stdout so training progress is visible immediately.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.loss import SelfSupervisedVFILoss
from models.model import EventVFIModel
from utils.config import load_config
from utils.data_preparation import AugmentConfig, SpatialConfig
from utils.debug_visualization import save_cycle_debug_outputs
from utils.get_train_data import build_left_aligned_dataset
from utils.metrics import summarize_reconstruction_metrics


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("overfit")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

_TENSOR_KEYS = ("I0", "I1", "I2", "E01", "E12", "E0515")


def _collate_fn(batch):
    out = {}
    for k in _TENSOR_KEYS:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Dataset builder (training split only, no augmentation)
# ---------------------------------------------------------------------------

def _build_tiny_dataset(cfg, num_samples: int, log: logging.Logger):
    spatial = SpatialConfig()
    if str(getattr(cfg, "resize", "") or "").strip():
        h, w = [int(x) for x in str(cfg.resize).split(",")]
        spatial = SpatialConfig(resize_hw=(h, w))
        print(f"  Resize          : {h}x{w}")
        log.info("Spatial resize: %dx%d", h, w)
    if str(getattr(cfg, "crop", "") or "").strip():
        h, w = [int(x) for x in str(cfg.crop).split(",")]
        spatial = SpatialConfig(resize_hw=spatial.resize_hw, crop_hw=(h, w))
        print(f"  Crop            : {h}x{w}")
        log.info("Spatial crop: %dx%d", h, w)

    print("  Building dataset ...")
    log.info("Building dataset from: %s", cfg.train_frame_dir)
    t0 = time.perf_counter()
    ds = build_left_aligned_dataset(
        sequence_name=getattr(cfg, "sequence_name", "overfit"),
        frame_dir=cfg.train_frame_dir,
        frame_timestamps_csv=cfg.train_frame_csv,
        events_path=cfg.train_events_path,
        num_event_bins=int(cfg.num_event_bins),
        event_window_us=float(getattr(cfg, "event_window_us", 1000.0)),
        sample_step=int(getattr(cfg, "sample_step", 1)),
        spatial=spatial,
        augment=AugmentConfig(),          # no augmentation: repeatable overfitting
        strict_validation=bool(getattr(cfg, "strict_validation", True)),
        rng_seed=0,
    )
    elapsed = time.perf_counter() - t0

    n = min(num_samples, len(ds))
    print(f"  Dataset total   : {len(ds)}  using first {n} samples  ({elapsed:.1f}s)")
    log.info("Dataset built: %d total samples, using %d  (%.1fs)", len(ds), n, elapsed)
    return Subset(ds, list(range(n)))


# ---------------------------------------------------------------------------
# Single step
# ---------------------------------------------------------------------------

def _step(model, criterion, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
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
    metrics = summarize_reconstruction_metrics(out["pred_1_cyc"].detach(), batch["I1"])
    return out, loss_dict, metrics, batch


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_overfit() -> bool:
    log = _setup_logger(Path("logs/overfit_test.log"))
    log.info("=" * 60)
    log.info("Overfit test started")

    cfg = load_config("event_vfi_overfit")
    log.info("Config loaded: event_vfi_overfit")

    num_samples = int(getattr(cfg, "num_samples",  4))
    steps       = int(getattr(cfg, "steps",        300))
    batch_size  = int(getattr(cfg, "batch_size",   2))
    lr          = float(getattr(cfg, "lr",         1e-3))
    log_every   = int(getattr(cfg, "log_every",    10))
    save_vis    = bool(getattr(cfg, "save_vis",    False))
    vis_every   = int(getattr(cfg, "vis_every",    50))
    vis_dir     = Path(str(getattr(cfg, "vis_dir", "logs/overfit_vis")))

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Device          : {device}")
    print(f"  Config section  : event_vfi_overfit")
    print(f"  event_window_us : {getattr(cfg, 'event_window_us', 1000)} µs")
    print(f"  num_samples     : {num_samples}")
    print(f"  steps           : {steps}")
    print(f"  batch_size      : {batch_size}")
    print(f"  lr              : {lr}")
    print(f"  log_every       : {log_every}")
    print(f"  save_vis        : {save_vis}  (every {vis_every} steps -> {vis_dir})")
    log.info("Device: %s | samples: %d | steps: %d | batch: %d | lr: %g",
             device, num_samples, steps, batch_size, lr)

    # ── Dataset ────────────────────────────────────────────────────────────
    ds = _build_tiny_dataset(cfg, num_samples, log)
    effective_batch = min(batch_size, num_samples)
    loader = DataLoader(
        ds,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_fn,
        drop_last=False,
    )
    print(f"  DataLoader      : batch_size={effective_batch}  batches_per_epoch={len(loader)}")
    log.info("DataLoader ready: batch_size=%d, batches_per_epoch=%d", effective_batch, len(loader))

    # ── Model ──────────────────────────────────────────────────────────────
    print("  Loading model ...")
    model_section = str(getattr(cfg, "model_section", "event_vfi_model"))
    model = EventVFIModel.from_config(model_section).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model           : {model_section}  ({n_params:,} params, {n_trainable:,} trainable)")
    log.info("Model: %s | params: %d | trainable: %d", model_section, n_params, n_trainable)

    # ── Loss (no smoothness: allows overfitting) ───────────────────────────
    lambda_char   = float(getattr(cfg, "lambda_char",   1.0))
    lambda_ssim   = float(getattr(cfg, "lambda_ssim",   0.2))
    lambda_smooth = float(getattr(cfg, "lambda_smooth", 0.0))
    criterion = SelfSupervisedVFILoss(
        lambda_char=lambda_char,
        lambda_ssim=lambda_ssim,
        lambda_smooth=lambda_smooth,
        lambda_perc=0.0,
        lambda_pseudo=0.0,
    ).to(device)
    print(f"  Loss weights    : char={lambda_char}  ssim={lambda_ssim}  smooth={lambda_smooth}  perc=0  pseudo=0")
    log.info("Loss: char=%.3f ssim=%.3f smooth=%.3f perc=0 pseudo=0",
             lambda_char, lambda_ssim, lambda_smooth)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log.info("Optimizer: Adam lr=%g", lr)

    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
        log.info("Visualizations will be saved to: %s", vis_dir)

    # ── Training loop ──────────────────────────────────────────────────────
    losses: List[float] = []
    psnrs:  List[float] = []

    model.train()
    print(f"\n  {'Step':>6}  {'loss':>10}  {'char':>10}  {'ssim_l':>10}  {'psnr_dB':>8}  {'ssim':>8}  {'grad_norm':>10}  {'ms/step':>8}")
    print("  " + "-" * 90, flush=True)
    log.info("Training loop started: %d steps", steps)

    global_step = 0
    run_start = time.perf_counter()
    while global_step < steps:
        for batch in loader:
            global_step += 1
            if global_step > steps:
                break

            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            _, loss_dict, metrics, batch_dev = _step(model, criterion, batch, device)
            loss_dict["loss_total"].backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            step_ms = (time.perf_counter() - step_start) * 1000

            loss_val = float(loss_dict["loss_total"].detach())
            psnr_val = metrics["psnr"]
            losses.append(loss_val)
            psnrs.append(psnr_val)

            print(
                f"  {global_step:>6}"
                f"  {loss_val:>10.5f}"
                f"  {float(loss_dict['loss_char'].detach()):>10.5f}"
                f"  {float(loss_dict['loss_ssim'].detach()):>10.5f}"
                f"  {psnr_val:>8.2f}"
                f"  {metrics['ssim']:>8.4f}"
                f"  {grad_norm:>10.4f}"
                f"  {step_ms:>8.1f}",
                flush=True,
            )
            log.debug("step=%d loss=%.5f char=%.5f ssim=%.5f psnr=%.2f ssim=%.4f grad_norm=%.4f ms=%.1f",
                      global_step, loss_val,
                      float(loss_dict['loss_char'].detach()),
                      float(loss_dict['loss_ssim'].detach()),
                      psnr_val, metrics['ssim'], grad_norm, step_ms)

            # periodic running-average summary
            if global_step % log_every == 0:
                window = losses[-log_every:]
                avg_loss = sum(window) / len(window)
                avg_psnr = sum(psnrs[-log_every:]) / len(psnrs[-log_every:])
                elapsed  = time.perf_counter() - run_start
                eta_s    = elapsed / global_step * (steps - global_step)
                print(f"  -- [{global_step}/{steps}]  avg_loss={avg_loss:.5f}  avg_psnr={avg_psnr:.2f} dB"
                      f"  elapsed={elapsed:.0f}s  ETA={eta_s:.0f}s")
                log.info("[%d/%d] avg_loss=%.5f avg_psnr=%.2f elapsed=%.0fs ETA=%.0fs",
                         global_step, steps, avg_loss, avg_psnr, elapsed, eta_s)

            if save_vis and global_step % vis_every == 0:
                with torch.no_grad():
                    model.eval()
                    vis_out = model(
                        I0=batch_dev["I0"], I1=batch_dev["I1"], I2=batch_dev["I2"],
                        E01=batch_dev["E01"], E12=batch_dev["E12"], E0515=batch_dev["E0515"],
                        return_debug=True,
                    )
                    save_cycle_debug_outputs(
                        out_dir=vis_dir,
                        step=global_step,
                        sample={k: batch_dev[k] for k in _TENSOR_KEYS if k in batch_dev},
                        outputs=vis_out,
                    )
                    model.train()
                print(f"  -- Saved visualizations for step {global_step} -> {vis_dir}")
                log.info("Saved visualizations for step %d to %s", global_step, vis_dir)

    # ── Results ────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - run_start
    print("\n" + "=" * 60)
    print("Overfitting summary")
    print("=" * 60)

    initial_loss = losses[0]
    final_loss   = losses[-1]
    min_loss     = min(losses)
    initial_psnr = psnrs[0]
    final_psnr   = psnrs[-1]
    best_psnr    = max(psnrs)

    loss_ratio = final_loss / (initial_loss + 1e-12)
    psnr_gain  = final_psnr - initial_psnr

    print(f"  Total time      : {total_time:.1f}s  ({total_time/steps*1000:.1f} ms/step avg)")
    print(f"  Loss : {initial_loss:.5f} -> {final_loss:.5f}  (ratio {loss_ratio:.3f},  min={min_loss:.5f})")
    print(f"  PSNR : {initial_psnr:.2f} dB -> {final_psnr:.2f} dB  (gain {psnr_gain:+.2f} dB, best {best_psnr:.2f} dB)")

    s = lambda ok: "[PASS]" if ok else "[FAIL]"
    loss_ok = loss_ratio < 0.5
    psnr_ok = psnr_gain > 3.0
    print(f"  {s(loss_ok)} Loss dropped by >50%")
    print(f"  {s(psnr_ok)} PSNR increased by >3 dB")

    passed = loss_ok and psnr_ok
    verdict = "PASS" if passed else "FAIL"
    print(f"\n  Overall: {verdict}")

    log.info("=" * 60)
    log.info("RESULTS: %s | time=%.1fs | loss %.5f->%.5f (ratio=%.3f, min=%.5f) | psnr %.2f->%.2f (gain=%+.2f, best=%.2f)",
             verdict, total_time,
             initial_loss, final_loss, loss_ratio, min_loss,
             initial_psnr, final_psnr, psnr_gain, best_psnr)
    log.info("loss_ok=%s  psnr_ok=%s", loss_ok, psnr_ok)
    log.info("Log saved to: logs/overfit_test.log")
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        passed = run_overfit()
    except Exception:
        _log = logging.getLogger("overfit")
        msg = traceback.format_exc()
        print("\n[FAIL] Exception during overfit test:")
        print(msg)
        _log.error("Exception during overfit test:\n%s", msg)
        sys.exit(1)

    sys.exit(0 if passed else 1)