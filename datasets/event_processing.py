"""Event frame processing utilities for hybrid high-speed microscopy.

Consolidated module for all event-related algorithms:
  - Voxel grid construction from raw events (torch)
  - Optical-flow backward warping (torch)
  - Event accumulation → image rendering (numpy)
  - Image format conversion and saving helpers
  - Edge detection with rising/falling polarity (numpy + cv2)
  - Event density / polarity maps (numpy)
  - Edge velocity estimation from image gradients + event polarity (numpy)
  - Biphasic polarity switch detection for edge traversal analysis (numpy)
  - Event noise filtering: hot pixel, refractory, NN activity, polarity consistency
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# 1. Event representation — voxel grid  (torch)
# ---------------------------------------------------------------------------

def events_to_voxel_grid(events_t, events_x, events_y, events_p,
                         num_bins: int, height: int, width: int):
    """Convert a batch of events into a voxel grid tensor.

    Parameters
    ----------
    events_t : (N,) tensor of normalised timestamps in [0, 1]
    events_x : (N,) int tensor of x coordinates
    events_y : (N,) int tensor of y coordinates
    events_p : (N,) tensor of polarities (+1 / -1)
    num_bins : number of temporal bins B
    height, width : sensor resolution

    Returns
    -------
    voxel : (B, H, W) float tensor
    """
    import torch

    voxel = torch.zeros(num_bins, height, width, dtype=torch.float32,
                        device=events_t.device)
    if events_t.numel() == 0:
        return voxel

    # Distribute each event across two adjacent bins via linear interpolation
    t_scaled = events_t * (num_bins - 1)
    t_low = t_scaled.floor().long().clamp(0, num_bins - 1)
    t_high = (t_low + 1).clamp(0, num_bins - 1)
    w_high = t_scaled - t_low.float()
    w_low = 1.0 - w_high

    idx_y = events_y.long()
    idx_x = events_x.long()

    voxel.index_put_((t_low, idx_y, idx_x), events_p * w_low, accumulate=True)
    voxel.index_put_((t_high, idx_y, idx_x), events_p * w_high, accumulate=True)
    return voxel


# ---------------------------------------------------------------------------
# 3. Event accumulation → image rendering  (numpy)
# ---------------------------------------------------------------------------

def window_image(np, signed, contrast: float, bit_depth: int):
    """Convert a signed event-count array into a viewable grayscale image.

    Parameters
    ----------
    np       : numpy module reference
    signed   : (H, W) int array — net event counts (on − off)
    contrast : scaling factor applied to event counts
    bit_depth : 8 or 16

    Returns
    -------
    img : (H, W) uint8 or uint16 array
    """
    if bit_depth == 8:
        img = 128.0 + contrast * signed.astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    # 16-bit visualization
    img16 = 32768.0 + (contrast * 256.0) * signed.astype(np.float32)
    return np.clip(img16, 0, 65535).astype(np.uint16)


def bgr_to_gray_u8(np, img_bgr):
    """Convert a BGR image to single-channel uint8 using luma weights.

    Parameters
    ----------
    np      : numpy module reference
    img_bgr : (H, W, 3) uint8 array in BGR order

    Returns
    -------
    gray : (H, W) uint8 array
    """
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    y = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(y, 0, 255).astype(np.uint8)


def save_event_image(np, out_path: Path, save_pil: bool, pil_image,
                     img, bit_depth: int = 8, fmt: str = "pgm") -> None:
    """Save an event-frame image to disk.

    Parameters
    ----------
    np        : numpy module reference
    out_path  : destination Path
    save_pil  : True → use Pillow (PNG / TIFF)
    pil_image : PIL.Image module (or None when save_pil is False)
    img       : (H, W) or (H, W, 3) numpy array
    bit_depth : 8 or 16
    fmt       : output format hint ("pgm", "png", "tiff")
    """
    if save_pil:
        if img.ndim == 3:
            # Pillow expects RGB order, OpenEB frame is BGR.
            img = img[:, :, ::-1]
        if bit_depth == 8:
            mode = "L" if img.ndim == 2 else "RGB"
            pil_image.fromarray(img, mode=mode).save(out_path)
        else:
            pil_image.fromarray(img, mode="I;16").save(out_path)
        return

    if img.ndim == 3:
        img = bgr_to_gray_u8(np, img)

    with out_path.open("wb") as f:
        h, w = img.shape
        f.write(b"P5\n")
        f.write(f"{w} {h}\n255\n".encode("ascii"))
        f.write(img.tobytes(order="C"))


# ---------------------------------------------------------------------------
# 4. Edge detection with rising / falling polarity  (numpy + cv2)
# ---------------------------------------------------------------------------

def detect_edges(img, kernel_size=3, grad_threshold=0.0):
    """Compute spatial gradient and separate rising / falling edges.

    Rising edge  = intensity increases along gradient direction (dark → bright).
    Falling edge = intensity decreases (bright → dark).

    Parameters
    ----------
    img            : (H, W) or (H, W, 3) uint8 image (BGR or grayscale)
    kernel_size    : Sobel kernel size (3, 5, 7 …); larger = wider edge response
    grad_threshold : absolute gradient threshold; pixels below are zeroed

    Returns
    -------
    dict with keys:
        magnitude  : (H, W) float32 — gradient magnitude √(gx² + gy²)
        direction  : (H, W) float32 — gradient angle in radians (atan2)
        grad_x     : (H, W) float32 — horizontal gradient (Sobel dx)
        grad_y     : (H, W) float32 — vertical gradient (Sobel dy)
        rising     : (H, W) float32 — positive gradient magnitude (dark→bright)
        falling    : (H, W) float32 — negative gradient magnitude (bright→dark)
    """
    import numpy as np
    import cv2

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=kernel_size)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=kernel_size)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)

    # Rising = component of gradient that is positive (dark→bright)
    # Falling = component that is negative (bright→dark)
    # Combine both axes: take the positive / negative parts per axis,
    # then compute magnitude of each.
    rising = np.sqrt(np.maximum(gx, 0) ** 2 + np.maximum(gy, 0) ** 2)
    falling = np.sqrt(np.maximum(-gx, 0) ** 2 + np.maximum(-gy, 0) ** 2)

    if grad_threshold > 0:
        mask = magnitude < grad_threshold
        magnitude[mask] = 0
        rising[mask] = 0
        falling[mask] = 0

    return {
        "magnitude": magnitude,
        "direction": direction,
        "grad_x": gx,
        "grad_y": gy,
        "rising": rising,
        "falling": falling,
    }


# ---------------------------------------------------------------------------
# 5. Event density and polarity maps  (numpy)
# ---------------------------------------------------------------------------

def compute_event_density(events_x, events_y, events_p, height, width):
    """Build per-pixel event count and net-polarity maps from raw events.

    Parameters
    ----------
    events_x : (N,) int array — x coordinates
    events_y : (N,) int array — y coordinates
    events_p : (N,) int array — polarity (+1 ON, 0 or -1 OFF)
    height, width : sensor resolution

    Returns
    -------
    dict with keys:
        count_map    : (H, W) int32  — total event count per pixel
        polarity_map : (H, W) float32 — net polarity sum per pixel
                       positive = more ON events, negative = more OFF
        on_map       : (H, W) int32  — ON event count per pixel
        off_map      : (H, W) int32  — OFF event count per pixel
    """
    import numpy as np

    x = np.asarray(events_x, dtype=np.int64)
    y = np.asarray(events_y, dtype=np.int64)
    p = np.asarray(events_p, dtype=np.float32)

    count_map = np.zeros((height, width), dtype=np.int32)
    polarity_map = np.zeros((height, width), dtype=np.float32)
    on_map = np.zeros((height, width), dtype=np.int32)
    off_map = np.zeros((height, width), dtype=np.int32)

    if x.size == 0:
        return {
            "count_map": count_map,
            "polarity_map": polarity_map,
            "on_map": on_map,
            "off_map": off_map,
        }

    np.add.at(count_map, (y, x), 1)
    np.add.at(polarity_map, (y, x), p)

    on_mask = p > 0
    np.add.at(on_map, (y[on_mask], x[on_mask]), 1)

    off_mask = p <= 0
    np.add.at(off_map, (y[off_mask], x[off_mask]), 1)

    return {
        "count_map": count_map,
        "polarity_map": polarity_map,
        "on_map": on_map,
        "off_map": off_map,
    }


# ---------------------------------------------------------------------------
# 6. Edge velocity estimation from image gradient + event polarity  (numpy)
# ---------------------------------------------------------------------------

def estimate_edge_velocity(grad_mag, grad_dir, event_density, polarity_map,
                           contrast_threshold=0.1, grad_min=1.0):
    """Estimate per-pixel edge-normal velocity from frame gradient and events.

    Uses the event camera generative model:
        event_rate ∝ |∇I| · v_n
    so:
        v_n = event_density / |∇I|

    The polarity map sign indicates motion direction relative to the gradient:
        positive polarity → edge moving in gradient direction  (dark→bright advancing)
        negative polarity → edge moving against gradient       (bright→dark advancing)

    Parameters
    ----------
    grad_mag       : (H, W) float32 — spatial gradient magnitude from detect_edges
    grad_dir       : (H, W) float32 — gradient angle (radians) from detect_edges
    event_density  : (H, W) int/float — total event count per pixel
    polarity_map   : (H, W) float — net polarity sum per pixel
    contrast_threshold : minimum contrast sensitivity C for the event camera model
    grad_min       : minimum gradient magnitude; pixels below are masked to avoid
                     division-by-zero in flat regions

    Returns
    -------
    dict with keys:
        speed      : (H, W) float32 — edge-normal speed magnitude (px / window)
        velocity_x : (H, W) float32 — x-component of edge-normal velocity
        velocity_y : (H, W) float32 — y-component of edge-normal velocity
        mask       : (H, W) bool    — True where estimate is valid
                     (gradient strong enough)
    """

    grad_mag = np.asarray(grad_mag, dtype=np.float32)
    grad_dir = np.asarray(grad_dir, dtype=np.float32)
    event_density = np.asarray(event_density, dtype=np.float32)
    polarity_map = np.asarray(polarity_map, dtype=np.float32)

    # Valid only where gradient is strong enough
    mask = grad_mag >= grad_min

    speed = np.zeros_like(grad_mag)
    speed[mask] = (event_density[mask] * contrast_threshold) / grad_mag[mask]

    # Direction: polarity sign determines whether motion is along or against
    # the gradient direction.  Normalise polarity to [-1, +1].
    direction_sign = np.zeros_like(polarity_map)
    nonzero = event_density > 0
    direction_sign[nonzero] = np.clip(
        polarity_map[nonzero] / event_density[nonzero], -1.0, 1.0
    )

    # Normal direction unit vector (perpendicular to edge = along gradient)
    nx = np.cos(grad_dir)
    ny = np.sin(grad_dir)

    velocity_x = speed * direction_sign * nx
    velocity_y = speed * direction_sign * ny

    return {
        "speed": speed,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# 7. Biphasic polarity switch detection  (numpy)
# ---------------------------------------------------------------------------

def detect_polarity_switches(events_t, events_x, events_y, events_p,
                             height, width, num_temporal_bins=4,
                             min_events_per_half=2):
    """Detect per-pixel biphasic polarity transitions (edge traversals).

    When an edge crosses a pixel it produces a characteristic biphasic signal:
        0 → +1 → −1 → 0   bright-side-first (bright edge approaching)
        0 → −1 → +1 → 0   dark-side-first   (dark edge approaching)

    The function splits each pixel's event stream at its temporal midpoint and
    compares the net polarity of the early half vs. the late half.

    Parameters
    ----------
    events_t : (N,) float array — timestamps (any unit, must be monotonic)
    events_x : (N,) int array   — x coordinates
    events_y : (N,) int array   — y coordinates
    events_p : (N,) float/int   — polarity (+1 ON, −1 OFF)
    height, width : sensor resolution
    num_temporal_bins : int — temporal bins used for the per-pixel polarity
        profile (more bins = finer temporal resolution of the switch)
    min_events_per_half : int — minimum events in each half for a pixel to be
        considered (suppresses noise)

    Returns
    -------
    dict with keys:
        switch_type     : (H, W) int8
                           0 = no switch or insufficient data
                          +1 = positive→negative  (0→+→−→0, bright edge first)
                          −1 = negative→positive  (0→−→+→0, dark edge first)
        switch_time     : (H, W) float32 — normalised time [0,1] of the polarity
                          crossover (NaN where switch_type == 0)
        switch_strength : (H, W) float32 — abs(early_pol − late_pol), a proxy
                          for edge contrast × event density
        polarity_profile: (num_temporal_bins, H, W) float32 — net polarity in
                          each temporal bin (useful for visualisation)
    """
    t = np.asarray(events_t, dtype=np.float64)
    x = np.asarray(events_x, dtype=np.int64)
    y = np.asarray(events_y, dtype=np.int64)
    p = np.asarray(events_p, dtype=np.float32)

    switch_type = np.zeros((height, width), dtype=np.int8)
    switch_time = np.full((height, width), np.nan, dtype=np.float32)
    switch_strength = np.zeros((height, width), dtype=np.float32)
    profile = np.zeros((num_temporal_bins, height, width), dtype=np.float32)

    if t.size == 0:
        return {
            "switch_type": switch_type,
            "switch_time": switch_time,
            "switch_strength": switch_strength,
            "polarity_profile": profile,
        }

    # Normalise timestamps to [0, 1]
    t_min, t_max = t.min(), t.max()
    t_range = t_max - t_min
    if t_range == 0:
        t_norm = np.zeros_like(t)
    else:
        t_norm = (t - t_min) / t_range

    # --- Build temporal polarity profile ---
    bin_idx = np.clip(
        (t_norm * num_temporal_bins).astype(np.int64), 0, num_temporal_bins - 1
    )
    for b in range(num_temporal_bins):
        mask_b = bin_idx == b
        if mask_b.any():
            np.add.at(profile[b], (y[mask_b], x[mask_b]), p[mask_b])

    # --- Detect switches: compare first-half vs second-half polarity ---
    half = num_temporal_bins // 2
    early_pol = profile[:half].sum(axis=0)   # (H, W)
    late_pol = profile[half:].sum(axis=0)    # (H, W)

    # Count events per half for the min-events filter
    early_count = np.zeros((height, width), dtype=np.int32)
    late_count = np.zeros((height, width), dtype=np.int32)
    early_mask = bin_idx < half
    late_mask = bin_idx >= half
    np.add.at(early_count, (y[early_mask], x[early_mask]), 1)
    np.add.at(late_count, (y[late_mask], x[late_mask]), 1)

    has_enough = (early_count >= min_events_per_half) & (late_count >= min_events_per_half)

    # Positive→negative: early_pol > 0 and late_pol < 0
    pos_neg = has_enough & (early_pol > 0) & (late_pol < 0)
    switch_type[pos_neg] = 1

    # Negative→positive: early_pol < 0 and late_pol > 0
    neg_pos = has_enough & (early_pol < 0) & (late_pol > 0)
    switch_type[neg_pos] = -1

    switched = pos_neg | neg_pos
    switch_strength[switched] = np.abs(early_pol[switched] - late_pol[switched])

    # --- Estimate crossover time per pixel ---
    # Walk temporal bins and find where cumulative polarity crosses zero.
    cum_pol = np.cumsum(profile, axis=0)  # (num_temporal_bins, H, W)
    sy, sx = np.where(switched)
    for i in range(len(sy)):
        py, px = sy[i], sx[i]
        cpol = cum_pol[:, py, px]
        # Find first bin where sign changes
        signs = np.sign(cpol)
        sign_changes = np.where(np.diff(signs) != 0)[0]
        if len(sign_changes) > 0:
            b0 = sign_changes[0]
            # Linear interpolation between bin b0 and b0+1
            v0, v1 = cpol[b0], cpol[b0 + 1]
            denom = v1 - v0
            if denom != 0:
                frac = -v0 / denom
            else:
                frac = 0.5
            switch_time[py, px] = (b0 + frac + 0.5) / num_temporal_bins
        else:
            switch_time[py, px] = 0.5

    return {
        "switch_type": switch_type,
        "switch_time": switch_time,
        "switch_strength": switch_strength,
        "polarity_profile": profile,
    }


# ---------------------------------------------------------------------------
# 8. Event noise filters  (numpy)
# ---------------------------------------------------------------------------

def filter_hot_pixels(events_t, events_x, events_y, events_p,
                      height, width, hot_pixel_freq=1000.0):
    """Remove events from hot pixels (abnormally high firing rate).

    A pixel is considered "hot" if its firing rate exceeds
    ``median_rate + hot_pixel_freq`` events/second across the full stream.

    Parameters
    ----------
    events_t, events_x, events_y, events_p : (N,) arrays
    height, width : sensor resolution
    hot_pixel_freq : float — maximum allowed event rate (events/sec).
        Pixels exceeding this are masked.

    Returns
    -------
    mask : (N,) bool — True for events to **keep**
    hot_mask : (H, W) bool — True for hot pixels (for diagnostics)
    """
    t = np.asarray(events_t, dtype=np.float64)
    x = np.asarray(events_x, dtype=np.int64)
    y = np.asarray(events_y, dtype=np.int64)

    if t.size == 0:
        return np.ones(0, dtype=bool), np.zeros((height, width), dtype=bool)

    duration = t.max() - t.min()
    if duration <= 0:
        return np.ones(t.size, dtype=bool), np.zeros((height, width), dtype=bool)

    count = np.zeros((height, width), dtype=np.int64)
    np.add.at(count, (y, x), 1)

    rate = count.astype(np.float64) / (duration * 1e-6)  # events per second (timestamps in µs)
    hot_mask = rate > hot_pixel_freq

    keep = ~hot_mask[y, x]
    return keep, hot_mask


def filter_refractory(events_t, events_x, events_y, events_p,
                      height, width, refractory_us=1000.0):
    """Suppress events that re-trigger the same pixel within a refractory period.

    If a pixel fires again within ``refractory_us`` of its last accepted event,
    the new event is dropped.

    Parameters
    ----------
    events_t : (N,) float array — timestamps in µs, **must be sorted ascending**
    events_x, events_y, events_p : (N,) arrays
    height, width : sensor resolution
    refractory_us : float — minimum inter-event interval per pixel (µs)

    Returns
    -------
    mask : (N,) bool — True for events to **keep**
    """
    t = np.asarray(events_t, dtype=np.float64)
    x = np.asarray(events_x, dtype=np.int64)
    y = np.asarray(events_y, dtype=np.int64)

    last_ts = np.full((height, width), -np.inf, dtype=np.float64)
    keep = np.ones(t.size, dtype=bool)

    for i in range(t.size):
        yi, xi = y[i], x[i]
        if t[i] - last_ts[yi, xi] < refractory_us:
            keep[i] = False
        else:
            last_ts[yi, xi] = t[i]

    return keep


def filter_activity_nn(events_t, events_x, events_y, events_p,
                       height, width, delta_t_us=5000.0):
    """Nearest-neighbor (NN) activity / correlation filter.

    An event is kept only if at least one of its 8-connected spatial neighbors
    also fired within ``±delta_t_us`` of its own timestamp.  Background-activity
    noise is spatially uncorrelated so this removes the vast majority of it.

    Parameters
    ----------
    events_t : (N,) float array — timestamps in µs, **must be sorted ascending**
    events_x, events_y, events_p : (N,) arrays
    height, width : sensor resolution
    delta_t_us : float — temporal neighborhood radius (µs)

    Returns
    -------
    mask : (N,) bool — True for events to **keep**
    """
    t = np.asarray(events_t, dtype=np.float64)
    x = np.asarray(events_x, dtype=np.int64)
    y = np.asarray(events_y, dtype=np.int64)

    # Timestamp surface: last event time at each pixel
    last_ts = np.full((height, width), -np.inf, dtype=np.float64)
    keep = np.zeros(t.size, dtype=bool)

    # 8-connected neighbor offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    for i in range(t.size):
        yi, xi, ti = y[i], x[i], t[i]

        # Check neighbors
        has_neighbor = False
        for dy, dx in offsets:
            ny_, nx_ = yi + dy, xi + dx
            if 0 <= ny_ < height and 0 <= nx_ < width:
                if abs(ti - last_ts[ny_, nx_]) <= delta_t_us:
                    has_neighbor = True
                    break

        keep[i] = has_neighbor
        last_ts[yi, xi] = ti

    return keep


def filter_polarity_consistency(events_t, events_x, events_y, events_p,
                                height, width, delta_t_us=5000.0,
                                min_agreement=0.6):
    """Filter events by spatial polarity coherence.

    Noise has random polarity, while real edge events have spatially coherent
    polarity among neighbors.  For each event, check the recent polarity of
    its 8-connected neighbors; keep the event only if a sufficient fraction
    of active neighbors share its polarity.

    Parameters
    ----------
    events_t : (N,) float array — timestamps in µs, **must be sorted ascending**
    events_x, events_y, events_p : (N,) arrays (polarity +1 / −1)
    height, width : sensor resolution
    delta_t_us : float — temporal neighborhood radius (µs)
    min_agreement : float in (0, 1] — fraction of active neighbors that must
        share the event's polarity to keep it (0.6 = 60 %)

    Returns
    -------
    mask : (N,) bool — True for events to **keep**
    """
    t = np.asarray(events_t, dtype=np.float64)
    x = np.asarray(events_x, dtype=np.int64)
    y = np.asarray(events_y, dtype=np.int64)
    p = np.asarray(events_p, dtype=np.float32)

    last_ts = np.full((height, width), -np.inf, dtype=np.float64)
    last_pol = np.zeros((height, width), dtype=np.float32)
    keep = np.zeros(t.size, dtype=bool)

    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    for i in range(t.size):
        yi, xi, ti, pi = y[i], x[i], t[i], p[i]

        same = 0
        total = 0
        for dy, dx in offsets:
            ny_, nx_ = yi + dy, xi + dx
            if 0 <= ny_ < height and 0 <= nx_ < width:
                if abs(ti - last_ts[ny_, nx_]) <= delta_t_us:
                    total += 1
                    if last_pol[ny_, nx_] * pi > 0:  # same sign
                        same += 1

        if total > 0 and (same / total) >= min_agreement:
            keep[i] = True
        elif total == 0:
            keep[i] = False  # isolated = noise

        last_ts[yi, xi] = ti
        last_pol[yi, xi] = pi

    return keep


# ---------------------------------------------------------------------------
# 9. Combined noise filter pipeline  (numpy)
# ---------------------------------------------------------------------------

def filter_events(events_t, events_x, events_y, events_p,
                  height, width,
                  hot_pixel_freq=1000.0,
                  refractory_us=1000.0,
                  nn_delta_t_us=5000.0,
                  polarity_consistency=False,
                  polarity_delta_t_us=5000.0,
                  polarity_min_agreement=0.6):
    """Apply the full noise-filter pipeline in recommended order.

        raw → hot pixel → refractory → NN activity → (optional) polarity

    Parameters
    ----------
    events_t, events_x, events_y, events_p : (N,) arrays
        Input events.  ``events_t`` must be in µs and sorted ascending.
    height, width : sensor resolution
    hot_pixel_freq : float — hot pixel rate threshold (events/sec). Set 0 to skip.
    refractory_us : float — refractory period (µs). Set 0 to skip.
    nn_delta_t_us : float — NN activity time window (µs). Set 0 to skip.
    polarity_consistency : bool — whether to apply polarity consistency filter
    polarity_delta_t_us : float — time window for polarity filter
    polarity_min_agreement : float — neighbor agreement fraction

    Returns
    -------
    dict with keys:
        events_t, events_x, events_y, events_p : filtered arrays
        kept      : int — events remaining
        removed   : int — events removed
        hot_mask  : (H, W) bool — hot pixel map (None if skipped)
        stages    : dict — events remaining after each stage
    """
    t = np.asarray(events_t, dtype=np.float64)
    x = np.asarray(events_x, dtype=np.int64)
    y = np.asarray(events_y, dtype=np.int64)
    p = np.asarray(events_p, dtype=np.float32)

    n_orig = t.size
    stages = {"original": n_orig}
    hot_mask_out = None

    # 1. Hot pixel
    if hot_pixel_freq > 0 and t.size > 0:
        keep, hot_mask_out = filter_hot_pixels(t, x, y, p, height, width,
                                               hot_pixel_freq=hot_pixel_freq)
        t, x, y, p = t[keep], x[keep], y[keep], p[keep]
    stages["after_hot_pixel"] = t.size

    # 2. Refractory
    if refractory_us > 0 and t.size > 0:
        keep = filter_refractory(t, x, y, p, height, width,
                                 refractory_us=refractory_us)
        t, x, y, p = t[keep], x[keep], y[keep], p[keep]
    stages["after_refractory"] = t.size

    # 3. NN activity
    if nn_delta_t_us > 0 and t.size > 0:
        keep = filter_activity_nn(t, x, y, p, height, width,
                                  delta_t_us=nn_delta_t_us)
        t, x, y, p = t[keep], x[keep], y[keep], p[keep]
    stages["after_nn_activity"] = t.size

    # 4. Polarity consistency (optional)
    if polarity_consistency and t.size > 0:
        keep = filter_polarity_consistency(
            t, x, y, p, height, width,
            delta_t_us=polarity_delta_t_us,
            min_agreement=polarity_min_agreement,
        )
        t, x, y, p = t[keep], x[keep], y[keep], p[keep]
    stages["after_polarity"] = t.size

    return {
        "events_t": t,
        "events_x": x,
        "events_y": y,
        "events_p": p,
        "kept": t.size,
        "removed": n_orig - t.size,
        "hot_mask": hot_mask_out,
        "stages": stages,
    }
