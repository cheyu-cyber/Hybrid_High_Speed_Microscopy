# Hybrid High-Speed Microscopy

## Configuration Reference (`config.json`)

All project utilities read their parameters from `config.json` at the project root. Each top-level key corresponds to a specific tool or module.

---

### `raw_to_frames`

Converts a `.raw` event recording into accumulated image frames.  
Script: `utils/raw_to_frames.py`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_raw` | string | — | Path to the input `.raw` event file. |
| `out_dir` | string | — | Output directory for generated frames. |
| `fps` | float | `0` | Output frame rate (Hz). Mutually exclusive with `accum_us`. If set, `period_us = 1e6 / fps`. |
| `accum_us` | int | `0` | Accumulation window duration in µs. If 0, defaults to `period_us`. |
| `period_us` | int | `0` | Spacing between consecutive frame windows in µs. Overrides `fps` if > 0. |
| `delta_t_us` | int | `1000` | Internal iterator chunk size in µs. Controls how many µs of raw events are read per loop iteration. Does **not** affect output timing. |
| `max_frames` | int | `0` | Stop after this many frames. 0 = no limit. |
| `start_us` | int | `0` | Start timestamp in µs — events before this are skipped. |
| `duration_us` | int | `0` | Maximum duration to process in µs (0 = entire file). |
| `polarity` | string | `"both"` | Which event polarities to accumulate: `"both"`, `"on"`, or `"off"`. |
| `render_mode` | string | `"count_gray"` | `"count_gray"` — linear count-to-grayscale mapping. `"metavision_dark"` — OpenEB built-in palette rendering. |
| `metavision_gray` | bool | `false` | When `render_mode` is `"metavision_dark"`, use the Gray palette instead of the Dark palette. |
| `to` | string | `"pgm"` | Output image format: `"pgm"`, `"png"`, or `"tiff"`. PNG/TIFF require Pillow. |
| `bit_depth` | int | `8` | Bit depth for PNG/TIFF output (8 or 16). PGM is always 8-bit. |
| `contrast` | float | `4.0` | Visualization gain for mapping event counts to pixel intensity. Higher = brighter output for the same event count. Ignored in `metavision_dark` mode. |

**Timing relationships:**
- `accum_us == period_us` → non-overlapping windows
- `accum_us < period_us` → gaps between windows
- `accum_us > period_us` → overlapping windows

---

### `video_processing`

Processes conventional video frames alongside event data.  
Library: `datasets/video_processing.py`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `video_path` | string | — | Path to the input video file (e.g. `.avi`). |
| `raw_path` | string | — | Path to the corresponding `.raw` event file. |
| `output_path` | string | — | Path for the output high-speed video. |
| `output_fps` | int | `60` | Frame rate of the output video. |
| `blend_alpha` | float | `0.5` | Alpha blending weight when merging video frames with event frames (0 = all video, 1 = all events). |
| `grayscale` | bool | `false` | Convert output to grayscale. |

---

### `video_to_frames`

Extracts individual frames from a video file.  
Script: `utils/video_to_frames.py`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_video` | string | — | Path to the input video file. |
| `out_dir` | string | — | Output directory for extracted frames. |
| `format` | string | `"png"` | Output image format: `"png"`, `"jpg"`, etc. |
| `grayscale` | bool | `false` | Save frames as grayscale. |
| `max_frames` | int | `0` | Stop after this many frames. 0 = no limit. |
| `start_frame` | int | `0` | Skip this many frames before starting extraction. |

---

### `model`

Parameters for the event-guided frame interpolation neural network.  
Module: `models/model.py`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_bins` | int | `5` | Number of temporal bins in the event voxel grid representation. |
| `image_channels` | int | `1` | Number of input image channels (1 = grayscale, 3 = RGB). |
| `base_channels` | int | `32` | Base channel count for the U-Net encoder. Doubles at each depth level. |
| `unet_depth` | int | `4` | Number of encoder/decoder levels in the U-Net. |
| `learning_rate` | float | `1e-4` | Optimizer learning rate. |
| `batch_size` | int | `4` | Training batch size. |
| `num_epochs` | int | `100` | Number of training epochs. |
| `loss` | string | `"l1"` | Loss function: `"l1"` or `"l2"`. |
| `checkpoint_dir` | string | — | Directory to save model checkpoints. |
| `resume_checkpoint` | string | `""` | Path to a checkpoint to resume training from. Empty = train from scratch. |

---

### `raw_to_edge_frames`

Pipeline: raw events → denoise → edge detection → velocity estimation → 3-panel visualisation.  
Script: `utils/raw_to_edge_frames.py`

#### I/O & Timing

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_raw` | string | — | Path to the input `.raw` event file. |
| `out_dir` | string | — | Output directory for edge visualisation frames. |
| `fps` | float | `0` | Output frame rate (Hz). Sets `period_us = 1e6 / fps`. |
| `accum_us` | int | `0` | Accumulation window duration in µs. If 0, defaults to `period_us`. |
| `period_us` | int | `0` | Spacing between consecutive frame windows in µs. Overrides `fps` if > 0. |
| `delta_t_us` | int | `1000` | Internal iterator chunk size in µs. |
| `max_frames` | int | `0` | Stop after this many frames. 0 = no limit. |
| `start_us` | int | `0` | Start timestamp in µs. |
| `duration_us` | int | `0` | Max duration to process in µs (0 = entire file). |

#### Denoising

Applied in order: hot pixel → refractory → nearest-neighbour activity → (optional) polarity consistency.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `hot_pixel_freq` | float | `1000.0` | Pixels firing above this rate (Hz) over the window are removed as hot pixels. |
| `refractory_us` | float | `1000.0` | Minimum time (µs) between consecutive events at the same pixel. Faster repeats are suppressed. |
| `nn_delta_t_us` | float | `5000.0` | Time window (µs) for nearest-neighbour activity check. An event is kept only if a neighbouring pixel also fired within this interval. |
| `polarity_consistency` | bool | `false` | Enable polarity consistency filter (extra stage). |
| `polarity_delta_t_us` | float | `5000.0` | Time window (µs) for polarity consistency check among spatial neighbours. |
| `polarity_min_agreement` | float | `0.6` | Minimum fraction of neighbours that must agree on polarity to keep an event (0–1). |

#### Edge Detection

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `edge_kernel_size` | int | `5` | Sobel kernel size for spatial gradient computation (3, 5, or 7). |
| `grad_threshold` | float | `10.0` | Minimum gradient magnitude to classify a pixel as an edge. |
| `contrast_threshold` | float | `0.15` | Threshold on normalised contrast used inside `estimate_edge_velocity` to filter weak edges. |
| `grad_min` | float | `5.0` | Floor value for gradient magnitude in the velocity denominator to avoid division-by-near-zero. |

#### Visualisation

Output is a 3-panel composite image (left to right): denoised event frame | speed heatmap (JET) | velocity quiver arrows.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `to` | string | `"png"` | Output image format. |
| `speed_max_clip` | float | `0` | Clamp speed heatmap to this maximum value. 0 = auto-scale to per-frame max. |
| `arrow_step` | int | `16` | Pixel spacing between quiver arrows. Smaller = denser arrows. |
| `arrow_scale` | float | `5.0` | Multiplier for arrow length. Larger = longer arrows for the same velocity. |

Arrow colour encodes movement direction via the HSV hue wheel:
- **Red** → rightward
- **Green** → downward
- **Cyan** → leftward
- **Magenta** → upward
