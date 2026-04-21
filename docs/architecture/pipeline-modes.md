# Pipeline Modes

This backend supports runtime mode selection via `pipeline_mode` on `POST /process/{job_id}`.

## Modes

### `fast`
- Goal: lower end-to-end latency, target near-realtime ratio on capable hardware.
- Detector: higher stride, reduced annotation write frequency, lower imgsz.
- Habitat: sampled frame classification.

### `quality`
- Goal: maximize tracking continuity and detection consistency.
- Detector: lower stride, dense frame outputs, higher imgsz.
- Habitat: full-frame classification on saved frames.

### `default`
- Uses existing environment defaults (`EA_*` variables).

## Why This Structure

- Keeps one API/process flow and avoids process branching complexity.
- Allows CPU fallback and optional GPU acceleration using the same code path.
- Makes SLA explicit: `fast` for turnaround, `quality` for reporting.

## Evaluation Target

Primary runtime metric:

- `processing_ratio = processing_time_seconds / video_duration_seconds`

Primary tracking quality metrics:

- ID switches
- stable unique tracks
- detection confidence distribution
