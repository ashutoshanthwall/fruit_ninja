"""Simulate the hand -> camera -> filter -> cursor pipeline and score it.

Reproduces the numbers in the README's latency table. Run from the project root:

    python tools/simulate_latency.py

Model: a scripted hand path (a fast slash, a medium swipe, a slow move, with
rests between) is sampled by a 30 FPS camera, with Gaussian noise added to
each measurement. Each frame becomes available PROC seconds after capture.
The cursor is sampled at 240 Hz and compared with the true hand position.
CAM_LAT (motion before the frame timestamp) is an assumption, not a
measurement; display latency is not modelled.

Metrics (pixels assume a 1920 px wide screen):
  lag          delay that best aligns the cursor with the true path while moving
  jitter       standard deviation of the cursor while the hand rests
  overshoot    how far the cursor passes the end of the fast slash
  steppiness   RMS second difference of the 240 Hz cursor path while moving
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoothing import MotionPredictor, OneEuroFilter  # noqa: E402

PX = 1920
FPS, CAM_LAT, PROC, RATE = 30, 0.015, 0.020, 240
SEGMENTS = [(1.0, 0.25, 0.1, 0.9), (2.0, 0.5, 0.9, 0.2), (3.2, 1.5, 0.2, 0.5)]  # start, duration, from, to
RESTS = [(0.5, 1.0), (1.6, 2.0), (2.9, 3.2), (5.0, 5.5)]
T_END = 5.5


def truth(t):
    t = np.asarray(t, float)
    x = np.full_like(t, 0.1)
    for t0, dur, a, b in SEGMENTS:
        s = np.clip((t - t0) / dur, 0, 1)
        x = np.where(t >= t0, a + (b - a) * (10 * s**3 - 15 * s**4 + 6 * s**5), x)  # minimum-jerk
    return x


def moving(t):
    return np.any([(t >= t0) & (t <= t0 + d + 0.05) for t0, d, _, _ in SEGMENTS], axis=0)


def run(filter_args, predictor_args, sigma, seed):
    """predictor_args=None reproduces the old pipeline: cursor moves only on new frames."""
    rng = np.random.default_rng(seed)
    frames = np.arange(0, T_END, 1 / FPS)
    meas = truth(frames - CAM_LAT) + rng.normal(0, sigma, frames.size)
    f = OneEuroFilter(*filter_args)
    filt = [float(f(m, t)) for m, t in zip(meas, frames)]
    pred = MotionPredictor(**predictor_args) if predictor_args is not None else None

    taus = np.arange(0.3, T_END, 1 / RATE)
    out, k = [], -1
    for tau in taus:
        while k + 1 < frames.size and frames[k + 1] + PROC <= tau:
            k += 1
            if pred:
                pred.update([filt[k], 0.5], frames[k], frames[k] + PROC)
        out.append(filt[k] if pred is None else float(pred.step(tau)[0]))
    out = np.array(out)

    mv = moving(taus)
    lags = np.arange(0, 0.2, 0.002)
    lag = lags[np.argmin([np.mean((out[mv] - truth(taus[mv] - L)) ** 2) for L in lags])]
    jitter = np.mean([np.std(out[(taus >= a) & (taus < b)]) for a, b in RESTS])
    overshoot = max(0.0, out[(taus > 1.25) & (taus < 2.0)].max() - 0.9)
    steppiness = np.sqrt(np.mean((np.diff(out, 2)[mv[2:]] * PX) ** 2))
    return lag * 1000, jitter * PX, overshoot * PX, steppiness


def main():
    configs = {
        "before: 30 Hz updates, 1euro(1, 20, 1)": ((1.0, 20.0, 1.0), None),
        "now: 240 Hz + bounded prediction": ((1.5, 40.0, 3.0), {}),
        "now, --max-predict 0": ((1.5, 40.0, 3.0), {"max_predict": 0}),
        "unbounded prediction (rejected)": ((1.5, 40.0, 3.0), {"max_predict": 1.0}),
    }
    for sigma_px in (2.9, 4.8, 7.7):
        print(f"measurement noise {sigma_px} px")
        for name, (fa, pa) in configs.items():
            r = np.mean([run(fa, pa, sigma_px / PX, seed) for seed in range(6)], axis=0)
            print(f"  {name:42s} lag {r[0]:4.0f} ms  jitter {r[1]:4.2f} px  "
                  f"overshoot {r[2]:5.1f} px  steppiness {r[3]:5.1f}")


if __name__ == "__main__":
    main()
