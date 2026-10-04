"""One Euro filter for cursor smoothing.

Reference: G. Casiez, N. Roussel, D. Vogel, "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012.

A plain moving average forces a trade-off: strong smoothing removes jitter
but makes fast motion lag behind the finger, which is fatal for slicing.
The One Euro filter adapts its cutoff frequency to the speed of the signal:
heavy smoothing when the finger is almost still (jitter is most visible),
light smoothing when it moves fast (lag is most visible).
"""

import math

import numpy as np


def _alpha(cutoff_hz, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff=1.5, beta=40.0, d_cutoff=3.0):
        """
        min_cutoff: cutoff (Hz) at rest. Lower = less jitter, more lag when slow.
        beta: how quickly the cutoff rises with speed. Higher = less lag when fast.
        d_cutoff: cutoff (Hz) used to smooth the speed estimate itself. The
            paper's 1 Hz reacts slowly at the start of a swipe.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x, t):
        """Filter sample `x` (scalar or array) taken at time `t` (seconds)."""
        x = np.asarray(x, dtype=float)
        if self._t is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x

        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        speed = float(np.linalg.norm(self._dx))
        a = _alpha(self.min_cutoff + self.beta * speed, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


class MotionPredictor:
    """Turns ~30 Hz hand positions into a smooth cursor path at any output rate.

    Moving the cursor only when a camera frame arrives makes it jump every
    33 ms. Between frames we instead continue along the current velocity
    ("dead reckoning"), for at most one frame period, so the cursor glides
    continuously until the next measurement takes over.

    Extrapolation overshoots when the hand stops suddenly, and at rest it
    would amplify jitter. So it is
      - scaled by a gain that is ~0 when the hand is slow and ~1 when fast,
      - capped at `max_predict` (fraction of the screen),
    and a short exponential ease hides the small correction when each new
    frame arrives.

    These defaults were picked in a simulation of a 30 FPS camera with 20 ms
    of processing: compared with updating only on new frames, the path is
    about 5x less steppy and ~5 ms less laggy, at the cost of overshooting
    sudden stops of very fast swipes by up to `max_predict`. Predicting
    further ahead cut lag more but overshot far worse.
    """

    def __init__(self, max_predict=0.04, ease_s=0.010, velocity_smoothing=0.8, gain_speed=0.3):
        """
        max_predict: largest extrapolation, as a fraction of the screen. 0 disables it.
        ease_s: time constant of the output easing. Higher = smoother, laggier.
        velocity_smoothing: weight of the newest velocity sample (0..1).
        gain_speed: speed (screens/s) at which extrapolation is half on.
        """
        self.max_predict = max_predict
        self.ease_s = ease_s
        self.velocity_smoothing = velocity_smoothing
        self.gain_speed = gain_speed
        self.reset()

    def reset(self):
        self._pos = self._vel = self._t = self._arrived = None
        self._period = 1 / 30
        self._out = self._out_t = None

    @property
    def active(self):
        return self._pos is not None

    def update(self, pos, t_capture, now):
        """Add a filtered position from a frame captured at `t_capture` that
        became available at `now`."""
        pos = np.asarray(pos, dtype=float)
        if self._t is None:
            self._vel = np.zeros_like(pos)
        elif t_capture <= self._t:
            return
        else:
            dt = t_capture - self._t
            a = self.velocity_smoothing
            self._vel = a * (pos - self._pos) / dt + (1 - a) * self._vel
            self._period = 0.8 * self._period + 0.2 * min(max(dt, 1 / 120), 0.1)
        self._pos, self._t, self._arrived = pos, t_capture, now

    def target(self, now):
        # Only bridge the gap to the next expected frame. If frames stop
        # coming (hand lost), the cursor stops instead of drifting away.
        horizon = min(max(now - self._arrived, 0.0), self._period)
        speed2 = float(self._vel @ self._vel)
        gain = speed2 / (speed2 + self.gain_speed ** 2) if speed2 > 0 else 0.0
        offset = gain * horizon * self._vel
        dist = float(np.linalg.norm(offset))
        if dist > self.max_predict:
            offset *= self.max_predict / dist
        return self._pos + offset

    def step(self, now):
        """Cursor position (0..1, 0..1) to show at time `now`."""
        tgt = self.target(now)
        if self._out is None or self.ease_s <= 0:
            self._out = tgt
        else:
            k = 1.0 - math.exp(-max(now - self._out_t, 0.0) / self.ease_s)
            self._out = self._out + k * (tgt - self._out)
        self._out_t = now
        self._out = np.clip(self._out, 0.0, 1.0)
        return self._out
