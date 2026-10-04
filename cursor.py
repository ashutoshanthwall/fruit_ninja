"""High-rate cursor output on its own thread.

The camera gives ~30 positions per second; screens refresh at 60-144 Hz.
This thread moves the mouse at `rate_hz` using MotionPredictor to fill in
between camera frames. It is also the only thread that touches the mouse,
so moves and button presses can never interleave from two threads.
"""

import threading
import time

import numpy as np

from smoothing import MotionPredictor


class CursorThread:
    def __init__(self, mouse, predictor=None, rate_hz=240):
        self.mouse = mouse
        self.predictor = predictor or MotionPredictor()
        self.period = 1.0 / rate_hz
        self.error = None
        self._lock = threading.Lock()
        self._want_pressed = False
        self._enabled = True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cursor", daemon=True)
        self._thread.start()

    # --- called from the main (tracking) thread ---------------------------
    def feed(self, pos, t_capture):
        with self._lock:
            self.predictor.update(pos, t_capture, time.monotonic())

    def lost(self):
        """Hand not visible: forget motion so re-acquiring starts fresh."""
        with self._lock:
            self.predictor.reset()

    def set_pressed(self, down):
        with self._lock:
            self._want_pressed = down

    def set_enabled(self, enabled):
        """Disabled = leave the mouse alone (paused)."""
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._want_pressed = False
                self.predictor.reset()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.mouse.close()  # releases the button if still held

    # --- cursor thread ----------------------------------------------------
    def _run(self):
        last_sent = None
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                with self._lock:
                    pos = self.predictor.step(now) if self._enabled and self.predictor.active else None
                    want = self._want_pressed
                if pos is not None and (last_sent is None or np.abs(pos - last_sent).max() > 1e-5):
                    self.mouse.move(float(pos[0]), float(pos[1]))
                    last_sent = pos.copy()
                if want and not self.mouse.pressed:
                    self.mouse.press()
                elif not want and self.mouse.pressed:
                    self.mouse.release()

                next_tick += self.period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:  # fell behind (e.g. system busy): don't try to catch up
                    next_tick = time.monotonic()
        except Exception as e:  # surfaced to the main thread via .error
            self.error = e
