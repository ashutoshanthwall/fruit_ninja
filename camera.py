"""Webcam capture on a background thread that always serves the newest frame.

Reading the camera in the main loop has two latency costs: the loop sits
idle in cap.read() while it could be working, and frames queued in the
driver while the loop was busy get processed late. Here a thread reads
continuously and keeps only the latest frame, so detection always runs on
the freshest image available.
"""

import os
import struct
import sys
import threading
import time

import cv2

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

# Linux V4L2 control "Exposure, Dynamic Framerate" (V4L2_CID_EXPOSURE_AUTO_PRIORITY).
# When on, the camera may lower its frame rate to expose longer in dim light:
# 30 -> 15 FPS doubles the delay of every hand position.
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903
_V4L2_CONTROL = struct.Struct("=Ii")
VIDIOC_G_CTRL = (3 << 30) | (_V4L2_CONTROL.size << 16) | (ord("V") << 8) | 27
VIDIOC_S_CTRL = (3 << 30) | (_V4L2_CONTROL.size << 16) | (ord("V") << 8) | 28


def v4l2_control(index, cid, value=None):
    """Read (value=None) or set a V4L2 control on /dev/video<index>.
    Returns the previous value, or None if unsupported."""
    if fcntl is None or not sys.platform.startswith("linux"):
        return None
    try:
        fd = os.open(f"/dev/video{index}", os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        old = _V4L2_CONTROL.unpack(fcntl.ioctl(fd, VIDIOC_G_CTRL, _V4L2_CONTROL.pack(cid, 0)))[1]
        if value is not None and value != old:
            fcntl.ioctl(fd, VIDIOC_S_CTRL, _V4L2_CONTROL.pack(cid, value))
        return old
    except OSError:
        return None
    finally:
        os.close(fd)


class LatestFrameCamera:
    def __init__(self, index=0, width=640, height=480, fps=30, fixed_fps=True):
        """fixed_fps: stop the camera from halving its frame rate in dim light
        (Linux only; the original setting is restored on release). The image
        gets darker instead."""
        self._index = index
        self._restore_auto_priority = None
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. Is it connected or used by another app? Try --camera 1.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if fixed_fps:
            old = v4l2_control(index, V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)
            if old:  # was on: remember to switch it back
                self._restore_auto_priority = old

        self._cond = threading.Condition()
        self._frame = None
        self._time = 0.0
        self._seq = 0
        self._last_read = 0
        self._failed = False
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _capture_time(self):
        """When the frame was captured, on the time.monotonic() clock.

        The Linux V4L2 backend reports the driver's buffer timestamp, taken on
        the same monotonic clock and earlier than the moment read() returns.
        Other backends use other clocks, so anything implausible falls back
        to "now".
        """
        now = time.monotonic()
        driver = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return driver if 0 <= now - driver < 0.5 else now

    def _run(self):
        while self._running:
            ok, frame = self._cap.read()
            with self._cond:
                if not ok:
                    self._failed = True
                    self._cond.notify_all()
                    return
                self._frame, self._time = frame, self._capture_time()
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout=2.0):
        """Wait for a frame newer than the last one returned.

        Returns (frame, capture_time, frames_skipped), or None if the camera
        failed or stopped delivering frames.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._last_read or self._failed, timeout):
                return None
            if self._seq == self._last_read:  # failed
                return None
            skipped = self._seq - self._last_read - 1
            self._last_read = self._seq
            return self._frame, self._time, skipped

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()
        if self._restore_auto_priority is not None:
            v4l2_control(self._index, V4L2_CID_EXPOSURE_AUTO_PRIORITY, self._restore_auto_priority)
