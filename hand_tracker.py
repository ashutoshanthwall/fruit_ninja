"""Hand tracking with the MediaPipe Tasks HandLandmarker.

The original project used cvzone, which is built on the legacy
`mediapipe.solutions` API. That API was removed in MediaPipe 0.10.30, and
older MediaPipe releases have no wheels for Python 3.12+, so cvzone-based
code no longer installs on a current Python. This module talks to the
supported Tasks API directly.
"""

import hashlib
import tempfile
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# Pinned model version + checksum, so every run uses the same weights.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path=DEFAULT_MODEL_PATH):
    """Download the hand landmark model on first run and verify its checksum."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmark model to {path} ...")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        digest = _sha256(tmp_path)
        if digest != MODEL_SHA256:
            raise RuntimeError(f"Model checksum mismatch (got {digest}); refusing to use it.")
        tmp_path.replace(path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the hand model: {e}\n"
            f"Download it manually from {MODEL_URL} and save it as {path}"
        ) from e
    return path


class HandTracker:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            # VIDEO mode re-uses the previous frame's hand position instead of
            # running the palm detector on every frame: faster and steadier.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, bgr_frame, timestamp_ms):
        """Return 21 (x, y, z) landmarks in pixel units, or None if no hand.

        z is scaled by the frame width, which MediaPipe documents as roughly
        the same scale as x.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        h, w = bgr_frame.shape[:2]
        return [(lm.x * w, lm.y * h, lm.z * w) for lm in result.hand_landmarks[0]]

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_hand(img, landmarks, color=(0, 200, 0)):
    pts = [(int(x), int(y)) for x, y, _ in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(img, p, 3, (255, 255, 255), -1)
