"""Gesture logic: turns hand landmarks into finger states and stable mouse actions.

Nothing here touches the camera or the mouse, so it can be unit-tested
with hand-written landmark data (see tests/).
"""

import math
import time

# MediaPipe hand landmark indices (21 points per hand).
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_TIP = 5, 8
PINKY_MCP = 17
PALM = (0, 5, 9, 13, 17)

# (tip, pip) landmark pairs for the four long fingers.
FINGER_JOINTS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def fingers_up(landmarks, previous=None, band=0.08):
    """Return {"thumb", "index", "middle", "ring", "pinky"} -> bool.

    `landmarks` is a list of 21 (x, y, z) points in pixel units.

    A finger counts as extended when its tip is farther from the wrist than
    its middle (PIP) joint. Comparing distances instead of "tip.y < pip.y"
    keeps working when the hand is tilted or pointing sideways.

    Pass the previous result as `previous` to add hysteresis: a finger then
    only changes state once the distance ratio is clearly past 1 (by
    `band`). This stops a half-bent finger from flickering between frames
    without waiting extra frames to confirm every change.
    """
    wrist = landmarks[WRIST]
    pinky_mcp = landmarks[PINKY_MCP]
    ratios = {
        name: math.dist(wrist, landmarks[tip]) / max(math.dist(wrist, landmarks[pip]), 1e-6)
        for name, (tip, pip) in FINGER_JOINTS.items()
    }
    # A folded thumb tucks its tip towards the pinky side of the palm.
    ratios["thumb"] = (math.dist(landmarks[THUMB_TIP], pinky_mcp)
                       / max(math.dist(landmarks[THUMB_IP], pinky_mcp), 1e-6))

    state = {}
    for name, ratio in ratios.items():
        if previous is None:
            state[name] = ratio > 1.0
        elif previous[name]:
            state[name] = ratio > 1.0 - band
        else:
            state[name] = ratio > 1.0 + band
    return state


def tracking_point(landmarks, point="knuckle"):
    """The (x, y) that steers the cursor.

    knuckle: base of the index finger (default). It does not move when the
        finger curls, so ending a slice doesn't drag the cursor downwards,
        and it jitters less than the fingertip.
    palm: average of the wrist and the four knuckles; the steadiest point.
    tip: the index fingertip.
    """
    if point == "tip":
        idx = (INDEX_TIP,)
    elif point == "palm":
        idx = PALM
    else:
        idx = (INDEX_MCP,)
    xs = [landmarks[i][0] for i in idx]
    ys = [landmarks[i][1] for i in idx]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def is_slice_gesture(fingers):
    """Index finger out, middle/ring/pinky curled. The thumb is ignored on
    purpose: its up/down reading is the noisiest of the five and requiring it
    made the slice flicker on and off."""
    return fingers["index"] and not (fingers["middle"] or fingers["ring"] or fingers["pinky"])


def is_open_palm(fingers):
    return fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]


class SliceController:
    """Debounces the per-frame slice gesture into mouse press/release actions.

    Finger hysteresis already removes most flicker, so a slice starts on the
    first frame. Releasing waits for a second frame so one odd reading can't
    cut a slice in half, and fast swipes often blur the hand enough that it
    is briefly not detected at all, so a lost hand keeps the button down
    for a few frames.
    """

    def __init__(self, press_frames=1, release_frames=2, lost_frames=5):
        self.press_frames = press_frames
        self.release_frames = release_frames
        self.lost_frames = lost_frames
        self.reset()

    def reset(self):
        self.pressed = False
        self._on = self._off = self._lost = 0

    def update(self, gesture):
        """`gesture` is True/False for a detected hand, or None if no hand.
        Returns "press", "release" or None."""
        if gesture is None:
            self._lost += 1
            self._on = 0
            if self.pressed and self._lost >= self.lost_frames:
                self.pressed = False
                return "release"
            return None

        self._lost = 0
        if gesture:
            self._on += 1
            self._off = 0
            if not self.pressed and self._on >= self.press_frames:
                self.pressed = True
                return "press"
        else:
            self._off += 1
            self._on = 0
            if self.pressed and self._off >= self.release_frames:
                self.pressed = False
                return "release"
        return None


class HoldToggle:
    """Fires once when a gesture has been held for `hold_seconds`, then waits
    for the gesture to be dropped before it can fire again."""

    def __init__(self, hold_seconds=1.0, clock=time.monotonic):
        self.hold_seconds = hold_seconds
        self._clock = clock
        self._since = None
        self._fired = False

    def update(self, active):
        if not active:
            self._since = None
            self._fired = False
            return False
        now = self._clock()
        if self._since is None:
            self._since = now
        if not self._fired and now - self._since >= self.hold_seconds:
            self._fired = True
            return True
        return False

    def progress(self):
        """0..1 while the gesture is being held, for drawing a progress bar."""
        if self._since is None or self._fired:
            return 0.0
        return min(1.0, (self._clock() - self._since) / self.hold_seconds)
