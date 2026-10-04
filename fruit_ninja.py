"""Control the mouse with your hand to play Fruit Ninja-style games.

Point with only your index finger to hold the left button down (slice).
Hold an open palm for one second to pause/resume. Press q in the terminal
(or q/Esc in the preview window) to quit.
"""

import argparse
import statistics
import sys
import time

import cv2

from camera import LatestFrameCamera
from cursor import CursorThread
from gestures import HoldToggle, SliceController, fingers_up, is_open_palm, is_slice_gesture, tracking_point
from hand_tracker import HandTracker, draw_hand, ensure_model
from mouse_control import EmergencyStop, make_mouse
from preview import PreviewWindow, draw_hud
from smoothing import MotionPredictor, OneEuroFilter
from terminal_keys import TerminalKeys

WINDOW = "Fruit Ninja CV"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--width", type=int, default=640, help="requested camera width")
    p.add_argument("--height", type=int, default=480, help="requested camera height")
    p.add_argument("--margin", type=float, default=0.1,
                   help="fraction of the frame on each side outside the active area (default: 0.1)")
    p.add_argument("--mouse", choices=["auto", "pyautogui", "uinput", "none"], default="auto",
                   help="mouse backend; 'none' only previews tracking (default: auto)")
    p.add_argument("--point", choices=["knuckle", "palm", "tip"], default="knuckle",
                   help="hand point that steers the cursor (default: knuckle of the index finger)")
    p.add_argument("--min-cutoff", type=float, default=1.5,
                   help="smoothing at rest; lower = steadier but laggier (default: 1.5)")
    p.add_argument("--beta", type=float, default=40.0,
                   help="smoothing speed response; higher = less lag on fast swipes (default: 40)")
    p.add_argument("--max-predict", type=float, default=0.04,
                   help="max motion prediction as a fraction of the screen; 0 = off (default: 0.04)")
    p.add_argument("--ease-ms", type=float, default=10.0,
                   help="cursor easing between frames; higher = smoother, laggier (default: 10)")
    p.add_argument("--rate", type=int, default=240, help="cursor updates per second (default: 240)")
    p.add_argument("--no-top", action="store_true", help="don't keep the preview window on top")
    p.add_argument("--allow-low-fps", action="store_true",
                   help="let the camera drop to ~15 FPS in dim light for a brighter image (Linux)")
    p.add_argument("--stats", action="store_true", help="print timing statistics every 5 seconds")
    args = p.parse_args()
    if not 0 <= args.margin < 0.5:
        p.error("--margin must be in [0, 0.5)")
    return args


def to_screen_fraction(x, y, frame_w, frame_h, margin):
    """Map a point inside the active rectangle of the camera frame to 0..1.

    The active rectangle is smaller than the frame because the hand tracker
    loses the hand when it touches the frame edge; shrinking the area means
    the screen edges are reachable while the hand is still fully visible.
    """
    def scale(value, size):
        lo, hi = margin * size, (1 - margin) * size
        return min(1.0, max(0.0, (value - lo) / (hi - lo)))

    return scale(x, frame_w), scale(y, frame_h)


class Stats:
    """Collects per-frame timings so latency can be measured, not guessed."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.detect_ms, self.age_ms, self.intervals, self.skipped = [], [], [], 0
        self.hands = 0
        self._last_capture = None

    def add(self, t_capture, detect_ms, age_ms, skipped, hand):
        self.hands += bool(hand)
        if self._last_capture is not None:
            self.intervals.append(t_capture - self._last_capture)
        self._last_capture = t_capture
        self.detect_ms.append(detect_ms)
        self.age_ms.append(age_ms)
        self.skipped += skipped

    def summary(self):
        if len(self.intervals) < 2:
            return "not enough frames yet"

        def median_p95(xs):
            return statistics.median(xs), sorted(xs)[int(0.95 * (len(xs) - 1))]

        cam_fps = 1.0 / statistics.median(self.intervals)
        d, d95 = median_p95(self.detect_ms)
        a, a95 = median_p95(self.age_ms)
        return (f"camera {cam_fps:4.1f} FPS | detection {d:4.1f} ms (p95 {d95:4.1f}) | "
                f"frame age at cursor update {a:4.1f} ms (p95 {a95:4.1f}) | "
                f"hand found {100 * self.hands / len(self.detect_ms):3.0f}% of frames | "
                f"frames skipped {self.skipped}")


def main():
    args = parse_args()
    try:
        model_path = ensure_model()
        mouse = make_mouse(args.mouse)
        camera = LatestFrameCamera(args.camera, args.width, args.height, fixed_fps=not args.allow_low_fps)
    except RuntimeError as e:
        sys.exit(str(e))

    predictor = MotionPredictor(max_predict=args.max_predict, ease_s=args.ease_ms / 1000)
    cursor = CursorThread(mouse, predictor, rate_hz=args.rate)
    smoother = OneEuroFilter(min_cutoff=args.min_cutoff, beta=args.beta)
    slicer = SliceController()
    pause_toggle = HoldToggle(hold_seconds=1.0)
    stats, total = Stats(), Stats()
    paused = False
    fingers = None
    fps = 0.0
    last_frame = last_report = time.monotonic()
    preview = None

    try:
        with HandTracker(model_path) as tracker, TerminalKeys() as keys:
            preview = PreviewWindow(WINDOW, on_top=not args.no_top)
            print("Running. Point with your index finger to slice.\n"
                  "Hold an open palm for 1 s to pause/resume. Press q here to quit.")
            while True:
                if cursor.error:
                    raise cursor.error
                got = camera.read()
                if got is None:
                    print("Camera stopped delivering frames.", file=sys.stderr)
                    break
                frame, t_capture, skipped = got
                frame = cv2.flip(frame, 1)  # mirror, so moving right moves the cursor right
                h, w = frame.shape[:2]

                t0 = time.monotonic()
                landmarks = tracker.detect(frame, t_capture * 1000)
                detect_ms = (time.monotonic() - t0) * 1000
                # Passing the previous reading enables per-finger hysteresis.
                fingers = fingers_up(landmarks, previous=fingers) if landmarks else None

                if pause_toggle.update(fingers is not None and is_open_palm(fingers)):
                    paused = not paused
                    slicer.reset()
                    cursor.set_enabled(not paused)
                    print("Paused: mouse is yours again." if paused else "Resumed.")

                if fingers is None:
                    smoother.reset()
                    cursor.lost()
                    gesture = None
                else:
                    gesture = is_slice_gesture(fingers)
                    if not paused:
                        x, y = tracking_point(landmarks, args.point)
                        pos = smoother(to_screen_fraction(x, y, w, h, args.margin), t_capture)
                        cursor.feed(pos, t_capture)
                age_ms = (time.monotonic() - t_capture) * 1000
                for s in (stats, total):
                    s.add(t_capture, detect_ms, age_ms, skipped, landmarks)

                if not paused:
                    action = slicer.update(gesture)
                    if action:
                        cursor.set_pressed(action == "press")

                # --- preview, after the cursor update so it adds no latency ---
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - last_frame, 1e-6)
                last_frame = now
                if paused:
                    status, color = "PAUSED (open palm to resume)", (0, 200, 255)
                elif slicer.pressed:
                    status, color = "SLICING", (0, 0, 255)
                elif fingers is None:
                    status, color = "no hand", (180, 180, 180)
                else:
                    status, color = "tracking", (0, 255, 0)
                if landmarks:
                    draw_hand(frame, landmarks)
                    x, y = tracking_point(landmarks, args.point)
                    cv2.circle(frame, (int(x), int(y)), 8, color, -1)
                draw_hud(frame, args.margin, status, color, fps, fingers, pause_toggle.progress())
                preview.show(frame)

                if args.stats and now - last_report >= 5:
                    print(stats.summary())
                    stats.reset()
                    last_report = now

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or "q" in keys.pressed().lower():
                    break
                if not preview.is_open():
                    break
    except EmergencyStop:
        print("Fail-safe triggered (mouse moved to a screen corner). Stopping.", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        # Never leave the left button stuck down, however the loop ended.
        cursor.close()
        camera.release()
        if preview:
            preview.close()
        else:
            cv2.destroyAllWindows()
        print("Session:", total.summary())
        print("Exiting.")


if __name__ == "__main__":
    main()
