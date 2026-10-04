"""Tests for the camera-free logic. Run with:  python -m unittest discover tests"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fruit_ninja import to_screen_fraction  # noqa: E402
from gestures import (HoldToggle, SliceController, fingers_up, is_open_palm,  # noqa: E402
                      is_slice_gesture, tracking_point)
from smoothing import MotionPredictor, OneEuroFilter  # noqa: E402

# Finger base (MCP) positions for an upright right hand, wrist at (0, 0),
# image y growing downwards. Each finger is MCP, PIP, DIP, TIP.
_BASES = {"thumb": (-30, -30), "index": (-20, -80), "middle": (0, -85), "ring": (20, -80), "pinky": (38, -70)}
_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def make_hand(extended, angle_deg=0.0):
    """Synthetic 21-landmark hand. Extended fingers continue straight away
    from the wrist; curled ones fold back towards the palm. The whole hand is
    rotated by `angle_deg` to check rotation invariance."""
    pts = [(0.0, 0.0)]
    for name in _ORDER:
        bx, by = _BASES[name]
        if name == "thumb":
            if name in extended:
                chain = [(bx, by), (bx - 15, by - 15), (bx - 28, by - 28), (bx - 40, by - 40)]
            else:  # tucked across the palm towards the pinky
                chain = [(bx, by), (bx + 10, by - 15), (bx + 30, by - 25), (bx + 50, by - 30)]
        else:
            norm = math.hypot(bx, by)
            dx, dy = bx / norm, by / norm
            if name in extended:
                chain = [(bx + dx * 25 * k, by + dy * 25 * k) for k in range(4)]
            else:
                # PIP sticks out, then the tip folds back below the knuckle.
                chain = [(bx, by), (bx + dx * 25, by + dy * 25), (bx + dx * 15, by + dy * 15),
                         (bx - dx * 10, by - dy * 10)]
        pts.extend(chain)
    a = math.radians(angle_deg)
    return [(x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a), 0.0) for x, y in pts]


class FingersUpTest(unittest.TestCase):
    def test_states_at_many_rotations(self):
        cases = [set(), {"index"}, {"thumb", "index"}, {"index", "middle"}, set(_ORDER)]
        for extended in cases:
            for angle in (0, 45, 90, 180, -60):
                with self.subTest(extended=extended, angle=angle):
                    got = fingers_up(make_hand(extended, angle))
                    self.assertEqual({f for f, up in got.items() if up}, extended)

    def test_slice_gesture_ignores_thumb(self):
        self.assertTrue(is_slice_gesture(fingers_up(make_hand({"index"}))))
        self.assertTrue(is_slice_gesture(fingers_up(make_hand({"thumb", "index"}))))
        self.assertFalse(is_slice_gesture(fingers_up(make_hand({"index", "middle"}))))
        self.assertFalse(is_slice_gesture(fingers_up(make_hand(set()))))

    def test_hysteresis_holds_state_near_threshold(self):
        # Index finger exactly at the tipping point (tip as far from the wrist as the PIP).
        hand = make_hand({"index"})
        wrist, pip = hand[0], hand[6]
        scale = math.dist(wrist, pip) / math.dist(wrist, hand[8]) * 1.03  # ratio 1.03
        hand[8] = tuple(c * scale for c in hand[8])
        up, down = dict.fromkeys(_ORDER, True), dict.fromkeys(_ORDER, False)
        self.assertTrue(fingers_up(hand, previous=up)["index"])     # stays up
        self.assertFalse(fingers_up(hand, previous=down)["index"])  # stays down
        self.assertTrue(fingers_up(hand)["index"])                  # no hysteresis: > 1

    def test_tracking_points(self):
        hand = make_hand({"index"})
        self.assertEqual(tracking_point(hand, "tip"), hand[8][:2])
        self.assertEqual(tracking_point(hand, "knuckle"), hand[5][:2])
        palm = tracking_point(hand, "palm")
        self.assertAlmostEqual(palm[0], sum(hand[i][0] for i in (0, 5, 9, 13, 17)) / 5)

    def test_knuckle_does_not_move_when_finger_curls(self):
        self.assertEqual(tracking_point(make_hand({"index"}), "knuckle"),
                         tracking_point(make_hand(set()), "knuckle"))

    def test_open_palm(self):
        self.assertTrue(is_open_palm(fingers_up(make_hand(set(_ORDER)))))
        self.assertFalse(is_open_palm(fingers_up(make_hand({"index"}))))


class SliceControllerTest(unittest.TestCase):
    def run_frames(self, ctrl, frames):
        return [ctrl.update(g) for g in frames]

    def test_press_needs_consistent_frames(self):
        c = SliceController(press_frames=2)
        self.assertEqual(self.run_frames(c, [True, False, True, True]), [None, None, None, "press"])

    def test_single_glitch_does_not_release(self):
        c = SliceController(press_frames=1, release_frames=3)
        actions = self.run_frames(c, [True, False, True, False, False, False])
        self.assertEqual(actions, ["press", None, None, None, None, "release"])

    def test_brief_hand_loss_keeps_slice(self):
        c = SliceController(press_frames=1, lost_frames=4)
        actions = self.run_frames(c, [True, None, None, None, True])
        self.assertEqual(actions, ["press", None, None, None, None])
        self.assertTrue(c.pressed)

    def test_long_hand_loss_releases(self):
        c = SliceController(press_frames=1, lost_frames=3)
        self.assertEqual(self.run_frames(c, [True, None, None, None]), ["press", None, None, "release"])


class HoldToggleTest(unittest.TestCase):
    def test_fires_once_per_hold(self):
        t = [0.0]
        h = HoldToggle(hold_seconds=1.0, clock=lambda: t[0])
        fired = []
        for now, active in [(0, True), (0.5, True), (1.0, True), (1.5, True), (2.0, False), (2.1, True), (3.2, True)]:
            t[0] = now
            fired.append(h.update(active))
        self.assertEqual(fired, [False, False, True, False, False, False, True])


class OneEuroTest(unittest.TestCase):
    def test_reduces_jitter_when_still(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=20.0)
        noisy = [0.5 + (0.003 if i % 2 else -0.003) for i in range(60)]
        out = [float(f(x, i / 30)) for i, x in enumerate(noisy)]
        self.assertLess(max(out[30:]) - min(out[30:]), 0.25 * (max(noisy) - min(noisy)))

    def test_little_lag_when_fast(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=20.0)
        # Swipe across the whole screen in 0.3 s at 30 FPS.
        for i in range(10):
            out = float(f(i / 9, i / 30))
        self.assertGreater(out, 0.85)


class MotionPredictorTest(unittest.TestCase):
    def feed_constant_velocity(self, p, speed, frames=10, fps=30):
        for i in range(frames):
            t = i / fps
            p.update([0.2 + speed * t, 0.5], t, t + 0.02)
        return t

    def test_glides_between_frames_when_moving(self):
        p = MotionPredictor(ease_s=0)
        t = self.feed_constant_velocity(p, speed=2.0)
        now = t + 0.02
        a, b = p.step(now + 0.005)[0], p.step(now + 0.015)[0]
        self.assertGreater(b, a)                    # keeps moving between frames
        self.assertGreater(a, 0.2 + 2.0 * t)        # ahead of the last measurement

    def test_still_hand_is_not_extrapolated(self):
        p = MotionPredictor(ease_s=0)
        for i in range(10):
            p.update([0.5 + (0.001 if i % 2 else -0.001), 0.5], i / 30, i / 30 + 0.02)
        out = p.step(9 / 30 + 0.05)
        self.assertAlmostEqual(out[0], 0.501, delta=0.0005)  # jitter not amplified

    def test_prediction_is_capped_and_stops(self):
        p = MotionPredictor(max_predict=0.03, ease_s=0)
        t = self.feed_constant_velocity(p, speed=5.0)
        last = 0.2 + 5.0 * t
        far_future = p.step(t + 1.0)[0]               # frames stopped arriving
        self.assertLessEqual(far_future - last, 0.03 + 1e-9)

    def test_disabled_prediction_follows_measurements(self):
        p = MotionPredictor(max_predict=0, ease_s=0)
        t = self.feed_constant_velocity(p, speed=2.0)
        self.assertAlmostEqual(p.step(t + 0.03)[0], 0.2 + 2.0 * t)


class ScreenMappingTest(unittest.TestCase):
    def test_active_area_maps_to_full_screen_and_clamps(self):
        self.assertEqual(to_screen_fraction(64, 48, 640, 480, 0.1), (0.0, 0.0))
        self.assertEqual(to_screen_fraction(576, 432, 640, 480, 0.1), (1.0, 1.0))
        self.assertEqual(to_screen_fraction(320, 240, 640, 480, 0.1), (0.5, 0.5))
        self.assertEqual(to_screen_fraction(-50, 900, 640, 480, 0.1), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
