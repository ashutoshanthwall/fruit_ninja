"""Camera preview window: resizable, always on top, remembers where you put it.

OpenCV's pip wheels draw windows with Qt over X11 (XWayland on Wayland
desktops). OpenCV's own "topmost" flag is only implemented on Windows and
macOS, so on Linux we set the standard X11 _NET_WM_STATE_ABOVE hint, which
KDE, GNOME and other window managers honour.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2

GEOMETRY_FILE = Path(__file__).resolve().parent / ".preview_window.json"
DEFAULT_SIZE = (480, 360)


class _X11Window:
    """Finds our own X11 window and sets/reads its "keep above" state."""

    def __init__(self, title):
        from Xlib import X, display, protocol
        self._X, self._protocol = X, protocol
        self.title = title
        self.d = display.Display()
        self.root = self.d.screen().root
        atom = self.d.intern_atom
        self.net_state, self.above = atom("_NET_WM_STATE"), atom("_NET_WM_STATE_ABOVE")
        self.pid_atom, self.name_atom = atom("_NET_WM_PID"), atom("_NET_WM_NAME")
        self.client_list = atom("_NET_CLIENT_LIST")
        self.win = None

    def _find(self):
        X = self._X
        clients = self.root.get_full_property(self.client_list, X.AnyPropertyType)
        for wid in (clients.value if clients else []):
            win = self.d.create_resource_object("window", wid)
            pid = win.get_full_property(self.pid_atom, X.AnyPropertyType)
            if not pid or pid.value[0] != os.getpid():
                continue
            name = win.get_full_property(self.name_atom, 0)
            value = name.value if name else ""
            if isinstance(value, bytes):
                value = value.decode(errors="ignore")
            if not value or self.title in value:
                return win
        return None

    def is_above(self):
        self.win = self.win or self._find()
        if self.win is None:
            return False
        state = self.win.get_full_property(self.net_state, self._X.AnyPropertyType)
        return bool(state) and self.above in state.value

    def request_above(self):
        self.win = self.win or self._find()
        if self.win is None:
            return
        X = self._X
        # _NET_WM_STATE_ADD = 1, source indication 1 = normal application
        ev = self._protocol.event.ClientMessage(window=self.win, client_type=self.net_state,
                                                data=(32, [1, self.above, 0, 1, 0]))
        self.root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        self.d.flush()

    def close(self):
        self.d.close()


class PreviewWindow:
    def __init__(self, title, on_top=True):
        self.title = title
        self._shown = 0
        self._started = time.monotonic()
        self._x11 = None
        self._topmost_done = not on_top
        self._offset = (0, 0)
        self._requested_pos = None
        # NORMAL: user-resizable. KEEPRATIO: no stretching. GUI_NORMAL: no Qt toolbar.
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        geo = self._load_geometry()
        w, h = (geo["w"], geo["h"]) if geo else DEFAULT_SIZE
        cv2.resizeWindow(title, w, h)
        if geo:
            self._requested_pos = (geo["x"], geo["y"])
            cv2.moveWindow(title, *self._requested_pos)

    def show(self, frame):
        cv2.imshow(self.title, frame)
        self._shown += 1
        if self._shown == 3:
            self._calibrate_offset()
        if not self._topmost_done and self._shown % 3 == 0:
            self._topmost_done = self._make_topmost()

    def _make_topmost(self):
        """Returns True once done (or given up). Called every few frames.

        A window manager ignores "keep above" requests that arrive before it
        has finished managing a new window, so on X11 we re-read the window
        state and repeat the request until it has actually taken effect.
        """
        if not sys.platform.startswith("linux"):
            try:  # implemented by OpenCV on Windows and macOS
                cv2.setWindowProperty(self.title, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
            return True
        if time.monotonic() - self._started > 10:
            print("Could not keep the preview on top; your window manager may not support it.",
                  file=sys.stderr)
            return True
        try:
            if self._x11 is None:
                self._x11 = _X11Window(self.title)
            if self._x11.is_above():
                self._x11.close()
                self._x11 = None
                return True
            self._x11.request_above()
        except Exception:  # no Xlib, no X server, ...: a cosmetic feature, never crash
            return True
        return False

    def _calibrate_offset(self):
        # moveWindow() positions the outer frame, but getWindowImageRect()
        # reports the image area inside the title bar. Measure the difference
        # once so the saved position doesn't creep down on every run.
        if self._requested_pos is None:
            return
        x, y, _, _ = cv2.getWindowImageRect(self.title)
        self._offset = (x - self._requested_pos[0], y - self._requested_pos[1])

    def is_open(self):
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self):
        if self._x11 is not None:
            self._x11.close()
        if self._shown and self.is_open():
            x, y, w, h = cv2.getWindowImageRect(self.title)
            if w > 50 and h > 50:
                self._save_geometry(x - self._offset[0], y - self._offset[1], w, h)
        cv2.destroyAllWindows()

    @staticmethod
    def _load_geometry():
        try:
            geo = json.loads(GEOMETRY_FILE.read_text())
            return {k: int(geo[k]) for k in ("x", "y", "w", "h")}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _save_geometry(x, y, w, h):
        try:
            GEOMETRY_FILE.write_text(json.dumps({"x": x, "y": y, "w": w, "h": h}))
        except OSError:
            pass


def draw_hud(img, margin, status, color, fps, fingers, palm_progress):
    h, w = img.shape[:2]
    cv2.rectangle(img, (int(margin * w), int(margin * h)),
                  (int((1 - margin) * w), int((1 - margin) * h)), (255, 0, 255), 2)
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, f"{fps:4.1f} FPS", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if fingers:
        order = ("thumb", "index", "middle", "ring", "pinky")
        bits = " ".join("1" if fingers[f] else "0" for f in order)
        cv2.putText(img, f"fingers: {bits}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if palm_progress > 0:
        cv2.rectangle(img, (10, 45), (10 + int(200 * palm_progress), 55), (0, 200, 255), -1)
