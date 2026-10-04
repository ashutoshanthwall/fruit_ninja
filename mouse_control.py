"""Mouse output backends.

All backends take cursor positions as fractions of the screen (0..1), so the
game loop does not need to know the screen resolution.

- pyautogui: Windows, macOS and Linux on X11.
- uinput:    Linux on Wayland, where pyautogui (X11) cannot move the real
             cursor. Creates a virtual absolute pointer through /dev/uinput,
             the same mechanism virtual machines use for their mouse.
- none:      track and preview only, never touch the mouse.
"""

import os
import struct
import sys
import time

try:
    import fcntl  # Unix only; needed by the uinput backend
except ImportError:
    fcntl = None


class EmergencyStop(Exception):
    """The user triggered the pyautogui fail-safe (real mouse in a screen corner)."""


class _Mouse:
    pressed = False

    def move(self, u, v):
        raise NotImplementedError

    def _button(self, down):
        raise NotImplementedError

    def press(self):
        if not self.pressed:
            self._button(True)
            self.pressed = True

    def release(self):
        if self.pressed:
            self._button(False)
            self.pressed = False

    def close(self):
        self.release()


class NullMouse(_Mouse):
    def move(self, u, v):
        pass

    def _button(self, down):
        pass


class PyAutoGuiMouse(_Mouse):
    def __init__(self):
        import pyautogui

        # By default pyautogui sleeps 0.1 s after *every* call, which adds
        # visible lag to each press and release.
        pyautogui.PAUSE = 0
        # Kept on as an emergency stop: slam the real mouse into a screen
        # corner to abort. move() never targets a corner itself, so hand
        # tracking alone can't trigger it.
        pyautogui.FAILSAFE = True
        self._pg = pyautogui
        self.width, self.height = pyautogui.size()

    def move(self, u, v):
        x = 1 + round(u * (self.width - 3))
        y = 1 + round(v * (self.height - 3))
        try:
            self._pg.moveTo(x, y)
        except self._pg.FailSafeException as e:
            raise EmergencyStop from e

    def _button(self, down):
        try:
            (self._pg.mouseDown if down else self._pg.mouseUp)()
        except self._pg.FailSafeException as e:
            raise EmergencyStop from e

    def close(self):
        # The final release must not be blocked by the fail-safe itself.
        self._pg.FAILSAFE = False
        super().close()


# --- Linux uinput (pure Python, no compiled dependencies) -------------------
# Constants from <linux/input-event-codes.h> and <linux/uinput.h>.
EV_SYN, EV_KEY, EV_ABS = 0x00, 0x01, 0x03
SYN_REPORT = 0
BTN_LEFT, BTN_RIGHT = 0x110, 0x111
ABS_X, ABS_Y = 0x00, 0x01
BUS_VIRTUAL = 0x06

_ABS_SETUP = struct.Struct("=H2x6i")      # struct uinput_abs_setup (28 bytes)
_DEV_SETUP = struct.Struct("=4H80sI")     # struct uinput_setup (92 bytes)
_EVENT = struct.Struct("@llHHi")          # struct input_event


def _IOW(nr, size):
    return (1 << 30) | (size << 16) | (ord("U") << 8) | nr


UI_DEV_CREATE = (ord("U") << 8) | 1
UI_DEV_DESTROY = (ord("U") << 8) | 2
UI_DEV_SETUP = _IOW(3, _DEV_SETUP.size)
UI_ABS_SETUP = _IOW(4, _ABS_SETUP.size)
UI_SET_EVBIT = _IOW(100, 4)
UI_SET_KEYBIT = _IOW(101, 4)
UI_SET_ABSBIT = _IOW(103, 4)

UINPUT_HELP = """\
No write access to /dev/uinput, which is needed to control the mouse on Wayland.
Grant it to the logged-in user (not to a whole group) with a udev rule:

  echo 'KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"' \\
    | sudo tee /etc/udev/rules.d/70-uinput.rules
  sudo udevadm control --reload && sudo udevadm trigger --name-match=uinput

Or run with --mouse none to only preview the tracking."""


class UInputMouse(_Mouse):
    ABS_MAX = 65535

    def __init__(self, name=b"Fruit Ninja CV pointer"):
        try:
            self._fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
        except (PermissionError, FileNotFoundError) as e:
            raise RuntimeError(UINPUT_HELP) from e
        try:
            fcntl.ioctl(self._fd, UI_SET_EVBIT, EV_KEY)
            for btn in (BTN_LEFT, BTN_RIGHT):
                fcntl.ioctl(self._fd, UI_SET_KEYBIT, btn)
            fcntl.ioctl(self._fd, UI_SET_EVBIT, EV_ABS)
            for axis in (ABS_X, ABS_Y):
                fcntl.ioctl(self._fd, UI_SET_ABSBIT, axis)
                # value, min, max, fuzz, flat, resolution
                fcntl.ioctl(self._fd, UI_ABS_SETUP, _ABS_SETUP.pack(axis, 0, 0, self.ABS_MAX, 0, 0, 0))
            fcntl.ioctl(self._fd, UI_DEV_SETUP, _DEV_SETUP.pack(BUS_VIRTUAL, 0x1209, 0x0001, 1, name, 0))
            fcntl.ioctl(self._fd, UI_DEV_CREATE)
        except OSError:
            os.close(self._fd)
            raise
        # Give udev and the compositor a moment to pick up the new device.
        time.sleep(0.5)

    def _emit(self, *events):
        data = b"".join(_EVENT.pack(0, 0, t, c, v) for t, c, v in events)
        os.write(self._fd, data + _EVENT.pack(0, 0, EV_SYN, SYN_REPORT, 0))

    def move(self, u, v):
        self._emit((EV_ABS, ABS_X, round(u * self.ABS_MAX)), (EV_ABS, ABS_Y, round(v * self.ABS_MAX)))

    def _button(self, down):
        self._emit((EV_KEY, BTN_LEFT, int(down)))

    def close(self):
        if self._fd is None:
            return
        super().close()
        fcntl.ioctl(self._fd, UI_DEV_DESTROY)
        os.close(self._fd)
        self._fd = None


def make_mouse(backend="auto"):
    if backend == "auto":
        on_wayland = sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE") == "wayland"
        backend = "uinput" if on_wayland else "pyautogui"
    if backend == "uinput":
        return UInputMouse()
    if backend == "pyautogui":
        return PyAutoGuiMouse()
    if backend == "none":
        return NullMouse()
    raise ValueError(f"unknown mouse backend: {backend}")
