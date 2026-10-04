"""Read single key presses from the terminal without blocking the game loop.

Lets you quit by switching to the terminal and pressing q, instead of having
to focus the OpenCV preview window.
"""

import os
import sys

try:
    import select
    import termios
    import tty
except ImportError:  # Windows
    termios = None
    import msvcrt


class TerminalKeys:
    def __enter__(self):
        self._old = None
        self._fd = None
        if termios and sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            # cbreak: keys arrive immediately without Enter; Ctrl+C still works.
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def pressed(self):
        """Return the characters typed since the last call ("" if none)."""
        if termios is None:
            chars = ""
            while msvcrt.kbhit():
                chars += msvcrt.getwch()
            return chars
        if self._fd is None or not select.select([self._fd], [], [], 0)[0]:
            return ""
        # os.read, not sys.stdin.read: the text wrapper would buffer input
        # that select() then no longer sees.
        return os.read(self._fd, 64).decode(errors="ignore")
