# 🍉 Fruit Ninja with Computer Vision

Play Fruit Ninja-style games (any game where you slice by dragging the mouse) with your index finger. A webcam tracks your hand with MediaPipe and the program moves and clicks the real mouse for you.

This project does **not** include a game. Open one in another window (for example a browser Fruit Ninja clone), then start this program.

---

## ▶️ Setup and run

Requires a webcam and a Python version that has MediaPipe ≥ 0.10.30 wheels (tested on 3.11 and 3.14).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python fruit_ninja.py
```

On the first run the hand model (`hand_landmarker.task`, about 7.5 MB) is downloaded into `models/` and its SHA-256 checksum is checked.

### Linux on Wayland

pyautogui can only move the cursor on X11. On Wayland (the default on current Fedora, Ubuntu and others) the program creates a virtual pointer through `/dev/uinput` instead. Give your user access once:

```bash
sudo ./setup_wayland_mouse.sh            # undo with: sudo ./setup_wayland_mouse.sh --remove
```

`uaccess` gives access only to the user at the active session, not to a group. Even so, any program you run can then inject keyboard and mouse input. Remove the rule file if you stop using the project.

---

## 🎮 Controls

| Gesture / key                           | Action                              |
| --------------------------------------- | ----------------------------------- |
| Move hand                               | Move cursor (index-finger knuckle)  |
| Only index finger up                    | Hold left button (slice)            |
| Any other hand shape                    | Release button                      |
| Open palm held for 1 second             | Pause / resume mouse control        |
| `q` in the terminal (or preview window) | Quit                                |
| Real mouse into a screen corner         | Emergency stop (pyautogui backend)  |

The thumb is ignored for the slice gesture, so pointing with the thumb up or down both work. Thumb detection was the least reliable part and made slicing flicker.

The camera preview stays on top of other windows. Drag and resize it wherever it fits around the game (handy for screen recording). It reopens at the same place and size next time.

Typical session: start `python fruit_ninja.py` in a terminal, switch to the game in your browser and play. Hold an open palm to pause and get your mouse back. Switch to the terminal and press `q` to quit.

### Options

```text
--camera N          camera index (default 0)
--mouse BACKEND     auto | pyautogui | uinput | none   ('none' = preview only)
--point P           knuckle | palm | tip: what steers the cursor (default knuckle)
--margin F          border of the frame that is outside the active area (default 0.1)
--min-cutoff F      smoothing when still; lower = steadier, laggier (default 1.5)
--beta F            smoothing response to speed; higher = less lag on fast swipes (default 40)
--max-predict F     motion prediction cap, fraction of the screen; 0 = off (default 0.04)
--ease-ms F         easing between camera frames; higher = smoother, laggier (default 10)
--rate N            cursor updates per second (default 240)
--no-top            don't keep the preview window on top
--allow-low-fps     let the camera halve its frame rate in dim light for a brighter image (Linux)
--stats             print camera FPS, detection time and frame age every 5 s
```

Start with `--mouse none` to see how well your hand is tracked without handing over the mouse.

---

## 🧠 How it works

```
camera thread (newest frame only, driver timestamps)
  → mirror → MediaPipe HandLandmarker (21 points)
      ├─ knuckle → active-area mapping → One Euro filter ─┐
      └─ finger states (hysteresis) → slice gesture ──────┤
                                                          ▼
                     cursor thread, 240 Hz: bounded prediction + easing → mouse
```

- **Camera thread** (`camera.py`): reads continuously and keeps only the newest frame, so detection never works on a stale queued image. Frames carry the driver's capture timestamp, so velocities use the real time between frames.
- **Hand tracking** (`hand_tracker.py`): the MediaPipe Tasks `HandLandmarker` in video mode, on the CPU. The GPU delegate measured slower here (23.7 ms vs 15.8 ms per frame).
- **Tracking point** (`gestures.py`): the cursor follows the index finger's knuckle, not its tip. The tip drops towards the palm when you curl the finger to stop slicing, which used to drag a stray slash downwards. The knuckle doesn't move then, and it jitters less.
- **Finger states**: a finger counts as up when its tip is farther from the wrist than its middle joint, which works for tilted and sideways hands. Hysteresis stops a half-bent finger from flickering, so a slice can start on the first frame instead of waiting for confirmation.
- **Active area**: the purple rectangle in the preview. It maps to the whole screen, so you reach the screen edges while the hand is still fully in view.
- **Smoothing** (`smoothing.py`): the [One Euro filter](https://gery.casiez.net/1euro/) (Casiez, Roussel and Vogel, CHI 2012) smooths heavily when the hand is still and lightly when it moves fast.
- **Cursor thread** (`cursor.py`): the camera gives 30 positions per second, so the cursor used to jump every 33 ms. It now moves 240 times per second, gliding along the current velocity between frames (capped, and only for fast motion) and easing into each new measurement. It is the only thread that touches the mouse.
- **Safety**: the button is always released when the program exits, crashes or loses the camera.

### What to expect from latency

Measured on the development laptop (i7-13620H, built-in Chicony USB2.0 webcam, 30 FPS maximum, dim room) with `--stats`:

| | Before | Now |
|---|---|---|
| Camera frame rate in dim light | 14.7 FPS | 31.2 FPS |
| Cursor updates per second | 30 | ~230 |
| Steppiness of the cursor path (simulated) | 47 | 9 (≈5× smoother) |
| Cursor jitter at rest (simulated, 4.8 px noise) | 1.50 px | 1.74 px |
| Overshoot when a very fast slash stops (simulated) | 6 px | ~40 px |

**The camera frame rate was the biggest problem.** Many webcams turn on "Exposure, Dynamic Framerate" by default, which lets them halve the frame rate to expose longer in dim light. At 15 FPS every hand position arrives 67 ms apart instead of 33. On Linux the program switches this off while it runs (and restores your setting on exit), so the camera holds 30 FPS and the image gets a little darker instead. In a dim room the hand was still found in 99–100% of frames once in view. If tracking struggles in very dark rooms, add light, or use `--allow-low-fps`.

**What's left is hardware.** A frame is ~29 ms old when the webcam delivers it (exposure plus USB transfer; this camera's compressed MJPEG mode was worse, 16 FPS), and hand detection takes ~23 ms. Prediction hides a little of that. Simulated rows come from `python tools/simulate_latency.py`, which documents its assumptions; removing the prediction cap there cut lag further but overshot fast slashes by ~90 px. A 60 FPS webcam and good lighting are the remaining big improvements.

## 🧪 Tests

The gesture, debounce, smoothing, prediction and mapping logic has unit tests that need no camera:

```bash
python -m unittest discover tests
```

---

## ⚠️ Limitations

- Lighting, background and camera quality strongly affect tracking. Expect poor results in dim light.
- With 30 FPS cameras the fastest swipes are sampled sparsely. Prediction smooths the path between samples but can't recover curves that happened between frames.
- Default smoothing and prediction values were tuned in simulation, not on recorded hand motion. Tune them for your camera and lighting.
- "Always on top" uses an X11 window hint. Tested on KDE Plasma (Wayland session, via XWayland). Most window managers honour it, but a *fullscreen* (F11) browser is usually stacked above it. Maximise the browser window instead of making it fullscreen.
- Multi-monitor setups: pyautogui maps to the primary screen. uinput maps to the whole desktop area.

---

## 🛠️ Technologies

Python · OpenCV · MediaPipe Tasks · NumPy · pyautogui · Linux uinput

The original version used `cvzone`, which relies on the old `mediapipe.solutions` API. MediaPipe removed that API in 0.10.30, and older MediaPipe versions have no wheels for Python 3.12+. The original pins therefore don't install on a current Python.

---

## Credits

- Original project: [Fruit Ninja Game with Computer Vision](https://github.com/ayushcpp/Fruit-Ninja-Game-with-Computer-Vision-EASY-ONE_-) by Ayush Gairola (MIT License).
- This version: rewritten on the MediaPipe Tasks API, with motion smoothing and prediction, a low-latency camera pipeline, Wayland support, an always-on-top preview and tests, by [ashutoshanthwall](https://github.com/ashutoshanthwall).
