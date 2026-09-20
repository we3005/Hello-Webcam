# Hello Webcam!

A real-time hand gesture recognition tool that uses Media pipe for tracking, with custom gestures.

**Recognizes:**
- Thumbs Up 👍
- Peace Sign ✌️
- Korean Finger Heart 🫰 
- OK Sign 👌 
- Prayer Hands 🙏 (often has a difficult time recognizing, you may need to form more of a cup than a prayer)
- Index Raised ☝️ 

**On-screen display**

Three things show up when a gesture is recognized, each in its own spot so
they don't compete for the same space:
- A label in the top-left
- An emoji in the top-right corner
- A hand-drawn illustration of the gesture (by Albert), stacked down the left side

**Approach Reasoning**

Instead of training a classifier, each gesture is defined by rules over the
21 hand landmarks MediaPipe returns per hand:

1. Every finger is classified into one of three "curl states" based on the
   angles at the knuckles:
   - `extended` — straight and pointing outwards
   - `hooked` — bent at the first knuckle 
   - `curled` — folded into a fist
2. Each gesture is a pattern over those curl states, plus a
   distance check.

The Index Raised gesture also runs a lightweight MediaPipe Face Detection
pass (only while that gesture is active, to save CPU) and uses its eye and
ear keypoints to size, rotate, and place a pair of glasses over your eyes 
(so that you can look like the 🤓 emoji)

**Assets**

```
assets/
  glasses.png            — glasses image (needs a transparent background)
  drawings/
    thumbup.png           — Thumbs Up
    peace.png             — Peace Sign
    heart.png             — Korean Finger Heart
    ok.png                — OK Sign
    pray.png              — Prayer Hands
    nerd.png              — Index Raised
```


## Credits

Cat drawings by Albert Rao (my friend).


## Emoji icons

Each recognized gesture also gets an emoji  (☝️, 👍, etc.) in the
top-right corner. Since OpenCV's built-in text drawing can't render emoji,
these are pre-rendered once at startup with Pillow using whichever
color-emoji font your OS provides (so this can change) and then cached
as images. If no color-emoji font can be found on your system, the badges
are skipped, but the app will still run


## Setup

> **Note:** Python 3.10 or 3.11 is required due to MediaPipe constraints. If your default system Python is 3.13, jump to the **Python 3.13 & Environment Setup Troubleshooting** section below before installing!

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Notes:**
- Requires a working webcam and OS permission for camera access.
- This runs as a local desktop script (uses `cv2.imshow`), so it needs a real
  Python environment with a display, not a browser sandbox.


## Common Errors

**Pinned MediaPipe version**

`requirements.txt` pins `mediapipe==0.10.21` on purpose. Starting around
mediapipe 0.10.30 (and in the 1.0.x releases), Google removed the legacy
`mediapipe.solutions` module this project relies on (`mp.solutions.hands`,
`mp.solutions.face_detection`, `mp.solutions.drawing_utils`) in favor of a
newer `mediapipe.tasks` API. Installing an unpinned or newer `mediapipe`
will fail with `AttributeError: module 'mediapipe' has no attribute
'solutions'`. If you need a newer mediapipe for another reason, the fix is
to rewrite the detection setup in `main.py` against the `mediapipe.tasks`
Task API instead — the gesture logic in `gesture_recognizer.py` would not
need to change, since it only works with landmark coordinates.

**Python 3.13**

MediaPipe does not yet publish official PyPI wheels for Python 3.13 (frowny face), so
`pip install mediapipe` will fail if your virtual environment is on 3.13.
`main.py` detects this and exits with a clear message rather than a raw
import error.

The fix is to run in a 3.11 or 3.12 environment, even if 3.13 is
your system default elsewhere — it won't affect anything else on your
machine. Easiest ways to get one:


## How to fix the errors mentioned:

**Using [uv](https://docs.astral.sh/uv/) (recommended, no separate Python install needed):**
```bash
uv venv --python 3.11 venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
uv pip install -r requirements.txt
python main.py
```
Install uv (if you get zsh: command not found: uv): curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
Close and reopen your terminal window after running the installer, and follow the instructions above to complete the setup.

Note: a `uv venv` environment does not include `pip` itself (uv manages
packages directly). Use `uv pip install ...` as shown above rather than
plain `pip install ...` inside it — or `python -m pip ...`, which will
report "No module named pip" the same way.

**Using [pyenv](https://github.com/pyenv/pyenv) (recommended for macOS):**
```bash
pyenv install 3.11.9
```

Enable shell integration and set Python 3.11:
If running pyenv shell 3.11.9 gives pyenv: shell integration not enabled, run:
```
eval "$(pyenv init -)"
pyenv shell 3.11.9
```

Avoid active virtual environment conflicts:
If your terminal prompt shows an active environment like (.venv), running python -m venv
will still use Python 3.13. You must deactivate all active environments first:
deactivate

Verify version and create virtual environment:
Ensure python --version outputs 3.11.9 before creating the environment:

```
python --version    # Must show Python 3.11.9
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Using Homebrew (macOS)**

Install Python 3.10 via Homebrew:
```
brew install python@3.10
```

Create virtual environment using the Homebrew binary path:
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Using conda:**
```bash
conda create -n gesture-app python=3.11
conda activate gesture-app
pip install -r requirements.txt
python main.py
```


## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug overlay (shows each finger's curl state) |
| `m` | Toggle mirror mode |


## Tuning

Gestures can be difficult to register from certain angles, lighting, or hand
sizes than others. If a gesture isn't triggering (or triggers too easily),
turn on debug mode (`d`) to see the live curl state of each finger, then
adjust thresholds in `gesture_recognizer.py`:

- `analyze_hand()` — thresholds for `extended` / `hooked` / `curled`
  (angle cutoffs `155`, `100`, and reach ratios `1.05`, `0.75`).
- `is_peace_sign()` — `spread > 0.35` controls how wide the "V" must be.
- `is_finger_heart()` — `pinch < 0.35` controls how close thumb/index must be.
- `is_ok_sign()` — same `pinch < 0.35` threshold as the finger heart; the
  two gestures are told apart by the other three fingers' curl state.
- `is_index_point()` — no distance threshold, just curl states; adjust the
  underlying `extended`/`curled` cutoffs in `analyze_hand()` if it isn't
  triggering reliably.
- `is_prayer_hands()` — `tip_gap < 0.5` controls how close the four
  fingertip pairs must be, and `wrist_gap < 1.0` controls how close the two
  wrists must be.

Overlay-specific tuning lives in `overlay.py`:
- `draw_glasses_image()` — `target_width = eye_dist / 0.7` controls how
  wide the glasses render relative to the detected eye distance; lower the
  `0.7` divisor to make them larger, raise it to make them smaller.
- `PANEL_WIDTH` — width in pixels of the left-side drawing panel cards.
- `ICON_SIZE` — size in pixels of the top-right emoji badges.
- `PANEL_BG` / `PANEL_ALPHA` / `ACCENT_SINGLE` / `ACCENT_TWO_HAND` — the
  color palette for the rounded label cards and panel backgrounds.

All thresholds are normalized by hand size, so they should hold up
welL across distances from the camera, but camera angle, occlusion, 
and individual hand proportions will still affect accuracy some.


For new two-hand gestures, follow the pattern of `is_prayer_hands()`, which
receives the list of both detected `HandInfo` objects. For new single-hand
gestures, follow the pattern of the other `is_*` functions in
`gesture_recognizer.py`. To give a new gesture its on-screen display, add
an entry to `GESTURE_EMOJI` (for the badge) and to `DRAWING_FILES` in
`overlay.py` (pointing at a new file under `assets/drawings/`) — the label
card needs no changes either way.
