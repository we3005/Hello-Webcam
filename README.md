# Hello Webcam!

A real-time hand gesture recognition tool that uses Media pipe for tracking, with custom gestures.

**Recognizes:**
- Thumbs Up 👍
- Peace Sign ✌️
- Korean Finger Heart 🫰 
- OK Sign 👌 
- Prayer Hands 🙏 (often has a difficult time recognizing, you may need to form more of a cup than a prayer)
- Index Raised ☝️ 
- Waving 👋 (motion-based — swing an open hand side to side)
- Hand Raised 🙋 (open palm held up at face level; shows a ♂️/♀️ badge from a face scan)

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

MediaPipe Face Detection runs on every frame (it's a small, fast model). The
Index Raised gesture uses its eye and ear keypoints to size, rotate, and place
a pair of glasses over your eyes (so that you can look like the 🤓 emoji); the
same detection also provides the face box for the gender guess and the height
that counts as "raised" (see below).

**Waving and Hand Raised**

Waving can't be recognized from a single frame, so `motion.py` adds a time
dimension. `WaveDetector` keeps the last ~1.2 s of each open hand's
fingertip position and calls it a wave once the hand has swung back and forth
at least 3 times, each swing at least 0.4 palm-lengths wide (measured in palm
lengths so it works at any distance, and so landmark jitter doesn't count).
Fingertips are tracked rather than the palm because a wave pivots at the wrist
or elbow, so the palm barely moves while the fingertips sweep a wide arc. A
short glitch in the open-hand check (motion blur) or a dropped detection frame
doesn't reset a wave in progress. A wave beats every other single-hand
gesture. With debug mode on (`d`), each hand shows `wave swings: n/3`, which is
handy for tuning.

Hand Raised is a pose rule in `gesture_recognizer.py`: an open palm (no finger
folded in), fingers pointing roughly up, with the middle fingertip above the
middle of your detected face (or in the top 40% of the frame if no face is
visible). It's checked after the finger-shape gestures, so it never steals a
more specific pose like the peace sign.

**Gender detection**

The raised-hand emoji reflects a guess made from your face: 🙋 with a ♂️ or ♀️
badge (the real 🙋‍♂️ / 🙋‍♀️ images in `assets/emoji/`). `gender.py` crops the face MediaPipe found and runs the Levi & Hassner
gender CNN through OpenCV's built-in `cv2.dnn` (no new pip packages). Guesses
are averaged over the last ~20 samples and only switch when the average
clearly crosses the middle, so it doesn't flicker.

This is a guess about appearance from a photo-like crop, not a fact about
anyone. It only picks between two labels, glasses or a raised hand covering
part of the face can throw it off, and it can be plain wrong. That's why
pressing `g` lets you set male/female by hand. Nothing is saved or sent
anywhere; frames are only processed in memory.

*Model files (required for auto mode):* put these two files in `assets/models/`
```
assets/models/gender_deploy.prototxt
assets/models/gender_net.caffemodel     (large file, listed in .gitignore)
```
They come from Levi & Hassner's project page
(http://www.openu.ac.il/home/hassner/projects/cnn_agegender/)
and are also bundled in several public "Gender-and-Age-Detection" repos on
GitHub (for example `smahesh29/Gender-and-Age-Detection`). Check the
license/terms of whichever copy you use. If the files are missing the app
still runs: the raised-hand emoji is just the neutral 🙋 until you press `g`.

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
  emoji/
    hand_raising_male.png   — 🙋‍♂️ shown for a male guess
    hand_raising_female.png — 🙋‍♀️ shown for a female guess
  models/
    gender_deploy.prototxt  — gender network definition (you add these two)
    gender_net.caffemodel   — gender network weights
```
Waving and Hand Raised have no cat drawings yet; to add them, draw
`wave.png` / `raise.png`, put them in `assets/drawings/`, and add
`"Waving": "wave.png"` and `"Hand Raised": "raise.png"` to `DRAWING_FILES` in
`overlay.py`.


## Credits

Cat drawings by Albert Rao (my friend).

Gender model: Gil Levi and Tal Hassner, *Age and Gender Classification Using
Convolutional Neural Networks*, IEEE Workshop on Analysis and Modeling of Faces
and Gestures (AMFG) at CVPR 2015.


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


## How To Fix This Error:
Follow the instructions by typing the commands into a terminal window—either inside your code editor (VS Code, PyCharm, etc.) unless specified otherwise, or in your system's standalone terminal (PowerShell on Windows, Terminal/Bash/Zsh on macOS/Linux) when noted.

**Using [uv](https://docs.astral.sh/uv/) (recommended, no separate Python install needed):**

For macOS/Linux:
```
uv venv --python 3.11 venv
source venv/bin/activate
uv pip install -r requirements.txt
python main.py
```

For Windows:
```
uv venv --python 3.11 venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

**Install uv (if you get zsh: command not found: uv)**

Open Powershell on Windows or Terminal on macOS/Linux:
```
winget install --id=astral-sh.uv -e # On Windows
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh # on macOS/Linux
```
Close and reopen your terminal window after running the installer, then go back into the project to complete the setup.

Note: a `uv venv` environment does not include `pip` itself (uv manages
packages directly). Use `uv pip install ...` as shown above rather than
plain `pip install ...` inside it — or `python -m pip ...`, which will
report "No module named pip" the same way.

**Using [pyenv](https://github.com/pyenv/pyenv) (recommended for macOS/Linux):**
Install pyenv:

macOS (via Homebrew):
```
brew install pyenv
```

Linux:
```
curl [https://pyenv.run](https://pyenv.run) | bash
```

```
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
Note: Requires Miniconda / Anaconda to be installed

```
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
| `g` | Gender for the raised-hand emoji: auto (face scan) → male → female |
| `f` | Toggle the face box / gender tag |


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

Motion and position tuning:
- `motion.py` — `WAVE_MIN_SWING` (how wide each swing must be, in palm
  lengths; raise it if idle hand movement triggers waving), `WAVE_MIN_SWINGS`
  (how many swings), and `WAVE_WINDOW_S` (how long the history window is,
  which also sets how quickly "Waving" switches off after you stop).
- `main.py` — `RAISE_FACE_FRACTION` (how far down the face box the fingertip
  must be *above*: 0 = top of the face, 0.5 = middle, 1 = chin) and
  `NO_FACE_RAISE_FRACTION` (the fallback line when no face is visible).
- `gesture_recognizer.py` — `is_upright(max_tilt_deg=50)` controls how far the
  hand may lean and still count as raised; `is_open_palm(min_extended=3)`
  controls how straight the fingers must be.
- `gender.py` — `HISTORY` (samples averaged), `SWITCH_HIGH` / `SWITCH_LOW`
  (how far the average must move before the answer changes), `FACE_MARGIN`
  (padding around the face crop).

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
card needs no changes either way. Gendered variants of an emoji are added by
listing them in `GENDER_VARIANTS` in `gesture_recognizer.py`.
