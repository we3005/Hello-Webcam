# Webcam Gesture Recognizer

Real-time hand gesture recognition using [MediaPipe Hands](https://developers.google.com/mediapipe)
for landmark tracking, with custom geometric rules to classify gestures.

**Recognizes:**
- 👍 Thumbs Up
- ✌️ Peace Sign
- 🦅 Claw
- 🫰 Korean Finger Heart (one hand, thumb + index pinched)
- 💕 Double Heart Hands (two hands crossed into a heart)

## Why this approach

Instead of training a classifier, each gesture is defined by rules over the
21 hand landmarks MediaPipe returns per hand:

1. Every finger is classified into one of three **curl states** based on the
   angles at its knuckles (not raw x/y position, so it works even if your
   hand is rotated — e.g. a sideways thumbs-up):
   - `extended` — straight, pointing out
   - `hooked` — bent at the first knuckle but not folded in (claw shape)
   - `curled` — folded into the palm (fist-like)
2. Each gesture is then just a pattern over those curl states, plus a
   distance check where needed (e.g. thumb tip near index tip for the
   finger heart).

This runs fast on CPU, needs no training data, and is easy to tune (see below).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Notes:**
- Requires a working webcam and OS permission for camera access.
- This runs as a local desktop script (uses `cv2.imshow`) — it needs a real
  Python environment with a display, not a browser sandbox.

### Python 3.13

MediaPipe does not yet publish official PyPI wheels for Python 3.13 (this is
a known, currently-open limitation on their end — see
[issue #6159](https://github.com/google-ai-edge/mediapipe/issues/6159)), so
`pip install mediapipe` will fail if your virtual environment is on 3.13.
`main.py` detects this and exits with a clear message rather than a raw
import error.

The fix is to run *this app* in a 3.11 or 3.12 environment, even if 3.13 is
your system default elsewhere — it won't affect anything else on your
machine. Easiest ways to get one:

**Using [uv](https://docs.astral.sh/uv/) (recommended, no separate Python install needed):**
```bash
uv venv --python 3.11 venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
uv pip install -r requirements.txt
python main.py
```

**Using [pyenv](https://github.com/pyenv/pyenv):**
```bash
pyenv install 3.11.9
pyenv exec python3.11 -m venv venv
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

Check MediaPipe's [PyPI page](https://pypi.org/project/mediapipe/) periodically —
once they ship 3.13 wheels, a plain 3.13 venv will work fine and this
workaround won't be needed.

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug overlay (shows each finger's curl state) |
| `m` | Toggle mirror mode |

## Tuning

Gestures are recognized less reliably from certain angles, lighting, or hand
sizes than others. If a gesture isn't triggering (or triggers too easily),
turn on debug mode (`d`) to see the live curl state of each finger, then
adjust thresholds in `gesture_recognizer.py`:

- `analyze_hand()` — thresholds for `extended` / `hooked` / `curled`
  (angle cutoffs `155`, `100`, and reach ratios `1.05`, `0.75`).
- `is_peace_sign()` — `spread > 0.35` controls how wide the "V" must be.
- `is_claw()` — `spread > 0.9` controls how spread the claw fingers must be.
- `is_finger_heart()` — `pinch < 0.35` controls how close thumb/index must be.
- `is_double_heart()` — `cross1/cross2 < 0.6` controls how close the two
  hands' crossed fingertips must be.

All thresholds are normalized by hand size, so they should hold up
reasonably well across distances from the camera — but camera angle,
occlusion, and individual hand proportions will still affect accuracy some.

## Extending

To add a new gesture:
1. Write a new `is_my_gesture(hand: HandInfo) -> bool` function in
   `gesture_recognizer.py` using `hand.curls`, `hand.tip(finger)`, and
   `hand.norm_dist(a, b)`.
2. Add it to the `if/elif` chain in `classify_frame()`.

For two-hand gestures, follow the pattern of `is_double_heart()`, which
receives the list of both detected `HandInfo` objects.
