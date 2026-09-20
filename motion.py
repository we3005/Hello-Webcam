"""
Motion-based gesture detection (things a single frame can't tell you).

The pose rules in gesture_recognizer.py look at one frame at a time, so they
can't distinguish a hand that is *waving* from a hand that is simply held
open — the difference is movement over time. WaveDetector keeps a short,
time-based history of each hand's horizontal position and reports a wave
once the hand has swung back and forth enough times.

How a wave is recognized
------------------------
1. The hand must be an open palm (the caller passes `eligible`); a fist or
   a pointing finger that happens to move sideways is not a wave.
2. The horizontal position of the palm centre is sampled every frame and
   only the last WAVE_WINDOW_S seconds are kept.
3. A "swing" is a sideways move of at least WAVE_MIN_SWING palm lengths
   followed by a move back the other way. Measuring in palm lengths (not
   pixels) keeps this independent of how far the hand is from the camera,
   and ignoring anything smaller keeps landmark jitter from counting.
4. WAVE_MIN_SWINGS or more swings inside the window = waving. Because the
   window is short, the wave label switches off shortly after you stop.
"""

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Tuple

WAVE_WINDOW_S = 1.2      # seconds of history considered
WAVE_MIN_SWING = 0.5     # smallest counted sideways move, in palm lengths
WAVE_MIN_SWINGS = 3      # e.g. right -> left -> right


def count_swings(xs: List[float], threshold: float) -> int:
    """Count alternating-direction moves of at least `threshold` in `xs`.

    The first sufficiently large move counts as one swing, and every later
    reversal (a move of at least `threshold` back the other way) adds one.
    Wobble smaller than `threshold` never registers.
    """
    if len(xs) < 2:
        return 0
    swings = 0
    direction = 0        # +1 moving right, -1 moving left, 0 not decided yet
    extreme = xs[0]      # furthest point reached in the current direction
    for x in xs[1:]:
        if direction == 0:
            if abs(x - extreme) >= threshold:
                direction = 1 if x > extreme else -1
                extreme = x
                swings += 1
        elif direction == 1:
            if x > extreme:
                extreme = x
            elif extreme - x >= threshold:
                direction = -1
                extreme = x
                swings += 1
        else:
            if x < extreme:
                extreme = x
            elif x - extreme >= threshold:
                direction = 1
                extreme = x
                swings += 1
    return swings


class WaveDetector:
    """Tracks horizontal hand movement per hand and reports waving."""

    def __init__(self, window_s: float = WAVE_WINDOW_S,
                 min_swing: float = WAVE_MIN_SWING,
                 min_swings: int = WAVE_MIN_SWINGS):
        self.window_s = window_s
        self.min_swing = min_swing
        self.min_swings = min_swings
        # key -> deque of (timestamp, x_pixels, palm_scale_pixels)
        self.samples: Dict[str, Deque[Tuple[float, float, float]]] = defaultdict(deque)

    def update(self, key: str, x: float, scale: float, now: float, eligible: bool) -> bool:
        """Record one frame for hand `key`; return True if it is waving.

        `x` is the palm centre's horizontal pixel position, `scale` the palm
        length in pixels, `now` a timestamp in seconds, and `eligible`
        whether the hand is currently in a wave-able pose (open palm).
        """
        history = self.samples[key]
        if not eligible:
            history.clear()
            return False

        history.append((now, x, scale))
        while history and now - history[0][0] > self.window_s:
            history.popleft()
        if len(history) < 4:
            return False

        xs = [s[1] for s in history]
        avg_scale = sum(s[2] for s in history) / len(history)
        return count_swings(xs, self.min_swing * avg_scale) >= self.min_swings

    def drop_missing(self, present: Iterable[str]) -> None:
        """Forget hands that weren't seen this frame so a hand that leaves
        and re-enters doesn't inherit stale movement history."""
        present = set(present)
        for key in list(self.samples):
            if key not in present:
                del self.samples[key]