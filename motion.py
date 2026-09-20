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
2. The horizontal position of the fingertips is sampled every frame and
   only the last WAVE_WINDOW_S seconds are kept. Fingertips are used (not the
   palm) because waving pivots at the wrist or elbow, so the fingertips sweep
   through by far the biggest arc.
3. A "swing" is a sideways move of at least WAVE_MIN_SWING palm lengths
   followed by a move back the other way. Measuring in palm lengths (not
   pixels) keeps this independent of how far the hand is from the camera,
   and ignoring anything smaller keeps landmark jitter from counting.
4. WAVE_MIN_SWINGS or more swings inside the window = waving. Because the
   window is short, the wave label switches off shortly after you stop.
"""

from collections import defaultdict, deque
from typing import Deque, Dict, Hashable, List, Tuple

WAVE_WINDOW_S = 1.2      # seconds of history considered
WAVE_MIN_SWING = 0.4     # smallest counted sideways move, in palm lengths
WAVE_MIN_SWINGS = 3      # e.g. right -> left -> right
WAVE_GRACE_S = 0.35      # a hand may leave the open-palm pose this long (motion blur) without losing its history


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
                 min_swings: int = WAVE_MIN_SWINGS,
                 grace_s: float = WAVE_GRACE_S):
        self.window_s = window_s
        self.min_swing = min_swing
        self.min_swings = min_swings
        self.grace_s = grace_s
        # key -> deque of (timestamp, x_pixels, palm_scale_pixels)
        self.samples: Dict[Hashable, Deque[Tuple[float, float, float]]] = defaultdict(deque)
        self._not_eligible_since: Dict[Hashable, float] = {}
        self.last_swings: Dict[Hashable, int] = {}     # for the debug overlay

    def update(self, key: Hashable, x: float, scale: float, now: float, eligible: bool) -> bool:
        """Record one frame for hand `key`; return True if it is waving.

        `x` is the fingertips' horizontal pixel position, `scale` the palm
        length in pixels, `now` a timestamp in seconds, and `eligible`
        whether the hand is currently in a wave-able pose (open palm).

        Fast waving blurs the hand and can make the pose check flicker for a
        frame or two, so a hand only loses its history after being ineligible
        for `grace_s` seconds in a row.
        """
        history = self.samples[key]
        if eligible:
            self._not_eligible_since.pop(key, None)
            history.append((now, x, scale))
        else:
            since = self._not_eligible_since.setdefault(key, now)
            if now - since > self.grace_s:
                history.clear()
                self.last_swings[key] = 0
                return False
            # brief dropout: keep the history, just don't add a sample

        while history and now - history[0][0] > self.window_s:
            history.popleft()
        if len(history) < 4:
            self.last_swings[key] = 0
            return False

        xs = [s[1] for s in history]
        avg_scale = sum(s[2] for s in history) / len(history)
        swings = count_swings(xs, self.min_swing * avg_scale)
        self.last_swings[key] = swings
        return swings >= self.min_swings

    def forget_stale(self, now: float) -> None:
        """Drop hands not seen for a whole window. (Expiring by time rather
        than by "missing this frame" means a dropped detection frame or a
        left/right label flip in the middle of a wave doesn't reset it.)"""
        for key in list(self.samples):
            history = self.samples[key]
            if not history or now - history[-1][0] > self.window_s:
                del self.samples[key]
                self._not_eligible_since.pop(key, None)
                self.last_swings.pop(key, None)