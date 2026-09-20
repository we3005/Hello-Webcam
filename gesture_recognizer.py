"""
Hand landmark utilities and gesture classification logic.

Uses MediaPipe Hands to obtain 21 3D landmarks per hand, then applies
rotation-invariant, angle-based heuristics to determine finger curl
state and recognize a fixed set of gestures:

    - Thumbs Up                   ("Thumbs Up")           👍
    - Peace Sign                  ("Peace Sign")           ✌️
    - Korean Finger Heart         ("Korean Finger Heart")  🫰  (one hand)
    - OK Sign                     ("OK Sign")              👌
    - Prayer Hands                ("Prayer Hands")         🙏  (two hands)
    - Index Raised                ("Index Raised")         ☝️
    - Hand Raised                 ("Hand Raised")          🙋  (open palm held up at face level)
    - Waving                      ("Waving")               👋  (open palm swung side to side —
                                                             detected over time in motion.py)

The heuristics are based on joint angles rather than raw x/y screen
positions, so they keep working when the hand is rotated (e.g. a
sideways thumbs-up), which naive "is fingertip above knuckle" checks
do not handle well.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe Hands topology)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB = (1, 2, 3, 4)     # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)     # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)

FINGERS = {
    "thumb": THUMB,
    "index": INDEX,
    "middle": MIDDLE,
    "ring": RING,
    "pinky": PINKY,
}

# Emoji shown alongside each recognized gesture's on-screen label.
# main.py renders these next to the label text when a color-emoji font
# is available on the system (see _find_emoji_font / _build_icon_cache).
GESTURE_EMOJI = {
    "Thumbs Up": "\U0001F44D",             # 👍
    "Peace Sign": "\u270C\uFE0F",           # ✌️
    "Korean Finger Heart": "\U0001FAF0",    # 🫰
    "OK Sign": "\U0001F44C",                # 👌
    "Prayer Hands": "\U0001F64F",           # 🙏
    "Index Raised": "\u261D\uFE0F",         # ☝️
    "Waving": "\U0001F44B",                # 👋
    "Hand Raised": "\U0001F64B",           # 🙋
}

WAVE_LABEL = "Waving"
RAISED_LABEL = "Hand Raised"

# Gender signs composited onto the raised-hand emoji (overlay.build_icon_cache
# adds a "Hand Raised (Male)" / "Hand Raised (Female)" variant for each).
GENDER_VARIANTS = {
    RAISED_LABEL: {
        "Male": "\u2642\uFE0F",     # ♂️
        "Female": "\u2640\uFE0F",   # ♀️
    },
}


class Curl(Enum):
    EXTENDED = "extended"   # straight, pointing out
    HOOKED = "hooked"       # bent at the knuckle but not folded into the palm
    CURLED = "curled"       # folded into the palm (fist-like)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC (at vertex b) in degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


@dataclass
class HandInfo:
    """Processed information about a single detected hand."""
    points: np.ndarray                     # (21, 3) landmark coordinates
    handedness: str                        # "Left" or "Right" (as reported by MediaPipe)
    curls: Dict[str, Curl] = field(default_factory=dict)
    scale: float = 1.0                     # normalization factor (palm size)

    def tip(self, finger: str) -> np.ndarray:
        return self.points[FINGERS[finger][3]]

    def norm_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return _dist(a, b) / self.scale


def analyze_hand(points: np.ndarray, handedness: str) -> HandInfo:
    """Compute per-finger curl state for one hand's 21 landmarks."""
    wrist = points[WRIST]
    scale = _dist(wrist, points[MIDDLE[1]])  # wrist -> middle MCP: a stable "palm size"
    scale = max(scale, 1e-6)

    curls: Dict[str, Curl] = {}
    for name, (j0, j1, j2, j3) in FINGERS.items():
        p0, p1, p2, p3 = points[j0], points[j1], points[j2], points[j3]
        angle_mid = _angle(p0, p1, p2)   # angle at the first knuckle
        angle_tip = _angle(p1, p2, p3)   # angle at the second knuckle
        tip_far = _dist(wrist, p3) / scale
        base_far = _dist(wrist, p1) / scale

        straight = angle_mid > 155 and angle_tip > 150
        reaches_out = tip_far > base_far * 1.05

        if straight and reaches_out:
            curls[name] = Curl.EXTENDED
        elif angle_mid > 100 and tip_far > base_far * 0.75:
            curls[name] = Curl.HOOKED
        else:
            curls[name] = Curl.CURLED

    return HandInfo(points=points, handedness=handedness, curls=curls, scale=scale)


# ---------------------------------------------------------------------------
# Single-hand gesture detectors
# ---------------------------------------------------------------------------

def is_thumbs_up(hand: HandInfo) -> bool:
    c = hand.curls
    if c["thumb"] != Curl.EXTENDED:
        return False
    if any(c[f] != Curl.CURLED for f in ("index", "middle", "ring", "pinky")):
        return False
    wrist = hand.points[WRIST]
    thumb_reach = hand.norm_dist(wrist, hand.tip("thumb"))
    folded_reach = max(hand.norm_dist(wrist, hand.tip(f)) for f in ("index", "middle", "ring", "pinky"))
    return thumb_reach > folded_reach * 0.9


def is_peace_sign(hand: HandInfo) -> bool:
    c = hand.curls
    if c["index"] != Curl.EXTENDED or c["middle"] != Curl.EXTENDED:
        return False
    if c["ring"] != Curl.CURLED or c["pinky"] != Curl.CURLED:
        return False
    spread = hand.norm_dist(hand.tip("index"), hand.tip("middle"))
    return spread > 0.35  # index/middle separated into a "V"


def is_finger_heart(hand: HandInfo) -> bool:
    """Korean one-hand 'finger heart' (🫰) — thumb and index tips pinched
    together, remaining fingers folded down."""
    c = hand.curls
    if any(c[f] != Curl.CURLED for f in ("middle", "ring", "pinky")):
        return False
    if c["thumb"] == Curl.EXTENDED and c["index"] == Curl.EXTENDED:
        return False  # a fully open pinch/L-shape, not a crossed heart
    pinch = hand.norm_dist(hand.tip("thumb"), hand.tip("index"))
    return pinch < 0.35


def is_ok_sign(hand: HandInfo) -> bool:
    """OK hand sign (👌) — thumb and index pinched into a circle, with the
    other three fingers extended and fanned out (the opposite of the finger
    heart's folded-down fingers, which is what distinguishes the two)."""
    c = hand.curls
    if any(c[f] != Curl.EXTENDED for f in ("middle", "ring", "pinky")):
        return False
    if c["thumb"] == Curl.EXTENDED and c["index"] == Curl.EXTENDED:
        return False  # open hand, not a pinched circle
    pinch = hand.norm_dist(hand.tip("thumb"), hand.tip("index"))
    return pinch < 0.35


def is_index_point(hand: HandInfo) -> bool:
    """Single index finger raised (☝️) — index extended, thumb and the
    remaining fingers folded into a loose fist."""
    c = hand.curls
    if c["index"] != Curl.EXTENDED:
        return False
    if c["thumb"] == Curl.EXTENDED:
        return False
    if any(c[f] != Curl.CURLED for f in ("middle", "ring", "pinky")):
        return False
    return True


def palm_center(hand: HandInfo) -> np.ndarray:
    """Centre of the palm (wrist + the four finger knuckles) — steadier to
    track than a fingertip, which is what motion.WaveDetector follows."""
    return hand.points[[WRIST, INDEX[0], MIDDLE[0], RING[0], PINKY[0]]].mean(axis=0)


def is_open_palm(hand: HandInfo, min_extended: int = 3) -> bool:
    """Open hand: none of the four fingers folded into the palm and at least
    `min_extended` of them straight. The thumb is ignored since it sits in
    very different places for waving, greeting, and raising a hand."""
    four = ("index", "middle", "ring", "pinky")
    c = hand.curls
    if any(c[f] == Curl.CURLED for f in four):
        return False
    return sum(1 for f in four if c[f] == Curl.EXTENDED) >= min_extended


def is_upright(hand: HandInfo, max_tilt_deg: float = 50.0) -> bool:
    """True if the fingers point roughly up in the image (wrist -> middle
    knuckle within `max_tilt_deg` of vertical)."""
    v = hand.points[MIDDLE[0]][:2] - hand.points[WRIST][:2]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return False
    cos_tilt = -v[1] / n   # image y grows downward, so "up" is (0, -1)
    return cos_tilt >= math.cos(math.radians(max_tilt_deg))


def is_hand_raised(hand: HandInfo, raise_y: float) -> bool:
    """Open palm held upright with its middle fingertip above `raise_y`
    (a pixel row — main.py passes the middle of the detected face, or a
    fixed fraction of the frame height when no face is visible)."""
    if not is_open_palm(hand) or not is_upright(hand):
        return False
    return float(hand.tip("middle")[1]) < raise_y


# ---------------------------------------------------------------------------
# Two-hand gesture detectors
# ---------------------------------------------------------------------------

def is_prayer_hands(hands: List[HandInfo]) -> bool:
    """Two-hand prayer / namaste pose (🙏) — both palms held flat together,
    fingers extended and lined up, wrists close to one another."""
    if len(hands) != 2:
        return False
    h1, h2 = hands
    for h in (h1, h2):
        extended_count = sum(1 for f in FINGERS if h.curls[f] == Curl.EXTENDED)
        if extended_count < 4:
            return False
    scale = (h1.scale + h2.scale) / 2
    tip_gap = sum(
        _dist(h1.tip(f), h2.tip(f)) for f in ("index", "middle", "ring", "pinky")
    ) / (4 * scale)
    wrist_gap = _dist(h1.points[WRIST], h2.points[WRIST]) / scale
    return tip_gap < 0.5 and wrist_gap < 1.0


def classify_frame(
    hands: List[HandInfo],
    waving_hands: Optional[Set[str]] = None,
    raise_y: Optional[float] = None,
) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """
    Returns (two_hand_gesture_or_None, [(handedness, single_hand_gesture), ...])

    waving_hands: handedness labels that motion.WaveDetector currently reports
        as waving. Waving wins over every other single-hand gesture.
    raise_y: pixel row a hand must reach above to count as "Hand Raised";
        None disables that gesture. It is checked last, after the finger-shape
        gestures, so a raised open palm never steals a more specific pose.
    """
    waving_hands = waving_hands or set()
    two_hand_result = "Prayer Hands" if is_prayer_hands(hands) else None

    per_hand_results = []
    if two_hand_result is None:
        for h in hands:
            label = None
            if h.handedness in waving_hands:
                label = WAVE_LABEL
            elif is_thumbs_up(h):
                label = "Thumbs Up"
            elif is_peace_sign(h):
                label = "Peace Sign"
            elif is_finger_heart(h):
                label = "Korean Finger Heart"
            elif is_ok_sign(h):
                label = "OK Sign"
            elif is_index_point(h):
                label = "Index Raised"
            elif raise_y is not None and is_hand_raised(h, raise_y):
                label = RAISED_LABEL
            if label:
                per_hand_results.append((h.handedness, label))

    return two_hand_result, per_hand_results