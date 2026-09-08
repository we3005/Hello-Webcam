"""
Hand landmark utilities and gesture classification logic.

Uses MediaPipe Hands to obtain 21 3D landmarks per hand, then applies
rotation-invariant, angle-based heuristics to determine finger curl
state and recognize a fixed set of gestures:

    - Thumbs Up
    - Peace Sign
    - Claw
    - Korean Finger Heart (one hand)
    - Double Heart Hands (two hands)

The heuristics are based on joint angles rather than raw x/y screen
positions, so they keep working when the hand is rotated (e.g. a
sideways thumbs-up), which naive "is fingertip above knuckle" checks
do not handle well.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

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


class Curl(Enum):
    EXTENDED = "extended"   # straight, pointing out
    HOOKED = "hooked"       # bent at the knuckle but not folded into the palm (claw)
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


def is_claw(hand: HandInfo) -> bool:
    c = hand.curls
    hook_fingers = ("index", "middle", "ring", "pinky")
    if any(c[f] != Curl.HOOKED for f in hook_fingers):
        return False
    tips = [hand.tip(f) for f in hook_fingers]
    spread = sum(hand.norm_dist(tips[i], tips[i + 1]) for i in range(len(tips) - 1))
    return spread > 0.9  # fingers spread apart, not bunched


def is_finger_heart(hand: HandInfo) -> bool:
    """Korean one-hand 'finger heart' — thumb and index tips pinched together,
    remaining fingers folded down."""
    c = hand.curls
    if any(c[f] != Curl.CURLED for f in ("middle", "ring", "pinky")):
        return False
    if c["thumb"] == Curl.EXTENDED and c["index"] == Curl.EXTENDED:
        return False  # a fully open pinch/L-shape, not a crossed heart
    pinch = hand.norm_dist(hand.tip("thumb"), hand.tip("index"))
    return pinch < 0.35


# ---------------------------------------------------------------------------
# Two-hand gesture detectors
# ---------------------------------------------------------------------------

def is_double_heart(hands: List[HandInfo]) -> bool:
    """Two-hand heart — each hand's thumb tip meets the other hand's index tip."""
    if len(hands) != 2:
        return False
    h1, h2 = hands
    for h in (h1, h2):
        if any(h.curls[f] != Curl.CURLED for f in ("middle", "ring", "pinky")):
            return False
    scale = (h1.scale + h2.scale) / 2
    cross1 = _dist(h1.tip("thumb"), h2.tip("index")) / scale
    cross2 = _dist(h2.tip("thumb"), h1.tip("index")) / scale
    return cross1 < 0.6 and cross2 < 0.6


def classify_frame(hands: List[HandInfo]) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """
    Returns (two_hand_gesture_or_None, [(handedness, single_hand_gesture), ...])
    """
    two_hand_result = "Double Heart Hands" if is_double_heart(hands) else None

    per_hand_results = []
    if two_hand_result is None:
        for h in hands:
            label = None
            if is_thumbs_up(h):
                label = "Thumbs Up"
            elif is_peace_sign(h):
                label = "Peace Sign"
            elif is_finger_heart(h):
                label = "Korean Finger Heart"
            elif is_claw(h):
                label = "Claw"
            if label:
                per_hand_results.append((h.handedness, label))

    return two_hand_result, per_hand_results
