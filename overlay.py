"""
On-screen decoration helpers: styled HUD labels, gesture emoji badges, the
image-based glasses overlay for the "Index Raised" (☝️) gesture, and the
left-side panel of hand-drawn gesture art.

Emoji rendering
----------------
OpenCV's built-in text drawing (cv2.putText) only supports a small Hershey
vector font with no emoji glyphs, so to show real emoji we render them once
at startup with Pillow, using whichever color-emoji font the OS provides,
and cache the RGBA image patches. Each video frame then just alpha-blits
the cached patch — no per-frame font rendering. If no color-emoji font can
be found (font missing, or an old Pillow without embedded-color glyph
support), icon rendering is skipped entirely and gesture labels fall back
to text-only, rather than crashing the app.

The raised-hand emoji comes in gendered variants. Composing the real
"person raising hand + male/female sign" ZWJ sequence needs a text-shaping
engine (libraqm) that many Pillow installs don't have, so instead the plain
🙋 is rendered and a small ♂️ / ♀️ badge is composited onto its corner. That
looks the same on every OS/Pillow build. That badge version is only a
fallback now: if `assets/emoji/hand_raising_male.png` / `..._female.png` exist,
load_gender_icons() uses those real 🙋‍♂️ / 🙋‍♀️ images instead.

Assets
------
`assets/glasses.png` and the six drawings in `assets/drawings/` are loaded
once at startup with alpha preserved (cv2.IMREAD_UNCHANGED). The glasses
are resized and rotated per frame to match the detected face; the drawings
are resized once to a fixed panel width since they don't need to track
anything moving.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# ---------------------------------------------------------------------------
# Color palette / shared HUD styling
# ---------------------------------------------------------------------------
PANEL_BG = (32, 28, 40)          # dark plum, semi-transparent panel fill
PANEL_ALPHA = 0.62
TEXT_COLOR = (240, 240, 245)
SHADOW_COLOR = (0, 0, 0)
ACCENT_SINGLE = (255, 176, 59)   # warm amber accent bar — per-hand gestures
ACCENT_TWO_HAND = (255, 110, 199)  # pink accent bar — two-hand gestures
CREDIT_COLOR = (200, 200, 210)


def _rounded_panel(frame: np.ndarray, x: int, y: int, w: int, h: int,
                    radius: int = 14, color=PANEL_BG, alpha: float = PANEL_ALPHA) -> None:
    """Alpha-blend a filled rounded rectangle onto the frame in place."""
    x, y = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x2 <= x or y2 <= y:
        return
    radius = max(0, min(radius, (x2 - x) // 2, (y2 - y) // 2))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x + radius, y), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x2, y2 - radius), color, -1)
    for cx, cy in ((x + radius, y + radius), (x2 - radius, y + radius),
                   (x + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int],
               accent=ACCENT_SINGLE) -> Tuple[int, int, int, int]:
    """Draw a gesture label as a rounded, semi-transparent card with a
    colored accent bar and shadowed text. Returns the card's (x, y, w, h)
    in case a caller wants to lay something out relative to it."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.75, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    pad_x, pad_y, bar_w = 16, 10, 5
    card_x, card_y = x, y - th - pad_y * 2
    card_w, card_h = tw + pad_x * 2 + bar_w, th + pad_y * 2

    _rounded_panel(frame, card_x, card_y, card_w, card_h)
    cv2.rectangle(frame, (card_x, card_y), (card_x + bar_w, card_y + card_h), accent, -1)

    text_x, text_y = card_x + bar_w + pad_x, card_y + card_h - pad_y - 3
    cv2.putText(frame, text, (text_x + 1, text_y + 2), font, scale, SHADOW_COLOR, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (text_x, text_y), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)
    return card_x, card_y, card_w, card_h


def draw_hint_bar(frame: np.ndarray, text: str, origin: Tuple[int, int]) -> None:
    """Small muted status text (FPS, key hints) with a soft shadow — no
    background card, so it stays unobtrusive along the bottom edge."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (x + 1, y + 1), font, 0.5, SHADOW_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, 0.5, CREDIT_COLOR, 1, cv2.LINE_AA)


def draw_face_tag(frame: np.ndarray, box: Tuple[int, int, int, int], text: str,
                   accent=ACCENT_SINGLE) -> None:
    """Outline the scanned face and stick a small label card above it
    (e.g. "Female 87%" or "Scanning...")."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), accent, 2, cv2.LINE_AA)
    if text:
        # draw_label puts its card above the given origin; keep it on-screen.
        draw_label(frame, text, (x, max(y - 6, 50)), accent=accent)


# ---------------------------------------------------------------------------
# Emoji badges (top-right corner)
# ---------------------------------------------------------------------------
_EMOJI_FONT_CANDIDATES = [
    "/System/Library/Fonts/Apple Color Emoji.ttc",       # macOS
    "C:\\Windows\\Fonts\\seguiemj.ttf",                   # Windows
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux (Noto, if installed)
]
_CANDIDATE_STRIKE_SIZES = [160, 137, 136, 128, 118, 109, 96, 64, 48, 40, 32, 20]
ICON_SIZE = 120
ICON_MARGIN = 24


def _find_emoji_font():
    if not _PIL_AVAILABLE:
        return None
    for path in _EMOJI_FONT_CANDIDATES:
        for size in _CANDIDATE_STRIKE_SIZES:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


def _render_emoji_image(font, text: str):
    """Render one emoji string to a transparent RGBA Pillow image."""
    canvas_size = font.size + 8
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((0, 0), text, font=font, embedded_color=True)
    return img


def _add_corner_badge(icon, glyph):
    """Composite `glyph` (an emoji image) onto the bottom-right corner of
    `icon`, on a small white disc so it stays readable over any video."""
    size = icon.width
    badge = int(size * 0.48)
    disc = Image.new("RGBA", (badge, badge), (0, 0, 0, 0))
    ImageDraw.Draw(disc).ellipse((0, 0, badge - 1, badge - 1), fill=(255, 255, 255, 235))

    bbox = glyph.getbbox()
    if bbox:
        glyph = glyph.crop(bbox)
        inner = int(badge * 0.68)
        scale = inner / max(glyph.width, glyph.height)
        glyph = glyph.resize(
            (max(int(glyph.width * scale), 1), max(int(glyph.height * scale), 1)),
            Image.LANCZOS,
        )
        disc.alpha_composite(glyph, dest=((badge - glyph.width) // 2, (badge - glyph.height) // 2))

    out = icon.copy()
    out.alpha_composite(disc, dest=(size - badge, size - badge))
    return out


def build_icon_cache(gesture_emoji: Dict[str, str],
                      gender_variants: Optional[Dict[str, Dict[str, str]]] = None
                      ) -> Dict[str, np.ndarray]:
    """Pre-render each gesture's emoji to an RGBA patch, once at startup.
    Returns an empty dict if emoji rendering isn't available.

    gender_variants maps a base gesture label to {gender: sign_emoji}. For
    each pair an extra "<label> (<gender>)" icon is cached: the base emoji
    with the gender sign badged onto its corner."""
    font = _find_emoji_font()
    if font is None:
        return {}

    cache: Dict[str, np.ndarray] = {}
    resized: Dict[str, "Image.Image"] = {}
    try:
        for label, emoji in gesture_emoji.items():
            img = _render_emoji_image(font, emoji).resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS)
            resized[label] = img
            cache[label] = np.array(img)  # RGBA

        for base_label, signs in (gender_variants or {}).items():
            if base_label not in resized:
                continue
            for gender, sign_emoji in signs.items():
                badged = _add_corner_badge(resized[base_label], _render_emoji_image(font, sign_emoji))
                cache[f"{base_label} ({gender})"] = np.array(badged)
    except TypeError:
        return {}   # Pillow too old for embedded_color glyphs
    return cache


# Where load_gender_icons looks, in order: filename prefixes per base gesture
# ("<prefix>_male.png" / "<prefix>_female.png"), inside these folders.
GENDER_ICON_PREFIXES = {"Hand Raised": ("hand_raising", "hand_raised")}
GENDER_ICON_DIRS = (
    os.path.join(ASSETS_DIR, "emoji"),
    ASSETS_DIR,
    os.path.dirname(ASSETS_DIR),      # project root
)


def load_gender_icons(gender_variants: Dict[str, Dict[str, str]]) -> Dict[str, np.ndarray]:
    """Load ready-made gendered emoji images (e.g. hand_raising_female.png,
    the real 🙋‍♀️) into icon-cache entries.

    Emoji fonts on a lot of setups can't draw combined "person + gender"
    emoji (that needs a text-shaping library most Pillow installs lack), so
    these are plain PNGs instead. Entries loaded here replace the
    badge-composite fallback from build_icon_cache. For each gesture/gender
    the first match wins, checking each prefix in GENDER_ICON_PREFIXES across
    each folder in GENDER_ICON_DIRS; anything not found prints a one-line note
    saying where it looked."""
    cache: Dict[str, np.ndarray] = {}
    if not _PIL_AVAILABLE:
        return cache
    for base_label, signs in gender_variants.items():
        prefixes = GENDER_ICON_PREFIXES.get(base_label, (base_label.lower().replace(" ", "_"),))
        for gender in signs:
            path = next(
                (os.path.join(d, f"{prefix}_{gender.lower()}.png")
                 for prefix in prefixes for d in GENDER_ICON_DIRS
                 if os.path.isfile(os.path.join(d, f"{prefix}_{gender.lower()}.png"))),
                None,
            )
            if path is None:
                print(f"Gender emoji image not found for '{base_label} ({gender})' - using the badge "
                      f"fallback. Looked for {prefixes[0]}_{gender.lower()}.png in {GENDER_ICON_DIRS[0]}")
                continue
            img = Image.open(path).convert("RGBA")
            scale = ICON_SIZE / max(img.size)
            img = img.resize((max(int(img.width * scale), 1), max(int(img.height * scale), 1)), Image.LANCZOS)
            canvas = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
            canvas.alpha_composite(img, dest=((ICON_SIZE - img.width) // 2, (ICON_SIZE - img.height) // 2))
            cache[f"{base_label} ({gender})"] = np.array(canvas)
    return cache


def overlay_rgba(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Alpha-blit an RGBA (Pillow channel order) patch onto a BGR frame."""
    h, w = patch.shape[:2]
    fh, fw = frame.shape[:2]
    if x < 0 or y < 0 or x + w > fw or y + h > fh:
        return
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    patch_bgr = patch[:, :, [2, 1, 0]].astype(np.float32)
    region = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (alpha * patch_bgr + (1 - alpha) * region).astype(np.uint8)


def overlay_bgra(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Alpha-blit a BGRA (OpenCV channel order) patch onto a BGR frame.
    Unlike overlay_rgba, out-of-bounds placement is clipped rather than
    skipped, since the glasses/drawings can legitimately run to an edge."""
    h, w = patch.shape[:2]
    fh, fw = frame.shape[:2]
    src_x0, src_y0 = max(0, -x), max(0, -y)
    dst_x0, dst_y0 = max(0, x), max(0, y)
    dst_x1, dst_y1 = min(fw, x + w), min(fh, y + h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
    region = patch[src_y0:src_y1, src_x0:src_x1]
    if region.shape[2] < 4:
        frame[dst_y0:dst_y1, dst_x0:dst_x1] = region[:, :, :3]
        return
    alpha = region[:, :, 3:4].astype(np.float32) / 255.0
    src_bgr = region[:, :, :3].astype(np.float32)
    dst = frame[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    frame[dst_y0:dst_y1, dst_x0:dst_x1] = (alpha * src_bgr + (1 - alpha) * dst).astype(np.uint8)


def draw_gesture_icons(frame: np.ndarray, icon_cache: Dict[str, np.ndarray],
                        active_labels: List[str]) -> None:
    """Large emoji badge per active, distinct gesture, stacked in the
    top-right corner."""
    if not icon_cache:
        return
    frame_h, frame_w = frame.shape[:2]
    slot, seen = 0, set()
    for label in active_labels:
        if label in seen or label not in icon_cache:
            continue
        seen.add(label)
        icon = icon_cache[label]
        size = icon.shape[0]
        x = frame_w - size - ICON_MARGIN
        y = ICON_MARGIN + slot * (size + ICON_MARGIN)
        overlay_rgba(frame, icon, x, y)
        slot += 1


# ---------------------------------------------------------------------------
# Image-based glasses overlay (Index Raised gesture)
# ---------------------------------------------------------------------------

def load_glasses_image(filename: str = "glasses.png") -> Optional[np.ndarray]:
    """Load the glasses PNG (with alpha) once at startup. Returns None if
    the file is missing so the app can keep running without the overlay."""
    path = os.path.join(ASSETS_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        return None
    return img


def draw_glasses_image(frame: np.ndarray, detection, glasses_img: np.ndarray,
                        w_px: int, h_px: int) -> None:
    """Resize, rotate, and place the glasses image over a detected face
    using its eye keypoints — width matched to eye-to-eye distance, and
    rotated to match head tilt."""
    kp = detection.location_data.relative_keypoints
    if len(kp) < 2:
        return
    right_eye, left_eye = kp[0], kp[1]
    rx, ry = right_eye.x * w_px, right_eye.y * h_px
    lx, ly = left_eye.x * w_px, left_eye.y * h_px
    eye_dist = max(math.hypot(lx - rx, ly - ry), 1.0)

    src_h, src_w = glasses_img.shape[:2]
    # The glasses image's lens-to-lens span is roughly 70% of its own width;
    # scale so that span matches the detected eye distance.
    target_width = eye_dist / 0.7
    scale = target_width / src_w
    new_w, new_h = max(int(src_w * scale), 1), max(int(src_h * scale), 1)
    resized = cv2.resize(glasses_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    angle_deg = math.degrees(math.atan2(ly - ry, lx - rx))
    center = (new_w / 2, new_h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        resized, rot_mat, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )

    mid_x, mid_y = (rx + lx) / 2, (ry + ly) / 2
    x, y = int(mid_x - new_w / 2), int(mid_y - new_h / 2)
    overlay_bgra(frame, rotated, x, y)


# ---------------------------------------------------------------------------
# Left-side hand-drawn gesture panel
# ---------------------------------------------------------------------------
DRAWING_FILES = {
    "Thumbs Up": "thumbup.png",
    "Peace Sign": "peace.png",
    "Korean Finger Heart": "heart.png",
    "OK Sign": "ok.png",
    "Prayer Hands": "pray.png",
    "Index Raised": "nerd.png",
}
PANEL_WIDTH = 190
PANEL_MARGIN = 20
CREDIT_TEXT = "Cat drawings: Albert Rao (my friend)"


def build_drawing_cache(drawing_files: Dict[str, str] = DRAWING_FILES) -> Dict[str, np.ndarray]:
    """Load and resize each gesture's drawing once at startup (fixed panel
    width, alpha preserved). Missing files are skipped rather than
    crashing the app."""
    cache: Dict[str, np.ndarray] = {}
    for label, filename in drawing_files.items():
        path = os.path.join(ASSETS_DIR, "drawings", filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        new_w = PANEL_WIDTH
        new_h = max(int(h * (new_w / w)), 1)
        cache[label] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cache


def draw_drawing_panel(frame: np.ndarray, drawing_cache: Dict[str, np.ndarray],
                        active_labels: List[str], start_y: int = PANEL_MARGIN,
                        bottom_margin: int = 50) -> None:
    """Stack each active gesture's drawing down the left side of the frame,
    starting at start_y (so callers can place it below any text labels),
    each on its own rounded card, with a small credit line beneath the
    last one. Stops adding cards (and skips the credit line) rather than
    overlapping the bottom hint bar if the frame is too short to fit
    everything — this can happen with two simultaneous single-hand
    gestures on a low-resolution camera."""
    if not drawing_cache:
        return
    frame_h = frame.shape[0]
    y = start_y
    seen = set()
    for label in active_labels:
        if label in seen or label not in drawing_cache:
            continue
        img = drawing_cache[label]
        h, w = img.shape[:2]
        card_pad = 10
        if y + h + card_pad * 2 > frame_h - bottom_margin:
            break
        seen.add(label)
        _rounded_panel(frame, PANEL_MARGIN - card_pad, y - card_pad,
                        w + card_pad * 2, h + card_pad * 2, radius=16)
        overlay_bgra(frame, img, PANEL_MARGIN, y)
        y += h + card_pad * 2 + 14

    if seen and y + 20 <= frame_h - bottom_margin:
        draw_hint_bar(frame, CREDIT_TEXT, (PANEL_MARGIN, y + 4))