"""
Face-based gender guess used to pick the raised-hand emoji variant.

This uses the Levi & Hassner gender CNN (Caffe model) through OpenCV's
built-in `cv2.dnn` module, so no extra Python packages are needed — only
the two model files placed in `assets/models/` (see the README's "Gender
detection" section):

    gender_deploy.prototxt
    gender_net.caffemodel

The face location comes from the MediaPipe face detector that main.py is
already running; this module only crops that box and classifies it.

Reliability
-----------
The network only sees a face crop and outputs two classes (Male / Female).
It is a guess about appearance, not a fact about the person: glasses,
occlusion (like a raised hand), lighting, and camera angle all affect it, and
it can simply be wrong. To keep it from flickering, predictions are averaged
over the last HISTORY samples, and the shown answer only switches when the
average moves clearly past the middle (see SWITCH_HIGH / SWITCH_LOW). main.py
also lets the user override the guess by hand (the `g` key).

If the model files are missing, `available` is False and the app carries on
with the neutral emoji rather than failing.
"""

import os
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "models")
PROTO_FILE = "gender_deploy.prototxt"
WEIGHTS_FILE = "gender_net.caffemodel"

# Preprocessing the network was trained with.
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
INPUT_SIZE = (227, 227)

FACE_MARGIN = 0.25    # expand the detector's tight box by this fraction per side
MIN_FACE_PX = 60      # skip faces smaller than this (too little detail)
HISTORY = 20          # predictions averaged together
MIN_SAMPLES = 3       # predictions needed before any answer is shown
SWITCH_HIGH = 0.60    # average P(female) at/above this -> Female
SWITCH_LOW = 0.40     # average P(female) at/below this -> Male
                      # in between, the previous answer is kept


class GenderEstimator:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.net = None
        self.history: Deque[float] = deque(maxlen=HISTORY)   # P(female) per sample
        self._label: Optional[str] = None

        proto = os.path.join(models_dir, PROTO_FILE)
        weights = os.path.join(models_dir, WEIGHTS_FILE)
        if os.path.isfile(proto) and os.path.isfile(weights):
            try:
                self.net = cv2.dnn.readNet(weights, proto)
            except cv2.error as err:
                print(f"Could not load the gender model ({err}); "
                      "falling back to the neutral raised-hand emoji.")

    @property
    def available(self) -> bool:
        return self.net is not None

    @property
    def label(self) -> Optional[str]:
        """"Male", "Female", or None if there isn't an answer yet."""
        return self._label

    @property
    def confidence(self) -> float:
        """Averaged probability of the currently shown label (0.5-1.0)."""
        if not self.history or self._label is None:
            return 0.0
        p_female = sum(self.history) / len(self.history)
        return p_female if self._label == "Female" else 1.0 - p_female

    def reset(self) -> None:
        """Forget everything so the next face gets a fresh scan."""
        self.history.clear()
        self._label = None

    def update(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> None:
        """Classify the face inside `face_box` = (x, y, w, h) in pixels.

        `frame` should be the clean camera frame (before any overlays are
        drawn), otherwise hand landmarks or labels end up in the crop.
        """
        if self.net is None:
            return
        x, y, w, h = face_box
        if min(w, h) < MIN_FACE_PX:
            return

        fh, fw = frame.shape[:2]
        mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
        x0, y0 = max(x - mx, 0), max(y - my, 0)
        x1, y1 = min(x + w + mx, fw), min(y + h + my, fh)
        face = frame[y0:y1, x0:x1]
        if face.size == 0:
            return

        blob = cv2.dnn.blobFromImage(face, 1.0, INPUT_SIZE, MODEL_MEAN_VALUES, swapRB=False)
        self.net.setInput(blob)
        preds = self.net.forward()[0]        # [P(male), P(female)]
        self.history.append(float(preds[1]))
        self._refresh_label()

    def _refresh_label(self) -> None:
        if len(self.history) < MIN_SAMPLES:
            return
        p_female = sum(self.history) / len(self.history)
        if p_female >= SWITCH_HIGH:
            self._label = "Female"
        elif p_female <= SWITCH_LOW:
            self._label = "Male"
        elif self._label is None:
            # First answer landed in the "unsure" band: take the nearer side.
            self._label = "Female" if p_female >= 0.5 else "Male"
        # else: still unsure -> keep showing the previous answer