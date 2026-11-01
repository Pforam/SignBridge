"""
SignBridge Gesture Recognition Configuration and Thresholds.
Centralizes all state machine and validation parameters for easy tuning.

Speed comes from removing waiting, not from loosening the accuracy checks.
"""

# --- Fast commit (calibrated for ~30 FPS camera feed) -----------------------
NEUTRAL_HOLD_FRAMES = 4            # ~0.13s of two open palms to arm (was 10)
STABILITY_FRAMES_K = 3             # 3 consecutive identical valid predictions (~0.1s) to commit (was 5)
INSTANT_COMMIT_CONFIDENCE = 0.92   # one frame at/above this (passing margin + centroid checks) commits immediately
MAX_MISS_FRAMES = 1                # invalid/dropped frames tolerated mid-streak before the streak resets

# --- Lock, then hold steady (replaces COOLDOWN_FRAMES / COOLDOWN_MS) ---------
MIN_LOCK_MS = 250                  # minimum time the locked sign stays visible
STEADY_MS = 350                    # hand must be steady this long to re-arm
STEADY_MOTION_THRESHOLD = 0.015    # mean landmark movement per frame (normalized 0..1) counted as steady
MAX_COOLDOWN_MS = 1500             # failsafe: never stay locked longer than this
REQUIRE_NEUTRAL_BETWEEN_SIGNS = False  # True = need two open palms before every sign

# --- Hand presence / same-sign repeat protection -----------------------------
HAND_LOST_FRAMES = 3               # frames without an active hand = "left the frame"
RELEASE_FRAMES = 4                 # frames of no/other sign after a lock before the same sign may repeat

CONFIDENCE_THRESHOLD = 0.75        # top-class probability threshold for static letters (reject unknowns)
CONFIDENCE_THRESHOLD_RELAXED = 0.65 # relaxed threshold when prediction has decisive margin
CONFIDENCE_THRESHOLD_WORD = 0.50   # probability threshold for 57-class temporal word sequences
MARGIN_THRESHOLD = 0.10            # margin between top-1 and top-2 probabilities for letters
MARGIN_THRESHOLD_RELAXED = 0.20    # decisive margin between top-1 and top-2 for confident classification
MARGIN_THRESHOLD_WORD = 0.06       # margin between top-1 and top-2 for words
DEFAULT_MAX_CENTROID_DISTANCE = 11.5  # fallback maximum Euclidean distance to class centroid

# --- Hand tracking & idle thresholds -----------------------------------------
MIN_HAND_SPAN = 0.04               # minimum bounding box span for active hand
IDLE_WRIST_Y_THRESHOLD = 0.88      # wrist below 88% of frame = resting/idle hand
TRACKING_CONFIDENCE_MIN = 0.35     # minimum confidence for landmark validity

# --- Debug --------------------------------------------------------------------
DEBUG_OVERLAY_ENABLED = False

# --- States and UI labels -----------------------------------------------------
NEUTRAL = "neutral"
ARMED = "armed"
CAPTURING = "capturing"
COOLDOWN = "cooldown"  # locked, hold steady

STATUS_LABELS = {
    NEUTRAL: "Show neutral to continue",
    ARMED: "Ready - show the next sign",
    CAPTURING: "Recognising...",
    COOLDOWN: "Sign locked - hold your hand steady",
}
