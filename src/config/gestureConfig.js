/**
 * SignBridge Gesture Recognition Configuration & Thresholds.
 * Centralizes all state machine timings, confidence limits, and stability parameters.
 *
 * Calibrated for a ~30 FPS camera feed. Everything tunable lives here.
 */

export const GESTURE_CONFIG = {
  // 1. Fast commit (speed comes from removing waiting, not from loosening accuracy checks)
  NEUTRAL_HOLD_FRAMES: 4,            // Frames (~0.13s) of 'two open palms' to arm (was 10)
  STABILITY_FRAMES_K: 3,             // Consecutive identical valid predictions (~0.1s) to commit (was 5)
  INSTANT_COMMIT_CONFIDENCE: 0.92,   // A single frame at/above this (and passing margin + centroid checks) commits immediately
  MAX_MISS_FRAMES: 1,                // Invalid/dropped frames tolerated mid-streak before the streak resets

  // 2. Lock, then hold steady (replaces the old fixed COOLDOWN_MS = 1000)
  MIN_LOCK_MS: 250,                  // Minimum time the locked sign stays visible
  STEADY_MS: 350,                    // Hand must be steady this long to re-arm
  STEADY_MOTION_THRESHOLD: 0.015,    // Mean landmark movement per frame (normalized 0..1) counted as "steady"
  MAX_COOLDOWN_MS: 1500,             // Failsafe: never stay locked longer than this
  REQUIRE_NEUTRAL_BETWEEN_SIGNS: false, // true = need the two-open-palms gesture before every sign

  // 3. Hand presence / same-sign repeat protection
  HAND_LOST_FRAMES: 3,               // Frames without an active hand before it counts as "left the frame"
  RELEASE_FRAMES: 4,                 // Frames of no/other sign after a lock before the same sign may be accepted again

  // 4. Confidence & Ambiguity Thresholds (calibrated for high accuracy and robust recognition)
  CONFIDENCE_THRESHOLD_LETTER: 0.75, // Top-class probability threshold for static letters
  CONFIDENCE_THRESHOLD_LETTER_RELAXED: 0.65, // Relaxed threshold when decisive margin (gap) is demonstrated
  CONFIDENCE_THRESHOLD_WORD: 0.50,   // Probability threshold for 57-class whole-word sequences
  MARGIN_THRESHOLD: 0.10,            // Minimum gap between top-1 and top-2 probabilities for letters
  MARGIN_THRESHOLD_RELAXED: 0.20,    // Decisive margin between top-1 and top-2 for confident classification
  MARGIN_THRESHOLD_WORD: 0.06,       // Minimum gap for words
  MIN_MOTION_VELOCITY: 0.025,        // Minimum wrist-relative joint displacement velocity for word gating
  MAX_WORD_ENTROPY: 2.20,            // Maximum distribution entropy tolerated before rejection
  MAX_CENTROID_DISTANCE: 11.5,       // Maximum Euclidean distance to class centroid

  // 5. Resting / Idle Hand Detection
  IDLE_WRIST_Y_THRESHOLD: 0.88,      // Wrist below 88% of camera frame treated as resting on desk
  MIN_HAND_SPAN: 0.035,              // Minimum bounding box dimension for an intentional gesture

  // 6. Debug & Developer Tools
  DEBUG_OVERLAY_ENABLED: false,      // Display live debug metrics HUD overlay on camera
};

export const GESTURE_STATES = {
  NEUTRAL: 'neutral',
  ARMED: 'armed',
  CAPTURING: 'capturing',
  COOLDOWN: 'cooldown', // "locked, hold steady"
};

export const GESTURE_STATUS_LABELS = {
  neutral: 'Show neutral to continue',
  armed: 'Ready - show the next sign',
  capturing: 'Recognising...',
  cooldown: 'Sign locked - hold your hand steady',
};
