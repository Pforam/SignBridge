import { useState, useRef, useCallback, useEffect } from 'react';
import { GESTURE_CONFIG, GESTURE_STATES, GESTURE_STATUS_LABELS } from '../config/gestureConfig';

/**
 * Gesture Recognition State Machine with Temporal Smoothing & Dual Mode (Letter & Word).
 *
 * Lifecycle: NEUTRAL -> ARMED -> CAPTURING -> COOLDOWN -> back to NEUTRAL
 * Re-arm base gesture: Two open palms facing camera held for N consecutive frames.
 */

const API_BASE = '/api';

/**
 * Fixed-capacity ring buffer — O(1) push, zero shifting, zero allocation after init.
 * Replaces frameBufferRef's plain array + Array.shift() pattern.
 */
class RingBuffer {
  constructor(capacity) {
    this._cap = capacity;
    this._buf = new Array(capacity);
    this._head = 0; // next write slot
    this._len  = 0;
  }
  push(item) {
    this._buf[this._head] = item;
    this._head = (this._head + 1) % this._cap;
    if (this._len < this._cap) this._len++;
  }
  get length() { return this._len; }
  /** Most-recently pushed item. */
  last()  { return this._len === 0 ? undefined : this._buf[(this._head - 1 + this._cap) % this._cap]; }
  /** Second-to-last item. */
  last2() { return this._len < 2  ? undefined : this._buf[(this._head - 2 + this._cap) % this._cap]; }
  /** Return items in insertion order (oldest → newest). */
  toArray() {
    if (this._len === 0) return [];
    const start = (this._head - this._len + this._cap) % this._cap;
    if (start + this._len <= this._cap) {
      return this._buf.slice(start, start + this._len);
    }
    return this._buf.slice(start).concat(this._buf.slice(0, this._head));
  }
  clear() { this._head = 0; this._len = 0; }
}

const STATES = {
  IDLE: 'idle',
  SEARCHING: 'searching',
  DETECTED: 'detected',
  TRACKING: 'tracking',
  RECOGNISING: 'recognising',
  STABLE: 'stable',
};

const STATUS_LABELS = {
  idle: 'Ready',
  searching: 'Searching for hands...',
  detected: 'Hands Detected',
  tracking: 'Tracking Keypoints...',
  recognising: 'Recognising Sign...',
  stable: 'Sign Locked',
};

const WORD_SEQUENCE_LENGTH = 30;

export function isSingleFingerExtended(pts, tipIdx, pipIdx, mcpIdx) {
  const wx = pts[0], wy = pts[1];
  const tx = pts[tipIdx * 3], ty = pts[tipIdx * 3 + 1];
  const px = pts[pipIdx * 3], py = pts[pipIdx * 3 + 1];
  const mx = pts[mcpIdx * 3], my = pts[mcpIdx * 3 + 1];

  const dTipW = Math.hypot(tx - wx, ty - wy);
  const dPipW = Math.hypot(px - wx, py - wy);
  const dTipM = Math.hypot(tx - mx, ty - my);
  const dPipM = Math.hypot(px - mx, py - my);

  return (dTipW > 1.15 * dPipW) && (dTipM > dPipM);
}

export function isThumbExtended(pts) {
  const wx = pts[0], wy = pts[1];
  const tx = pts[4 * 3], ty = pts[4 * 3 + 1];
  const mx = pts[2 * 3], my = pts[2 * 3 + 1];
  const ix = pts[5 * 3], iy = pts[5 * 3 + 1];

  const dTipW = Math.hypot(tx - wx, ty - wy);
  const dMcpW = Math.hypot(mx - wx, my - wy);
  const dTipIdx = Math.hypot(tx - ix, ty - iy);

  return (dTipW > 1.10 * dMcpW) && (dTipIdx > 0.04);
}

export function isPalmFacingCamera(pts) {
  const wx = pts[0], wy = pts[1], wz = pts[2] || 0;
  const ix = pts[5 * 3], iy = pts[5 * 3 + 1], iz = pts[5 * 3 + 2] || 0;
  const px = pts[17 * 3], py = pts[17 * 3 + 1], pz = pts[17 * 3 + 2] || 0;

  const v1x = ix - wx, v1y = iy - wy, v1z = iz - wz;
  const v2x = px - wx, v2y = py - wy, v2z = pz - wz;

  const crossZ = v1x * v2y - v1y * v2x;
  const palmWidth = Math.hypot(ix - px, iy - py);
  return palmWidth > 0.030 && Math.abs(crossZ) > 0.0008;
}

export function isNeutralGesture(landmarks) {
  if (!landmarks || landmarks.length < 126) return false;
  const leftHand = landmarks.slice(0, 63);
  const rightHand = landmarks.slice(63, 126);

  const hasLeft = leftHand.some((v, i) => i % 3 === 0 && (v !== 0 || leftHand[i + 1] !== 0));
  const hasRight = rightHand.some((v, i) => i % 3 === 0 && (v !== 0 || rightHand[i + 1] !== 0));
  if (!hasLeft || !hasRight) return false;

  for (const hand of [leftHand, rightHand]) {
    if (!isPalmFacingCamera(hand)) return false;
    if (!isThumbExtended(hand)) return false;
    if (!isSingleFingerExtended(hand, 8, 6, 5)) return false;
    if (!isSingleFingerExtended(hand, 12, 10, 9)) return false;
    if (!isSingleFingerExtended(hand, 16, 14, 13)) return false;
    if (!isSingleFingerExtended(hand, 20, 18, 17)) return false;
  }
  return true;
}

export function computeLandmarkMotion(curr, prev) {
  if (!curr || !prev || curr.length < 126 || prev.length < 126) return 0;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < 42; i++) {
    const idx = i * 3;
    const x1 = curr[idx], y1 = curr[idx + 1];
    const x2 = prev[idx], y2 = prev[idx + 1];
    if ((x1 !== 0 || y1 !== 0) && (x2 !== 0 || y2 !== 0)) {
      sum += Math.hypot(x1 - x2, y1 - y2);
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

class LatencyProfiler {
  constructor(windowSize = 30) {
    this.windowSize = windowSize;
    this.history = [];
  }
  record(metrics) {
    this.history.push({ ...metrics, time: performance.now() });
    if (this.history.length > this.windowSize) {
      this.history.shift();
    }
  }
  getStats() {
    if (this.history.length === 0) return { avg: 0, p95: 0, latest: 0 };
    const totals = this.history.map(h => h.total || 0).sort((a, b) => a - b);
    const sum = totals.reduce((acc, v) => acc + v, 0);
    const avg = Math.round(sum / totals.length);
    const p95Idx = Math.min(totals.length - 1, Math.floor(totals.length * 0.95));
    const p95 = Math.round(totals[p95Idx]);
    const latest = Math.round(totals[totals.length - 1] || 0);
    return { avg, p95, latest };
  }
}

let audioContextInstance = null;
export function playLockFeedback() {
  try {
    if (typeof window !== 'undefined' && 'navigator' in window && typeof navigator.vibrate === 'function') {
      navigator.vibrate(15);
    }
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {
      if (!audioContextInstance || audioContextInstance.state === 'closed') {
        audioContextInstance = new AudioCtx();
      }
      if (audioContextInstance.state === 'suspended') {
        audioContextInstance.resume();
      }
      const osc = audioContextInstance.createOscillator();
      const gain = audioContextInstance.createGain();
      osc.type = 'sine';
      const t = audioContextInstance.currentTime;
      osc.frequency.setValueAtTime(880, t);
      osc.frequency.exponentialRampToValueAtTime(1320, t + 0.04);
      gain.gain.setValueAtTime(0.04, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.04);
      osc.connect(gain);
      gain.connect(audioContextInstance.destination);
      osc.start(t);
      osc.stop(t + 0.04);
    }
  } catch {
    // Non-blocking fallback
  }
}

export function isIdleHand(landmarks) {
  if (!landmarks || landmarks.length < 126) return true;
  let activeHands = 0;
  for (let h = 0; h < 2; h++) {
    const offset = h * 63;
    const wx = landmarks[offset], wy = landmarks[offset + 1];
    if (wx !== 0 || wy !== 0) {
      if (wy > GESTURE_CONFIG.IDLE_WRIST_Y_THRESHOLD) continue;
      let minX = 1, maxX = 0, minY = 1, maxY = 0;
      for (let i = 0; i < 21; i++) {
        const x = landmarks[offset + i * 3];
        const y = landmarks[offset + i * 3 + 1];
        if (x !== 0 || y !== 0) {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
      if ((maxX - minX) < GESTURE_CONFIG.MIN_HAND_SPAN || (maxY - minY) < GESTURE_CONFIG.MIN_HAND_SPAN) {
        continue;
      }
      activeHands++;
    }
  }
  return activeHands === 0;
}

function isValidHandGeometry(landmarks) {
  if (!landmarks || landmarks.length < 126) return false;
  let activePoints = 0;
  let xMin = 1, xMax = 0, yMin = 1, yMax = 0;
  for (let i = 0; i < 42; i++) {
    const x = landmarks[i * 3];
    const y = landmarks[i * 3 + 1];
    if (x !== 0 || y !== 0) {
      activePoints++;
      xMin = Math.min(xMin, x);
      xMax = Math.max(xMax, x);
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    }
  }
  if (activePoints < 21) return false;
  if ((xMax - xMin) < 0.03 || (yMax - yMin) < 0.03) return false;
  return true;
}

function formatWordForSentence(rawWord) {
  if (!rawWord) return '';
  // Convert compound names with underscores (e.g. 'GOOD_MORNING' -> 'GOOD MORNING') into natural words
  return rawWord.replace(/_/g, ' ').trim();
}

function isHandActive(frame, offset) {
  return Boolean(frame && (frame[offset] !== 0 || frame[offset + 1] !== 0));
}

function getFrameHandCentroid(frame, offset) {
  const indices = [0, 5, 9, 13, 17];
  let x = 0;
  let y = 0;
  for (const idx of indices) {
    x += frame[offset + idx * 3];
    y += frame[offset + idx * 3 + 1];
  }
  return { x: x / 5, y: y / 5 };
}

function swapHandsInFrame(frame) {
  const swapped = new Array(126);
  for (let i = 0; i < 63; i++) {
    swapped[i] = frame[63 + i];
    swapped[63 + i] = frame[i];
  }
  return swapped;
}

export function useGestureRecognition({ enabled = false, initialMode = 'letter', _videoElement = null, onSendMessage = null } = {}) {
  const [recognitionMode, setRecognitionMode] = useState(initialMode); // 'letter' | 'word'
  const [status, setStatus] = useState(STATES.IDLE);
  const [detectedLetter, setDetectedLetter] = useState(null);
  const [detectedWord, setDetectedWord] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [sentenceBuffer, setSentenceBuffer] = useState('');
  const [allScores, setAllScores] = useState({});
  const [handInfo, setHandInfo] = useState(null); // { count, label }
  const [guidance, setGuidance] = useState(null);
  const [wordBufferCount, setWordBufferCount] = useState(0);
  const [availableWords, setAvailableWords] = useState([
    'AGAIN', 'BAD', 'BOY', 'BYE_BYE', 'CHILD', 'CORRECT', 'DAY', 'DEAF', 'DIFFICULT', 'DOCTOR',
    'EASY', 'FEAR', 'FOOD', 'GIRL', 'GOOD', 'GOOD_AFTERNOON', 'GOOD_EVENING', 'GOOD_MORNING', 'GOOD_NIGHT',
    'HE', 'HEARING', 'HELLO', 'HELP', 'HOW_ARE_YOU', 'IM_FINE', 'INDIA', 'I_DONT_UNDERSTAND',
    'LANGUAGE', 'MAN', 'ME', 'MORNING', 'MY_NAME_IS', 'NAMASTE', 'NO', 'NO_FEAR', 'PEACE',
    'PLEASE', 'PRACTICE', 'REMEMBER', 'SHE', 'SIGN', 'SORRY', 'STRONG', 'TEACHER', 'THANK_YOU',
    'THANK_YOU_VERY_MUCH', 'THIN', 'UNDERSTAND', 'WASHROOM', 'WATER', 'WEAK', 'WELCOME', 'WHERE',
    'WOMAN', 'WRONG', 'YES', 'YOU'
  ]);

  // Inactivity / Hand-Drop Auto-Send states (5-second countdown timer)
  const [autoSendEnabled, setAutoSendEnabled] = useState(true);
  const [autoSendTimeoutMs, setAutoSendTimeoutMs] = useState(5000);
  const [inactivityCountdown, setInactivityCountdown] = useState(null);
  const [isAutoSending, setIsAutoSending] = useState(false);

  // Synchronous Sentence Buffer & Ref
  const sentenceBufferRef = useRef('');
  sentenceBufferRef.current = sentenceBuffer;
  const setSentenceBufferSync = useCallback((updater) => {
    setSentenceBuffer(prev => {
      const next = typeof updater === 'function' ? updater(prev) : updater;
      sentenceBufferRef.current = next;
      return next;
    });
  }, []);

  // Auto-Space Between Words states & stable refs
  const [autoSpaceWords, setAutoSpaceWords] = useState(true);
  const [autoWordSpaceDelayMs, setAutoWordSpaceDelayMs] = useState(2500);
  const autoSpaceTimerRef = useRef(null);
  const autoSpaceWordsRef = useRef(autoSpaceWords);
  const autoWordSpaceDelayMsRef = useRef(autoWordSpaceDelayMs);

  // Clear all pending timers on unmount (ponytail: prevent memory leaks and post-unmount updates)
  useEffect(() => {
    return () => {
      if (autoSpaceTimerRef.current) clearTimeout(autoSpaceTimerRef.current);
      if (autoSendTimerRef.current) clearTimeout(autoSendTimerRef.current);
      if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    };
  }, []);

  useEffect(() => {
    autoSpaceWordsRef.current = autoSpaceWords;
  }, [autoSpaceWords]);

  useEffect(() => {
    autoWordSpaceDelayMsRef.current = autoWordSpaceDelayMs;
  }, [autoWordSpaceDelayMs]);

  const clearAutoSpaceTimer = useCallback(() => {
    if (autoSpaceTimerRef.current) {
      clearTimeout(autoSpaceTimerRef.current);
      autoSpaceTimerRef.current = null;
    }
  }, []);

  // Fast "Lock, then Hold Steady" State Machine
  const initialGestureState = GESTURE_CONFIG.REQUIRE_NEUTRAL_BETWEEN_SIGNS ? GESTURE_STATES.NEUTRAL : GESTURE_STATES.ARMED;
  const [gestureState, setGestureState] = useState(initialGestureState);
  const gestureStateRef = useRef(initialGestureState);
  const [lockedSign, setLockedSign] = useState(null);
  const lockedSignRef = useRef(null);
  const [steadyProgress, setSteadyProgress] = useState(0.0);
  const steadyProgressRef = useRef(0.0);
  const steadyDurationMsRef = useRef(0.0);
  const lockedAtMsRef = useRef(0);
  const prevLandmarksRef = useRef(null);
  const lastUpdateMsRef = useRef(0);
  const releaseFramesRef = useRef(GESTURE_CONFIG.RELEASE_FRAMES);
  const missFramesRef = useRef(0);
  const lastCommittedSignRef = useRef(null);
  const profilerRef = useRef(new LatencyProfiler(30));
  const [latencyMetrics, setLatencyMetrics] = useState({ avg: 0, p95: 0, latest: 0 });

  const neutralHoldCountRef = useRef(0);
  const candidateStreakRef = useRef(0);
  const currentCandidateRef = useRef(null);
  const recentLetterVotesRef = useRef([]); // sliding window for majority voting
  const pendingWordRef = useRef(null); // holds captured word until return to neutral

  // Status label for UI
  const statusLabel = gestureState === GESTURE_STATES.COOLDOWN
    ? (lockedSign ? `Detected: ${lockedSign} - Hold your hand steady` : (GESTURE_STATUS_LABELS[gestureState] || 'Sign locked'))
    : (GESTURE_STATUS_LABELS[gestureState] || 'Ready');

  // Temporal smoothing & buffering refs
  const consecutiveLetterRef = useRef({ letter: null, count: 0 });
  const consecutiveWordRef = useRef({ word: null, count: 0 });
  const lastCommitRef = useRef(0);
  const lastCommittedItemRef = useRef(null);
  const frameBufferRef = useRef(new RingBuffer(WORD_SEQUENCE_LENGTH));
  const wordInferenceCooldownRef = useRef(0);
  const letterInferenceCooldownRef = useRef(0);
  const isRequestPendingRef = useRef(false);
  const isWordRequestPendingRef = useRef(false);
  const handsMissingCountRef = useRef(0);
  const bodyAnchorsRef = useRef(null);
  const wsRef = useRef(null);
  const pendingWsRequestsRef = useRef(new Map());
  const pendingReqIdRef = useRef(0);

  // Debug Overlay Metrics
  const [debugOverlay, setDebugOverlay] = useState(GESTURE_CONFIG.DEBUG_OVERLAY_ENABLED);
  const [debugInfo, setDebugInfo] = useState({
    state: 'neutral',
    confidence: 0,
    margin: 0,
    distance: 0,
    neutralStreak: 0,
    candidateStreak: 0,
    isNeutral: false,
    isIdle: true,
  });
  const toggleDebugOverlay = useCallback(() => setDebugOverlay(prev => !prev), []);

  // Stabilize detected sign and onSendMessage across renders
  const detectedRef = useRef(null);
  useEffect(() => {
    detectedRef.current = detectedWord || detectedLetter;
  }, [detectedWord, detectedLetter]);

  const onSendMessageRef = useRef(onSendMessage);
  useEffect(() => {
    onSendMessageRef.current = onSendMessage;
  }, [onSendMessage]);

  // Inactivity auto-send timers & refs
  const autoSendTimerRef = useRef(null);
  const countdownIntervalRef = useRef(null);
  const isAutoSendingRef = useRef(false);

  const cancelInactivityCountdown = useCallback(() => {
    if (autoSendTimerRef.current) {
      clearTimeout(autoSendTimerRef.current);
      autoSendTimerRef.current = null;
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    setInactivityCountdown(null);
  }, []);

  // Cancel any running countdown immediately if user disables auto-send
  useEffect(() => {
    if (!autoSendEnabled) {
      cancelInactivityCountdown();
    }
  }, [autoSendEnabled, cancelInactivityCountdown]);

  const triggerAutoSend = useCallback(async () => {
    const rawText = sentenceBufferRef.current.trim();
    if (!rawText || isAutoSendingRef.current) {
      cancelInactivityCountdown();
      return;
    }

    isAutoSendingRef.current = true;
    setIsAutoSending(true);
    cancelInactivityCountdown();

    try {
      let speechText = rawText;
      // 1. Refine with LLM (Groq / Gemini)
      try {
        const res = await fetch(`${API_BASE}/llm/refine`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: rawText })
        });
        if (res.ok) {
          const data = await res.json();
          if (data.refined_sentence && data.refined_sentence.trim()) {
            speechText = data.refined_sentence.trim();
          }
        }
      } catch (refineErr) {
        console.debug('Auto-send LLM refine notice:', refineErr);
      }

      // Sanitize text from any residual thinking tokens
      speechText = speechText
        .replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '')
        .replace(/^(?:Here'?s (?:a )?thinking process:?|Thinking Process:?)[\s\S]*?(?=\n\n|\n|$)/gi, '')
        .trim() || rawText;

      // 2. Speak aloud via Web Speech TTS
      if (speechText && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel(); // Stop any previous speech
        const utterance = new SpeechSynthesisUtterance(speechText);
        utterance.rate = 0.95;
        window.speechSynthesis.speak(utterance);
      }

      // 3. Send message to the Dialogue Chat Thread
      if (onSendMessageRef.current) {
        onSendMessageRef.current(speechText, 'human');
      }

      // 4. Remove only the sent prefix from draft buffer, keeping any newly signed characters
      setSentenceBufferSync(prev => {
        const trimmed = prev.trim();
        if (trimmed.startsWith(rawText)) {
          return trimmed.slice(rawText.length).trimStart();
        }
        return '';
      });
    } catch (e) {
      console.warn('Auto-send execution error:', e);
    } finally {
      isAutoSendingRef.current = false;
      setIsAutoSending(false);
      cancelInactivityCountdown();
    }
  }, [cancelInactivityCountdown, setSentenceBufferSync]);

  /**
   * Start or maintain the 5-second countdown timer when no gesture is detected.
   * If forceRestart is true, restarts the timer at 5.0s.
   * If already running and forceRestart is false, lets the timer continue counting down.
   */
  const startOrResetInactivityTimer = useCallback((forceRestart = false) => {
    const rawText = sentenceBufferRef.current.trim();
    if (!autoSendEnabled || !rawText || isAutoSendingRef.current) {
      return;
    }

    // If timer is already running and forceRestart is false, do not disrupt the countdown
    if (autoSendTimerRef.current && !forceRestart) {
      return;
    }

    // Reset any existing timer & interval
    if (autoSendTimerRef.current) {
      clearTimeout(autoSendTimerRef.current);
      autoSendTimerRef.current = null;
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }

    const startTime = Date.now();
    const targetTime = startTime + autoSendTimeoutMs;

    setInactivityCountdown((autoSendTimeoutMs / 1000).toFixed(1));

    countdownIntervalRef.current = setInterval(() => {
      const remainingMs = targetTime - Date.now();
      if (remainingMs <= 0) {
        if (countdownIntervalRef.current) {
          clearInterval(countdownIntervalRef.current);
          countdownIntervalRef.current = null;
        }
        setInactivityCountdown('0.0');
      } else {
        setInactivityCountdown((remainingMs / 1000).toFixed(1));
      }
    }, 100);

    autoSendTimerRef.current = setTimeout(() => {
      autoSendTimerRef.current = null;
      triggerAutoSend();
    }, autoSendTimeoutMs);
  }, [autoSendEnabled, autoSendTimeoutMs, triggerAutoSend]);

  const handleHandsDropped = useCallback(() => {
    // If auto-space is enabled, automatically insert space after the last completed word
    if (autoSpaceWordsRef.current) {
      clearAutoSpaceTimer();
      setSentenceBufferSync(prev => {
        const trimmed = prev.trimEnd();
        if (trimmed.length > 0 && !prev.endsWith(' ')) {
          return trimmed + ' ';
        }
        return prev;
      });
      lastCommittedItemRef.current = null;
    }

    // No gesture detected (hands dropped): start or continue 5-second countdown
    startOrResetInactivityTimer(false);
  }, [startOrResetInactivityTimer, clearAutoSpaceTimer, setSentenceBufferSync]);

  // Fetch model information & available word classes
  useEffect(() => {
    async function fetchModelInfo() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          const data = await res.json();
          if (data.word_recognizer_available) {
            const infoRes = await fetch(`${API_BASE}/words/info`);
            if (infoRes.ok) {
              const infoData = await infoRes.json();
              if (infoData.labels && infoData.labels.length > 0) {
                setAvailableWords(infoData.labels);
              }
            }
          }
        }
      } catch (e) {
        console.warn('Could not fetch backend model info:', e);
      }
    }
    fetchModelInfo();
  }, []);

  // Reset state when disabled or when mode switches
  useEffect(() => {
    if (!enabled) {
      setStatus(STATES.IDLE);
      setDetectedLetter(null);
      setDetectedWord(null);
      setConfidence(0);
      consecutiveLetterRef.current = { letter: null, count: 0 };
      consecutiveWordRef.current = { word: null, count: 0 };
      lastCommittedItemRef.current = null;
      frameBufferRef.current.clear();
      setWordBufferCount(0);
      cancelInactivityCountdown();
    } else {
      setStatus(STATES.SEARCHING);
      frameBufferRef.current.clear();
      setWordBufferCount(0);
    }
  }, [enabled, recognitionMode, cancelInactivityCountdown]);

  // Persistent WebSocket connection for sub-2ms low-latency inference
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;
    let isCancelled = false;

    function connect() {
      if (isCancelled) return;
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Connect via same-origin /ws (handled by Vite proxy in dev, Nginx in Docker/k8s)
        const wsUrl = window.location.port === '5000'
          ? `${protocol}//${window.location.hostname}:5001`
          : `${protocol}//${window.location.host}/ws`;

        ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            const reqId = data.req_id;
            if (reqId !== undefined && pendingWsRequestsRef.current.has(reqId)) {
              const resolver = pendingWsRequestsRef.current.get(reqId);
              pendingWsRequestsRef.current.delete(reqId);
              resolver(data);
            }
          } catch (e) {
            console.debug('WS parse notice:', e);
          }
        };

        ws.onclose = () => {
          if (wsRef.current === ws) {
            wsRef.current = null;
          }
          if (!isCancelled) {
            reconnectTimeout = setTimeout(connect, 2000);
          }
        };

        ws.onerror = () => {
          try { ws.close(); } catch {}
        };
      } catch {
        if (!isCancelled) {
          reconnectTimeout = setTimeout(connect, 3000);
        }
      }
    }

    connect();

    return () => {
      isCancelled = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (ws) {
        try { ws.close(1000, 'unmount'); } catch {}
      }
      wsRef.current = null;
      pendingWsRequestsRef.current.clear();
    };
  }, []);

  /**
   * Send inference request over WebSocket pipe (1-3ms) with seamless HTTP fallback.
   */
  const predictViaSocketOrHttp = useCallback(async (payload, httpEndpoint) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      return new Promise((resolve) => {
        const reqId = ++pendingReqIdRef.current;
        payload.req_id = reqId;

        const timeout = setTimeout(async () => {
          pendingWsRequestsRef.current.delete(reqId);
          // Fallback to HTTP if socket request times out
          try {
            const res = await fetch(`${API_BASE}${httpEndpoint}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            });
            resolve(res.ok ? await res.json() : null);
          } catch {
            resolve(null);
          }
        }, 300);

        pendingWsRequestsRef.current.set(reqId, (data) => {
          clearTimeout(timeout);
          resolve(data);
        });

        ws.send(JSON.stringify(payload));
      });
    }

    // Default HTTP fallback
    try {
      const res = await fetch(`${API_BASE}${httpEndpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      return res.ok ? await res.json() : null;
    } catch {
      return null;
    }
  }, []);

  /**
   * Process a frame of 126 landmark floats from MediaPipe Holistic (with upper-body anchors).
   */
  const processLandmarks = useCallback(async (landmarks, handCount = 0, handedness = null, holisticData = null) => {
    if (!enabled) return null;

    if (holisticData && holisticData.bodyAnchors) {
      bodyAnchorsRef.current = holisticData.bodyAnchors;
    }

    const frameStart = performance.now();
    const captureTime = (holisticData && holisticData.captureTime) || frameStart;

    const hasActiveLandmarks = Boolean(handCount > 0 && isValidHandGeometry(landmarks));
    const nowTime = Date.now();

    // Compute motion between frames
    const motion = hasActiveLandmarks ? computeLandmarkMotion(landmarks, prevLandmarksRef.current) : 0;
    if (hasActiveLandmarks) {
      prevLandmarksRef.current = landmarks;
    }
    const dt = lastUpdateMsRef.current > 0 ? Math.max(1, Math.min(100, nowTime - lastUpdateMsRef.current)) : 33;
    lastUpdateMsRef.current = nowTime;

    // Update hand status with tracking grace period
    if (handCount > 0 && hasActiveLandmarks) {
      handsMissingCountRef.current = 0;
      const label = handedness || (handCount >= 2 ? 'Both Hands (ISL)' : 'Single Hand');
      setHandInfo(p => (p && p.count === handCount && p.label === label ? p : { count: handCount, label }));
      setStatus(prev => (prev === STATES.DETECTED ? prev : STATES.DETECTED));
    } else {
      handsMissingCountRef.current += 1;
      releaseFramesRef.current += 1;

      // In COOLDOWN state: if hand lost for HAND_LOST_FRAMES, immediately re-arm!
      if (gestureStateRef.current === GESTURE_STATES.COOLDOWN && handsMissingCountRef.current >= GESTURE_CONFIG.HAND_LOST_FRAMES) {
        const nextState = GESTURE_CONFIG.REQUIRE_NEUTRAL_BETWEEN_SIGNS ? GESTURE_STATES.NEUTRAL : GESTURE_STATES.ARMED;
        gestureStateRef.current = nextState;
        setGestureState(nextState);
        lockedSignRef.current = null;
        setLockedSign(null);
        steadyProgressRef.current = 0;
        setSteadyProgress(0);
        steadyDurationMsRef.current = 0;
        releaseFramesRef.current = GESTURE_CONFIG.RELEASE_FRAMES;
      }

      // Grace period: allow up to 12 dropped tracking frames (~400ms) without clearing the word buffer
      if (handsMissingCountRef.current < 12 && frameBufferRef.current.length > 0) {
        return null; // Retain buffer during momentary tracking flicker
      }
      setHandInfo(null);
      setStatus(prev => (prev === STATES.SEARCHING ? prev : STATES.SEARCHING));
      setDetectedLetter(null);
      setDetectedWord(null);
      setConfidence(0);
      setAllScores({});
      setGuidance(prev => (prev === 'Show your hands inside the camera frame.' ? prev : 'Show your hands inside the camera frame.'));
      
      if (gestureStateRef.current !== GESTURE_STATES.COOLDOWN) {
        const nextState = GESTURE_CONFIG.REQUIRE_NEUTRAL_BETWEEN_SIGNS ? GESTURE_STATES.NEUTRAL : GESTURE_STATES.ARMED;
        gestureStateRef.current = nextState;
        setGestureState(nextState);
        neutralHoldCountRef.current = 0;
        currentCandidateRef.current = null;
        candidateStreakRef.current = 0;
        missFramesRef.current = 0;
      }

      lastCommittedItemRef.current = null;
      frameBufferRef.current.clear();
      setWordBufferCount(0);

      // Trigger hand-drop countdown if sentence buffer has words
      handleHandsDropped();
      return null;
    }

    // Evaluate Neutral & Idle hand condition for State Machine
    const isNeutral = isNeutralGesture(landmarks);
    const isIdle = isIdleHand(landmarks);
    const curState = gestureStateRef.current;

    // 1. COOLDOWN state transition (Locked, Hold Steady)
    if (curState === GESTURE_STATES.COOLDOWN) {
      const elapsed = nowTime - lockedAtMsRef.current;

      const isReleasing = isNeutral || (motion > GESTURE_CONFIG.STEADY_MOTION_THRESHOLD);
      if (isReleasing) {
        releaseFramesRef.current += 1;
      }

      // Failsafe: never stay locked longer than MAX_COOLDOWN_MS
      if (elapsed >= GESTURE_CONFIG.MAX_COOLDOWN_MS) {
        const nextState = GESTURE_CONFIG.REQUIRE_NEUTRAL_BETWEEN_SIGNS ? GESTURE_STATES.NEUTRAL : GESTURE_STATES.ARMED;
        gestureStateRef.current = nextState;
        setGestureState(nextState);
        lockedSignRef.current = null;
        setLockedSign(null);
        steadyDurationMsRef.current = 0;
        steadyProgressRef.current = 0;
        setSteadyProgress(0);
      } else {
        if (elapsed >= GESTURE_CONFIG.MIN_LOCK_MS) {
          if (releaseFramesRef.current >= GESTURE_CONFIG.RELEASE_FRAMES && isReleasing) {
            const nextState = GESTURE_CONFIG.REQUIRE_NEUTRAL_BETWEEN_SIGNS ? GESTURE_STATES.NEUTRAL : GESTURE_STATES.ARMED;
            gestureStateRef.current = nextState;
            setGestureState(nextState);
            lockedSignRef.current = null;
            setLockedSign(null);
            steadyDurationMsRef.current = 0;
            steadyProgressRef.current = 0;
            setSteadyProgress(0);
          } else {
            if (motion <= GESTURE_CONFIG.STEADY_MOTION_THRESHOLD) {
              steadyDurationMsRef.current += dt;
            } else {
              steadyDurationMsRef.current = Math.max(0, steadyDurationMsRef.current - 20);
            }
            const prog = Math.min(1.0, steadyDurationMsRef.current / GESTURE_CONFIG.STEADY_MS);
            steadyProgressRef.current = prog;
            setSteadyProgress(prog);

            if (prog >= 1.0) {
              const nextState = GESTURE_CONFIG.REQUIRE_NEUTRAL_BETWEEN_SIGNS ? GESTURE_STATES.NEUTRAL : GESTURE_STATES.ARMED;
              gestureStateRef.current = nextState;
              setGestureState(nextState);
              lockedSignRef.current = null;
              setLockedSign(null);
              steadyDurationMsRef.current = 0;
              steadyProgressRef.current = 0;
              setSteadyProgress(0);
            }
          }
        } else {
          steadyProgressRef.current = 0;
          setSteadyProgress(0);
        }
      }
    }
    // 2. NEUTRAL state transition (holding two open palms to re-arm)
    else if (curState === GESTURE_STATES.NEUTRAL) {
      currentCandidateRef.current = null;
      candidateStreakRef.current = 0;
      missFramesRef.current = 0;

      if (isNeutral) {
        neutralHoldCountRef.current += 1;
        if (neutralHoldCountRef.current >= GESTURE_CONFIG.NEUTRAL_HOLD_FRAMES) {
          gestureStateRef.current = GESTURE_STATES.ARMED;
          setGestureState(GESTURE_STATES.ARMED);
          neutralHoldCountRef.current = 0;
        }
      } else {
        neutralHoldCountRef.current = 0;
      }
    }
    // 3. ARMED state transition
    else if (curState === GESTURE_STATES.ARMED) {
      currentCandidateRef.current = null;
      candidateStreakRef.current = 0;
      missFramesRef.current = 0;

      if (hasActiveLandmarks && !isIdle && !isNeutral) {
        clearAutoSpaceTimer();
        gestureStateRef.current = GESTURE_STATES.CAPTURING;
        setGestureState(GESTURE_STATES.CAPTURING);
      }
    }
    // 4. CAPTURING state transitions
    else if (curState === GESTURE_STATES.CAPTURING) {
      if ((GESTURE_CONFIG.REQUIRE_NEUTRAL_BETWEEN_SIGNS && isNeutral) || isNeutral || isIdle) {
        // If in word mode and we captured a valid word, commit upon returning to NEUTRAL / resting
        if (recognitionMode === 'word' && pendingWordRef.current) {
          const pending = pendingWordRef.current;
          pendingWordRef.current = null;
          const formattedWord = formatWordForSentence(pending.word);
          const isDifferentWord = (lastCommittedItemRef.current !== pending.word);
          const canRepeat = (nowTime - lastCommitRef.current > 1800);

          if (nowTime - lastCommitRef.current > 900 && (isDifferentWord || canRepeat)) {
            setStatus(STATES.STABLE);
            lockedSignRef.current = pending.word;
            setLockedSign(pending.word);
            clearAutoSpaceTimer();
            setSentenceBufferSync(prev => {
              const clean = prev.trim();
              return clean.length > 0 ? `${clean} ${formattedWord}` : formattedWord;
            });
            lastCommitRef.current = nowTime;
            lastCommittedItemRef.current = pending.word;
            playLockFeedback();
          }
        }

        gestureStateRef.current = GESTURE_STATES.NEUTRAL;
        setGestureState(GESTURE_STATES.NEUTRAL);
        neutralHoldCountRef.current = 1;
        currentCandidateRef.current = null;
        candidateStreakRef.current = 0;
        missFramesRef.current = 0;
        recentLetterVotesRef.current = [];
      }
    }

    setDebugInfo(prev => ({
      ...prev,
      state: gestureStateRef.current,
      neutralStreak: neutralHoldCountRef.current,
      candidateStreak: candidateStreakRef.current,
      isNeutral,
      isIdle
    }));

    // Adaptive Guidance for Single-Hand vs Dual-Hand Gestures
    if (handCount === 1) {
      const g1 = 'Single hand detected (e.g. C, L, O, V, numbers 0–9)';
      setGuidance(prev => (prev === g1 ? prev : g1));
    } else if (handCount >= 2) {
      const guidanceMsg = (landmarks && landmarks.length >= 126 ? analyzeQuality(landmarks) : null) || 'Dual-hand ISL tracking active';
      setGuidance(prev => (prev === guidanceMsg ? prev : guidanceMsg));
    }

    setStatus(prev => (prev === STATES.TRACKING ? prev : STATES.TRACKING));

    // ─────────────────────────────────────────────────────────────────────────
    // MODE 1: WORD SEQUENCE RECOGNITION (Sliding 30-Frame Window)
    // ─────────────────────────────────────────────────────────────────────────
    if (recognitionMode === 'word') {
      // Maintain hand slot continuity across momentary occlusion or crossing
      let frameToPush = landmarks;
      if (frameBufferRef.current.length > 0 && landmarks && landmarks.length >= 126) {
        const prevFrame = frameBufferRef.current.last();
        const prevHasL = isHandActive(prevFrame, 0);
        const prevHasR = isHandActive(prevFrame, 63);
        const curHasL = isHandActive(landmarks, 0);
        const curHasR = isHandActive(landmarks, 63);

        if (prevHasL && prevHasR && curHasL && curHasR) {
          const prevL = getFrameHandCentroid(prevFrame, 0);
          const prevR = getFrameHandCentroid(prevFrame, 63);
          const curL = getFrameHandCentroid(landmarks, 0);
          const curR = getFrameHandCentroid(landmarks, 63);

          const distDirect = Math.hypot(prevL.x - curL.x, prevL.y - curL.y) + Math.hypot(prevR.x - curR.x, prevR.y - curR.y);
          const distCross = Math.hypot(prevL.x - curR.x, prevL.y - curR.y) + Math.hypot(prevR.x - curL.x, prevR.y - curL.y);

          if (distCross + 0.05 < distDirect && distDirect > 0.15) {
            frameToPush = swapHandsInFrame(landmarks);
          }
        } else if (!prevHasL && prevHasR && curHasL && !curHasR) {
          const prevR = getFrameHandCentroid(prevFrame, 63);
          const curL = getFrameHandCentroid(landmarks, 0);
          if (Math.hypot(prevR.x - curL.x, prevR.y - curL.y) < 0.20) {
            frameToPush = swapHandsInFrame(landmarks);
          }
        } else if (prevHasL && !prevHasR && !curHasL && curHasR) {
          const prevL = getFrameHandCentroid(prevFrame, 0);
          const curR = getFrameHandCentroid(landmarks, 63);
          if (Math.hypot(prevL.x - curR.x, prevL.y - curR.y) < 0.20) {
            frameToPush = swapHandsInFrame(landmarks);
          }
        }
      }

      // Append current frame to rolling sequence buffer
      frameBufferRef.current.push(frameToPush); // O(1) — ring buffer, no shift
      const currentBufLen = frameBufferRef.current.length;
      if (currentBufLen % 3 === 0 || currentBufLen >= WORD_SEQUENCE_LENGTH) {
        setWordBufferCount(currentBufLen);
      }

      // Start inference only while CAPTURING state is active and sequence has viable length (>= 10 frames)
      const now = Date.now();
      const isCapturing = (gestureStateRef.current === GESTURE_STATES.CAPTURING);
      const hasViableSequence = isCapturing && (frameBufferRef.current.length >= 10);
      const isCooldownOver = (now - wordInferenceCooldownRef.current >= 200); // ponytail: 200ms halves round-trips; lower when model inference < 50ms

      if (hasViableSequence && isCooldownOver && !isWordRequestPendingRef.current) {
        wordInferenceCooldownRef.current = now;
        isWordRequestPendingRef.current = true;

        // If sequence has 15-29 frames, linearly resample to exactly 30 frames for the model
        let framesToSend = frameBufferRef.current.toArray();
        if (framesToSend.length < WORD_SEQUENCE_LENGTH) {
          const N = framesToSend.length;
          const indices = Array.from({ length: WORD_SEQUENCE_LENGTH }, (_, i) =>
            Math.min(N - 1, Math.round(i * (N - 1) / (WORD_SEQUENCE_LENGTH - 1)))
          );
          framesToSend = indices.map(idx => framesToSend[idx]);
        }

        try {
          const prediction = await predictViaSocketOrHttp({
            type: 'word',
            frames: framesToSend,
            body_anchors: bodyAnchorsRef.current,
          }, '/translate/word');

          if (!prediction) {
            startOrResetInactivityTimer(false);
            return null;
          }

          // Handle confidence rejection from backend
          if (prediction.rejected || !prediction.word || prediction.word === '?' || prediction.confidence < GESTURE_CONFIG.CONFIDENCE_THRESHOLD_WORD) {
            setStatus(prev => (prev === STATES.TRACKING ? prev : STATES.TRACKING));
            setDetectedWord(null);
            setDetectedLetter(null);
            setConfidence(prev => (Math.abs(prev - (prediction.confidence || 0)) < 0.02 ? prev : (prediction.confidence || 0)));
            setGuidance(prev => (prev === 'Low confidence — adjust your hand position.' ? prev : 'Low confidence — adjust your hand position.'));
            consecutiveWordRef.current = { word: null, count: 0 };
            // No gesture detected: start or maintain 5-second countdown
            startOrResetInactivityTimer(false);
            return prediction;
          }

          // Active gesture detected: pause/cancel countdown
          cancelInactivityCountdown();
          setStatus(prev => (prev === STATES.RECOGNISING ? prev : STATES.RECOGNISING));
          setDetectedWord(prediction.word);
          setDetectedLetter(null);
          setConfidence(prediction.confidence);
          setAllScores(prediction.all_scores || {});

          // Temporal candidate tracking & stroke completion gating for whole words
          if (prediction.confidence >= GESTURE_CONFIG.CONFIDENCE_THRESHOLD_WORD) {
            pendingWordRef.current = {
              word: prediction.word,
              confidence: prediction.confidence,
              timestamp: now,
            };
            const prev = consecutiveWordRef.current;
            if (prev.word === prediction.word) {
              prev.count += 1;
            } else {
              consecutiveWordRef.current = { word: prediction.word, count: 1 };
            }

            // Measure recent kinetic velocity to detect gesture stroke completion / deceleration
            let strokeEnergy = 0;
            if (frameBufferRef.current.length >= 2) {
              const currF = frameBufferRef.current.last();
              const prevF = frameBufferRef.current.last2();
              let diffSum = 0;
              let validPts = 0;
              for (let p = 0; p < Math.min(currF.length, prevF.length); p += 3) {
                if (currF[p] !== 0 || prevF[p] !== 0) {
                  const dx = currF[p] - prevF[p];
                  const dy = currF[p + 1] - prevF[p + 1];
                  const dz = currF[p + 2] - prevF[p + 2];
                  diffSum += Math.sqrt(dx * dx + dy * dy + dz * dz);
                  validPts++;
                }
              }
              strokeEnergy = validPts > 0 ? diffSum / validPts : 0;
            }

            // Commit word when gesture decelerates (stroke completion) or has strong confidence / stability
            const isStrokeComplete = strokeEnergy < 0.080 || prediction.confidence >= 0.70 || consecutiveWordRef.current.count >= 2;

            if (isStrokeComplete && (consecutiveWordRef.current.count >= 2 || prediction.confidence >= 0.75)) {
              const formattedWord = formatWordForSentence(prediction.word);
              const isDifferentWord = (lastCommittedItemRef.current !== prediction.word);
              const canRepeatSameWord = (now - lastCommitRef.current > 1800);

              if (now - lastCommitRef.current > 1100 && (isDifferentWord || canRepeatSameWord)) {
                setStatus(STATES.STABLE);
                lockedSignRef.current = prediction.word;
                setLockedSign(prediction.word);
                clearAutoSpaceTimer();
                setSentenceBufferSync(prev => {
                  const clean = prev.trim();
                  // Automatically place space between words to form a word-level sentence
                  return clean.length > 0 ? `${clean} ${formattedWord}` : formattedWord;
                });
                lastCommitRef.current = now;
                lastCommittedItemRef.current = prediction.word;
                consecutiveWordRef.current = { word: null, count: 0 };
                frameBufferRef.current.clear(); // Clear buffer after successful word lock
                setWordBufferCount(0);
                playLockFeedback();
                // Word locked: start 5-second countdown to automatically send draft message
                startOrResetInactivityTimer(true);
              }
            }
          } else {
            consecutiveWordRef.current = { word: null, count: 0 };
          }

          return prediction;
        } catch (e) {
          console.warn('Word recognition error:', e);
          startOrResetInactivityTimer(false);
          return null;
        } finally {
          isWordRequestPendingRef.current = false;
        }
      } else {
        // Viable sequence not reached or cooldown active: check if waiting
        if (!detectedRef.current) {
          startOrResetInactivityTimer(false);
        }
      }
      return null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MODE 2: STATIC LETTER RECOGNITION (A-Z)
    // ─────────────────────────────────────────────────────────────────────────


    // Only execute letter recognition when in CAPTURING (or transitioning from ARMED)
    if (gestureStateRef.current === GESTURE_STATES.NEUTRAL || gestureStateRef.current === GESTURE_STATES.COOLDOWN) {
      setDetectedLetter(null);
      return null;
    }

    const now = Date.now();
    if (now - letterInferenceCooldownRef.current < 35 || isRequestPendingRef.current) return null;
    letterInferenceCooldownRef.current = now;
    isRequestPendingRef.current = true;

    try {
      const r = v => Math.round(v * 1e4) / 1e4;
      const roundedLandmarks = landmarks.map(r);
      const prediction = await predictViaSocketOrHttp({
        type: 'letter',
        landmarks: roundedLandmarks,
        body_anchors: bodyAnchorsRef.current,
      }, '/translate');

      if (!prediction) {
        startOrResetInactivityTimer(false);
        return null;
      }

      const conf = prediction.confidence || 0;
      const margin = prediction.margin !== undefined ? prediction.margin : 1.0;
      const dist = prediction.centroid_distance || 0;
      const decisionStart = performance.now();

      setDebugInfo({
        state: gestureStateRef.current,
        confidence: conf,
        margin: margin,
        distance: dist,
        neutralStreak: neutralHoldCountRef.current,
        candidateStreak: candidateStreakRef.current,
        isNeutral,
        isIdle
      });

      // Handle confidence rejection, margin rejection, or out-of-dataset centroid rejection
      const isHighConf = conf >= GESTURE_CONFIG.CONFIDENCE_THRESHOLD_LETTER && margin >= GESTURE_CONFIG.MARGIN_THRESHOLD;
      const isHighMargin = conf >= (GESTURE_CONFIG.CONFIDENCE_THRESHOLD_LETTER_RELAXED || 0.65) && margin >= (GESTURE_CONFIG.MARGIN_THRESHOLD_RELAXED || 0.20);
      const isVal = Boolean(
        !prediction.rejected &&
        prediction.letter &&
        prediction.letter !== '?' &&
        (isHighConf || isHighMargin) &&
        dist <= GESTURE_CONFIG.MAX_CENTROID_DISTANCE
      );

      if (!isVal) {
        setStatus(prev => (prev === STATES.TRACKING ? prev : STATES.TRACKING));
        setDetectedLetter(null);
        setDetectedWord(null);
        setConfidence(conf);
        setAllScores(prediction.all_scores || {});
        releaseFramesRef.current += 1;
        missFramesRef.current += 1;
        if (missFramesRef.current > GESTURE_CONFIG.MAX_MISS_FRAMES) {
          candidateStreakRef.current = 0;
          currentCandidateRef.current = null;
          recentLetterVotesRef.current = [];
        }
        startOrResetInactivityTimer(false);
        return prediction;
      }

      // Valid gesture detected: pause/cancel countdown and space timer
      cancelInactivityCountdown();
      clearAutoSpaceTimer();

      // Only inside CAPTURING state do we commit
      if (gestureStateRef.current === GESTURE_STATES.CAPTURING) {
        const label = prediction.letter;
        const isSameAsLast = (label === lastCommittedSignRef.current);
        const canRepeat = (!isSameAsLast) || (releaseFramesRef.current >= GESTURE_CONFIG.RELEASE_FRAMES);

        if (canRepeat) {
          let shouldCommit = false;

          // Push to sliding voting window (N = 5 frames)
          const votes = recentLetterVotesRef.current;
          votes.push({ label, conf, time: now });
          if (votes.length > 5) votes.shift();

          // Temporary debug log of top-3 predictions per frame for H
          if (label === 'H' || (prediction.all_scores && 'H' in prediction.all_scores)) {
            const sortedTop3 = Object.entries(prediction.all_scores || {})
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([letter, score]) => `${letter}:${score.toFixed(3)}`);
            console.debug(`[Letter H Debug] Top-1: ${label} (${conf.toFixed(3)}) | Top-3: [${sortedTop3.join(', ')}]`);
          }

          // Rule 1: Instant commit on single high-confidence frame
          if (conf >= GESTURE_CONFIG.INSTANT_COMMIT_CONFIDENCE) {
            shouldCommit = true;
          } else {
            // Rule 2: Temporal majority voting across sliding window of N=5 frames
            const counts = {};
            let majorityLabel = null;
            let maxCount = 0;
            for (const v of votes) {
              counts[v.label] = (counts[v.label] || 0) + 1;
              if (counts[v.label] > maxCount) {
                maxCount = counts[v.label];
                majorityLabel = v.label;
              }
            }

            // Require majority (>= 3 out of 5 frames, 60%) or K consecutive identical frames
            if (majorityLabel === label && maxCount >= 3) {
              shouldCommit = true;
            } else if (label === currentCandidateRef.current) {
              candidateStreakRef.current += 1;
              missFramesRef.current = 0;
              if (candidateStreakRef.current >= GESTURE_CONFIG.STABILITY_FRAMES_K) {
                shouldCommit = true;
              }
            } else {
              currentCandidateRef.current = label;
              candidateStreakRef.current = 1;
              missFramesRef.current = 0;
            }
          }

          // Display detected letter preview
          setStatus(STATES.RECOGNISING);
          setDetectedLetter(label);
          setDetectedWord(null);
          setConfidence(conf);
          setAllScores(prediction.all_scores || {});

          if (shouldCommit) {
            // LOCK SIGN!
            lockedSignRef.current = label;
            setLockedSign(label);
            lastCommittedSignRef.current = label;
            lastCommittedItemRef.current = label;
            releaseFramesRef.current = 0;
            lockedAtMsRef.current = now;
            steadyDurationMsRef.current = 0;
            steadyProgressRef.current = 0;
            setSteadyProgress(0);

            // Move to COOLDOWN (Locked, Hold Steady)
            gestureStateRef.current = GESTURE_STATES.COOLDOWN;
            setGestureState(GESTURE_STATES.COOLDOWN);
            currentCandidateRef.current = null;
            candidateStreakRef.current = 0;
            missFramesRef.current = 0;
            recentLetterVotesRef.current = [];

            setStatus(STATES.STABLE);
            clearAutoSpaceTimer();
            setSentenceBufferSync(prevBuf => prevBuf + label);
            lastCommitRef.current = now;
            playLockFeedback();
            startOrResetInactivityTimer(true);

            if (autoSpaceWordsRef.current) {
              autoSpaceTimerRef.current = setTimeout(() => {
                setSentenceBufferSync(prev => {
                  const trimmed = prev.trimEnd();
                  return trimmed.length > 0 && !prev.endsWith(' ') ? trimmed + ' ' : prev;
                });
                lastCommittedItemRef.current = null;
              }, autoWordSpaceDelayMsRef.current);
            }
          }
        }
      }

      // Record latency metrics
      const decisionEnd = performance.now();
      profilerRef.current.record({
        captureToModel: Math.max(0, frameStart - captureTime),
        inference: Math.max(0, decisionStart - frameStart),
        decision: Math.max(0, decisionEnd - decisionStart),
        total: Math.max(0, decisionEnd - captureTime),
      });
      if (Math.random() < 0.2) {
        setLatencyMetrics(profilerRef.current.getStats());
      }

      return prediction;
    } catch (e) {
      console.warn('Letter recognition API error:', e);
      startOrResetInactivityTimer(false);
      return null;
    } finally {
      isRequestPendingRef.current = false;
    }
  }, [enabled, recognitionMode, cancelInactivityCountdown, startOrResetInactivityTimer, handleHandsDropped, clearAutoSpaceTimer, setSentenceBufferSync]);

  // Sentence buffer actions
  const undoLetter = useCallback(() => {
    setSentenceBufferSync(prev => {
      const trimmed = prev.trimEnd();
      const lastSpaceIdx = trimmed.lastIndexOf(' ');
      const next = lastSpaceIdx !== -1 ? trimmed.substring(0, lastSpaceIdx + 1) : prev.slice(0, -1);
      if (!next.trim()) {
        cancelInactivityCountdown();
      } else {
        startOrResetInactivityTimer(true);
      }
      return next;
    });
  }, [setSentenceBufferSync, cancelInactivityCountdown, startOrResetInactivityTimer]);

  const deleteLetter = useCallback(() => {
    setSentenceBufferSync(prev => {
      const next = prev.slice(0, -1);
      if (!next.trim()) {
        cancelInactivityCountdown();
      } else {
        startOrResetInactivityTimer(true);
      }
      return next;
    });
  }, [setSentenceBufferSync, cancelInactivityCountdown, startOrResetInactivityTimer]);

  const clearBuffer = useCallback(() => {
    clearAutoSpaceTimer();
    cancelInactivityCountdown();
    setSentenceBufferSync('');
    consecutiveLetterRef.current = { letter: null, count: 0 };
    consecutiveWordRef.current = { word: null, count: 0 };
    lastCommittedItemRef.current = null;
    frameBufferRef.current.clear();
    setWordBufferCount(0);
  }, [clearAutoSpaceTimer, cancelInactivityCountdown, setSentenceBufferSync]);

  const addSpace = useCallback(() => {
    setSentenceBufferSync(prev => (prev.endsWith(' ') ? prev : prev + ' '));
  }, [setSentenceBufferSync]);

  const updateSentence = useCallback((newText) => {
    setSentenceBufferSync(newText);
    if (!newText.trim()) {
      cancelInactivityCountdown();
    } else {
      startOrResetInactivityTimer(true);
    }
  }, [setSentenceBufferSync, cancelInactivityCountdown, startOrResetInactivityTimer]);

  const refineSentence = useCallback(async () => {
    const rawText = sentenceBufferRef.current.trim();
    if (!rawText) return null;
    try {
      const res = await fetch(`${API_BASE}/llm/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: rawText })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.refined_sentence) {
          setSentenceBufferSync(data.refined_sentence);
          startOrResetInactivityTimer(true);
          return data;
        }
      }
    } catch (e) {
      console.warn('Refine LLM error:', e);
    }
    return null;
  }, [setSentenceBufferSync, startOrResetInactivityTimer]);

  const commitSentence = useCallback(() => {
    const text = sentenceBufferRef.current.trim();
    cancelInactivityCountdown();
    setSentenceBufferSync('');
    return text;
  }, [cancelInactivityCountdown, setSentenceBufferSync]);

  const sendSentence = useCallback(async () => {
    const rawText = sentenceBufferRef.current.trim();
    if (!rawText || isAutoSendingRef.current) return;
    cancelInactivityCountdown();
    clearAutoSpaceTimer();
    isAutoSendingRef.current = true;
    setIsAutoSending(true);
    let textToSend = rawText;
    try {
      const res = await fetch(`${API_BASE}/llm/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: rawText })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.refined_sentence && data.refined_sentence.trim()) {
          textToSend = data.refined_sentence.trim();
        }
      }
    } catch (e) {
      console.warn('Manual send refine notice:', e);
    } finally {
      isAutoSendingRef.current = false;
      setIsAutoSending(false);
    }
    textToSend = textToSend.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '').trim() || rawText;
    if (onSendMessage) {
      onSendMessage(textToSend, 'human');
    }
    setSentenceBufferSync(prev => {
      const trimmed = prev.trim();
      if (trimmed.startsWith(rawText)) {
        return trimmed.slice(rawText.length).trimStart();
      }
      return '';
    });
  }, [cancelInactivityCountdown, clearAutoSpaceTimer, onSendMessage, setSentenceBufferSync]);

  return {
    // Gesture State Machine & Diagnostics
    gestureState,
    statusLabel,
    lockedSign,
    steadyProgress,
    latencyMetrics,
    debugOverlay,
    toggleDebugOverlay,
    debugInfo,

    // Mode
    recognitionMode,
    setRecognitionMode,

    // State
    status,
    detectedLetter,
    detectedWord,
    detectedSign: detectedWord || detectedLetter,
    confidence,
    allScores,
    handInfo,
    guidance,
    sentenceBuffer,
    wordBufferCount,
    wordBufferMax: WORD_SEQUENCE_LENGTH,
    availableWords,
    // Auto-Space Between Words states
    autoSpaceWords,
    setAutoSpaceWords,
    autoWordSpaceDelayMs,
    setAutoWordSpaceDelayMs,
    // Auto-Send / Inactivity states
    autoSendEnabled,
    setAutoSendEnabled,
    autoSendTimeoutMs,
    setAutoSendTimeoutMs,
    inactivityCountdown,
    lastAutoSpoken: null,
    isAutoSending,

    // Actions
    processLandmarks,
    undoLetter,
    deleteLetter,
    clearBuffer,
    addSpace,
    updateSentence,
    refineSentence,
    commitSentence,
    sendSentence,
    triggerAutoSend,
    cancelInactivityCountdown,
  };
}

/**
 * Analyze landmark quality and return guidance message if needed.
 */
function analyzeQuality(landmarks) {
  if (!landmarks || landmarks.length < 126) {
    return 'Hand not fully visible. Move closer to camera.';
  }

  let xMin = 1, xMax = 0, yMin = 1, yMax = 0;

  for (let i = 0; i < 42; i++) {
    const x = landmarks[i * 3];
    const y = landmarks[i * 3 + 1];
    if (x > 0 || y > 0) {
      xMin = Math.min(xMin, x);
      xMax = Math.max(xMax, x);
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    }
  }

  const xRange = xMax - xMin;
  const yRange = yMax - yMin;

  if (xRange < 0.05 && yRange < 0.05) {
    return 'Move hands closer to camera.';
  }

  if (xMin < 0.02 || xMax > 0.98 || yMin < 0.02 || yMax > 0.98) {
    return 'Hands partially outside frame. Center both hands.';
  }

  if (xRange < 0.1 && yRange < 0.1) {
    return 'Spread fingers clearly for better recognition.';
  }

  return null;
}