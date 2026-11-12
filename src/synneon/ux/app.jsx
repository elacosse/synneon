import React, { useState, useEffect, useRef, useCallback } from "react";
import { Liveline } from "liveline";
import { ChevronDown, Mic, Minus, Plus, GripHorizontal, LayoutGrid, Check } from "lucide-react";

// ── Emotion Wheel ────────────────────────────────────────────────────────────
const EMOTION_WHEEL = [
  { id: "happy", label: "Happy", color: "#f5d100", startAngle: -90, subs: ["Joyful", "Confident"] },
  { id: "trust", label: "Trust", color: "#6b8c21", startAngle: -45, subs: ["Peaceful", "Hopeful"] },
  { id: "fear", label: "Fear", color: "#4ab8f0", startAngle: 0, subs: ["Cautious", "Anxious"] },
  { id: "surprise", label: "Surprise", color: "#00d4c8", startAngle: 45, subs: ["Amazed", "Confused"] },
  { id: "sad", label: "Sad", color: "#7b6fcd", startAngle: 90, subs: ["Unhappy", "Hurt"] },
  { id: "disgust", label: "Disgust", color: "#b930c0", startAngle: 135, subs: ["Disapproval", "Dislike"] },
  { id: "anger", label: "Anger", color: "#cc2200", startAngle: 180, subs: ["Critical", "Frustrated"] },
  { id: "anticipation", label: "Anticipation", color: "#ff6600", startAngle: 225, subs: ["Interested", "Eager"] },
];

function sectorPath(cx, cy, r1, r2, aDeg1, aDeg2) {
  const toRad = (d) => (d * Math.PI) / 180;
  const a1 = toRad(aDeg1),
    a2 = toRad(aDeg2);
  const large = aDeg2 - aDeg1 > 180 ? 1 : 0;
  const x1i = cx + r1 * Math.cos(a1),
    y1i = cy + r1 * Math.sin(a1);
  const x2i = cx + r1 * Math.cos(a2),
    y2i = cy + r1 * Math.sin(a2);
  const x1o = cx + r2 * Math.cos(a1),
    y1o = cy + r2 * Math.sin(a1);
  const x2o = cx + r2 * Math.cos(a2),
    y2o = cy + r2 * Math.sin(a2);
  return `M${x1i},${y1i} A${r1},${r1},0,${large},1,${x2i},${y2i} L${x2o},${y2o} A${r2},${r2},0,${large},0,${x1o},${y1o} Z`;
}

const formatTime = (ms) => {
  const totalCentiseconds = Math.floor(ms / 10);
  const centiseconds = totalCentiseconds % 100;
  const seconds = Math.floor(totalCentiseconds / 100) % 60;
  const minutes = Math.floor(totalCentiseconds / 6000);
  const hours = Math.floor(totalCentiseconds / 360000);

  const pad = (n) => n.toString().padStart(2, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}:${pad(centiseconds)}`;
};

// Sleek Apple-style minimize/expand button
const WindowButton = ({ minimized, onClick }) => (
  <button
    onClick={(e) => {
      e.stopPropagation();
      onClick();
    }}
    className="group relative w-3 h-3 rounded-full bg-zinc-700 hover:bg-zinc-500 transition-colors flex items-center justify-center shrink-0 focus:outline-none"
    title={minimized ? "Expand" : "Minimize"}
  >
    {minimized ? (
      <Plus size={8} className="text-zinc-900 opacity-0 group-hover:opacity-100 transition-opacity" strokeWidth={4} />
    ) : (
      <Minus size={8} className="text-zinc-900 opacity-0 group-hover:opacity-100 transition-opacity" strokeWidth={4} />
    )}
  </button>
);

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [currentProfile, setCurrentProfile] = useState("default");
  const [recordingTime, setRecordingTime] = useState(0);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [sampleRate, setSampleRate] = useState("44.1");

  // WebSocket and Hardware states
  const wsRef = useRef(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [neonStatus, setNeonStatus] = useState({ connected: false });
  const [pupilStreaming, setPupilStreaming] = useState(false);
  const [sessionData, setSessionData] = useState({ userId: null, sessionId: null });
  const [isTranscribing, setIsTranscribing] = useState(false);

  // Dashboard panel toggle states
  const [minimized, setMinimized] = useState({
    b1: false,
    b2: false,
    b3: false,
    b4: false,
    b5: false,
    b6: false,
    b7: false,
    b8: false,
    b11: false,
  });

  // Widget visibility state
  const [visibleBlocks, setVisibleBlocks] = useState({
    b1: true,
    b2: true,
    b3: false,
    b4: true,
    b5: true,
    b6: true,
    b7: true,
    b8: true,
    b11: true,
  });
  const [showWidgetMenu, setShowWidgetMenu] = useState(false);

  const widgetNames = {
    b6: "Connection",
    b1: "Audio",
    b2: "Transcript",
    b3: "Pupilometry",
    b4: "Abbreviation",
    b5: "Letter Grid",
    b7: "Emotion",
    b8: "Compass",
    b11: "Neon Capture",
  };

  // Drag and drop state
  const [blocksOrder, setBlocksOrder] = useState(["b2", "b1", "b6", "b11", "b4", "b5", "b8", "b3", "b7"]);
  const [draggedId, setDraggedId] = useState(null);
  const [dragOverId, setDragOverId] = useState(null);

  // Conversation memory and input state
  const [memoryLog, setMemoryLog] = useState([]);
  const [typedMessage, setTypedMessage] = useState("");
  const [expansionCandidates, setExpansionCandidates] = useState([]);
  const [isExpanding, setIsExpanding] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speakingText, setSpeakingText] = useState("");

  // Neon live capture state
  const [neonCapture, setNeonCapture] = useState(null);
  const [neonCaptureError, setNeonCaptureError] = useState(null);
  const [neonCaptureKey, setNeonCaptureKey] = useState(0);
  const [neoVizTab, setNeoVizTab] = useState("saliency_overlay");
  const [neonRecording, setNeonRecording] = useState(false);
  const [neonInferring, setNeonInferring] = useState(false);
  const [neonDuration, setNeonDuration] = useState(null);
  const [blinkFlash, setBlinkFlash] = useState(false);
  const [neonWorn, setNeonWorn] = useState(null);

  // Expand Context (b10) state
  const [expandCtx, setExpandCtx] = useState(null);
  const [expandCtxLoading, setExpandCtxLoading] = useState(false);
  const [expandCtxError, setExpandCtxError] = useState(null);
  const [expandInferring, setExpandInferring] = useState(false);
  const [showExpandRawCtx, setShowExpandRawCtx] = useState(false);
  const [expandRawPrompts, setExpandRawPrompts] = useState(null);

  const fetchExpandCtx = useCallback(async () => {
    const uid = sessionData.userId;
    const sid = sessionData.sessionId;
    if (!uid || !sid) {
      setExpandCtx(null);
      return;
    }
    setExpandCtxLoading(true);
    setExpandCtxError(null);
    try {
      const res = await fetch(
        `/context/expand-context?user_id=${encodeURIComponent(uid)}&session_id=${encodeURIComponent(sid)}`,
      );
      if (res.status === 404) {
        setExpandCtx(null);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setExpandCtx(await res.json());
    } catch (e) {
      setExpandCtxError(e.message);
    } finally {
      setExpandCtxLoading(false);
    }
  }, [sessionData.userId, sessionData.sessionId]);

  useEffect(() => {
    if (sessionData.sessionId) {
      setExpandCtx(null);
      setExpandCtxError(null);
      setExpandRawPrompts(null);
    }
  }, [sessionData.sessionId]);

  // Compass UI State
  const FULL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
  const [compassGroup, setCompassGroup] = useState(FULL_ALPHABET);
  const [compassHistory, setCompassHistory] = useState([]);

  const getCompassZones = useCallback((group) => {
    const total = group.length;
    if (total === 0) return { up: [], right: [], down: [], left: [] };
    const quarter = Math.ceil(total / 4);
    return {
      up: group.slice(0, quarter),
      right: group.slice(quarter, quarter * 2),
      down: group.slice(quarter * 2, quarter * 3),
      left: group.slice(quarter * 3),
    };
  }, []);

  const compassZones = getCompassZones(compassGroup);

  // References
  const isRecordingRef = useRef(false);
  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const startTimeRef = useRef(0);
  const chatScrollRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const compassInputRef = useRef(null);
  const emotionSvgRef = useRef(null);

  // Emotion wheel state
  const [emotionCursor, setEmotionCursor] = useState(null);

  // Pupilometry state
  const [leftPupilSize, setLeftPupilSize] = useState("4.20");
  const [rightPupilSize, setRightPupilSize] = useState("4.20");
  const [leftPupilData, setLeftPupilData] = useState([]);
  const [rightPupilData, setRightPupilData] = useState([]);

  const handleEmotionPointer = (e) => {
    const svg = emotionSvgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * 300;
    const sy = ((e.clientY - rect.top) / rect.height) * 300;
    const dx = sx - 150,
      dy = sy - 150;
    const r = Math.sqrt(dx * dx + dy * dy);
    if (r < 12 || r > 148) return;

    const angleDeg = (Math.atan2(dy, dx) * 180) / Math.PI;
    const normAngle = ((angleDeg % 360) + 360) % 360;
    const sectorIdx = Math.floor(((normAngle + 90) % 360) / 45);
    const emotion = EMOTION_WHEEL[sectorIdx % 8];

    let intensity, ring, subEmotion;
    if (r <= 100) {
      ring = "inner";
      intensity = (r - 12) / (100 - 12);
      subEmotion = null;
    } else {
      ring = "outer";
      intensity = (r - 104) / (148 - 104);
      const relAngle = (((angleDeg - emotion.startAngle) % 45) + 45) % 45;
      subEmotion = relAngle < 22.5 ? emotion.subs[0] : emotion.subs[1];
    }

    setEmotionCursor({
      x: sx,
      y: sy,
      emotion: emotion.label,
      subEmotion,
      intensity: Math.min(1, Math.max(0, intensity)),
      ring,
      color: emotion.color,
    });
  };

  // Compass UI Keyboard Listener
  useEffect(() => {
    compassInputRef.current = (direction) => {
      const targetGroup = compassZones[direction];
      if (!targetGroup || !targetGroup.length) return;
      if (targetGroup.length === 1) {
        setTypedMessage((prev) => prev + targetGroup[0].toLowerCase());
        setCompassGroup(FULL_ALPHABET);
        setCompassHistory([]);
      } else {
        setCompassHistory((prev) => [...prev, compassGroup]);
        setCompassGroup(targetGroup);
      }
    };

    const handleKeyDown = (e) => {
      // Only intercept if the widget is visible and expanded
      if (!visibleBlocks.b8 || minimized.b8) return;

      // Ignore if typing in an actual input field
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

      if (e.key === "ArrowUp") {
        e.preventDefault();
        compassInputRef.current("up");
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        compassInputRef.current("right");
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        compassInputRef.current("down");
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        compassInputRef.current("left");
      } else if (e.key === "Backspace") {
        e.preventDefault();
        if (compassHistory.length > 0) {
          const newHistory = [...compassHistory];
          setCompassGroup(newHistory.pop());
          setCompassHistory(newHistory);
        } else {
          setTypedMessage((prev) => prev.slice(0, -1));
        }
      } else if (e.key === "Enter") {
        e.preventDefault();
        setTypedMessage((prev) => prev + " ");
        setCompassGroup(FULL_ALPHABET);
        setCompassHistory([]);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [visibleBlocks.b8, minimized.b8, compassZones, compassGroup, compassHistory]);

  // Keyboard layout data
  const commonWords = [
    "a",
    "are",
    "can",
    "do",
    "for",
    "i",
    "in",
    "is",
    "it",
    "just",
    "me",
    "my",
    "of",
    "on",
    "that",
    "the",
    "this",
    "to",
    "we",
    "you",
  ];
  const lettersRow1 = ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"];
  const lettersRow2 = ["a", "s", "d", "f", "g", "h", "j", "k", "l"];
  const lettersRow3 = ["z", "x", "c", "v", "b", "n", "m"];

  // Scroll to bottom of chat when new message arrives
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [memoryLog, minimized.b2]);

  // Fetch audio input devices
  const getDevices = async () => {
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      const enumeratedDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = enumeratedDevices.filter((device) => device.kind === "audioinput");

      setDevices(audioInputs);
      if (audioInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(audioInputs[0].deviceId);
      }
    } catch (error) {
      console.error("Error fetching devices or permissions denied:", error);
    }
  };

  useEffect(() => {
    getDevices();
    return () => stopAudioContext();
  }, []);

  const [inputUserId, setInputUserId] = useState("");

  const initSession = async (customUserId = null) => {
    console.log(`[Session] Initializing session...`);
    try {
      const url = customUserId ? `/context/users?user_id=${encodeURIComponent(customUserId)}` : "/context/users";
      const res = await fetch(url, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        console.log(`[Session] Started session for User: ${data.user_id}`);
        setSessionData({ userId: data.user_id, sessionId: data.session_id });
        setInputUserId("");
        setMemoryLog([]);
      }
    } catch (err) {
      console.error("Failed to start session:", err);
    }
  };

  useEffect(() => {
    initSession();
  }, []);

  // Timer loop
  const updateTimer = useCallback(() => {
    if (!isRecordingRef.current) return;
    setRecordingTime(performance.now() - startTimeRef.current);
    animationFrameRef.current = requestAnimationFrame(updateTimer);
  }, []);

  // Audio visualizer loop
  const drawVisualizer = useCallback(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    ctx.clearRect(0, 0, width, height);
    ctx.beginPath();
    ctx.arc(cx, cy, 80, 0, 2 * Math.PI);
    ctx.strokeStyle = "#3f3f46";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 6]);
    ctx.stroke();
    ctx.setLineDash([]);

    const numBars = 64;
    const innerRadius = 95;
    const maxBarLength = 45;

    ctx.lineWidth = 3.5;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#a1a1aa";

    let dataArray = new Uint8Array(numBars);
    if (isRecordingRef.current && analyserRef.current) {
      analyserRef.current.getByteFrequencyData(dataArray);
    }

    for (let i = 0; i < numBars; i++) {
      const rawValue = isRecordingRef.current ? dataArray[Math.floor(i * 1.5) % dataArray.length] : 0;
      const smoothValue = Math.pow(rawValue / 255, 1.2);
      const barHeight = 4 + smoothValue * maxBarLength;

      const angle = (i * (2 * Math.PI)) / numBars - Math.PI / 2;
      const x1 = cx + Math.cos(angle) * innerRadius;
      const y1 = cy + Math.sin(angle) * innerRadius;
      const x2 = cx + Math.cos(angle) * (innerRadius + barHeight);
      const y2 = cy + Math.sin(angle) * (innerRadius + barHeight);

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    if (isRecordingRef.current) {
      requestAnimationFrame(drawVisualizer);
    }
  }, []);

  const toggleRecording = useCallback(async () => {
    if (isRecordingRef.current) {
      setIsRecording(false);
      isRecordingRef.current = false;
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      stopAudioContext();
      console.log("[Audio] Stopped recording.");
      setTimeout(() => requestAnimationFrame(drawVisualizer), 50);
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ event: "recording_stopped" }));
      }
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true,
        });

        streamRef.current = stream;
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        analyserRef.current.smoothingTimeConstant = 0.8;

        const source = audioContextRef.current.createMediaStreamSource(stream);
        source.connect(analyserRef.current);
        setSampleRate((audioContextRef.current.sampleRate / 1000).toFixed(1));

        setIsRecording(true);
        isRecordingRef.current = true;
        setRecordingTime(0);
        startTimeRef.current = performance.now();
        audioChunksRef.current = [];

        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType });

          setIsTranscribing(true);
          try {
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = async () => {
              const base64data = reader.result.split(",")[1];

              console.log(`[Audio] Sending ${audioBlob.size} bytes for transcription...`);
              const res = await fetch("/audio/transcribe/base64", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  audio_base64: base64data,
                  media_type: mediaRecorder.mimeType || "audio/webm",
                  profile: "default",
                  save_to_memory: true,
                }),
              });

              if (res.ok) {
                const data = await res.json();
                const transcript = data.text.trim();
                console.log(`[Audio] Received transcription: "${transcript}"`);
                if (transcript) {
                  const newMsg = {
                    id: Date.now(),
                    role: "system",
                    text: transcript,
                    timestamp: formatTime(performance.now() % 360000),
                  };
                  setMemoryLog((prev) => [...prev, newMsg]);
                }
              }
              setIsTranscribing(false);
            };
          } catch (err) {
            console.error("Transcription failed:", err);
            setIsTranscribing(false);
          }
        };

        mediaRecorder.start(100);
        console.log("[Audio] Started recording via MediaRecorder.");

        startTimeRef.current = performance.now();
        requestAnimationFrame(updateTimer);
        requestAnimationFrame(drawVisualizer);
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ event: "recording_started" }));
        }
      } catch (error) {
        console.error("Error accessing microphone:", error);
        alert("Could not access microphone. Please ensure permissions are granted.");
      }
    }
  }, [selectedDeviceId, drawVisualizer]);

  const toggleRecordingRef = useRef(toggleRecording);
  useEffect(() => {
    toggleRecordingRef.current = toggleRecording;
  }, [toggleRecording]);

  useEffect(() => {
    // In development mode, Vite's proxy handles /ws via window.location.port
    // In production, we assume 8765 or the served port.
    const WS_PORT = window.location.port || "8765";
    const wsUrl = `ws://${window.location.hostname}:${WS_PORT}/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => {
      setWsConnected(false);
      setNeonStatus({ connected: false });
      setPupilStreaming(false);
    };

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.event === "app_config" && msg.profile) {
          setCurrentProfile(msg.profile);
        }
        if (msg.event === "neon_recording_started") {
          setNeonRecording(true);
          setNeonCaptureError(null);
          setNeonCapture(null);
          setNeonDuration(null);
        } else if (msg.event === "neon_recording_stopped") {
          setNeonRecording(false);
          setNeonInferring(true);
        } else if (msg.event === "neon_capture_processed") {
          setNeonDuration(msg.duration_s ?? null);
          setNeonCaptureKey((k) => k + 1);
        } else if (msg.event === "neon_capture_complete") {
          setNeonCapture(msg.result);
          setNeonInferring(false);
        } else if (msg.event === "neon_capture_error") {
          setNeonInferring(false);
          setNeonCaptureError(msg.detail);
          console.error("Neon capture failed:", msg.detail);
        } else if (msg.event === "blink_detected") {
          setBlinkFlash(true);
          if (window._blinkFlashTimer) clearTimeout(window._blinkFlashTimer);
          window._blinkFlashTimer = setTimeout(() => setBlinkFlash(false), 400);
        } else if (msg.event === "expand_inference_started") {
          setExpandInferring(true);
          setExpandCtx(null);
          setExpandCtxError(null);
        } else if (msg.event === "expand_inference_complete") {
          setExpandInferring(false);
          fetchExpandCtx();
        } else if (msg.event === "pupil_data") {
          setPupilStreaming(true);
          const t = msg.timestamp ?? Date.now() / 1000;
          setLeftPupilSize(msg.left_mm.toFixed(2));
          setRightPupilSize(msg.right_mm.toFixed(2));
          setLeftPupilData((prev) => {
            const next = [...prev, { time: t, value: msg.left_mm }];
            return next.length > 2400 ? next.slice(-2400) : next;
          });
          setRightPupilData((prev) => {
            const next = [...prev, { time: t, value: msg.right_mm }];
            return next.length > 2400 ? next.slice(-2400) : next;
          });
          if (window.pupilTimeoutId) clearTimeout(window.pupilTimeoutId);
          window.pupilTimeoutId = setTimeout(() => setPupilStreaming(false), 1000);
        } else if (msg.event === "app_config") {
          setVisibleBlocks((prev) => ({ ...prev, b3: !!msg.pupil }));
        } else if (msg.event === "neon_status") {
          setNeonStatus(msg);
        } else if (msg.event === "worn_status") {
          setNeonWorn(msg.worn);
        } else if (msg.event === "compass_gesture") {
          compassInputRef.current?.(msg.direction);
        }
      } catch (err) {
        console.error("Failed to parse WS message", err);
      }
    };

    wsRef.current = ws;
    return () => {
      ws.close();
    };
  }, []);

  const stopAudioContext = () => {
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    if (audioContextRef.current) audioContextRef.current.close();
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
  };

  useEffect(() => {
    if (!isRecording) drawVisualizer();
  }, [drawVisualizer, isRecording]);

  // Keyboard Handlers
  const handleKeyClick = (val) => setTypedMessage((prev) => prev === "" ? val : prev + " " + val);
  const handleBackspace = () => setTypedMessage((prev) => prev.slice(0, -1));
  const handleSend = async (moreCandidates = false) => {
    const isMore = typeof moreCandidates === "boolean" ? moreCandidates : false;
    if (typedMessage.trim() === "") return;
    setIsExpanding(true);
    setExpansionCandidates([]);
    setExpandRawPrompts(null);

    console.log(`[API] Requesting expansion for: "${typedMessage.trim()}"`);

    try {
      // NOTE: In the future, we can pull context from Redis and pass it in the payload.
      const res = await fetch(`/inference/expand?user_id=${encodeURIComponent(sessionData.userId || "default")}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tokens: typedMessage.trim(),
          profile: currentProfile,
          more_candidates: isMore,
          context: {
            emotion: emotionCursor
              ? {
                  label: (emotionCursor.subEmotion ?? emotionCursor.emotion).toLowerCase(),
                  intensity: emotionCursor.intensity,
                }
              : { label: "neutral", intensity: 0.0 },
            parsed_visual: neonCapture?.parsed_visual || { focused_object: null, detected_objects: [] },
            pragmatic: neonCapture?.pragmatic || { intent: "declarative", audience: "unknown", physical_setting: null },
          },
        }),
      });

      if (!res.ok) throw new Error("Failed to expand");
      const data = await res.json();
      console.log(`[API] Received ${data.candidates?.length || 0} expansion candidates.`);
      setExpansionCandidates(data.candidates || []);
      setExpandRawPrompts({ system_prompt: data.system_prompt || null, user_prompt: data.user_prompt || null });
    } catch (err) {
      console.error("Expansion error:", err);
      // Fallback: just send the raw message
      const newMsg = {
        id: Date.now(),
        role: "user",
        text: typedMessage.trim(),
        timestamp: formatTime(performance.now() % 360000),
      };
      setMemoryLog((prev) => [...prev, newMsg]);
      setTypedMessage("");
      setNeonCapture(null);
      setNeonDuration(null);
      setNeonCaptureKey(0);
    } finally {
      setIsExpanding(false);
    }
  };

  const handlePickCandidate = async (candidateText) => {
    const newMsg = {
      id: Date.now(),
      role: "user",
      text: candidateText,
      timestamp: formatTime(performance.now() % 360000),
    };
    setMemoryLog((prev) => [...prev, newMsg]);
    setTypedMessage("");
    setExpansionCandidates([]);
    setNeonCapture(null);
    setNeonDuration(null);
    setNeonCaptureKey(0);

    // Persist the selected expansion to Redis so it appears in future conversation context
    fetch(`/context/utterance?user_id=${encodeURIComponent(sessionData.userId || "default")}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: candidateText, role: "user", source: "expansion" }),
    }).catch((err) => console.error("[API] Failed to save utterance:", err));

    // Text goes in the JSON body, not the URL, so it stays out of access logs.
    const ttsBody = { text: candidateText };
    let displayText = candidateText;

    if (emotionCursor) {
      const label = (emotionCursor.subEmotion ?? emotionCursor.emotion).toLowerCase();
      const intensity = emotionCursor.intensity.toFixed(2);
      ttsBody.emotion_label = label;
      ttsBody.emotion_intensity = Number(intensity);
      if (emotionCursor.intensity >= 0.05) {
        displayText = `<emotion value="${label}" />${candidateText}`;
      }
    }

    console.log(`[API] Requesting TTS for: "${displayText}"`);
    setSpeakingText(displayText);
    setIsSpeaking(true);
    try {
      const res = await fetch("/audio/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ttsBody),
      });
      if (!res.ok) throw new Error("TTS failed");
      const blob = await res.blob();
      console.log(`[API] Received TTS audio block (${blob.size} bytes). Playing...`);
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.onended = () => URL.revokeObjectURL(url);
      await audio.play();
    } catch (err) {
      console.error("TTS error:", err);
    } finally {
      setIsSpeaking(false);
      setSpeakingText("");
    }
  };

  const handleCancelExpansion = () => {
    console.log("[API] Expansion cancelled by user.");
    setExpansionCandidates([]);
  };

  const toggleBlockVisibility = (id) => {
    setVisibleBlocks((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // --- HTML5 Drag and Drop Handlers ---
  const handleDragStart = (e, id) => {
    setDraggedId(id);
    e.dataTransfer.setData("text/plain", id);
    e.dataTransfer.effectAllowed = "move";
  };

  const handleDragEnter = (e, id) => {
    e.preventDefault();
    setDragOverId(id);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e, targetId) => {
    e.preventDefault();
    if (!draggedId || draggedId === targetId) {
      setDraggedId(null);
      setDragOverId(null);
      return;
    }
    setBlocksOrder((prev) => {
      const newOrder = [...prev];
      const draggedIdx = newOrder.indexOf(draggedId);
      const targetIdx = newOrder.indexOf(targetId);
      newOrder.splice(draggedIdx, 1);
      newOrder.splice(targetIdx, 0, draggedId);
      return newOrder;
    });
    setDraggedId(null);
    setDragOverId(null);
  };

  const handleDragEnd = () => {
    setDraggedId(null);
    setDragOverId(null);
  };

  // --- Dynamic Block Rendering ---
  const renderBlock = (id) => {
    const isDragged = draggedId === id;
    const isOver = dragOverId === id && draggedId !== id;

    // Smooth, glowing Apple-like borders and scale effects
    const commonWrapperClasses = `transition-all duration-300 transform-gpu border ${
      isDragged
        ? "opacity-40 scale-[0.98] border-zinc-600 shadow-none z-0"
        : isOver
          ? "border-blue-500 shadow-[0_0_30px_rgba(59,130,246,0.15)] scale-[1.01] z-50"
          : "border-white/5 shadow-2xl z-10"
    } bg-[#16161a] rounded-2xl relative flex flex-col`;

    const dragProps = {
      draggable: true,
      onDragStart: (e) => handleDragStart(e, id),
      onDragEnter: (e) => handleDragEnter(e, id),
      onDragOver: handleDragOver,
      onDrop: (e) => handleDrop(e, id),
      onDragEnd: handleDragEnd,
    };

    switch (id) {
      case "b1":
        return (
          <div
            key={id}
            {...dragProps}
            className={`flex-1 min-w-[320px] max-w-[360px] ${minimized.b1 ? "" : "aspect-[3/4]"} ${commonWrapperClasses}`}
          >
            <div className="flex justify-between items-center p-6 text-xs tracking-widest text-zinc-500 cursor-grab active:cursor-grabbing">
              <div className="flex items-center gap-3">
                <WindowButton minimized={minimized.b1} onClick={() => setMinimized((p) => ({ ...p, b1: !p.b1 }))} />
                <span>Audio</span>
                <GripHorizontal
                  size={14}
                  className="text-zinc-700 opacity-50 hover:opacity-100 transition-opacity ml-1"
                />
              </div>
              <div className="relative">
                <button
                  onClick={() => setShowDropdown(!showDropdown)}
                  className="flex items-center gap-2 hover:text-zinc-300 transition-colors focus:outline-none cursor-pointer"
                >
                  AUDIO_IN{" "}
                  <ChevronDown
                    size={14}
                    className={`transition-transform duration-300 ${showDropdown ? "rotate-180" : ""}`}
                  />
                </button>
                {showDropdown && (
                  <div className="absolute right-0 top-full mt-2 w-48 bg-[#1f1f24] border border-white/10 rounded-lg shadow-xl overflow-hidden z-20 animate-in fade-in slide-in-from-top-2 duration-200 cursor-default">
                    {devices.length === 0 ? (
                      <div className="p-3 text-xs text-zinc-500 text-center">No mics found</div>
                    ) : (
                      devices.map((device) => (
                        <button
                          key={device.deviceId}
                          onClick={() => {
                            setSelectedDeviceId(device.deviceId);
                            setShowDropdown(false);
                          }}
                          className={`w-full text-left px-3 py-2 text-xs flex items-center gap-2 hover:bg-white/5 transition-colors ${selectedDeviceId === device.deviceId ? "text-blue-400" : "text-zinc-400"}`}
                        >
                          <Mic size={12} />
                          <span className="truncate">
                            {device.label || `Microphone ${device.deviceId.slice(0, 4)}`}
                          </span>
                        </button>
                      ))
                    )}
                  </div>
                )}
              </div>
            </div>

            <div
              className={`flex flex-col flex-1 overflow-hidden transition-opacity duration-300 ${minimized.b1 ? "hidden opacity-0" : "opacity-100"}`}
            >
              <div className="flex-1 relative flex items-center justify-center cursor-default">
                <canvas
                  ref={canvasRef}
                  width={300}
                  height={300}
                  className="absolute inset-0 m-auto pointer-events-none"
                />
                <button
                  onClick={toggleRecording}
                  className={`relative z-10 w-[72px] h-[72px] rounded-full flex items-center justify-center transition-all duration-300 cursor-pointer ${
                    isRecording
                      ? "bg-transparent border-[2.5px] border-[#a0c0ff] shadow-[0_0_20px_rgba(160,192,255,0.2)]"
                      : "bg-transparent border-[2.5px] border-zinc-700 hover:border-zinc-500"
                  }`}
                >
                  <div
                    className={`transition-all duration-300 rounded-sm bg-[#ff6b52] ${
                      isRecording
                        ? "w-[22px] h-[22px] rounded-md shadow-[0_0_15px_rgba(255,107,82,0.4)]"
                        : "w-[28px] h-[28px] !rounded-full opacity-80 hover:opacity-100"
                    }`}
                  />
                </button>
              </div>
              <div className="p-6 pb-4 cursor-default">
                <h2 className="text-[10px] tracking-widest text-zinc-600 font-sans font-semibold mb-2">STATUS</h2>
                <div className="text-zinc-300 font-sans text-sm space-y-1">
                  <p className="transition-opacity duration-300">
                    {isRecording
                      ? "Secure link established."
                      : isTranscribing
                        ? "Processing audio data..."
                        : "System standing by."}
                  </p>
                  <p className={`transition-opacity duration-300 ${isRecording ? "opacity-100" : "opacity-40"}`}>
                    {isRecording ? "Capturing input stream..." : "Awaiting initialization command."}
                  </p>
                </div>
              </div>
              <div className="p-6 pt-4 border-t border-white/5 flex justify-between items-center text-xs text-zinc-500 cursor-default">
                <span className="w-24 text-left font-mono tabular-nums tracking-wider text-zinc-400">
                  {formatTime(recordingTime)}
                </span>
                <span className="w-24 text-right font-mono tracking-wider">{sampleRate} kHz</span>
              </div>
            </div>
          </div>
        );

      case "b2":
        return (
          <div
            key={id}
            {...dragProps}
            className={`flex-1 min-w-[320px] max-w-[360px] ${minimized.b2 ? "" : "aspect-[3/4]"} ${commonWrapperClasses}`}
          >
            <div
              className={`flex justify-between items-center p-6 text-xs tracking-widest text-zinc-500 cursor-grab active:cursor-grabbing ${minimized.b2 ? "" : "border-b"} border-white/5 bg-[#16161a]/95 z-10 rounded-t-2xl`}
            >
              <div className="flex items-center gap-3">
                <WindowButton minimized={minimized.b2} onClick={() => setMinimized((p) => ({ ...p, b2: !p.b2 }))} />
                <span>Transcript</span>
                <GripHorizontal
                  size={14}
                  className="text-zinc-700 opacity-50 hover:opacity-100 transition-opacity ml-1"
                />
              </div>
              <span>TRANSCRIPT</span>
            </div>

            <div
              className={`flex flex-col flex-1 overflow-hidden transition-opacity duration-300 ${minimized.b2 ? "hidden opacity-0" : "opacity-100"} cursor-default`}
            >
              <div
                ref={chatScrollRef}
                className="flex-1 overflow-y-auto p-6 space-y-6 flex flex-col scroll-smooth [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
              >
                {memoryLog.map((msg) => (
                  <div key={msg.id} className={`flex w-full ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`max-w-[85%] flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}>
                      <div
                        className={`px-4 py-2.5 text-[13px] font-sans leading-relaxed shadow-sm ${msg.role === "user" ? "bg-zinc-800 text-zinc-200 rounded-2xl rounded-br-sm" : "bg-[#111114] border border-white/5 text-zinc-400 rounded-2xl rounded-bl-sm"}`}
                      >
                        {msg.text}
                      </div>
                      <span className="text-[9px] text-zinc-600 mt-1.5 font-mono tracking-widest px-1">
                        {msg.timestamp}
                      </span>
                    </div>
                  </div>
                ))}
                {(isRecording || isTranscribing) && (
                  <div className="flex w-full justify-start animate-in fade-in duration-300">
                    <div className="max-w-[85%] flex flex-col items-start">
                      <div className="px-4 py-3 bg-[#111114] border border-white/5 rounded-2xl rounded-bl-sm flex items-center gap-1.5">
                        <div
                          className={`w-1.5 h-1.5 rounded-full animate-bounce ${isTranscribing ? "bg-blue-500" : "bg-zinc-500"}`}
                          style={{ animationDelay: "0ms" }}
                        />
                        <div
                          className={`w-1.5 h-1.5 rounded-full animate-bounce ${isTranscribing ? "bg-blue-500" : "bg-zinc-500"}`}
                          style={{ animationDelay: "150ms" }}
                        />
                        <div
                          className={`w-1.5 h-1.5 rounded-full animate-bounce ${isTranscribing ? "bg-blue-500" : "bg-zinc-500"}`}
                          style={{ animationDelay: "300ms" }}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <div className="p-4 border-t border-white/5 bg-[#16161a]">
                <div className="w-full h-10 bg-[#111114] border border-white/5 rounded-xl px-4 flex items-center text-xs text-zinc-600 font-sans shadow-inner overflow-hidden whitespace-nowrap relative">
                  {isRecording ? (
                    <span className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#ff6b52] animate-pulse" />
                      Listening to input stream...
                    </span>
                  ) : isTranscribing ? (
                    <span className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                      Transcribing audio...
                    </span>
                  ) : (
                    <span className="flex items-center gap-2 text-zinc-600">
                      <div className="w-1.5 h-1.5 rounded-full bg-zinc-700" />
                      System standing by.
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        );

      case "b3":
        return (
          <div
            key={id}
            {...dragProps}
            className={`flex-1 min-w-[320px] max-w-[360px] ${minimized.b3 ? "" : "aspect-[3/4]"} ${commonWrapperClasses}`}
          >
            <div className="flex justify-between items-center p-6 text-xs tracking-widest text-zinc-500 cursor-grab active:cursor-grabbing">
              <div className="flex items-center gap-3">
                <WindowButton minimized={minimized.b3} onClick={() => setMinimized((p) => ({ ...p, b3: !p.b3 }))} />
                <span>Pupilometry</span>
                <GripHorizontal
                  size={14}
                  className="text-zinc-700 opacity-50 hover:opacity-100 transition-opacity ml-1"
                />
              </div>
              <span>PUPIL_DATA</span>
            </div>

            <div
              className={`flex flex-col flex-1 overflow-hidden transition-opacity duration-300 ${minimized.b3 ? "hidden opacity-0" : "opacity-100"} cursor-default`}
            >
              <div className="flex-1 relative flex flex-col gap-3 p-5 pt-2">
                <div>
                  <div className="flex justify-between items-center mb-1 px-0.5">
                    <span className="text-[10px] text-red-400/60 tracking-[0.2em] font-sans font-medium">LEFT</span>
                    <span className="text-sm font-light text-red-100 tabular-nums">
                      {pupilStreaming ? leftPupilSize : "--"} <span className="text-[10px] text-red-400/40">mm</span>
                    </span>
                  </div>
                  <div style={{ height: 90 }}>
                    <Liveline
                      data={leftPupilData}
                      value={parseFloat(leftPupilSize)}
                      color="#ef4444"
                      theme="dark"
                      exaggerate
                      badge={false}
                      loading={!pupilStreaming}
                      formatValue={(v) => v.toFixed(2) + " mm"}
                      window={30}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1 px-0.5">
                    <span className="text-[10px] text-blue-400/60 tracking-[0.2em] font-sans font-medium">RIGHT</span>
                    <span className="text-sm font-light text-blue-100 tabular-nums">
                      {pupilStreaming ? rightPupilSize : "--"} <span className="text-[10px] text-blue-400/40">mm</span>
                    </span>
                  </div>
                  <div style={{ height: 90 }}>
                    <Liveline
                      data={rightPupilData}
                      value={parseFloat(rightPupilSize)}
                      color="#3b82f6"
                      theme="dark"
                      exaggerate
                      badge={false}
                      loading={!pupilStreaming}
                      formatValue={(v) => v.toFixed(2) + " mm"}
                      window={30}
                    />
                  </div>
                </div>
              </div>

              <div className="px-6 py-2">
                <h2 className="text-[10px] tracking-widest text-zinc-600 font-sans font-semibold mb-1">STATUS</h2>
                <div className="text-zinc-300 font-sans text-xs space-y-1">
                  <p className="transition-colors duration-300">Baseline established.</p>
                  <p
                    className={`transition-opacity duration-300 ${pupilStreaming ? "opacity-100 text-blue-400/80" : "opacity-40"}`}
                  >
                    {pupilStreaming ? "Tracking microsaccades & dilation..." : "Awaiting stimulus target."}
                  </p>
                </div>
              </div>

              <div className="p-5 pt-3 border-t border-white/5 flex justify-between items-center text-xs text-zinc-500">
                <span className="font-mono tracking-wider text-zinc-400">LIVELINE</span>
                <span className="font-mono tracking-wider flex items-center gap-2">
                  <div
                    className={`w-1.5 h-1.5 rounded-full transition-colors duration-300 ${pupilStreaming ? "bg-blue-400 animate-pulse" : "bg-zinc-700"}`}
                  />
                  SYNC
                </span>
              </div>
            </div>
          </div>
        );

      case "b4": {
        const b4sc = expandCtx?.steering_context || {};
        const b4emotion = b4sc.emotion || {};
        const b4visual = b4sc.parsed_visual || {};
        const b4pragmatic = b4sc.pragmatic || {};
        return (
          <div key={id} {...dragProps} className={`w-full max-w-[1128px] ${commonWrapperClasses}`}>
            <div
              className={`flex justify-between items-center text-xs tracking-widest text-zinc-500 cursor-grab active:cursor-grabbing p-6 sm:px-8 ${minimized.b4 ? "sm:py-6" : "sm:pt-8 sm:pb-4"}`}
            >
              <div className="flex items-center gap-3">
                <WindowButton minimized={minimized.b4} onClick={() => setMinimized((p) => ({ ...p, b4: !p.b4 }))} />
                <span>Abbreviation</span>
                <GripHorizontal
                  size={14}
                  className="text-zinc-700 opacity-50 hover:opacity-100 transition-opacity ml-1"
                />
              </div>
              <div className="flex items-center gap-3">
                {expandInferring ? (
                  <span className="text-[10px] px-2 py-0.5 rounded-full border border-yellow-800 text-yellow-400 bg-yellow-950/40 animate-pulse">
                    INFERRING
                  </span>
                ) : expandCtx ? (
                  <span className="text-[10px] px-2 py-0.5 rounded-full border border-emerald-800 text-emerald-400 bg-emerald-950/40">
                    DONE
                  </span>
                ) : null}
                <span className="text-zinc-600">CURRENT_BUFFER</span>
              </div>
            </div>
            <div
              className={`${minimized.b4 ? "hidden" : "block"} px-6 pb-6 sm:px-8 sm:pb-8 cursor-default flex flex-col gap-4`}
            >
              <div className="min-h-[80px] bg-[#111114] border border-white/5 rounded-xl p-5 flex flex-col justify-center shadow-inner overflow-hidden relative">
                {isExpanding ? (
                  <span className="text-xl md:text-3xl text-zinc-400 font-sans font-light tracking-wide break-words w-full flex items-center gap-3">
                    <div
                      className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"
                      style={{ animationDelay: "0ms" }}
                    />
                    <div
                      className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"
                      style={{ animationDelay: "150ms" }}
                    />
                    <div
                      className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"
                      style={{ animationDelay: "300ms" }}
                    />
                    <span className="ml-2">Expanding '{typedMessage}'...</span>
                  </span>
                ) : expansionCandidates.length > 0 ? (
                  <div className="w-full flex flex-col gap-3">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-zinc-500 font-sans uppercase tracking-widest">
                        Pick an expansion:
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleSend(true)}
                          className="text-xs bg-yellow-500/10 hover:bg-yellow-500/20 text-yellow-400 px-3 py-1 rounded-md transition-colors"
                        >
                          More Options
                        </button>
                        {expandRawPrompts && (expandRawPrompts.system_prompt || expandRawPrompts.user_prompt) && (
                          <button
                            onClick={() => setShowExpandRawCtx(true)}
                            className="text-xs bg-zinc-700/40 hover:bg-zinc-700/70 text-zinc-400 px-3 py-1 rounded-md transition-colors"
                          >
                            Inspect
                          </button>
                        )}
                        <button
                          onClick={handleCancelExpansion}
                          className="text-xs bg-red-500/10 hover:bg-red-500/20 text-red-400 px-3 py-1 rounded-md transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                    <div className="flex flex-col gap-2">
                      {expansionCandidates.map((cand, idx) => (
                        <button
                          key={idx}
                          onClick={() => handlePickCandidate(cand.text)}
                          className="w-full text-left p-4 bg-[#1f1f24] hover:bg-[#27272a] border border-white/10 rounded-lg text-zinc-200 font-sans text-lg transition-colors flex items-center gap-3"
                        >
                          <span className="text-zinc-500 font-mono text-sm">{idx + 1}.</span>
                          {cand.text}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <input
                    type="text"
                    value={typedMessage}
                    onChange={(e) => setTypedMessage(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        handleSend();
                      }
                    }}
                    className="text-xl md:text-3xl font-sans font-light tracking-wide w-full bg-transparent border-none outline-none text-zinc-100 caret-blue-500 placeholder:text-zinc-600 placeholder:italic"
                    placeholder="Awaiting keyboard input..."
                  />
                )}
              </div>
              {isSpeaking && (
                <div className="flex flex-col gap-1 pl-2">
                  <div className="text-xs text-blue-400 font-sans flex items-center gap-2 animate-pulse">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                    Synthesizing speech...
                  </div>
                  {speakingText && (
                    <div className="text-xs text-zinc-500 font-mono truncate max-w-full pl-4" title={speakingText}>
                      {speakingText}
                    </div>
                  )}
                </div>
              )}
              {expandInferring && (
                <div className="flex items-center gap-3 py-1">
                  <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
                  <span className="text-xs text-yellow-400/70">Running abbreviation expansion…</span>
                </div>
              )}
              {expandCtx && !expandInferring && (
                <div className="flex flex-col gap-4 pt-2 border-t border-white/5">
                  {/* Tokens */}
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-widest text-zinc-600">Tokens</span>
                    <span className="font-mono text-lg text-zinc-200">{expandCtx.tokens || "(none)"}</span>
                  </div>
                  {/* Steering Context Pills */}
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-widest text-zinc-600">Steering Context</span>
                    <div className="flex flex-wrap gap-2">
                      <span className="text-[10px] bg-zinc-800 border border-white/5 rounded-full px-2.5 py-1 text-zinc-300">
                        Emotion: {b4emotion.label || "—"}
                        {b4emotion.intensity != null ? ` (${Number(b4emotion.intensity).toFixed(2)})` : ""}
                      </span>
                      <span className="text-[10px] bg-zinc-800 border border-white/5 rounded-full px-2.5 py-1 text-zinc-300">
                        Visual: {b4visual.focused_object || "—"}
                      </span>
                      <span className="text-[10px] bg-zinc-800 border border-white/5 rounded-full px-2.5 py-1 text-zinc-300">
                        Pragmatic: {b4pragmatic.intent || "—"}
                      </span>
                    </div>
                  </div>
                  {/* Conversation */}
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-widest text-zinc-600">Conversation</span>
                    <pre className="text-xs whitespace-pre-wrap overflow-y-auto max-h-32 bg-[#0a0a0c] border border-white/5 rounded-xl p-3 text-zinc-400">
                      {expandCtx.conversation_history || "(none)"}
                    </pre>
                  </div>
                  {/* Memories */}
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-widest text-zinc-600">Memories</span>
                    <pre className="text-xs whitespace-pre-wrap overflow-y-auto max-h-24 bg-[#0a0a0c] border border-white/5 rounded-xl p-3 text-zinc-400">
                      {expandCtx.recent_memories || "(none)"}
                    </pre>
                  </div>
                  {/* Candidates */}
                  <div className="flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-widest text-zinc-600">Candidates</span>
                    <div className="flex flex-col gap-1">
                      {(expandCtx.candidates || []).map((c) => (
                        <div key={c.rank} className="flex gap-2 text-xs">
                          <span className="text-zinc-600 w-5 shrink-0">#{c.rank}</span>
                          <span className="text-zinc-300">{c.text}</span>
                        </div>
                      ))}
                      {(!expandCtx.candidates || expandCtx.candidates.length === 0) && (
                        <span className="text-xs text-zinc-600">(none)</span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      }

      case "b5":
        return (
          <div key={id} {...dragProps} className={`w-full max-w-[1128px] ${commonWrapperClasses}`}>
            <div
              className={`flex justify-between items-center text-xs tracking-widest text-zinc-500 cursor-grab active:cursor-grabbing p-6 sm:px-8 ${minimized.b5 ? "sm:py-6" : "sm:pt-8 sm:pb-6"}`}
            >
              <div className="flex items-center gap-3">
                <WindowButton minimized={minimized.b5} onClick={() => setMinimized((p) => ({ ...p, b5: !p.b5 }))} />
                <span>Letter Grid</span>
                <GripHorizontal
                  size={14}
                  className="text-zinc-700 opacity-50 hover:opacity-100 transition-opacity ml-1"
                />
              </div>
              <span>MANUAL_OVERRIDE</span>
            </div>

            <div
              className={`${minimized.b5 ? "hidden" : "flex flex-col gap-8 w-full"} px-6 pb-6 sm:px-8 sm:pb-8 cursor-default`}
            >
              <div className="flex w-full flex-col gap-3">
                <h3 className="text-[11px] tracking-widest text-zinc-400 font-sans font-semibold mb-1">
                  Common words:
                </h3>
                <div className="grid grid-cols-5 md:grid-cols-10 gap-2 md:gap-3">
                  {commonWords.map((word) => (
                    <button
                      key={word}
                      onClick={() => handleKeyClick(`[${word}] `)}
                      className="h-12 md:h-14 bg-[#27272a]/80 hover:bg-[#3f3f46] text-zinc-200 font-sans font-medium text-[15px] md:text-lg rounded-xl transition-all duration-75 active:scale-95 active:bg-zinc-600 border border-white/5 shadow-sm flex items-center justify-center cursor-pointer"
                    >
                      {word}
                    </button>
                  ))}
                </div>
                <div className="grid grid-cols-2 gap-2 md:gap-3 mt-1">
                  <button
                    onClick={() => handleKeyClick(", ")}
                    className="h-12 md:h-14 bg-[#27272a]/80 hover:bg-[#3f3f46] text-zinc-200 font-sans font-semibold text-lg md:text-xl rounded-xl transition-all duration-75 active:scale-95 active:bg-zinc-600 border border-white/5 shadow-sm flex items-center justify-center cursor-pointer"
                  >
                    ,
                  </button>
                  <button
                    onClick={() => handleKeyClick("? ")}
                    className="h-12 md:h-14 bg-[#27272a]/80 hover:bg-[#3f3f46] text-zinc-200 font-sans font-semibold text-lg md:text-xl rounded-xl transition-all duration-75 active:scale-95 active:bg-zinc-600 border border-white/5 shadow-sm flex items-center justify-center cursor-pointer"
                  >
                    ?
                  </button>
                </div>
              </div>

              <div className="flex w-full flex-col gap-3">
                <h3 className="text-[11px] tracking-widest text-zinc-400 font-sans font-semibold mb-1">Letters:</h3>
                <div className="flex flex-col gap-2 md:gap-3">
                  <div className="flex justify-center gap-2 md:gap-3">
                    {lettersRow1.map((l) => (
                      <button
                        key={l}
                        onClick={() => handleKeyClick(l)}
                        className="h-14 md:h-16 flex-1 bg-[#27272a]/80 hover:bg-[#3f3f46] text-zinc-200 font-sans font-medium text-xl md:text-2xl rounded-xl transition-all duration-75 active:scale-95 active:bg-zinc-600 border border-white/5 shadow-sm flex items-center justify-center pb-0.5 cursor-pointer"
                      >
                        {l}
                      </button>
                    ))}
                  </div>
                  <div className="flex justify-center gap-2 md:gap-3">
                    <div className="flex-[0.5]" />
                    {lettersRow2.map((l) => (
                      <button
                        key={l}
                        onClick={() => handleKeyClick(l)}
                        className="h-14 md:h-16 flex-1 bg-[#27272a]/80 hover:bg-[#3f3f46] text-zinc-200 font-sans font-medium text-xl md:text-2xl rounded-xl transition-all duration-75 active:scale-95 active:bg-zinc-600 border border-white/5 shadow-sm flex items-center justify-center pb-0.5 cursor-pointer"
                      >
                        {l}
                      </button>
                    ))}
                    <div className="flex-[0.5]" />
                  </div>
                  <div className="flex justify-center gap-2 md:gap-3">
                    <div className="flex-[1.5]" />
                    {lettersRow3.map((l) => (
                      <button
                        key={l}
                        onClick={() => handleKeyClick(l)}
                        className="h-14 md:h-16 flex-1 bg-[#27272a]/80 hover:bg-[#3f3f46] text-zinc-200 font-sans font-medium text-xl md:text-2xl rounded-xl transition-all duration-75 active:scale-95 active:bg-zinc-600 border border-white/5 shadow-sm flex items-center justify-center pb-0.5 cursor-pointer"
                      >
                        {l}
                      </button>
                    ))}
                    <div className="flex-[1.5]" />
                  </div>
                  <div className="grid grid-cols-2 gap-2 md:gap-3 mt-1 md:mt-2">
                    <button
                      onClick={handleBackspace}
                      className="h-14 md:h-16 bg-[#27272a]/80 hover:bg-[#3f3f46] text-zinc-200 font-sans font-semibold text-lg md:text-xl rounded-xl transition-all duration-75 active:scale-95 active:bg-zinc-600 border border-white/5 shadow-sm flex items-center justify-center cursor-pointer"
                    >
                      Backspace
                    </button>
                    <button
                      onClick={handleSend}
                      className="h-14 md:h-16 bg-[#27272a]/80 hover:bg-[#3f3f46] text-zinc-200 font-sans font-semibold text-lg md:text-xl rounded-xl transition-all duration-75 active:scale-95 active:bg-zinc-600 border border-white/5 shadow-sm flex items-center justify-center cursor-pointer"
                    >
                      Send
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case "b6":
        return (
          <div
            key={id}
            {...dragProps}
            className={`flex-1 min-w-[320px] max-w-[360px] ${minimized.b6 ? "" : "aspect-[3/4]"} ${commonWrapperClasses}`}
          >
            <div className="flex justify-between items-center p-6 text-xs tracking-widest text-zinc-500 cursor-grab active:cursor-grabbing">
              <div className="flex items-center gap-3">
                <WindowButton minimized={minimized.b6} onClick={() => setMinimized((p) => ({ ...p, b6: !p.b6 }))} />
                <span>Connection</span>
                <GripHorizontal
                  size={14}
                  className="text-zinc-700 opacity-50 hover:opacity-100 transition-opacity ml-1"
                />
              </div>
              <span>NEON_STATUS</span>
            </div>

            <div
              className={`flex flex-col flex-1 overflow-hidden transition-opacity duration-300 ${minimized.b6 ? "hidden opacity-0" : "opacity-100"} cursor-default`}
            >
              <div className="flex-1 p-6 flex flex-col justify-center gap-6">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-sans text-zinc-400">User ID</span>
                  <div className="flex items-center gap-2">
                    <span
                      className="text-xs font-mono text-zinc-300 truncate max-w-[120px]"
                      title={sessionData.userId || "Loading..."}
                    >
                      {sessionData.userId ? sessionData.userId.slice(0, 8) : "Loading..."}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-sans text-zinc-400">Session ID</span>
                  <div className="flex items-center gap-2">
                    <span
                      className="text-xs font-mono text-zinc-300 truncate max-w-[120px]"
                      title={sessionData.sessionId || "Loading..."}
                    >
                      {sessionData.sessionId ? sessionData.sessionId.slice(0, 8) : "Loading..."}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-sans text-zinc-400">Backend Status</span>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${wsConnected ? "bg-green-500" : "bg-red-500"}`} />
                    <span className="text-xs font-mono text-zinc-300">
                      {wsConnected ? "Connected" : "Disconnected"}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-sans text-zinc-400">Neon Hardware</span>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${neonStatus.connected ? "bg-green-500" : "bg-red-500"}`} />
                    <span className="text-xs font-mono text-zinc-300">
                      {neonStatus.connected ? neonStatus.device_addr : "No device"}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-sans text-zinc-400">Pupil Stream</span>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${pupilStreaming ? "bg-blue-500" : "bg-zinc-600"}`} />
                    <span className="text-xs font-mono text-zinc-300">{pupilStreaming ? "Active" : "Inactive"}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-sans text-zinc-400">Blink</span>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full transition-colors duration-150 ${blinkFlash ? "bg-cyan-400" : "bg-zinc-600"}`}
                    />
                    <span className="text-xs font-mono text-zinc-300">{blinkFlash ? "Detected" : "—"}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-sans text-zinc-400">Worn</span>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full ${neonWorn === true ? "bg-green-500" : neonWorn === false ? "bg-zinc-600" : "bg-zinc-700"}`}
                    />
                    <span className="text-xs font-mono text-zinc-300">
                      {neonWorn === null ? "—" : neonWorn ? "true" : "false"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case "b7":
        return (
          <div key={id} {...dragProps} className={`flex-1 min-w-[544px] max-w-[744px] ${commonWrapperClasses}`}>
            <div className="flex justify-between items-center p-6 text-xs tracking-widest text-zinc-500 cursor-grab active:cursor-grabbing">
              <div className="flex items-center gap-3">
                <WindowButton minimized={minimized.b7} onClick={() => setMinimized((p) => ({ ...p, b7: !p.b7 }))} />
                <span>Emotion</span>
                <GripHorizontal
                  size={14}
                  className="text-zinc-700 opacity-50 hover:opacity-100 transition-opacity ml-1"
                />
              </div>
              <span>AFFECT_STATE</span>
            </div>

            <div
              className={`flex flex-col flex-1 overflow-hidden transition-opacity duration-300 ${minimized.b7 ? "hidden opacity-0" : "opacity-100"} cursor-default`}
            >
              <div className="flex-1 flex items-center justify-center p-4">
                <svg
                  ref={emotionSvgRef}
                  viewBox="0 0 300 300"
                  className="w-full max-w-[640px] touch-none select-none"
                  onMouseDown={(e) => e.stopPropagation()}
                  onPointerDown={(e) => {
                    e.currentTarget.setPointerCapture(e.pointerId);
                    handleEmotionPointer(e);
                  }}
                  onPointerMove={(e) => {
                    if (e.buttons === 1) handleEmotionPointer(e);
                  }}
                >
                  {EMOTION_WHEEL.map((em) => {
                    const a1 = em.startAngle,
                      a2 = em.startAngle + 45;
                    const midAngle = a1 + 22.5;
                    const midRad = (midAngle * Math.PI) / 180;
                    const labelR = 69;
                    const lx = 150 + labelR * Math.cos(midRad);
                    const ly = 150 + labelR * Math.sin(midRad);
                    return (
                      <g key={em.id}>
                        <path
                          d={sectorPath(150, 150, 12, 100, a1, a2)}
                          fill={em.color}
                          stroke="#16161a"
                          strokeWidth="1.5"
                          style={{ cursor: "crosshair" }}
                        />
                        <path
                          d={sectorPath(150, 150, 104, 148, a1, a1 + 22.5)}
                          fill={em.color}
                          stroke="#16161a"
                          strokeWidth="1.5"
                          style={{ cursor: "crosshair", filter: "brightness(0.75)" }}
                        />
                        <path
                          d={sectorPath(150, 150, 104, 148, a1 + 22.5, a2)}
                          fill={em.color}
                          stroke="#16161a"
                          strokeWidth="1.5"
                          style={{ cursor: "crosshair", filter: "brightness(0.85)" }}
                        />
                        <text
                          x={lx}
                          y={ly}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize="6.5"
                          fontWeight="bold"
                          fill="rgba(0,0,0,0.72)"
                          transform={`rotate(${midAngle + 90}, ${lx}, ${ly})`}
                          style={{ pointerEvents: "none" }}
                        >
                          {em.label}
                        </text>
                        {em.subs.map((sub, si) => {
                          const subAngle = a1 + 11.25 + si * 22.5;
                          const subRad = (subAngle * Math.PI) / 180;
                          const subR = 126;
                          return (
                            <text
                              key={sub}
                              x={150 + subR * Math.cos(subRad)}
                              y={150 + subR * Math.sin(subRad)}
                              textAnchor="middle"
                              dominantBaseline="middle"
                              fontSize="6.5"
                              fontWeight="bold"
                              fill="rgba(0,0,0,0.65)"
                              transform={`rotate(${subAngle + 90}, ${150 + subR * Math.cos(subRad)}, ${150 + subR * Math.sin(subRad)})`}
                              style={{ pointerEvents: "none" }}
                            >
                              {sub}
                            </text>
                          );
                        })}
                      </g>
                    );
                  })}
                  <circle cx="150" cy="150" r="102" fill="none" stroke="#16161a" strokeWidth="1.5" />
                  <circle
                    cx="150"
                    cy="150"
                    r="12"
                    fill="#2a2a30"
                    stroke="#3f3f46"
                    strokeWidth="1"
                    style={{ cursor: "pointer" }}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={() => setEmotionCursor(null)}
                  />
                  {emotionCursor && (
                    <circle
                      cx={emotionCursor.x}
                      cy={emotionCursor.y}
                      r="6"
                      fill={emotionCursor.color}
                      stroke="white"
                      strokeWidth="2"
                      style={{ pointerEvents: "none", filter: "drop-shadow(0 0 4px rgba(255,255,255,0.4))" }}
                    />
                  )}
                </svg>
              </div>

              <div className="px-6 pb-5 border-t border-white/5 pt-4 space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-zinc-500">Emotion</span>
                  <span className="font-mono text-zinc-200" style={{ color: emotionCursor?.color }}>
                    {emotionCursor?.emotion ?? "—"}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-zinc-500">Sub</span>
                  <span className="font-mono text-zinc-400">{emotionCursor?.subEmotion ?? "—"}</span>
                </div>
                <div className="flex items-center gap-3 pt-1">
                  <span className="text-zinc-500 text-xs w-16">Intensity</span>
                  <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-75"
                      style={{
                        width: `${((emotionCursor?.intensity ?? 0) * 100).toFixed(1)}%`,
                        backgroundColor: emotionCursor?.color ?? "#3f3f46",
                      }}
                    />
                  </div>
                  <span className="text-zinc-300 font-mono text-xs w-8 text-right">
                    {emotionCursor ? emotionCursor.intensity.toFixed(2) : "—"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        );

      case "b8":
        const renderZone = (letters, colorClass, borderClass, label, arrow) => (
          <div
            className={`flex flex-col items-center justify-center p-4 rounded-xl border-2 ${borderClass} bg-[#111114] shadow-inner h-[160px] transition-all`}
          >
            <span className={`text-[10px] font-bold tracking-widest ${colorClass} mb-2`}>
              {arrow} {label}
            </span>
            {letters.length === 0 ? (
              <span className="text-zinc-600">—</span>
            ) : letters.length === 1 ? (
              <span className="text-white text-3xl font-bold underline decoration-zinc-500 underline-offset-4">
                {letters[0]}
              </span>
            ) : (
              <div className="flex flex-col items-center gap-1">
                <span className={`font-bold ${colorClass} tracking-[0.2em] text-sm`}>
                  {letters.slice(0, Math.ceil(letters.length / 2)).join(" ")}
                </span>
                <span className={`${colorClass} tracking-[0.2em] text-sm opacity-80`}>
                  {letters.slice(Math.ceil(letters.length / 2)).join(" ")}
                </span>
              </div>
            )}
          </div>
        );

        return (
          <div key={id} {...dragProps} className={`w-full max-w-[1128px] ${commonWrapperClasses}`}>
            <div
              className={`flex justify-between items-center text-xs tracking-widest text-zinc-500 cursor-grab active:cursor-grabbing p-6 sm:px-8 ${minimized.b8 ? "sm:py-6" : "sm:pt-8 sm:pb-4"}`}
            >
              <div className="flex items-center gap-3">
                <WindowButton minimized={minimized.b8} onClick={() => setMinimized((p) => ({ ...p, b8: !p.b8 }))} />
                <span>Compass</span>
                <GripHorizontal
                  size={14}
                  className="text-zinc-700 opacity-50 hover:opacity-100 transition-opacity ml-1"
                />
              </div>
              <span>COMPASS_UI</span>
            </div>

            <div className={`${minimized.b8 ? "hidden" : "flex flex-col"} px-6 pb-6 sm:px-8 sm:pb-8 cursor-default`}>
              <div className="grid gap-3 mx-auto" style={{ gridTemplateColumns: "repeat(3, 160px)" }}>
                <div />
                {renderZone(compassZones.up, "text-blue-400", "border-blue-500/30", "UP", "↑")}
                <div />

                {renderZone(compassZones.left, "text-green-400", "border-green-500/30", "LEFT", "←")}
                <div className="flex flex-col items-center justify-center p-2 rounded-xl border border-zinc-800 bg-[#0a0a0c] shadow-inner h-[160px]">
                  <span className="text-zinc-600 text-3xl mb-1">✛</span>
                  <span className="text-[10px] text-zinc-500 tracking-widest">
                    {compassHistory.length ? `DEPTH ${compassHistory.length}` : "ROOT"}
                  </span>
                </div>
                {renderZone(compassZones.right, "text-orange-400", "border-orange-500/30", "RIGHT", "→")}

                <div />
                {renderZone(compassZones.down, "text-purple-400", "border-purple-500/30", "DOWN", "↓")}
                <div />
              </div>

              <div className="mt-6 flex justify-between items-center text-[10px] text-zinc-600 uppercase tracking-widest max-w-[528px] mx-auto w-full">
                <span>↑↓←→ Navigate</span>
                <span>⌫ Back</span>
                <span>↵ Space</span>
              </div>
            </div>
          </div>
        );

      case "b11": {
        const neoVisual = neonCapture?.parsed_visual || {};
        const neoPragmatic = neonCapture?.pragmatic || {};
        const detectedObjects = neoVisual.detected_objects || [];

        const NEO_VIZ_TABS = [
          { value: "scene", label: "Scene" },
          { value: "saliency_overlay", label: "Saliency" },
          { value: "saliency_peaks", label: "Peaks" },
          { value: "gaze_trajectory", label: "Trajectory" },
          { value: "gaze_fixations", label: "Fixations" },
        ];

        const formatDuration = (s) => {
          if (s == null) return null;
          if (s < 60) return `${s.toFixed(1)}s`;
          const m = Math.floor(s / 60);
          const rem = Math.round(s % 60);
          return `${m}m ${rem}s`;
        };
        const durationLabel = formatDuration(neonDuration);

        // Header badge — four states
        const neoBadge = neonRecording
          ? { label: "RECORDING", cls: "border-red-800 text-red-400 bg-red-950/40 animate-pulse" }
          : neonInferring
            ? { label: "INFERRING", cls: "border-yellow-800 text-yellow-400 bg-yellow-950/40 animate-pulse" }
            : neonCapture
              ? { label: "CAPTURED", cls: "border-emerald-800 text-emerald-400 bg-emerald-950/40" }
              : { label: "WAITING", cls: "border-white/5 text-zinc-600" };

        // Whether we have Phase-1 data (images) to show
        const hasVisuals = neonCaptureKey > 0 && !neonRecording;

        return (
          <div key={id} {...dragProps} className={`w-full ${commonWrapperClasses}`}>
            {/* Header */}
            <div
              className={`flex justify-between items-center text-xs tracking-widest text-zinc-500 cursor-grab active:cursor-grabbing p-6 sm:px-8 ${minimized.b11 ? "sm:py-6" : "sm:pt-8 sm:pb-4"}`}
            >
              <div className="flex items-center gap-3">
                <WindowButton minimized={minimized.b11} onClick={() => setMinimized((p) => ({ ...p, b11: !p.b11 }))} />
                <span>Neon Capture</span>
                <GripHorizontal
                  size={14}
                  className="text-zinc-700 opacity-50 hover:opacity-100 transition-opacity ml-1"
                />
              </div>
              <div className="flex items-center gap-2">
                {(hasVisuals || neonCaptureError || neonCapture) && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setNeonCapture(null);
                      setNeonDuration(null);
                      setNeonCaptureError(null);
                      setNeonCaptureKey(0);
                    }}
                    className="text-[10px] px-2 py-0.5 rounded-full border border-red-800 text-red-400 bg-red-950/40 hover:bg-red-900/60 transition-colors uppercase tracking-widest cursor-pointer"
                  >
                    Clear
                  </button>
                )}
                <span className={`text-[10px] px-2 py-0.5 rounded-full border ${neoBadge.cls}`}>{neoBadge.label}</span>
              </div>
            </div>

            <div className={`${minimized.b11 ? "hidden" : "flex flex-col"} cursor-default`}>
              <div className="flex flex-col gap-4 px-6 pb-6 sm:px-8 sm:pb-8">
                {/* State: idle */}
                {!neonRecording && !neonInferring && !hasVisuals && (
                  <span className="text-xs text-zinc-600">No capture — double blink to start recording</span>
                )}

                {/* State: recording */}
                {neonRecording && (
                  <div className="flex items-center gap-3 py-2">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-xs text-red-400">Recording in progress — double blink to stop</span>
                  </div>
                )}

                {/* States: inferring or complete — show visuals as soon as Phase 1 is done */}
                {hasVisuals && (
                  <>
                    {/* Visualization tab bar */}
                    <div className="flex gap-1 flex-wrap">
                      {NEO_VIZ_TABS.map((tab) => (
                        <button
                          key={tab.value}
                          onClick={() => setNeoVizTab(tab.value)}
                          className={`text-[10px] uppercase tracking-widest px-2.5 py-1 rounded-lg transition-colors ${neoVizTab === tab.value ? "bg-white/10 text-zinc-200" : "text-zinc-600 hover:text-zinc-400 hover:bg-white/5"}`}
                        >
                          {tab.label}
                        </button>
                      ))}
                    </div>

                    {/* Visualization image */}
                    <div className="bg-[#0a0a0c] border border-white/5 rounded-xl overflow-hidden aspect-video flex items-center justify-center">
                      <img
                        key={`${neonCaptureKey}-${neoVizTab}`}
                        src={`/viz/live-frame?type=${neoVizTab}&k=${neonCaptureKey}`}
                        alt={neoVizTab}
                        className="w-full h-full object-contain"
                        onError={(e) => {
                          e.target.style.display = "none";
                        }}
                      />
                    </div>

                    {/* VLM result — inferring placeholder or final result */}
                    {neonInferring ? (
                      <div className="flex items-center gap-3 py-1">
                        <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
                        <span className="text-xs text-yellow-400/70">Running scene inference…</span>
                      </div>
                    ) : neonCaptureError ? (
                      <div className="flex items-start gap-3 py-1">
                        <span className="w-2 h-2 rounded-full bg-red-500 shrink-0 mt-1" />
                        <span className="text-xs text-red-400/90 break-words line-clamp-3" title={neonCaptureError}>
                          Error: {neonCaptureError}
                        </span>
                      </div>
                    ) : (
                      neonCapture && (
                        <>
                          {/* Parsed Visual */}
                          <div className="flex flex-col gap-2">
                            <span className="text-[10px] uppercase tracking-widest text-zinc-600">Parsed Visual</span>
                            <div className="flex flex-wrap gap-2">
                              {neoVisual.focused_object && (
                                <span className="text-[10px] bg-zinc-800 border border-white/5 rounded-full px-2.5 py-1 text-zinc-200">
                                  Focus: {neoVisual.focused_object}
                                </span>
                              )}
                              {detectedObjects.slice(0, 6).map((obj) => (
                                <span
                                  key={obj}
                                  className="text-[10px] bg-zinc-800/50 border border-white/5 rounded-full px-2.5 py-1 text-zinc-400"
                                >
                                  {obj}
                                </span>
                              ))}
                            </div>
                          </div>

                          {/* Pragmatic */}
                          <div className="flex flex-col gap-2">
                            <span className="text-[10px] uppercase tracking-widest text-zinc-600">Pragmatic</span>
                            <div className="flex flex-wrap gap-2">
                              {neoPragmatic.intent && (
                                <span className="text-[10px] bg-zinc-800 border border-white/5 rounded-full px-2.5 py-1 text-zinc-300">
                                  Intent: {neoPragmatic.intent}
                                </span>
                              )}
                              {neoPragmatic.audience && (
                                <span className="text-[10px] bg-zinc-800 border border-white/5 rounded-full px-2.5 py-1 text-zinc-300">
                                  Audience: {neoPragmatic.audience}
                                </span>
                              )}
                              {neoPragmatic.physical_setting && (
                                <span className="text-[10px] bg-zinc-800 border border-white/5 rounded-full px-2.5 py-1 text-zinc-300">
                                  Setting: {neoPragmatic.physical_setting}
                                </span>
                              )}
                            </div>
                          </div>
                        </>
                      )
                    )}
                  </>
                )}
              </div>
              {hasVisuals && durationLabel && (
                <div className="p-6 pt-4 sm:px-8 border-t border-white/5 flex justify-between items-center text-xs text-zinc-500 cursor-default mt-auto">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">Capture Length</span>
                  <span className="w-24 text-right font-mono tabular-nums tracking-wider text-zinc-400">
                    {durationLabel}
                  </span>
                </div>
              )}
            </div>
          </div>
        );
      }

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-[#111114] text-zinc-400 font-mono flex flex-col items-center justify-start gap-6 p-6 select-none overflow-x-hidden">
      {/* Top Header / Widget Menu */}
      <div className="w-full max-w-[1128px] flex justify-between items-center z-50 relative">
        <div className="text-zinc-600 text-[10px] tracking-[0.3em] font-semibold">SYSTEM_DASHBOARD</div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-[#16161a] border border-white/5 rounded-xl px-3 h-[38px] focus-within:border-white/20 transition-colors shadow-sm">
            <input
              type="text"
              placeholder="User ID..."
              value={inputUserId}
              onChange={(e) => setInputUserId(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") initSession(inputUserId || null);
              }}
              className="bg-transparent text-xs text-zinc-300 tracking-widest outline-none w-24 placeholder:text-zinc-600 h-full"
            />
            <button
              onClick={() => initSession(inputUserId || null)}
              className="flex items-center justify-center text-[10px] font-bold tracking-wider bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded px-2 py-1 transition-colors focus:outline-none"
              title="Start New Session"
            >
              NEW
            </button>
          </div>

          <div className="relative">
            <button
              onClick={() => setShowWidgetMenu(!showWidgetMenu)}
              className={`flex items-center gap-2 text-xs tracking-widest transition-all duration-300 px-4 h-[38px] rounded-xl border ${showWidgetMenu ? "bg-white/10 border-white/20 text-zinc-200" : "bg-[#16161a] border-white/5 text-zinc-500 hover:text-zinc-300 hover:bg-white/5"} shadow-sm focus:outline-none`}
            >
              <LayoutGrid size={14} /> WIDGETS{" "}
              <ChevronDown
                size={12}
                className={`transition-transform duration-300 ${showWidgetMenu ? "rotate-180" : ""}`}
              />
            </button>

            {showWidgetMenu && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setShowWidgetMenu(false)} />
                <div className="absolute right-0 top-full mt-3 w-56 bg-[#1f1f24]/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl overflow-hidden z-50 animate-in fade-in slide-in-from-top-2 duration-200">
                  <div className="px-4 py-3 text-[10px] tracking-widest text-zinc-500 border-b border-white/5 bg-black/20">
                    ACTIVE_MODULES
                  </div>
                  <div className="p-1.5 flex flex-col gap-0.5">
                    {Object.entries(widgetNames).map(([id, name]) => (
                      <button
                        key={id}
                        onClick={() => toggleBlockVisibility(id)}
                        className="w-full text-left px-3 py-2.5 text-xs flex items-center justify-between hover:bg-white/5 rounded-lg transition-colors text-zinc-300"
                      >
                        <span className={`${!visibleBlocks[id] && "opacity-50"}`}>{name}</span>
                        <div
                          className={`w-3.5 h-3.5 rounded-[4px] border ${visibleBlocks[id] ? "bg-blue-500 border-blue-500" : "border-zinc-600"} flex items-center justify-center transition-colors shadow-inner`}
                        >
                          {visibleBlocks[id] && <Check size={10} className="text-white" strokeWidth={4} />}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Main Grid area */}
      {/* Dynamic Masonry-style Flex Container */}
      <div className="w-full max-w-[1128px] flex flex-wrap items-start justify-center gap-6 mx-auto">
        {blocksOrder.filter((id) => visibleBlocks[id]).map(renderBlock)}
      </div>

      {/* Inspect LLM Context modal */}
      {showExpandRawCtx && expandRawPrompts && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onClick={() => setShowExpandRawCtx(false)}
        >
          <div
            className="relative w-full max-w-3xl max-h-[80vh] mx-4 bg-[#0f0f12] border border-white/10 rounded-2xl shadow-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
              <div className="flex flex-col gap-0.5">
                <span className="text-xs tracking-widest text-zinc-400">LLM CONTEXT INSPECTOR</span>
                {expandRawPrompts.model && (
                  <span className="text-[10px] text-zinc-600 font-mono">{expandRawPrompts.model}</span>
                )}
              </div>
              <button
                onClick={() => setShowExpandRawCtx(false)}
                className="text-zinc-500 hover:text-zinc-200 transition-colors text-lg leading-none"
              >
                ✕
              </button>
            </div>
            {/* Modal body */}
            <div className="flex flex-col gap-6 overflow-y-auto p-6">
              <div className="flex flex-col gap-2">
                <span className="text-[10px] uppercase tracking-widest text-zinc-600">System Prompt</span>
                <pre className="text-xs whitespace-pre-wrap font-mono text-zinc-300 bg-[#0a0a0c] border border-white/5 rounded-xl p-4 leading-relaxed">
                  {expandRawPrompts.system_prompt || "(none)"}
                </pre>
              </div>
              <div className="flex flex-col gap-2">
                <span className="text-[10px] uppercase tracking-widest text-zinc-600">User Prompt</span>
                <pre className="text-xs whitespace-pre-wrap font-mono text-zinc-300 bg-[#0a0a0c] border border-white/5 rounded-xl p-4 leading-relaxed">
                  {expandRawPrompts.user_prompt || "(none)"}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
