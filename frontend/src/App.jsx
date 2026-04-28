import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Navigate, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import L from "leaflet";

const BACKEND_BASE_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_BACKEND_BASE_URL) ||
  "http://localhost:8000";
const TILE_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_TILE_URL) ||
  "http://127.0.0.1:8080/styles/klokantech-basic/{z}/{x}/{y}.png";
const STORAGE_KEY = "ea-react-state";
const HABITAT_COLORS = ["#86a8ff", "#5f7dde", "#69b38a", "#d2a45f", "#5ea6c6", "#8f88de"];
const HABITAT_CLASS_COLORS = {
  forest: "#567d46",
  grassland: "#8fbf5f",
  shrubland: "#b88a52",
  water: "#5a8fcf",
  unknown: "#8f88de",
};
const PIPELINE_OPTIONS = [
  {
    value: "fast_trend",
    label: "Fast Trend",
    speed: "Fastest turnaround",
    continuity: "Light continuity",
    evidence: "Lean evidence set",
    note: "Quick review-oriented processing for rapid trend assessment.",
    useCase: "Best for rapid triage, patrol review, and quick trend checks.",
    emphasis: "Review first",
  },
  {
    value: "quality",
    label: "Quality",
    speed: "Heaviest runtime",
    continuity: "Strongest continuity",
    evidence: "Comprehensive evidence",
    note: "Highest-confidence analysis path with comprehensive processing.",
    useCase: "Best for final review, reporting, and continuity-sensitive analysis runs.",
    emphasis: "Highest confidence",
  },
];
const VISIBLE_MODEL_KEYS = ["labeling_data_v2_s", "roboflow_v1_fresh2"];
const MODEL_META = {
  labeling_data_v2_s: {
    label: "Research Model v3_2s",
    badge: "Recommended",
    note: "Primary internal model for most runs. Use this by default.",
  },
  roboflow_v1_fresh2: {
    label: "Online Model",
    badge: "Alternative",
    note: "External-style online baseline for comparison checks.",
  },
};
const PROCESSING_STAGES = [
  { label: "Preparing run", note: "Locking the source and warming the local pipeline.", accent: "Source check" },
  { label: "Indexing frames", note: "Sampling the video for the first pass.", accent: "Frame scan" },
  { label: "Tracking movement", note: "Linking detections across the route.", accent: "Continuity pass" },
  { label: "Building evidence", note: "Writing review frames and evidence assets.", accent: "Evidence output" },
  { label: "Almost there", note: "Finalizing the run bundle for results review.", accent: "Packaging" },
];
const PROCESSING_FINISH_STAGE = {
  label: "Results ready",
  note: "Wrapping the run and opening the results workspace.",
  accent: "Final handoff",
};
const STATUS_TONE = {
  complete: "status-complete",
  processing: "status-processing",
  cancel_requested: "status-processing",
  cancelled: "status-failed",
  queued: "status-queued",
  uploaded: "status-queued",
  failed: "status-failed",
};

const PAGE_TITLES = {
  "/": "Overview",
  "/analyze": "Analyze",
  "/results": "Results",
  "/review": "Results",
  "/history": "History",
};

function clsx(...parts) {
  return parts.filter(Boolean).join(" ");
}

function formatDate(value) {
  if (!value) {
    return "-";
  }
  const normalized = String(value)
    .replace(/^(\d{4}):(\d{2}):(\d{2})/, "$1-$2-$3")
    .replace(" ", "T");
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return new Intl.DateTimeFormat("en-MY", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function formatShortDate(value) {
  if (!value) {
    return "-";
  }
  const normalized = String(value)
    .replace(/^(\d{4}):(\d{2}):(\d{2})/, "$1-$2-$3")
    .replace(" ", "T");
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return new Intl.DateTimeFormat("en-MY", {
    month: "long",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

function formatTimeOnly(value) {
  if (!value) {
    return "-";
  }
  const normalized = String(value)
    .replace(/^(\d{4}):(\d{2}):(\d{2})/, "$1-$2-$3")
    .replace(" ", "T");
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }
  return new Intl.DateTimeFormat("en-MY", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function formatDuration(seconds) {
  const total = Number(seconds || 0);
  if (!Number.isFinite(total) || total <= 0) {
    return "-";
  }
  const mins = Math.floor(total / 60);
  const secs = Math.round(total % 60);
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function formatMetric(value, digits = 0) {
  const num = Number(value ?? 0);
  if (!Number.isFinite(num)) {
    return "-";
  }
  if (Math.abs(num) >= 1000 && digits === 0) {
    return num.toLocaleString("en-US");
  }
  return digits > 0 ? num.toFixed(digits) : String(Math.round(num));
}

function formatPercent(value, digits = 1) {
  const num = Number(value ?? 0);
  if (!Number.isFinite(num)) {
    return "-";
  }
  return `${num.toFixed(digits)}%`;
}

function formatFileSize(bytes) {
  const value = Number(bytes || 0);
  if (!Number.isFinite(value) || value <= 0) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const digits = unitIndex >= 2 ? 2 : 0;
  return `${size.toFixed(digits)} ${units[unitIndex]}`;
}

function formatJobIdCompact(value) {
  const raw = String(value || "");
  if (!raw) {
    return "No job yet";
  }
  if (raw.length <= 18) {
    return raw;
  }
  return `${raw.slice(0, 8)}...${raw.slice(-6)}`;
}

function sampleFrameIndices(frameIndices, maxCount = 8) {
  const indices = Array.isArray(frameIndices) ? frameIndices.filter((value) => Number.isFinite(Number(value))) : [];
  if (!indices.length) {
    return [];
  }
  if (indices.length <= maxCount) {
    return indices;
  }
  const out = [];
  const lastIndex = indices.length - 1;
  for (let step = 0; step < maxCount; step += 1) {
    const sourceIndex = Math.round((step / (maxCount - 1)) * lastIndex);
    out.push(indices[sourceIndex]);
  }
  return [...new Set(out)];
}

function inferFileKind(file) {
  const name = String(file?.name || "").toLowerCase();
  if (/\.(mp4|mov|avi)$/.test(name)) {
    return "Video";
  }
  if (/\.(jpg|jpeg|png)$/.test(name)) {
    return "Image";
  }
  return "File";
}

function inferMediaTypeFromFilename(name) {
  const lower = String(name || "").toLowerCase();
  if (/\.(mp4|mov|avi)$/.test(lower)) {
    return "Video";
  }
  if (/\.(jpg|jpeg|png)$/.test(lower)) {
    return "Image";
  }
  return "File";
}

function validateSingleSource(files) {
  const list = Array.from(files || []);
  if (!list.length) {
    return { file: null, error: "Choose one image or video to continue." };
  }
  if (list.length > 1) {
    return { file: null, error: "Only one image or one video can be uploaded per analysis run." };
  }
  const [file] = list;
  const name = String(file?.name || "").toLowerCase();
  if (!/\.(mp4|mov|avi|jpg|jpeg|png)$/.test(name)) {
    return { file: null, error: "Unsupported file type. Use MP4, MOV, AVI, JPG, JPEG, or PNG." };
  }
  return { file, error: "" };
}

function modelLabel(key) {
  return MODEL_META[key]?.label || key || "Unknown model";
}

function humanizeFrameLabel(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num < 0) {
    return "-";
  }
  return `Frame ${Math.max(1, num + 1)}`;
}

function detectionClassLabel(row) {
  const raw = String(row?.cls_label || "").trim();
  if (!raw) {
    return "";
  }
  const normalized = raw.toLowerCase();
  if (/^\d+$/.test(raw)) {
    return "";
  }
  if (/^class\s*\d+$/i.test(raw)) {
    return "";
  }
  if (normalized === "unknown" || normalized === "n/a" || normalized === "na" || normalized === "-") {
    return "";
  }
  return raw;
}

function normalizeAgeClassLabel(value) {
  const txt = String(value || "").trim().toLowerCase();
  if (txt.includes("baby") || txt.includes("calf")) {
    return "Baby";
  }
  if (txt.includes("juvenile") || txt.includes("subadult")) {
    return "Juvenile";
  }
  if (txt.includes("adult")) {
    return "Adult";
  }
  return "Unknown";
}

function parseTrackId(value) {
  if (value == null) {
    return null;
  }
  const txt = String(value).trim();
  if (/^\d+$/.test(txt)) {
    return Number(txt);
  }
  const digits = txt.replace(/\D+/g, "");
  return digits ? Number(digits) : null;
}

function detectionClassId(row) {
  const value = Number(row?.cls_id);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function detectionClassIdLabel(row) {
  const value = detectionClassId(row);
  return value == null ? "" : `Class ${value}`;
}

function detectionAgeLabel(row) {
  const label = normalizeAgeClassLabel(row?.cls_label);
  return label === "Unknown" ? "" : label;
}

function getDetectionsForFrame(detections, frameIndex) {
  const rows = Array.isArray(detections) ? detections : [];
  if (frameIndex == null) {
    return [];
  }
  const target = Number(frameIndex);
  const exact = rows.filter((row) => Number(row.frame_index) === target);
  return exact;
}

function habitatColorForLabel(label) {
  const raw = String(label || "").trim().toLowerCase();
  if (!raw) {
    return HABITAT_CLASS_COLORS.unknown;
  }
  if (raw.includes("forest")) {
    return HABITAT_CLASS_COLORS.forest;
  }
  if (raw.includes("grass")) {
    return HABITAT_CLASS_COLORS.grassland;
  }
  if (raw.includes("shrub") || raw.includes("bush")) {
    return HABITAT_CLASS_COLORS.shrubland;
  }
  if (raw.includes("water") || raw.includes("river") || raw.includes("wet")) {
    return HABITAT_CLASS_COLORS.water;
  }
  return HABITAT_CLASS_COLORS.unknown;
}

function sampleArray(items, maxCount = 4) {
  const list = Array.isArray(items) ? items : [];
  if (list.length <= maxCount) {
    return list;
  }
  const out = [];
  const lastIndex = list.length - 1;
  for (let step = 0; step < maxCount; step += 1) {
    const sourceIndex = Math.round((step / (maxCount - 1)) * lastIndex);
    out.push(list[sourceIndex]);
  }
  return [...new Set(out)];
}

function toRadians(value) {
  return (value * Math.PI) / 180;
}

function haversineKm(a, b) {
  const earthRadiusKm = 6371;
  const dLat = toRadians(b.lat - a.lat);
  const dLon = toRadians(b.lon - a.lon);
  const lat1 = toRadians(a.lat);
  const lat2 = toRadians(b.lat);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * earthRadiusKm * Math.asin(Math.sqrt(h));
}

function sanitizeGpsTrack(rows) {
  const points = (rows || [])
    .map((row) => ({
      lat: Number(row.lat),
      lon: Number(row.lon),
      ts: row.ts || null,
    }))
    .filter((row) => Number.isFinite(row.lat) && Number.isFinite(row.lon))
    .filter((row) => Math.abs(row.lat) <= 90 && Math.abs(row.lon) <= 180);

  const sanitized = [];
  for (const point of points) {
    const prev = sanitized[sanitized.length - 1];
    if (!prev) {
      sanitized.push(point);
      continue;
    }
    const segmentKm = haversineKm(prev, point);
    if (segmentKm < 0.003) {
      continue;
    }
    if (segmentKm > 2.5) {
      continue;
    }
    sanitized.push(point);
  }
  return sanitized;
}

function appStateModeFromMetrics(metrics) {
  const mode = String(metrics?.tracking_bundle?.mode || metrics?.pipeline_mode || "").trim().toLowerCase();
  return mode === "quality" ? "Quality" : mode === "fast_trend" || mode === "fast" ? "Fast Trend" : "-";
}

function normalizePipelineMode(value) {
  const mode = String(value || "").trim().toLowerCase();
  if (mode === "quality") {
    return "quality";
  }
  if (mode === "fast_trend" || mode === "fast") {
    return "fast_trend";
  }
  return "";
}

function modeLabelFromKey(value) {
  const meta = modeMeta(normalizePipelineMode(value));
  return meta?.label || "-";
}

function resolveJobModeKey(job, metrics) {
  return normalizePipelineMode(job?.pipeline_mode || metrics?.tracking_bundle?.mode || metrics?.pipeline_mode);
}

function resolveJobModelKey(job, metrics) {
  const explicit = String(job?.model_key || "").trim();
  if (explicit) {
    return explicit;
  }
  return appStateModelFromJob(job, metrics);
}

function hasMeaningfulTimelineRow(row) {
  if (!row) {
    return false;
  }
  const elephantId = Number(row.elephant_id || 0);
  const durationFrames = Number(row.duration_frames || 0);
  const firstSeen = Number(row.first_seen || 0);
  const lastSeen = Number(row.last_seen || 0);
  const ageClass = detectionClassLabel({ cls_label: row.age_class });
  return elephantId > 0 && (durationFrames > 1 || lastSeen > firstSeen || Boolean(ageClass));
}

function buildInterpretation({ mediaType, metrics, gpsRows, modeKey }) {
  const detections = formatMetric(metrics?.detections_count);
  const peak = formatMetric(metrics?.peak_herd_size);
  const average = formatMetric(metrics?.avg_herd_size, 1);
  const dominantHabitat = String(metrics?.dominant_habitat || "").trim();
  const modeLabel = modeLabelFromKey(modeKey);
  const lead =
    mediaType === "Image"
      ? `Single-image evidence run with ${detections} sampled detections in one inspection surface.`
      : `${modeLabel !== "-" ? `${modeLabel} ` : ""}video run with ${detections} detections, peaking at ${peak} visible elephants and averaging ${average}.`;
  if (dominantHabitat && dominantHabitat !== "-") {
    return `${lead} Dominant habitat reads as ${dominantHabitat}.`;
  }
  if (mediaType === "Video" && Array.isArray(gpsRows) && gpsRows.length) {
    return `${lead} GPS metadata is present, but route distance remains an estimated travel value only.`;
  }
  return lead;
}

function modeMeta(value) {
  return PIPELINE_OPTIONS.find((item) => item.value === value) || null;
}

function reportBackendReachable(detail = {}) {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new CustomEvent("ea-backend-reachable", { detail }));
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BACKEND_BASE_URL}${path}`, options);
  } catch (error) {
    throw error;
  }
  reportBackendReachable({ path, status: response.status });
  if (!response.ok) {
    let message = "";
    try {
      const payload = await response.json();
      message = payload?.detail || payload?.message || "";
    } catch {
      try {
        message = await response.text();
      } catch {
        message = "";
      }
    }
    throw new Error(message || `Request failed: ${response.status}`);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

function requestJobCancel(jobId, { keepalive = false } = {}) {
  if (!jobId) {
    return Promise.resolve(null);
  }
  return fetch(`${BACKEND_BASE_URL}/jobs/${jobId}/cancel`, {
    method: "POST",
    keepalive,
  }).catch(() => null);
}

function formatRelativeStatusTime(value) {
  const stamp = Number(value || 0);
  if (!Number.isFinite(stamp) || stamp <= 0) {
    return "";
  }
  const deltaSeconds = Math.max(0, Math.round((Date.now() - stamp) / 1000));
  if (deltaSeconds < 8) {
    return "just now";
  }
  if (deltaSeconds < 60) {
    return `${deltaSeconds}s ago`;
  }
  const minutes = Math.floor(deltaSeconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function deriveHealthPresentation(health) {
  const stage = String(health?.stage || "starting");
  if (stage === "healthy") {
    return {
      systemLabel: "System Ready",
      systemTone: "status-online",
      backendLabel: "Backend Healthy",
      backendTone: "status-online",
      showBanner: false,
      bannerTone: "",
      bannerTitle: "",
      bannerBody: "",
    };
  }
  if (stage === "degraded") {
    return {
      systemLabel: "System Degraded",
      systemTone: "status-warning",
      backendLabel: "Backend Unstable",
      backendTone: "status-warning",
      showBanner: true,
      bannerTone: "backend-banner-warning",
      bannerTitle: "Backend connection is unstable",
      bannerBody:
        health?.message || "Requests are intermittently failing. Review actions may still work, but new analysis runs are at risk.",
    };
  }
  if (stage === "offline") {
    return {
      systemLabel: "System Interrupted",
      systemTone: "status-failed",
      backendLabel: "Backend Offline",
      backendTone: "status-failed",
      showBanner: true,
      bannerTone: "backend-banner-danger",
      bannerTitle: "Backend is unavailable",
      bannerBody:
        health?.message || "The local backend is not responding. Start or restart the backend before launching another run.",
    };
  }
  return {
    systemLabel: "System Starting",
    systemTone: "status-warning",
    backendLabel: "Checking Backend",
    backendTone: "status-warning",
    showBanner: true,
    bannerTone: "backend-banner-neutral",
    bannerTitle: "Checking local backend",
    bannerBody: health?.message || "Waiting for the backend to respond on the local device.",
  };
}

function usePersistedAppState() {
  const [state, setState] = useState(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw
        ? JSON.parse(raw)
        : {
            selectedJobId: null,
            uploadedJobId: null,
            selectedModel: "labeling_data_v2_s",
            selectedMode: "fast_trend",
            lastProcessResponse: null,
          };
    } catch {
      return {
        selectedJobId: null,
        uploadedJobId: null,
        selectedModel: "labeling_data_v2_s",
        selectedMode: "fast_trend",
        lastProcessResponse: null,
      };
    }
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [state]);

  return [state, setState];
}

function useJobs(refreshKey) {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    api("/jobs")
      .then((payload) => {
        if (!active) {
          return;
        }
        setJobs(Array.isArray(payload) ? payload : []);
        setError("");
      })
      .catch((err) => {
        if (!active) {
          return;
        }
        setError(err.message || "Failed to load jobs.");
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  return { jobs, loading, error };
}

function useJobBundle(jobId, refreshKey) {
  const [state, setState] = useState({
    loading: false,
    error: "",
    metrics: null,
    results: null,
    detections: null,
    timeline: null,
  });

  useEffect(() => {
    if (!jobId) {
      setState({
        loading: false,
        error: "",
        metrics: null,
        results: null,
        detections: null,
        timeline: null,
      });
      return;
    }

    let active = true;
    setState((current) => ({ ...current, loading: true, error: "" }));

    Promise.all([
      api(`/metrics/${jobId}`),
      api(`/results/${jobId}`),
      api(`/detections/${jobId}`),
      api(`/timeline/${jobId}`),
    ])
      .then(([metrics, results, detections, timeline]) => {
        if (!active) {
          return;
        }
        setState({
          loading: false,
          error: "",
          metrics,
          results,
          detections,
          timeline,
        });
      })
      .catch((err) => {
        if (!active) {
          return;
        }
        setState((current) => ({
          ...current,
          loading: false,
          error: err.message || "Failed to load selected analysis bundle.",
        }));
      });

    return () => {
      active = false;
    };
  }, [jobId, refreshKey]);

  return state;
}

function usePreviewMeta(jobId, refreshKey, options = {}) {
  const [state, setState] = useState({
    loading: false,
    error: "",
    meta: null,
  });
  const pipelineMode = options.pipelineMode || "";
  const live = Boolean(options.live);

  useEffect(() => {
    if (!jobId) {
      setState({ loading: false, error: "", meta: null });
      return;
    }

    let active = true;
    let timer = null;
    const params = new URLSearchParams();
    if (pipelineMode) {
      params.set("pipeline_mode", pipelineMode);
    }
    const query = params.toString();
    const url = `/media/preview/${jobId}/meta${query ? `?${query}` : ""}`;

    const load = ({ background = false } = {}) => {
      if (!background) {
        setState((current) => ({ ...current, loading: true, error: "" }));
      }
      api(url)
        .then((meta) => {
          if (!active) {
            return;
          }
          setState({ loading: false, error: "", meta });
        })
        .catch((err) => {
          if (!active) {
            return;
          }
          setState({
            loading: false,
            error: err.message || "Failed to load frame preview metadata.",
            meta: null,
          });
        });
    };

    const handleVisibility = () => {
      if (document.visibilityState === "visible") {
        load({ background: true });
      }
    };

    load();
    if (live) {
      timer = window.setInterval(() => load({ background: true }), 2000);
    }
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      active = false;
      document.removeEventListener("visibilitychange", handleVisibility);
      if (timer) {
        window.clearInterval(timer);
      }
    };
  }, [jobId, refreshKey, pipelineMode, live]);

  return state;
}

function AppShell({ children, health, processingCount, onRetryHealth }) {
  const location = useLocation();
  const [, setRelativeClock] = useState(0);
  const healthUi = deriveHealthPresentation(health);
  const lastSeenLabel = formatRelativeStatusTime(health?.lastSuccessAt);
  useEffect(() => {
    const tick = () => setRelativeClock((value) => value + 1);
    const handleVisibility = () => {
      if (document.visibilityState === "visible") {
        tick();
      }
    };
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") {
        tick();
      }
    }, 15000);
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, []);
  const navItems = [
    { to: "/", label: "Overview", description: "System snapshot and recent activity." },
    { to: "/analyze", label: "Analyze", description: "Lock a source and launch a new run." },
    { to: "/results", label: "Results", description: "Read the current run summary and metrics." },
    { to: "/history", label: "History", description: "Browse the local archive of past runs." },
  ];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">EA</div>
          <div>
            <h1>Elephant Analytics</h1>
            <p>Wildlife Intelligence Platform</p>
          </div>
        </div>
        <nav className="topnav">
          {navItems.map((item) => {
            const active = location.pathname === item.to;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                title={item.description}
                data-description={item.description}
                className={clsx("topnav-link", active && "topnav-link-active")}
              >
                {item.label}
              </NavLink>
            );
          })}
        </nav>
        <div className="topbar-status">
          <span className={clsx("status-chip", healthUi.systemTone)}>{healthUi.systemLabel}</span>
          <span className={clsx("status-chip", healthUi.backendTone)}>
            {healthUi.backendLabel}
          </span>
          {processingCount > 0 ? (
            <span className="status-chip status-processing">{processingCount} Processing</span>
          ) : null}
        </div>
      </header>
      {healthUi.showBanner ? (
        <div className={clsx("backend-banner", healthUi.bannerTone)}>
          <div className="backend-banner-copy">
            <strong>{healthUi.bannerTitle}</strong>
            <p>{healthUi.bannerBody}</p>
          </div>
          <div className="backend-banner-meta">
            <span>{lastSeenLabel ? `Last stable response ${lastSeenLabel}` : "No stable backend response yet"}</span>
            <button className="button button-muted backend-banner-button" type="button" onClick={onRetryHealth}>
              Retry Connection
            </button>
          </div>
        </div>
      ) : null}
      <main className="workspace">{children}</main>
    </div>
  );
}

function PageHeader({ title, description, actions }) {
  return (
    <section className="page-header">
      <div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </section>
  );
}

function Panel({ title, subtitle, actions, children, className = "" }) {
  return (
    <section className={clsx("panel", className)}>
      {(title || subtitle || actions) && (
        <div className="panel-header">
          <div>
            {title ? <h3>{title}</h3> : null}
            {subtitle ? <p>{subtitle}</p> : null}
          </div>
          {actions ? <div className="panel-actions">{actions}</div> : null}
        </div>
      )}
      {children}
    </section>
  );
}

function MetricCard({ icon, value, label, detail }) {
  return (
    <article className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      {detail ? <div className="metric-detail">{detail}</div> : null}
    </article>
  );
}

function EmptyState({ title, body, action }) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      <p>{body}</p>
      {action}
    </div>
  );
}

function SegmentedControl({ items, value, onChange }) {
  return (
    <div className="segmented-control">
      {items.map((item) => (
        <button
          key={item.value}
          type="button"
          className={clsx("segmented-pill", value === item.value && "segmented-pill-active")}
          onClick={() => onChange(item.value)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

function FrameReviewPanel({
  jobId,
  detections,
  previewMeta,
  fps,
  title = "Frame Review",
  subtitle = "Drag through sampled frames and inspect saved detections.",
  compact = false,
  mediaType = "Video",
  modeKey = "",
}) {
  const frameIndices = previewMeta?.frame_indices || [];
  const [frameCursor, setFrameCursor] = useState(0);
  const [idQuery, setIdQuery] = useState("");
  const isImageMode = mediaType === "Image";
  const isFastVideoMode = !isImageMode && normalizePipelineMode(modeKey) === "fast_trend";
  const showDetectionList = isImageMode || !isFastVideoMode;

  useEffect(() => {
    setFrameCursor(0);
  }, [jobId]);

  const maxCursor = Math.max(frameIndices.length - 1, 0);
  const clampedCursor = Math.max(0, Math.min(frameCursor, maxCursor));
  const selectedFrameIndex = frameIndices[clampedCursor];

  const frameRows = useMemo(() => {
    return getDetectionsForFrame(detections, selectedFrameIndex);
  }, [detections, selectedFrameIndex]);

  const frameImageSrc = useMemo(() => {
    if (!jobId || selectedFrameIndex == null) {
      return "";
    }
    return `${BACKEND_BASE_URL}/media/frame/${jobId}/${selectedFrameIndex}`;
  }, [jobId, selectedFrameIndex]);

  const frameTimeLabel = useMemo(() => {
    if (selectedFrameIndex == null) {
      return "-";
    }
    return formatDuration((fps || 0) > 0 ? selectedFrameIndex / fps : selectedFrameIndex);
  }, [fps, selectedFrameIndex]);

  const visibleClassIds = useMemo(() => {
    const labels = frameRows.map((row) => detectionClassIdLabel(row)).filter(Boolean);
    return [...new Set(labels)];
  }, [frameRows]);
  const filteredFrameRows = useMemo(() => {
    const normalized = idQuery.trim().toLowerCase();
    if (!normalized) {
      return frameRows;
    }
    return frameRows.filter((row) => {
      const parsedId = parseTrackId(row.track_id);
      const idLabel = parsedId ? `id ${String(parsedId).padStart(2, "0")}` : "elephant";
      return idLabel.toLowerCase().includes(normalized);
    });
  }, [frameRows, idQuery]);

  return (
    <Panel title={title} subtitle={subtitle}>
      {!frameIndices.length ? (
        <EmptyState title="No saved preview frames" body="This run has no stored evidence frames to scrub through." />
      ) : (
        <div className={clsx("frame-review-layout", compact && "frame-review-layout-compact")}>
          <div className="frame-review-stage">
            <div className="frame-canvas">
              {frameImageSrc ? (
                <>
                  <img
                    src={frameImageSrc}
                    alt={`Frame ${selectedFrameIndex}`}
                    className="frame-image"
                  />
                </>
              ) : (
                <div className="frame-empty">Frame unavailable</div>
              )}
            </div>

            {!isImageMode ? (
              <div className="frame-scrubber">
                <input
                  type="range"
                  min={0}
                  max={maxCursor}
                  value={clampedCursor}
                  onChange={(event) => setFrameCursor(Number(event.target.value || 0))}
                />
                <div className="frame-scrubber-meta">
                  <span>{humanizeFrameLabel(selectedFrameIndex)}</span>
                  <span>{frameTimeLabel}</span>
                  <span>{frameRows.length} detections</span>
                </div>
              </div>
            ) : (
              <div className="frame-scrubber-meta frame-scrubber-meta-static">
                <span>Single image inspection</span>
                <span>{frameRows.length} detections</span>
              </div>
            )}
          </div>

          <div className="frame-sidebar">
            <div className="frame-sidebar-block">
              <span className="detail-label">{isImageMode ? "Current media" : "Selected frame"}</span>
              <strong>{isImageMode ? "Still image" : humanizeFrameLabel(selectedFrameIndex)}</strong>
            </div>
            {!isImageMode ? (
              <div className="frame-sidebar-block">
                <span className="detail-label">Sample time</span>
                <strong>{frameTimeLabel}</strong>
              </div>
            ) : null}
            <div className="frame-sidebar-block">
              <span className="detail-label">Visible detections</span>
              <strong>{frameRows.length}</strong>
            </div>
            {showDetectionList ? (
              <div className="frame-sidebar-list">
                <div className="frame-sidebar-filter">
                  <label htmlFor={`frame-id-filter-${jobId || "media"}`}>Find ID</label>
                  <input
                    id={`frame-id-filter-${jobId || "media"}`}
                    type="search"
                    placeholder="Type ID 01"
                    value={idQuery}
                    onChange={(event) => setIdQuery(event.target.value)}
                  />
                </div>
                {filteredFrameRows.length ? (
                  filteredFrameRows.slice(0, compact ? 6 : 12).map((row, index) => (
                    <div key={`${row.frame_index}-${row.track_id}-${index}`} className="frame-sidebar-row">
                      <div className="frame-sidebar-row-head">
                        <strong>{parseTrackId(row.track_id) ? `ID ${String(parseTrackId(row.track_id)).padStart(2, "0")}` : "Elephant"}</strong>
                      </div>
                      <div className="frame-sidebar-meta-grid">
                        <div>
                          <label>Confidence</label>
                          <span>{formatPercent(Number(row.conf || 0) * 100)}</span>
                        </div>
                        {!isImageMode ? (
                          <div>
                            <label>Frame</label>
                            <span>{humanizeFrameLabel(row.frame_index)}</span>
                          </div>
                        ) : null}
                        {detectionAgeLabel(row) ? (
                          <div>
                            <label>Label</label>
                            <span>{detectionAgeLabel(row)}</span>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="frame-sidebar-empty">
                    {idQuery.trim() ? "No detections match this ID." : "No detections in this saved frame."}
                  </div>
                )}
              </div>
            ) : (
              <div className="frame-sidebar-empty">Fast mode keeps current-frame evidence lightweight.</div>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

function ResultsSummaryHero({ job, metrics, gpsRows, navigate }) {
  const mediaType = inferMediaTypeFromFilename(job?.filename);
  const modelKey = resolveJobModelKey(job, metrics);
  const modeKey = resolveJobModeKey(job, metrics);
  const interpretation = buildInterpretation({ mediaType, metrics, gpsRows, modeKey });

  return (
    <Panel className="results-summary-hero">
      <div className="results-summary-hero-layout">
        <div className="results-summary-hero-copy">
          <span className="results-summary-kicker">Summary-first results brief</span>
          <div className="results-summary-headline">
            <h2>{job?.filename || "Selected run"}</h2>
            <div className="results-summary-head-actions">
              <span className="results-summary-status-pill">{job?.status || "complete"}</span>
            </div>
          </div>
          <p>{interpretation}</p>
        </div>

        <div className="results-summary-hero-meta">
          <div className="results-summary-meta-card results-summary-meta-card-primary">
            <span className="detail-label">Status</span>
            <strong>{job?.status || "complete"}</strong>
            <span>{modeLabelFromKey(modeKey) !== "-" ? modeLabelFromKey(modeKey) : "Ready for review"}</span>
          </div>
          <div className="results-summary-meta-card">
            <span className="detail-label">Source</span>
            <strong>{mediaType}</strong>
            <span>{gpsRows?.length ? "GPS available" : "No GPS metadata"}</span>
          </div>
          <div className="results-summary-meta-card">
            <span className="detail-label">Model</span>
            <strong>{modelLabel(modelKey)}</strong>
            <span>Active detection model</span>
          </div>
          <div className="results-summary-meta-card">
            <span className="detail-label">Dominant habitat</span>
            <strong>{metrics?.dominant_habitat && metrics?.dominant_habitat !== "-" ? metrics.dominant_habitat : "Unavailable"}</strong>
            <span>{metrics?.dominant_habitat_share ? `${formatPercent(metrics.dominant_habitat_share, 0)} coverage` : "No habitat coverage"}</span>
          </div>
          <div className="results-summary-meta-card">
            <span className="detail-label">Total detections</span>
            <strong>{formatMetric(metrics?.detections_count)}</strong>
          </div>
          <div className="results-summary-meta-card">
            <span className="detail-label">Average herd size</span>
            <strong>{formatMetric(metrics?.avg_herd_size, 1)}</strong>
          </div>
          <div className="results-summary-meta-card">
            <span className="detail-label">Peak herd size</span>
            <strong>{formatMetric(metrics?.peak_herd_size)}</strong>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function ResultsEvidencePreview({ jobId, filename, previewMeta, detections }) {
  const frameIndices = previewMeta?.frame_indices || [];
  const heroFrame = frameIndices[0];
  const [idQuery, setIdQuery] = useState("");
  const imageRows = useMemo(
    () => getDetectionsForFrame(detections, heroFrame),
    [detections, heroFrame]
  );
  const filteredImageRows = useMemo(() => {
    const normalized = idQuery.trim().toLowerCase();
    if (!normalized) {
      return imageRows;
    }
    return imageRows.filter((row) => {
      const parsedId = parseTrackId(row.track_id);
      const idLabel = parsedId ? `id ${String(parsedId).padStart(2, "0")}` : "elephant";
      return idLabel.toLowerCase().includes(normalized);
    });
  }, [imageRows, idQuery]);
  return (
    <Panel
      title="Evidence Preview"
      subtitle="Representative evidence with a scrollable ID list."
    >
      {!frameIndices.length ? (
        <EmptyState title="No saved evidence preview" body="This run does not yet have stored review frames." />
      ) : (
        <div className="frame-review-layout frame-review-layout-compact results-evidence-layout">
          <div className="frame-review-stage">
            <div className="frame-canvas frame-canvas-image">
              <img
                src={`${BACKEND_BASE_URL}/media/frame/${jobId}/${heroFrame}`}
                alt={`${filename || "Selected run"} representative evidence`}
                className="frame-image frame-image-static"
              />
            </div>
            <div className="frame-scrubber-meta frame-scrubber-meta-static results-evidence-stage-meta">
              <span>Representative image</span>
              <span>{filename || humanizeFrameLabel(heroFrame)}</span>
              <span>{imageRows.length} detections</span>
            </div>
          </div>
          <div className="frame-sidebar frame-sidebar-image">
            <div className="frame-sidebar-block">
              <span className="detail-label">Visible detections</span>
              <strong>{imageRows.length}</strong>
            </div>
            <div className="frame-sidebar-list frame-sidebar-list-scroll">
              <div className="frame-sidebar-filter">
                <label htmlFor={`image-id-filter-${jobId || "image"}`}>Find ID</label>
                <input
                  id={`image-id-filter-${jobId || "image"}`}
                  type="search"
                  placeholder="Type ID 01"
                  value={idQuery}
                  onChange={(event) => setIdQuery(event.target.value)}
                />
              </div>
              {filteredImageRows.length ? (
                filteredImageRows.map((row, index) => (
                  <div key={`${row.frame_index}-${row.track_id}-${index}`} className="frame-sidebar-row">
                    <div className="frame-sidebar-row-head">
                      <strong>{parseTrackId(row.track_id) ? `ID ${String(parseTrackId(row.track_id)).padStart(2, "0")}` : "Elephant"}</strong>
                    </div>
                    <div className="frame-sidebar-meta-grid">
                      <div>
                        <label>Confidence</label>
                        <span>{formatPercent(Number(row.conf || 0) * 100)}</span>
                      </div>
                      {detectionAgeLabel(row) ? (
                        <div>
                          <label>Label</label>
                          <span>{detectionAgeLabel(row)}</span>
                        </div>
                      ) : null}
                    </div>
                  </div>
                ))
              ) : (
                <div className="frame-sidebar-empty">
                  {idQuery.trim() ? "No detections match this ID." : "No detections stored for this evidence image."}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}

function SpatialMapPanel({ gpsRows }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const [mapError, setMapError] = useState("");
  const points = useMemo(
    () =>
      (gpsRows || [])
        .map((row) => [Number(row.lat), Number(row.lon)])
        .filter((pair) => Number.isFinite(pair[0]) && Number.isFinite(pair[1])),
    [gpsRows],
  );

  useEffect(() => {
    if (!containerRef.current || !points.length) {
      return undefined;
    }

    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }

    setMapError("");
    const map = L.map(containerRef.current, {
      zoomControl: true,
      attributionControl: true,
    });
    mapRef.current = map;

    L.tileLayer(TILE_URL, { maxZoom: 18 })
      .addTo(map)
      .on("tileerror", () => {
        setMapError("Tile server reachable, but map tiles failed to load.");
      });

    if (points.length === 1) {
      L.circleMarker(points[0], {
        radius: 7,
        color: "#63d69b",
        fillColor: "#63d69b",
        fillOpacity: 1,
      })
        .addTo(map)
        .bindPopup("Pinned location");
      map.setView(points[0], 15);
    } else {
      const line = L.polyline(points, {
        color: "#86a8ff",
        weight: 4,
        opacity: 0.9,
      }).addTo(map);
      map.fitBounds(line.getBounds(), { padding: [24, 24] });

      L.circleMarker(points[0], {
        radius: 6,
        color: "#63d69b",
        fillColor: "#63d69b",
        fillOpacity: 1,
      })
        .addTo(map)
        .bindPopup("Start");

      L.circleMarker(points[points.length - 1], {
        radius: 6,
        color: "#d6a64d",
        fillColor: "#d6a64d",
        fillOpacity: 1,
      })
        .addTo(map)
        .bindPopup("End");
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [gpsRows]);

  if (!gpsRows?.length) {
    return (
      <div className="map-placeholder">
        <div className="map-placeholder-mark">--</div>
        <p>No GPS route available</p>
        <span>This media item does not include embedded spatial metadata.</span>
      </div>
    );
  }

  return (
    <div className="spatial-map-wrap">
      {points.length > 1 ? (
        <div className="spatial-map-legend" aria-label="GPS route legend">
          <span className="spatial-map-legend-item">
            <span className="spatial-map-legend-dot spatial-map-legend-dot-start" />
            Start
          </span>
          <span className="spatial-map-legend-item">
            <span className="spatial-map-legend-dot spatial-map-legend-dot-end" />
            End
          </span>
        </div>
      ) : null}
      <div ref={containerRef} className="spatial-map-canvas" />
      {mapError ? <div className="inline-feedback feedback-error">{mapError}</div> : null}
    </div>
  );
}

function OverviewPage({ jobs, health, navigate }) {
  const recentRuns = jobs.filter((job) => String(job.status).toLowerCase() === "complete").slice(0, 4);
  const recentRunsKey = recentRuns.map((job) => String(job.id)).join(":");
  const [activeRecentIndex, setActiveRecentIndex] = useState(0);
  const activeRecentRun = recentRuns[activeRecentIndex] || recentRuns[0] || null;
  const [heroFramesByJob, setHeroFramesByJob] = useState({});
  const [activeHeroFrameIndex, setActiveHeroFrameIndex] = useState(0);

  useEffect(() => {
    setActiveRecentIndex(0);
  }, [recentRunsKey]);

  useEffect(() => {
    setActiveHeroFrameIndex(0);
  }, [activeRecentIndex, activeRecentRun?.id]);

  useEffect(() => {
    let active = true;
    const runIds = recentRuns.map((job) => String(job.id));
    if (!runIds.length) {
      setHeroFramesByJob({});
      return undefined;
    }

    Promise.all(
      recentRuns.map(async (job) => {
        try {
          const meta = await api(`/media/preview/${job.id}/meta`);
          return [String(job.id), sampleFrameIndices(meta?.frame_indices, 8)];
        } catch {
          return [String(job.id), []];
        }
      })
    ).then((entries) => {
      if (!active) {
        return;
      }
      setHeroFramesByJob((current) => {
        const next = {};
        entries.forEach(([jobId, frames]) => {
          next[jobId] = frames;
        });
        return next;
      });
    });

    return () => {
      active = false;
    };
  }, [recentRunsKey]);

  useEffect(() => {
    if (recentRuns.length <= 1) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setActiveRecentIndex((current) => (current + 1) % recentRuns.length);
    }, 4200);
    return () => window.clearInterval(timer);
  }, [recentRunsKey, recentRuns.length]);

  const activeHeroFrames = activeRecentRun ? heroFramesByJob[String(activeRecentRun.id)] || [] : [];

  useEffect(() => {
    if (activeHeroFrames.length <= 1) {
      return undefined;
    }
    let intervalId = null;
    const startDelay = window.setTimeout(() => {
      intervalId = window.setInterval(() => {
        setActiveHeroFrameIndex((current) => (current + 1) % activeHeroFrames.length);
      }, 3200);
    }, 3200);

    return () => {
      window.clearTimeout(startDelay);
      if (intervalId) {
        window.clearInterval(intervalId);
      }
    };
  }, [activeRecentRun?.id, activeHeroFrames.length]);

  const activeHeroFrame = activeHeroFrames[activeHeroFrameIndex] ?? activeHeroFrames[0] ?? null;
  const activeHeroFrameUrl =
    activeRecentRun && activeHeroFrame !== null
      ? `${BACKEND_BASE_URL}/media/frame/${activeRecentRun.id}/${activeHeroFrame}`
      : "";

  return (
    <>
      <section className="hero-layout">
        <div className="hero-copy">
          <div className="eyebrow">Professional Wildlife Analytics Platform</div>
          <h2>Evidence-backed elephant monitoring from aerial media</h2>
          <p>
            A local analysis workstation for turning drone imagery and video into herd metrics,
            habitat context, route intelligence, and review-ready evidence.
          </p>
          <div className="hero-actions">
            <button className="button button-primary" onClick={() => navigate("/analyze")}>
              Start New Analysis
            </button>
          </div>
        </div>
        <div className="hero-visual">
          <div className="hero-image-card">
            {activeHeroFrameUrl ? (
              <img
                key={`${activeRecentRun?.id}-${activeHeroFrame}`}
                src={activeHeroFrameUrl}
                alt={activeRecentRun ? `${activeRecentRun.filename} preview` : "Recent analysis preview"}
                className="hero-image-media"
              />
            ) : (
              <div
                className="hero-image-fallback"
                style={{
                  backgroundImage:
                    'linear-gradient(rgba(10, 14, 12, 0.16), rgba(10, 14, 12, 0.18)), url("/overview-elephants.jpg")',
                }}
              />
            )}
            <div className="hero-image-overlay">
              <div>
                <span>Recent Run</span>
                <strong>{activeRecentRun ? activeRecentRun.filename : "No analysis yet"}</strong>
              </div>
              <div className="hero-image-overlay-status">
                <span>Status</span>
                <strong>{activeRecentRun ? activeRecentRun.status : "-"}</strong>
              </div>
            </div>
            {recentRuns.length ? (
              <div className="hero-carousel-dots">
                {recentRuns.map((job, index) => (
                  <button
                    key={job.id}
                    type="button"
                    className={clsx("hero-carousel-dot", index === activeRecentIndex && "hero-carousel-dot-active")}
                    onClick={() => setActiveRecentIndex(index)}
                    aria-label={`Show recent run ${index + 1}`}
                  />
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </section>

      <Panel title="Platform Capabilities">
        <div className="capability-grid">
          <article className="capability-card">
            <span className="capability-icon">DT</span>
            <h4>Detection & Tracking</h4>
            <p>Frame-level elephant detection with confidence scoring and mode-aware track continuity.</p>
          </article>
          <article className="capability-card">
            <span className="capability-icon">HM</span>
            <h4>Herd Metrics</h4>
            <p>Herd size trends, peak observations, and sampled visibility summaries from processed runs.</p>
          </article>
          <article className="capability-card">
            <span className="capability-icon">HA</span>
            <h4>Habitat Analysis</h4>
            <p>Habitat distribution, dominant terrain, and timeline context when classification weights are available.</p>
          </article>
          <article className="capability-card">
            <span className="capability-icon">SC</span>
            <h4>Spatial Context</h4>
            <p>GPS extraction, route context, and map export when embedded metadata exists in the source media.</p>
          </article>
        </div>
      </Panel>

      <section className="overview-lower-grid">
        <Panel
          title="Recent Analyses"
          actions={
            <button className="text-link" onClick={() => navigate("/history")}>
              View All History
            </button>
          }
        >
          {recentRuns.length ? (
            <div className="recent-run-list">
              {recentRuns.map((job, index) => (
                <button
                  key={job.id}
                  type="button"
                  className={clsx("recent-run-row", index === activeRecentIndex && "recent-run-row-active")}
                  onClick={() => navigate("/results")}
                  onMouseEnter={() => setActiveRecentIndex(index)}
                >
                  <div className="recent-run-meta">
                    <h4>{job.filename}</h4>
                    <p>{formatDate(job.created_at)}</p>
                  </div>
                  <div className="recent-run-stats">
                    <strong>{formatMetric(job.duration_seconds, 0)}</strong>
                    <span>{job.duration_seconds ? "seconds" : "pending"}</span>
                  </div>
                  <span className={clsx("status-chip", STATUS_TONE[job.status] || "status-queued")}>
                    {job.status}
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <EmptyState title="No recent analyses" body="Upload the first video or image to start building a local archive." />
          )}
        </Panel>

        <Panel title="Local Analysis Environment">
          <div className="environment-grid">
            <div className="environment-item">
              <span className="status-dot status-dot-on" />
              <div>
                <strong>Processing Engine</strong>
                <p>{health.ok ? "Ready" : "Waiting for backend"}</p>
              </div>
            </div>
            <div className="environment-item">
              <span className="status-dot status-dot-on" />
              <div>
                <strong>Model Service</strong>
                <p>Active</p>
              </div>
            </div>
            <div className="environment-item">
              <span className="status-dot status-dot-off" />
              <div>
                <strong>Tile Server</strong>
                <p>Optional</p>
              </div>
            </div>
          </div>
        </Panel>
      </section>
    </>
  );
}

function AnalyzePage({ appState, setAppState, jobs, onRefreshJobs, navigate, health }) {
  const [models, setModels] = useState({});
  const [habitatModel, setHabitatModel] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [localPreviewUrl, setLocalPreviewUrl] = useState("");
  const [videoPosterUrl, setVideoPosterUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [processingStageIndex, setProcessingStageIndex] = useState(0);
  const [isFinishingRun, setIsFinishingRun] = useState(false);
  const fileInputRef = useRef(null);
  const processAbortRef = useRef(null);
  const processingRef = useRef(false);
  const activeProcessJobIdRef = useRef(null);
  const suppressAutoCancelRef = useRef(false);
  const etaSampleRef = useRef({ startedAt: 0 });
  const processingPreview = usePreviewMeta(appState.uploadedJobId || null, 0, {
    pipelineMode: appState.selectedMode,
    live: processing,
  });

  useEffect(() => {
    processingRef.current = processing;
  }, [processing]);

  useEffect(() => {
    activeProcessJobIdRef.current = appState.uploadedJobId || null;
  }, [appState.uploadedJobId]);

  useEffect(() => {
    if (!processing) {
      setProcessingStageIndex(0);
      setIsFinishingRun(false);
      return undefined;
    }

    setProcessingStageIndex(0);

    const startedAt = Date.now();
    const tick = window.setInterval(() => {
      const elapsed = Date.now() - startedAt;
      const stage = Math.min(
        PROCESSING_STAGES.length - 1,
        Math.floor((elapsed / 9000) % PROCESSING_STAGES.length)
      );
      setProcessingStageIndex(stage);
    }, 900);

    return () => window.clearInterval(tick);
  }, [processing]);

  useEffect(() => {
    api("/models")
      .then((payload) => {
        setModels(payload.models || {});
        setHabitatModel(payload.habitat_model || null);
      })
      .catch(() => {
        setModels({});
        setHabitatModel(null);
      });
  }, []);

  const availableModelEntries = useMemo(() => {
    const entries = Object.entries(models || {});
    const filtered = entries.filter(([key]) => VISIBLE_MODEL_KEYS.includes(key));
    if (filtered.length) {
      return filtered;
    }
    return [
      ["labeling_data_v2_s", ""],
      ["roboflow_v1_fresh2", ""],
    ];
  }, [models]);
  const analysisPipelineOptions = PIPELINE_OPTIONS;

  useEffect(() => {
    if (!analysisPipelineOptions.some((option) => option.value === appState.selectedMode)) {
      setAppState((current) => ({ ...current, selectedMode: "fast_trend" }));
    }
  }, [analysisPipelineOptions, appState.selectedMode, setAppState]);

  const activeJobId = appState.uploadedJobId || null;
  const activeJob = jobs.find((job) => String(job.id) === String(activeJobId));
  const activeJobStatus = String(activeJob?.status || "").toLowerCase();
  const selectedModeMeta = analysisPipelineOptions.find((option) => option.value === appState.selectedMode) || analysisPipelineOptions[0];
  const selectedFileMeta = selectedFile
    ? {
        name: selectedFile.name,
        size: formatFileSize(selectedFile.size),
        kind: inferFileKind(selectedFile),
      }
    : null;
  const isImageInput = selectedFileMeta?.kind === "Image";
  const isVideoInput = selectedFileMeta?.kind === "Video";
  const uploadWorkspaceState = processing
    ? "processing"
    : uploading
      ? "uploading"
      : selectedFileMeta
        ? "selected"
        : dragActive
          ? "drag"
          : "empty";
  const analysisState = error
    ? "error"
    : processing
      ? "running"
      : activeJob?.status === "complete"
        ? "completed"
        : activeJob?.status === "uploaded"
          ? "uploaded"
          : selectedFileMeta
            ? "locked"
            : "no_input";
  const analysisStateLabel =
    analysisState === "error"
      ? "Attention needed"
      : analysisState === "running"
        ? "Running"
        : analysisState === "completed"
          ? "Completed"
          : analysisState === "uploaded"
            ? "Uploaded"
            : analysisState === "locked"
              ? "Ready"
              : "No input";
  const analysisStateMessage =
    analysisState === "error"
      ? error
      : analysisState === "running"
        ? "Analysis is running in the local backend."
        : analysisState === "completed"
          ? "Run finished. Results are ready to review."
          : analysisState === "uploaded"
            ? "Source is uploaded and ready to run."
            : analysisState === "locked"
              ? isImageInput
                ? "Still image locked for this run."
                : "Source locked for this run."
              : "One image or one video per analysis run.";
  const backendUnavailable = health?.stage === "offline";
  const backendRecovering =
    processing &&
    !isFinishingRun &&
    ["starting", "degraded", "offline"].includes(String(health?.stage || ""));
  const transitionOverlay = isFinishingRun
    ? {
        kicker: "Results handoff",
        title: "Opening Results Workspace",
        body: "Packaging evidence, syncing the last metrics, and moving into the review surface.",
        tone: "analysis-transition-overlay-finish",
      }
    : backendRecovering
      ? {
          kicker: "Connection recovery",
          title: health?.stage === "offline" ? "Reconnecting to local backend" : "Stabilizing backend session",
          body:
            health?.stage === "offline"
              ? "The run is paused while the local backend comes back online. Your analysis context is being preserved."
              : "The backend is responding again. Holding this screen briefly to avoid a jarring jump during recovery.",
          tone: "analysis-transition-overlay-reconnect",
        }
      : null;

  useEffect(() => {
    if (!activeJobId) {
      if (!processingRef.current) {
        return;
      }
      setProcessing(false);
      return;
    }
    // Analyze can unmount while the backend keeps running. Rehydrate the local
    // processing shell from live job state when the page comes back.
    if (activeJobStatus === "processing" || activeJobStatus === "cancel_requested" || activeJobStatus === "uploaded") {
      if (!processingRef.current) {
        setProcessing(true);
      }
      return;
    }
    if (processingRef.current && ["complete", "failed", "cancelled"].includes(activeJobStatus)) {
      setProcessing(false);
    }
  }, [activeJobId, activeJobStatus]);
  const processingStage = isFinishingRun
    ? PROCESSING_FINISH_STAGE
    : PROCESSING_STAGES[processingStageIndex] || PROCESSING_STAGES[0];
  const processingMeta = processingPreview.meta;
  const savedFrameTarget = Number(processingMeta?.saved_frame_target || 0);
  const currentFrameCount = Number(processingMeta?.frame_count || 0);
  const hasMeasuredProgress = savedFrameTarget > 0;
  // Leave some headroom so the bar does not hit 100% before habitat/finalize
  // work has actually finished on the backend.
  const measuredProgressRatio = hasMeasuredProgress
    ? Math.max(0.04, Math.min(isFinishingRun ? 1 : 0.99, currentFrameCount / savedFrameTarget))
    : 0;
  const measuredProgressPercent = isFinishingRun ? 100 : Math.round(measuredProgressRatio * 100);
  const [etaSeconds, setEtaSeconds] = useState(null);

  useEffect(() => {
    if (!processing) {
      etaSampleRef.current = { startedAt: 0 };
      setEtaSeconds(null);
      return;
    }
    const now = Date.now();
    if (!etaSampleRef.current.startedAt) {
      etaSampleRef.current = { startedAt: now };
      setEtaSeconds(null);
      return;
    }
    if (!hasMeasuredProgress || isFinishingRun || currentFrameCount <= 1 || currentFrameCount >= savedFrameTarget) {
      setEtaSeconds(null);
      return;
    }
    const elapsedSeconds = (now - etaSampleRef.current.startedAt) / 1000;
    if (elapsedSeconds < 8) {
      setEtaSeconds(null);
      return;
    }
    const frameRate = currentFrameCount / elapsedSeconds;
    if (frameRate <= 0.15) {
      setEtaSeconds(null);
      return;
    }
    const remainingFrames = Math.max(savedFrameTarget - currentFrameCount, 0);
    const paddedEstimate = (remainingFrames / frameRate) * 1.12;
    const nextEta = Math.max(8, Math.min(paddedEstimate, 3600));
    setEtaSeconds(Number.isFinite(nextEta) ? Math.round(nextEta) : null);
  }, [processing, hasMeasuredProgress, isFinishingRun, currentFrameCount, savedFrameTarget]);

  function formatEtaLabel(totalSeconds) {
    const secs = Number(totalSeconds || 0);
    if (!Number.isFinite(secs) || secs <= 0) {
      return "Estimating remaining time";
    }
    if (secs < 60) {
      return `About ${Math.round(secs)}s remaining`;
    }
    const mins = Math.floor(secs / 60);
    const remSecs = Math.round(secs % 60);
    return remSecs ? `About ${mins}m ${remSecs}s remaining` : `About ${mins}m remaining`;
  }

  useEffect(() => {
    if (!selectedFile) {
      setLocalPreviewUrl("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setLocalPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    let active = true;
    let generatedPosterUrl = "";
    setVideoPosterUrl("");

    if (!selectedFile || inferFileKind(selectedFile) !== "Video") {
      return undefined;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;
    video.src = objectUrl;

    const finalize = () => {
      video.pause();
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(objectUrl);
    };

    const captureFrame = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth || 1;
        canvas.height = video.videoHeight || 1;
        const context = canvas.getContext("2d");
        if (!context) {
          finalize();
          return;
        }
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        generatedPosterUrl = canvas.toDataURL("image/jpeg", 0.88);
        if (active) {
          setVideoPosterUrl(generatedPosterUrl);
        }
      } catch {
        // Keep graceful fallback to direct video preview.
      } finally {
        finalize();
      }
    };

    video.addEventListener("loadeddata", () => {
      try {
        video.currentTime = Math.min(0.15, Math.max(video.duration * 0.02 || 0.1, 0.1));
      } catch {
        captureFrame();
      }
    });
    video.addEventListener("seeked", captureFrame);
    video.addEventListener("error", finalize);

    return () => {
      active = false;
      finalize();
    };
  }, [selectedFile]);

  function clearLocalSelection() {
    if (processingRef.current && activeProcessJobIdRef.current) {
      requestJobCancel(activeProcessJobIdRef.current);
    }
    processAbortRef.current?.abort();
    suppressAutoCancelRef.current = false;
    activeProcessJobIdRef.current = null;
    setSelectedFile(null);
    setDragActive(false);
    setLocalPreviewUrl("");
    setVideoPosterUrl("");
    setMessage("");
    setError("");
    setAppState((current) => ({
      ...current,
      uploadedJobId: null,
    }));
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function stageIncomingFiles(files) {
    const result = validateSingleSource(files);
    if (result.error) {
      setError(result.error);
      return;
    }
    setSelectedFile(result.file);
    if (inferFileKind(result.file) === "Image") {
      setAppState((current) => ({ ...current, selectedMode: "quality" }));
    }
    setMessage("Input locked for this run. Ready for local analysis.");
    setError("");
  }

  async function handleUpload(event) {
    event.preventDefault();
    if (!selectedFile) {
      setError("Choose one image or video before uploading.");
      return;
    }
    setUploading(true);
    setError("");
    setMessage("Uploading media to the local backend...");
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const payload = await api("/upload", {
        method: "POST",
        body: formData,
      });
      setAppState((current) => ({
        ...current,
        uploadedJobId: payload.job_id,
        selectedJobId: payload.job_id,
      }));
      setMessage("Source uploaded successfully.");
      onRefreshJobs();
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleProcess() {
    suppressAutoCancelRef.current = false;
    setIsFinishingRun(false);
    setProcessing(true);
    setError("");
    setMessage("Running local analysis. Larger videos can take several minutes.");
    const controller = new AbortController();
    processAbortRef.current = controller;
    try {
      let jobId = appState.uploadedJobId || null;

      // Match the Streamlit flow: if a file is currently selected, upload it as part
      // of the same Run Analysis action instead of requiring a separate step first.
      if (selectedFile) {
        setMessage("Uploading media to the local backend...");
        const formData = new FormData();
        formData.append("file", selectedFile);
        const uploadPayload = await api("/upload", {
          method: "POST",
          body: formData,
        });
        jobId = uploadPayload.job_id;
        setAppState((current) => ({
          ...current,
          uploadedJobId: jobId,
          selectedJobId: jobId,
        }));
        activeProcessJobIdRef.current = jobId;
        onRefreshJobs();
      }

      if (!jobId) {
        setError("Choose one media file before running analysis.");
        return;
      }

      setMessage("Processing media. This can take several minutes for larger videos.");
      const params = new URLSearchParams({
        model_key: appState.selectedModel,
        pipeline_mode: appState.selectedMode,
      });
      const payload = await api(`/process/${jobId}?${params.toString()}`, {
        method: "POST",
        signal: controller.signal,
      });
      await api(`/results/${jobId}`);
      setAppState((current) => ({
        ...current,
        uploadedJobId: jobId,
        selectedJobId: jobId,
        lastProcessResponse: payload,
      }));
      suppressAutoCancelRef.current = true;
      setIsFinishingRun(true);
      setMessage("Analysis complete. Opening results workspace...");
      onRefreshJobs();
      await new Promise((resolve) => window.setTimeout(resolve, isImageInput ? 1100 : 1300));
      startTransition(() => {
        setProcessing(false);
        setIsFinishingRun(false);
        navigate("/results");
      });
    } catch (err) {
      if (err?.name === "AbortError") {
        setMessage("Analysis cancellation requested.");
        return;
      }
      setError(err.message || "Processing failed.");
    } finally {
      processAbortRef.current = null;
      if (!suppressAutoCancelRef.current) {
        setProcessing(false);
      }
    }
  }

  async function handleCancelProcess() {
    const jobId = activeProcessJobIdRef.current || appState.uploadedJobId;
    if (!processing || !jobId) {
      return;
    }
    setMessage("Stopping analysis...");
    setIsFinishingRun(false);
    await requestJobCancel(jobId);
    processAbortRef.current?.abort();
  }

  return (
    <>
      <PageHeader
        title="New Run"
        description="Lock one source, choose a strategy, and launch analysis."
      />

      <section className="analysis-stage">
        <section className="analysis-hero-shell">
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.mov,.avi,.jpg,.jpeg,.png"
            className="analysis-hidden-input"
            onChange={(event) => stageIncomingFiles(event.target.files)}
          />

          {!selectedFileMeta ? (
            <div
              className={clsx("analysis-upload-hero", dragActive && "analysis-upload-hero-drag")}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={(event) => {
                event.preventDefault();
                if (event.currentTarget.contains(event.relatedTarget)) {
                  return;
                }
                setDragActive(false);
              }}
              onDrop={(event) => {
                event.preventDefault();
                setDragActive(false);
                stageIncomingFiles(event.dataTransfer?.files);
              }}
            >
              <div className="analysis-upload-orb">{dragActive ? "GO" : "UP"}</div>
              <div className="analysis-upload-copy">
                <span className="analysis-kicker">Upload workspace</span>
                <h3>{dragActive ? "Release to stage source" : "Drop one image or video here"}</h3>
                <p>One file per run. Selecting a new file replaces the current source.</p>
              </div>
              <div className="analysis-upload-footer">
                <div className="analysis-upload-tags">
                  <span>Single source</span>
                  <span>MP4, MOV, AVI</span>
                  <span>JPG, JPEG, PNG</span>
                </div>
                <button className="button button-primary" type="button" onClick={() => fileInputRef.current?.click()}>
                  Browse File
                </button>
              </div>
            </div>
          ) : (
            <section className="analysis-upload-locked">
              <div className="analysis-upload-locked-top">
                <div className="analysis-upload-preview">
                  {isImageInput && localPreviewUrl ? (
                    <img src={localPreviewUrl} alt={selectedFileMeta.name} className="analysis-upload-preview-media" />
                  ) : isVideoInput && (videoPosterUrl || localPreviewUrl) ? (
                    videoPosterUrl ? (
                      <img src={videoPosterUrl} alt={`${selectedFileMeta.name} preview`} className="analysis-upload-preview-media" />
                    ) : (
                      <video
                        src={localPreviewUrl}
                        className="analysis-upload-preview-media"
                        muted
                        playsInline
                        preload="metadata"
                      />
                    )
                  ) : (
                    <div className="analysis-upload-preview-fallback">{uploadWorkspaceState === "processing" ? "PR" : uploadWorkspaceState === "uploading" ? "UP" : "OK"}</div>
                  )}
                </div>
                <div className="analysis-upload-lock-copy">
                  <span className="analysis-kicker">Input locked</span>
                  <h3>{selectedFileMeta.name}</h3>
                  <div className={clsx("analysis-state-banner", `analysis-state-${analysisState}`)}>
                    <span className="analysis-state-label">{analysisStateLabel}</span>
                    <p>{analysisStateMessage}</p>
                  </div>
                </div>
                <div className="analysis-upload-lock-status">
                  <span className="detail-label">State</span>
                  <strong>{analysisStateLabel}</strong>
                </div>
              </div>

              <div className="analysis-upload-lock-meta">
                <div>
                  <span className="detail-label">Type</span>
                  <strong>{selectedFileMeta.kind}</strong>
                </div>
                <div>
                  <span className="detail-label">Size</span>
                  <strong>{selectedFileMeta.size}</strong>
                </div>
                <div>
                  <span className="detail-label">Rule</span>
                  <strong>One file per run</strong>
                </div>
              </div>

              <div className="analysis-upload-lock-actions">
                <button className="button button-muted" type="button" onClick={() => fileInputRef.current?.click()}>
                  Replace
                </button>
                <button className="button button-muted" type="button" onClick={clearLocalSelection}>
                  Remove
                </button>
              </div>
            </section>
          )}
        </section>

        <section className="analysis-side">
          <div className="analysis-model-bar">
            <div className="analysis-model-copy">
              <span className="analysis-kicker">Detection model</span>
              <strong>{modelLabel(appState.selectedModel)}</strong>
            </div>
            <div className="analysis-model-switcher" role="radiogroup" aria-label="Detection model">
              {availableModelEntries.map(([key]) => (
                <button
                  key={key}
                  type="button"
                  className={clsx(
                    "analysis-model-pill",
                    appState.selectedModel === key && "analysis-model-pill-active"
                  )}
                  onClick={() =>
                    setAppState((current) => ({ ...current, selectedModel: key }))
                  }
                  aria-pressed={appState.selectedModel === key}
                >
                  {key === "labeling_data_v2_s" ? "Research v3_2s" : "Online"}
                </button>
              ))}
            </div>
          </div>

          <section className="analysis-mode-shell">
            <div className="analysis-mode-head">
              <div>
                <span className="analysis-kicker">Processing mode</span>
                <h3>{isImageInput ? "Still Image" : selectedModeMeta.label}</h3>
              </div>
              {!isImageInput ? (
                <div className="analysis-mode-rail">
                  {analysisPipelineOptions.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      className={clsx("analysis-mode-pill", appState.selectedMode === option.value && "analysis-mode-pill-active")}
                      onClick={() => setAppState((current) => ({ ...current, selectedMode: option.value }))}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>

            <div className="analysis-mode-detail">
              <p className="analysis-mode-note">
                {isImageInput
                  ? "Single-frame analysis with full-resolution evidence output."
                  : selectedModeMeta.note}
              </p>
              <div className="analysis-mode-detail-grid">
                <div>
                  <span className="detail-label">Speed</span>
                  <strong>{isImageInput ? "Single pass" : selectedModeMeta.speed}</strong>
                </div>
                <div>
                  <span className="detail-label">Continuity</span>
                  <strong>{isImageInput ? "Not track-based" : selectedModeMeta.continuity}</strong>
                </div>
                <div>
                  <span className="detail-label">Evidence</span>
                  <strong>{isImageInput ? "Full still-frame evidence" : selectedModeMeta.evidence}</strong>
                </div>
                <div>
                  <span className="detail-label">Best for</span>
                  <strong>{isImageInput ? "Single-image inspection and static evidence review." : selectedModeMeta.useCase}</strong>
                </div>
              </div>
            </div>

            <div className="analysis-mode-caveat">
              {isImageInput ? "Still images do not use video continuity modes." : "Tracking continuity varies by mode."}
            </div>
          </section>

          <div className="analysis-run-cta">
            <button
              className="button button-primary button-wide"
              type="button"
              onClick={handleProcess}
              disabled={processing || backendUnavailable}
              title={backendUnavailable ? "Backend is offline. Restore the local backend before running analysis." : ""}
            >
              {processing ? "Running Analysis..." : "Run Analysis"}
            </button>
            {processing ? (
              <button className="button button-muted button-wide" type="button" onClick={handleCancelProcess}>
                Cancel Analysis
              </button>
            ) : null}
          </div>
        </section>
      </section>

      {processing ? (
        <section className="analysis-processing-band" aria-live="polite">
          <div className="analysis-processing-copy">
            <span className="analysis-kicker">Live processing</span>
            <div className="analysis-processing-headline">
              <strong>{processingStage.label}</strong>
              <span>{processingStage.accent}</span>
            </div>
            <p>{processingStage.note}</p>
          </div>
          <div className="analysis-processing-meter">
            <div className="analysis-processing-track">
              <div
                className={clsx(
                  "analysis-processing-fill",
                  hasMeasuredProgress && "analysis-processing-fill-measured"
                )}
                style={hasMeasuredProgress ? { width: `${measuredProgressPercent}%` } : undefined}
              />
            </div>
            <div className="analysis-processing-meta">
              <span>
                {hasMeasuredProgress
                  ? `${currentFrameCount} / ${savedFrameTarget} preview frames`
                  : "Live run status"}
              </span>
              <span>
                {isFinishingRun
                  ? "Finalizing results..."
                  : hasMeasuredProgress
                    ? etaSeconds != null
                      ? formatEtaLabel(etaSeconds)
                      : `${measuredProgressPercent}% of current frame pass`
                    : "Awaiting backend completion"}
              </span>
            </div>
          </div>
        </section>
      ) : null}

      {transitionOverlay ? (
        <section className={clsx("analysis-transition-overlay", transitionOverlay.tone)} aria-live="polite">
          <div className="analysis-transition-card">
            <span className="analysis-kicker">{transitionOverlay.kicker}</span>
            <div className="analysis-transition-headline">
              <strong>{transitionOverlay.title}</strong>
              <span className="analysis-transition-pulse" aria-hidden="true" />
            </div>
            <p>{transitionOverlay.body}</p>
            <div className="analysis-transition-track">
              <div className="analysis-transition-track-fill" />
            </div>
          </div>
        </section>
      ) : null}

      <section className="analysis-context-row">
        <div className="analysis-context-summary">
          <div className="analysis-context-primary">
            <span className="analysis-kicker">Current run context</span>
            <h3>{selectedFileMeta?.name || activeJob?.filename || "No source selected"}</h3>
            <p>
              {analysisState === "no_input"
                ? "Lock one source to prepare a new run."
                : `${modelLabel(appState.selectedModel)} - ${isImageInput ? "Still image analysis" : selectedModeMeta.label}`}
            </p>
          </div>
          <div className="analysis-context-meta">
            <div>
              <span className="detail-label">Active Job</span>
              <strong title={activeJobId || ""}>{formatJobIdCompact(activeJobId)}</strong>
            </div>
            <div>
              <span className="detail-label">State</span>
              <strong>{analysisStateLabel}</strong>
            </div>
            <div>
              <span className="detail-label">Habitat</span>
              <strong>{habitatModel ? "Available" : "Optional / Missing"}</strong>
            </div>
          </div>
        </div>
      </section>

      {backendUnavailable ? (
        <div className="inline-feedback feedback-error">
          Backend is offline. New analysis runs are paused until the local backend responds again.
        </div>
      ) : null}
      {message && !selectedFileMeta ? <div className="inline-feedback feedback-success">{message}</div> : null}
      {error ? <div className="inline-feedback feedback-error">{error}</div> : null}
    </>
  );
}

function ResultsPage({ selectedJobId, onSelectJobId, refreshKey, jobs, navigate }) {
  const bundle = useJobBundle(selectedJobId, refreshKey);
  const preview = usePreviewMeta(selectedJobId, refreshKey);
  const [activeTab, setActiveTab] = useState("trend");

  useEffect(() => {
    setActiveTab("trend");
  }, [selectedJobId]);

  const metrics = bundle.metrics;
  const results = bundle.results;
  const gpsRows = results?.gps || [];
  const selectedJob = useMemo(
    () => jobs.find((job) => String(job.id) === String(selectedJobId)),
    [jobs, selectedJobId]
  );
  const mediaType = inferMediaTypeFromFilename(selectedJob?.filename);
  const sanitizedGpsTrack = useMemo(() => sanitizeGpsTrack(gpsRows), [gpsRows]);
  const gpsDistanceKm = useMemo(() => {
    if (mediaType !== "Video" || sanitizedGpsTrack.length < 2) {
      return null;
    }
    let total = 0;
    for (let index = 1; index < sanitizedGpsTrack.length; index += 1) {
      total += haversineKm(sanitizedGpsTrack[index - 1], sanitizedGpsTrack[index]);
    }
    return total > 0 ? total : null;
  }, [mediaType, sanitizedGpsTrack]);

  const herdSeries = useMemo(() => {
    const rows = metrics?.herd_series || [];
    return rows.map((row) => ({
      frame: row.frame_index,
      count: row.herd_size,
      time: row.time_seconds != null ? Number(row.time_seconds).toFixed(1) : row.frame_index,
    }));
  }, [metrics]);

  const habitatDistribution = metrics?.habitat_distribution || [];
  const habitatTimeline = metrics?.habitat_timeline || [];
  const confidenceProfile = metrics?.fast_review_bundle?.run_diagnostics?.confidence_profile || null;
  const observedAgeComposition = metrics?.fast_review_bundle?.run_diagnostics?.observed_age_composition || [];
  const samplingCaveats = metrics?.fast_review_bundle?.run_diagnostics?.caveats || [];
  const resultDetections = bundle.detections?.detections || [];
  const confidenceSeries = useMemo(() => {
    const byFrame = new Map();
    resultDetections.forEach((row) => {
      const key = Number(row.frame_index || 0);
      if (!byFrame.has(key)) {
        byFrame.set(key, []);
      }
      byFrame.get(key).push(Number(row.conf || 0));
    });
    return Array.from(byFrame.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([frame, values]) => ({
        frame,
        avgConfidence: values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1),
      }));
  }, [resultDetections]);
  const confidenceBucketData = useMemo(() => {
    const total = Math.max(resultDetections.length, 1);
    const buckets = {
      "High (>=75%)": 0,
      "Medium (55-74%)": 0,
      "Low (<55%)": 0,
    };
    resultDetections.forEach((row) => {
      const conf = Number(row.conf || 0);
      if (conf >= 0.75) {
        buckets["High (>=75%)"] += 1;
      } else if (conf >= 0.55) {
        buckets["Medium (55-74%)"] += 1;
      } else {
        buckets["Low (<55%)"] += 1;
      }
    });
    return Object.entries(buckets).map(([bucket, count]) => ({
      bucket,
      share: (count / total) * 100,
    }));
  }, [resultDetections]);
  const ageCompositionData = useMemo(
    () =>
      observedAgeComposition.map((row) => ({
        age: row.age_class,
        share: Number(row.share || 0) * 100,
      })),
    [observedAgeComposition]
  );
  const habitatPieData = useMemo(
    () =>
      habitatDistribution.map((row) => ({
        habitat: row.habitat,
        percentage: Number(row.percentage || 0),
        count: Number(row.count || 0),
        fill: habitatColorForLabel(row.habitat),
      })),
    [habitatDistribution]
  );

  const tabs = [
    { value: "trend", label: "Herd Trends" },
    { value: "habitat", label: "Habitat" },
    { value: "spatial", label: "Spatial" },
    { value: "diagnostics", label: "Diagnostics" },
  ];

  return (
    <>
      <PageHeader title="Analysis Results" description="Executive summary and analytical findings." />

      {!selectedJobId ? (
        <EmptyState
          title="No analysis selected"
          body="Open a job from history or finish an analysis run to populate the result workspace."
          action={
            <button className="button button-primary" onClick={() => navigate("/history")}>
              Open History
            </button>
          }
        />
      ) : bundle.loading ? (
        <div className="inline-feedback">Loading analysis bundle...</div>
      ) : bundle.error ? (
        <div className="inline-feedback feedback-error">{bundle.error}</div>
      ) : (
        <>
          <ResultsSummaryHero job={selectedJob} metrics={metrics} gpsRows={gpsRows} navigate={navigate} />

          {mediaType === "Video" ? (
            <FrameReviewPanel
              jobId={selectedJobId}
              detections={resultDetections}
              previewMeta={preview.meta}
              fps={metrics?.fps}
              mediaType={mediaType}
              modeKey={resolveJobModeKey(selectedJob, metrics)}
              compact
              title="Live Evidence Preview"
              subtitle="Scrub sampled frames and inspect current-frame detections without leaving Results."
            />
          ) : (
            <ResultsEvidencePreview
              jobId={selectedJobId}
              filename={selectedJob?.filename}
              previewMeta={preview.meta}
              detections={resultDetections}
            />
          )}

          <SegmentedControl items={tabs} value={activeTab} onChange={setActiveTab} />

          {activeTab === "trend" ? (
            <Panel title="Visible Count Trend" subtitle="Sampled detections across processed frames.">
              {herdSeries.length ? (
                <div className="chart-shell">
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={herdSeries}>
                      <CartesianGrid stroke="rgba(148, 163, 184, 0.10)" vertical={false} />
                      <XAxis dataKey="frame" stroke="#73829d" tickLine={false} axisLine={false} />
                      <YAxis stroke="#73829d" tickLine={false} axisLine={false} />
                      <Tooltip
                        contentStyle={{
                          background: "#121722",
                          border: "1px solid rgba(148, 163, 184, 0.18)",
                          borderRadius: "14px",
                        }}
                      />
                      <Line type="monotone" dataKey="count" stroke="#89a9ff" strokeWidth={3} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState title="No herd trend data" body="This run does not have sampled visibility points to chart." />
              )}
            </Panel>
          ) : null}

          {activeTab === "habitat" ? (
            <section className="results-two-col">
              <Panel title="Habitat Distribution" subtitle="Coverage by habitat type.">
                {habitatPieData.length ? (
                  <div className="chart-shell">
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Tooltip
                          formatter={(value, _name, entry) => [`${formatMetric(value, 0)}%`, entry?.payload?.habitat || "Habitat"]}
                          contentStyle={{
                            background: "rgba(242, 246, 232, 0.96)",
                            border: "1px solid rgba(170, 194, 117, 0.5)",
                            borderRadius: "14px",
                            color: "#1f2617",
                            boxShadow: "0 18px 36px rgba(0, 0, 0, 0.18)",
                          }}
                          itemStyle={{ color: "#1f2617", fontWeight: 700 }}
                          labelStyle={{ color: "#5b6647", fontWeight: 600 }}
                        />
                        <Pie
                          data={habitatPieData}
                          dataKey="percentage"
                          nameKey="habitat"
                          innerRadius={76}
                          outerRadius={112}
                          paddingAngle={2}
                        >
                          {habitatPieData.map((row) => (
                            <Cell key={row.habitat} fill={row.fill} />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="results-habitat-legend">
                      {habitatPieData.map((row) => (
                        <div key={row.habitat} className="results-habitat-legend-row">
                          <span className="results-habitat-swatch" style={{ backgroundColor: row.fill }} />
                          <span>{row.habitat}</span>
                          <strong>{formatPercent(row.percentage, 0)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <EmptyState title="No habitat results" body="Habitat output is unavailable for this run or the model weight is missing." />
                )}
              </Panel>

              <Panel title="Dominant Habitat" subtitle="Primary environment classification.">
                <div className="dominant-card">
                  <div className="dominant-value">{metrics?.dominant_habitat || "-"}</div>
                  <div className="dominant-stats">
                    <div>
                      <span className="detail-label">Coverage</span>
                      <strong>{formatPercent(metrics?.dominant_habitat_share || 0, 0)}</strong>
                    </div>
                    <div>
                      <span className="detail-label">Samples</span>
                      <strong>{formatMetric(habitatPieData[0]?.count || 0)}</strong>
                    </div>
                  </div>
                </div>
              </Panel>

              <Panel
                title="Habitat Timeline"
                subtitle="Classification across the processed sequence."
                className="results-full-span"
              >
                {habitatTimeline.length ? (
                  <div className="timeline-list">
                    {habitatTimeline.map((segment, index) => (
                      <div key={`${segment.start}-${segment.end}-${index}`} className="timeline-row">
                        <div className="timeline-row-fill" style={{ width: `${Math.max(8, Math.min(100, ((segment.end - segment.start + 1) / Math.max((habitatTimeline.at(-1)?.end || 1), 1)) * 100))}%` }} />
                        <div className="timeline-row-meta">
                          <strong>{segment.habitat}</strong>
                          <span>
                            Frames {segment.start} - {segment.end}
                          </span>
                          <span>{formatPercent((segment.confidence || 0) * 100)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState title="No habitat timeline" body="Habitat rows were not generated for this run." />
                )}
              </Panel>
            </section>
          ) : null}

          {activeTab === "spatial" ? (
            <Panel title="Detection Map" subtitle="Pinned GPS location when available. Travel distance is only estimated for valid video routes.">
              <div className="spatial-summary-grid">
                <div>
                  <span className="detail-label">{mediaType === "Image" ? "Pinned location" : "Start point"}</span>
                  <strong>
                    {sanitizedGpsTrack.length >= 1
                      ? `${sanitizedGpsTrack[0]?.lat?.toFixed(5)}, ${sanitizedGpsTrack[0]?.lon?.toFixed(5)}`
                      : "Unavailable"}
                  </strong>
                </div>
                <div>
                  <span className="detail-label">{mediaType === "Image" ? "GPS status" : "End point"}</span>
                  <strong>
                    {mediaType === "Image"
                      ? sanitizedGpsTrack.length >= 1
                        ? "Single point available"
                        : "Unavailable"
                      : sanitizedGpsTrack.length >= 2
                      ? `${sanitizedGpsTrack.at(-1)?.lat?.toFixed(5)}, ${sanitizedGpsTrack.at(-1)?.lon?.toFixed(5)}`
                      : "Unavailable"}
                  </strong>
                </div>
                <div>
                  <span className="detail-label">Estimated travel distance</span>
                  <strong>{mediaType === "Video" && gpsDistanceKm != null ? `${formatMetric(gpsDistanceKm, 2)} km` : "Unavailable"}</strong>
                </div>
              </div>
              {sanitizedGpsTrack.length >= 1 ? (
                <SpatialMapPanel gpsRows={sanitizedGpsTrack} />
              ) : (
                <EmptyState
                  title="GPS location unavailable"
                  body={mediaType === "Image" ? "This image does not include a usable GPS location." : "This run has too few valid GPS points to show a reliable route or pinned location."}
                />
              )}
            </Panel>
          ) : null}

          {activeTab === "diagnostics" ? (
            <Panel title="Run Diagnostics" subtitle="Confidence, observed composition, and sampling caveats.">
              <div className="diagnostic-grid">
                <div className="diagnostic-quality-card">
                  <span className="detail-label">Quality Summary</span>
                  <strong>
                    {(confidenceProfile?.avg_confidence || 0) >= 0.75
                      ? "Stable detection confidence across this run."
                      : (confidenceProfile?.avg_confidence || 0) >= 0.6
                        ? "Confidence is moderate; review edge cases."
                        : "Confidence is mixed; treat this run as lower-certainty evidence."}
                  </strong>
                  <p>
                    {formatMetric(metrics?.detections_count)} detections with average confidence of{" "}
                    {formatPercent((confidenceProfile?.avg_confidence || 0) * 100)}.
                  </p>
                </div>
                <div>
                  <span className="detail-label">Avg confidence</span>
                  <strong>{formatPercent((confidenceProfile?.avg_confidence || 0) * 100)}</strong>
                </div>
                <div>
                  <span className="detail-label">High confidence</span>
                  <strong>{formatPercent((confidenceProfile?.high_confidence_share || 0) * 100)}</strong>
                </div>
              </div>

              <section className="results-two-col diagnostics-detail-grid">
                <Panel title="Detection Confidence Profile">
                  {confidenceBucketData.length ? (
                    <div className="chart-shell">
                      <ResponsiveContainer width="100%" height={320}>
                        <BarChart data={confidenceBucketData} layout="vertical" margin={{ left: 12, right: 18 }}>
                          <CartesianGrid stroke="rgba(148, 163, 184, 0.10)" horizontal={false} />
                          <XAxis type="number" stroke="#73829d" tickLine={false} axisLine={false} unit="%" />
                          <YAxis dataKey="bucket" type="category" stroke="#73829d" tickLine={false} axisLine={false} width={120} />
                          <Tooltip
                            formatter={(value) => `${Number(value).toFixed(1)}%`}
                            contentStyle={{
                              background: "#121722",
                              border: "1px solid rgba(148, 163, 184, 0.18)",
                              borderRadius: "14px",
                            }}
                          />
                          <Bar dataKey="share" fill="#90a0bd" radius={[0, 8, 8, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <EmptyState title="No confidence profile" body="Detection confidence buckets are unavailable for this run." />
                  )}
                </Panel>

                <Panel title="Observed Age Composition">
                  {ageCompositionData.length ? (
                    <div className="chart-shell">
                      <ResponsiveContainer width="100%" height={320}>
                        <BarChart data={ageCompositionData} layout="vertical" margin={{ left: 12, right: 18 }}>
                          <CartesianGrid stroke="rgba(148, 163, 184, 0.10)" horizontal={false} />
                          <XAxis type="number" stroke="#73829d" tickLine={false} axisLine={false} unit="%" />
                          <YAxis dataKey="age" type="category" stroke="#73829d" tickLine={false} axisLine={false} width={120} />
                          <Tooltip
                            formatter={(value) => `${Number(value).toFixed(1)}%`}
                            contentStyle={{
                              background: "#121722",
                              border: "1px solid rgba(148, 163, 184, 0.18)",
                              borderRadius: "14px",
                            }}
                          />
                          <Bar dataKey="share" fill="#90a0bd" radius={[0, 8, 8, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <EmptyState title="No age composition" body="Observed age composition is unavailable for this run." />
                  )}
                </Panel>
              </section>

              {confidenceSeries.length ? (
                <Panel title="Confidence Analysis" subtitle="Confidence trend across sampled frames.">
                  <div className="chart-shell">
                    <ResponsiveContainer width="100%" height={340}>
                      <LineChart data={confidenceSeries}>
                        <CartesianGrid stroke="rgba(148, 163, 184, 0.10)" vertical={false} />
                        <XAxis dataKey="frame" stroke="#73829d" tickLine={false} axisLine={false} />
                        <YAxis stroke="#73829d" tickLine={false} axisLine={false} domain={[0, 1]} />
                        <Tooltip
                          formatter={(value) => formatPercent(Number(value) * 100)}
                          contentStyle={{
                            background: "#121722",
                            border: "1px solid rgba(148, 163, 184, 0.18)",
                            borderRadius: "14px",
                          }}
                        />
                        <Line type="monotone" dataKey="avgConfidence" stroke="#7cc0ff" strokeWidth={3} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Panel>
              ) : null}

              {samplingCaveats.length ? (
                <Panel title="Sampling Caveats">
                  <div className="diagnostic-caveat-list">
                    {samplingCaveats.map((item, index) => (
                      <div key={`${item}-${index}`} className="diagnostic-caveat-row">
                        <span className="diagnostic-caveat-dot" />
                        <span>{item}</span>
                      </div>
                    ))}
                  </div>
                </Panel>
              ) : null}
            </Panel>
          ) : null}

        </>
      )}
    </>
  );
}

function HistoryPage({ jobs, loading, error, selectedJobId, onSelectJobId, onRefresh, navigate }) {
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [previewFramesByJob, setPreviewFramesByJob] = useState({});
  const [confirmIntent, setConfirmIntent] = useState(null);

  const filteredJobs = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return jobs;
    }
    return jobs.filter((job) => String(job.filename || "").toLowerCase().includes(normalized));
  }, [jobs, query]);

  useEffect(() => {
    let active = true;
    const targetJobs = filteredJobs.slice(0, 18);
    const missing = targetJobs.filter((job) => previewFramesByJob[String(job.id)] === undefined);
    if (!missing.length) {
      return undefined;
    }
    Promise.all(
      missing.map(async (job) => {
        try {
          const meta = await api(`/media/preview/${job.id}/meta`);
          const frameIndices = meta?.frame_indices || [];
          if (frameIndices.length) {
            return [String(job.id), `${BACKEND_BASE_URL}/media/frame/${job.id}/${frameIndices[0]}`];
          }
        } catch {
          return [String(job.id), null];
        }
        return [String(job.id), null];
      })
    ).then((entries) => {
      if (!active) {
        return;
      }
      setPreviewFramesByJob((current) => {
        const next = { ...current };
        entries.forEach(([jobId, src]) => {
          next[jobId] = src;
        });
        return next;
      });
    });
    return () => {
      active = false;
    };
  }, [filteredJobs, previewFramesByJob]);

  async function deleteSelected() {
    if (!selectedJobId) {
      return;
    }
    setBusy(true);
    try {
      await api("/admin/delete-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_ids: [selectedJobId] }),
      });
      onSelectJobId("");
      onRefresh();
    } finally {
      setBusy(false);
      setConfirmIntent(null);
    }
  }

  async function clearHistory() {
    setBusy(true);
    try {
      await api("/admin/clear-history", { method: "POST" });
      onSelectJobId("");
      onRefresh();
    } finally {
      setBusy(false);
      setConfirmIntent(null);
    }
  }

  return (
    <>
      <PageHeader title="Analysis Archive" description="Historical record and operational audit trail." />

      <section className="history-controls">
        <input
          className="search-input"
          type="search"
          placeholder="Search analyses..."
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <div className="page-actions">
          <button className="button button-muted" onClick={onRefresh}>
            Refresh
          </button>
          <button
            className="button button-danger"
            disabled={!selectedJobId || busy}
            onClick={() => setConfirmIntent("delete")}
          >
            Delete Selected
          </button>
          <button
            className="button button-danger ghost"
            disabled={!jobs.length || busy}
            onClick={() => setConfirmIntent("clear")}
          >
            Clear History
          </button>
        </div>
      </section>

      {loading ? <div className="inline-feedback">Loading analysis archive...</div> : null}
      {error ? <div className="inline-feedback feedback-error">{error}</div> : null}

      <div className="history-list">
        {filteredJobs.map((job) => (
          <button
            key={job.id}
            type="button"
            className={clsx("history-row", String(selectedJobId) === String(job.id) && "history-row-active")}
            onClick={() => onSelectJobId(String(job.id))}
          >
            <div className="history-row-select">
              <input
                type="checkbox"
                checked={String(selectedJobId) === String(job.id)}
                onChange={() => onSelectJobId(String(job.id))}
                onClick={(event) => event.stopPropagation()}
              />
            </div>
            <div className="history-row-main">
              <div className="history-row-thumb">
                {previewFramesByJob[String(job.id)] ? (
                  <img src={previewFramesByJob[String(job.id)]} alt={`${job.filename} preview`} className="history-row-thumb-image" />
                ) : (
                  <div className="history-row-thumb-empty">{job.filename?.match(/\.(jpg|jpeg|png)$/i) ? "IMG" : "VID"}</div>
                )}
              </div>
              <div className="history-row-copy">
              <div className="history-row-title">
                <strong>{job.filename}</strong>
                <span className={clsx("status-chip", STATUS_TONE[job.status] || "status-queued")}>{job.status}</span>
              </div>
              <div className="history-row-meta">
                <span>{formatTimeOnly(job.created_at)}</span>
                <span>{selectedJobModeHint(job)}</span>
                <span>{job.filename?.match(/\.(jpg|jpeg|png)$/i) ? "Image" : "Video"}</span>
              </div>
              </div>
            </div>
            <div className="history-row-stat">
              <strong>{formatMetric(job.duration_seconds, 0)}</strong>
              <span>{job.duration_seconds ? "seconds" : "pending"}</span>
            </div>
            <div className="history-row-action">
              <span>{formatShortDate(job.created_at)}</span>
            </div>
          </button>
        ))}
      </div>

      {!filteredJobs.length && !loading ? (
        <EmptyState title="No matching analyses" body="Try a different filename or date search." />
      ) : null}

      {selectedJobId ? (
        <div className="page-actions">
          <button className="button button-primary" onClick={() => navigate("/results")}>
            Open Selected in Results
          </button>
        </div>
      ) : null}

      {confirmIntent ? (
        <div className="modal-backdrop" onClick={() => (busy ? null : setConfirmIntent(null))}>
          <div className="confirm-modal" onClick={(event) => event.stopPropagation()}>
            <span className="detail-label">{confirmIntent === "delete" ? "Delete Selected Run" : "Clear Archive"}</span>
            <h3>{confirmIntent === "delete" ? "Delete this run from local history?" : "Clear all local history?"}</h3>
            <p>
              {confirmIntent === "delete"
                ? "This will remove the selected run and its saved evidence from local history."
                : "This will remove every saved run, preview frame, detection row, and local output in the archive."}
            </p>
            <div className="page-actions confirm-modal-actions">
              <button className="button button-muted" disabled={busy} onClick={() => setConfirmIntent(null)}>
                Cancel
              </button>
              <button
                className="button button-danger"
                disabled={busy}
                onClick={confirmIntent === "delete" ? deleteSelected : clearHistory}
              >
                {busy ? "Working..." : confirmIntent === "delete" ? "Delete Run" : "Clear All"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}

function appStateModelFromJob(_job, metrics) {
  const explicit = String(_job?.model_key || "").trim();
  if (explicit) {
    return explicit;
  }
  const trackingBundle = metrics?.tracking_bundle;
  if (trackingBundle && metrics?.results_bundle) {
    return "labeling_data_v2_s";
  }
  return "labeling_data_v2_s";
}

function modeMetaFromMetrics(metrics) {
  return modeLabelFromKey(resolveJobModeKey(null, metrics));
}

function selectedJobModeHint(job) {
  if (!job) {
    return "-";
  }
  if (String(job.status).toLowerCase() === "processing") {
    return "In progress";
  }
  return "Stored run";
}

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const [appState, setAppState] = usePersistedAppState();
  const [jobRefreshKey, setJobRefreshKey] = useState(0);
  const [health, setHealth] = useState({
    ok: false,
    stage: "starting",
    message: "Checking backend...",
    consecutiveFailures: 0,
    lastSuccessAt: 0,
    lastFailureAt: 0,
  });
  const { jobs, loading, error } = useJobs(jobRefreshKey);
  const processingCount = jobs.filter((job) => {
    const status = String(job.status).toLowerCase();
    return status === "processing" || status === "cancel_requested";
  }).length;

  useEffect(() => {
    let active = true;
    let timer = null;

    const checkHealth = () => {
      fetch(`${BACKEND_BASE_URL}/health`)
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`Health check failed: ${response.status}`);
          }
          return response.json();
        })
        .then((payload) => {
          if (!active) {
            return;
          }
          const ok = Boolean(payload?.ok);
          const now = Date.now();
          setHealth((current) => ({
            ...current,
            ok,
            stage: ok ? "healthy" : "degraded",
            message: ok
              ? "Backend reachable on the local device."
              : "Backend responded, but health checks are degraded.",
            consecutiveFailures: ok ? 0 : Math.max(1, Number(current.consecutiveFailures || 0)),
            lastSuccessAt: ok ? now : Number(current.lastSuccessAt || 0),
            lastFailureAt: ok ? Number(current.lastFailureAt || 0) : now,
          }));
        })
        .catch(() => {
          if (!active) {
            return;
          }
          const now = Date.now();
          setHealth((current) => {
            const failureCount = Number(current.consecutiveFailures || 0) + 1;
            const stage =
              failureCount >= 2 ? "offline" : current.lastSuccessAt ? "degraded" : "starting";
            return {
              ...current,
              ok: false,
              stage,
              message:
                stage === "offline"
                  ? `Unable to reach ${BACKEND_BASE_URL}. The backend may be down or crashed.`
                  : `Still trying to reach ${BACKEND_BASE_URL}.`,
              consecutiveFailures: failureCount,
              lastFailureAt: now,
            };
          });
        });
    };

    const handleReachable = () => {
      if (!active) {
        return;
      }
      const now = Date.now();
      setHealth((current) => ({
        ...current,
        ok: true,
        stage: "healthy",
        message: "Backend reachable on the local device.",
        consecutiveFailures: 0,
        lastSuccessAt: now,
      }));
    };

    const handleVisibility = () => {
      if (document.visibilityState === "visible") {
        checkHealth();
        setJobRefreshKey((value) => value + 1);
      }
    };

    window.addEventListener("ea-backend-reachable", handleReachable);
    document.addEventListener("visibilitychange", handleVisibility);
    checkHealth();
    timer = window.setInterval(checkHealth, 8000);

    return () => {
      active = false;
      window.removeEventListener("ea-backend-reachable", handleReachable);
      document.removeEventListener("visibilitychange", handleVisibility);
      if (timer) {
        window.clearInterval(timer);
      }
    };
  }, []);

  useEffect(() => {
    if (processingCount <= 0) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setJobRefreshKey((value) => value + 1);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [processingCount]);

  useEffect(() => {
    if (!appState.selectedJobId && jobs.length) {
      setAppState((current) => ({
        ...current,
        selectedJobId: String(jobs[0].id),
      }));
    }
  }, [jobs, appState.selectedJobId, setAppState]);

  useEffect(() => {
    const pageLabel = PAGE_TITLES[location.pathname] || "Elephant Analytics";
    document.title = pageLabel === "Elephant Analytics" ? pageLabel : `${pageLabel} | Elephant Analytics`;
  }, [location.pathname]);

  function refreshJobs() {
    setJobRefreshKey((value) => value + 1);
  }

  function retryHealthCheck() {
    setHealth((current) => ({
      ...current,
      stage: current.lastSuccessAt ? "degraded" : "starting",
      message: `Rechecking ${BACKEND_BASE_URL}...`,
    }));
    setJobRefreshKey((value) => value + 1);
    api("/health")
      .then((payload) => {
        const now = Date.now();
        setHealth((current) => ({
          ...current,
          ok: Boolean(payload?.ok),
          stage: payload?.ok ? "healthy" : "degraded",
          message: payload?.ok
            ? "Backend reachable on the local device."
            : "Backend responded, but health checks are degraded.",
          consecutiveFailures: payload?.ok ? 0 : Math.max(1, Number(current.consecutiveFailures || 0)),
          lastSuccessAt: payload?.ok ? now : Number(current.lastSuccessAt || 0),
          lastFailureAt: payload?.ok ? Number(current.lastFailureAt || 0) : now,
        }));
      })
      .catch(() => {
        const now = Date.now();
        setHealth((current) => ({
          ...current,
          ok: false,
          stage: "offline",
          message: `Unable to reach ${BACKEND_BASE_URL}. The backend may be down or crashed.`,
          consecutiveFailures: Number(current.consecutiveFailures || 0) + 1,
          lastFailureAt: now,
        }));
      });
  }

  function selectJobId(jobId) {
    setAppState((current) => ({
      ...current,
      selectedJobId: jobId || null,
    }));
  }

  return (
    <AppShell health={health} processingCount={processingCount} onRetryHealth={retryHealthCheck}>
      <Routes>
        <Route path="/" element={<OverviewPage jobs={jobs} health={health} navigate={navigate} />} />
        <Route
          path="/analyze"
          element={
            <AnalyzePage
              appState={appState}
              setAppState={setAppState}
              jobs={jobs}
              onRefreshJobs={refreshJobs}
              navigate={navigate}
              health={health}
            />
          }
        />
        <Route
          path="/results"
          element={
            <ResultsPage
              selectedJobId={appState.selectedJobId}
              onSelectJobId={selectJobId}
              refreshKey={jobRefreshKey}
              jobs={jobs}
              navigate={navigate}
            />
          }
        />
        <Route
          path="/review"
          element={<Navigate to="/results" replace />}
        />
        <Route
          path="/history"
          element={
            <HistoryPage
              jobs={jobs}
              loading={loading}
              error={error}
              selectedJobId={appState.selectedJobId}
              onSelectJobId={selectJobId}
              onRefresh={refreshJobs}
              navigate={navigate}
            />
          }
        />
      </Routes>
    </AppShell>
  );
}
