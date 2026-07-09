import React, { useEffect, useRef, useState, useCallback } from "react";
import { createChart, CandlestickSeries, LineSeries, createSeriesMarkers } from "lightweight-charts";
import {
  ComposedChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid, Area, AreaChart, Bar, BarChart, Cell,
} from "recharts";

/* ============================================================================
 * XAU/USD Scalping Bot — Dashboard
 * Dark theme, real-time via WebSocket, optimised for scalping.
 * ==========================================================================*/

// Empty API => same-origin: works behind the Vite dev proxy (dev) and the
// nginx reverse-proxy (docker/prod). Set VITE_API_URL to target a remote API.
const API = import.meta?.env?.VITE_API_URL || "";
const WS_URL =
  (API
    ? API.replace(/^http/, "ws")
    : (typeof window !== "undefined"
        ? (window.location.protocol === "https:" ? "wss:" : "ws:") +
          "//" + window.location.host
        : "ws://localhost:8000")) + "/ws";

const COLORS = {
  bg: "#0a0e17",
  panel: "#121826",
  panel2: "#0f1420",
  border: "#1f2937",
  text: "#e5e7eb",
  sub: "#8b95a7",
  green: "#16c784",
  red: "#ea3943",
  grey: "#6b7280",
  blue: "#3b82f6",
  amber: "#f59e0b",
  candleUp: "#16c784",
  candleDown: "#ea3943",
};

/* ----------------------------- helpers ---------------------------------- */
// Converts a UTC ISO string to the browser's local time (e.g. Paris = UTC+2)
const fmtLocalTime = (isoStr, secs = false) => {
  if (!isoStr) return "—";
  try {
    return new Date(isoStr).toLocaleTimeString("fr-FR", {
      hour: "2-digit", minute: "2-digit",
      ...(secs ? { second: "2-digit" } : {}),
    });
  } catch { return (isoStr || "").slice(11, secs ? 19 : 16); }
};

const fmt = (n, d = 2) =>
  n === null || n === undefined || isNaN(n) ? "—" : Number(n).toFixed(d);
// Formats large monetary values with thousands separators
const fmtUSD = (n, d = 0) =>
  n === null || n === undefined || isNaN(n)
    ? "—"
    : Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const money = (n) =>
  n === null || n === undefined || isNaN(n)
    ? "—"
    : (n >= 0 ? "+" : "") + "$" + Number(n).toFixed(2);
const pct = (n) =>
  n === null || n === undefined || isNaN(n)
    ? "—"
    : (n >= 0 ? "+" : "") + Number(n).toFixed(2) + "%";

const biasColor = (b) =>
  b === "LONG" ? COLORS.green : b === "SHORT" ? COLORS.red : COLORS.grey;

function useBeep() {
  const ctxRef = useRef(null);
  return useCallback((freq = 660, dur = 0.15) => {
    try {
      if (!ctxRef.current)
        ctxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const ctx = ctxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.frequency.value = freq;
      osc.type = "sine";
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + dur);
    } catch (e) {
      /* audio not allowed yet */
    }
  }, []);
}

/* ============================= TradingView chart ========================== */
const getChartH = () => (typeof window !== "undefined" && window.innerWidth <= 768 ? 240 : 480);

function TvChart({ candles, markers, levels, orderBlocks, position, symbol, livePrice }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef({});
  const srLinesRef = useRef([]);
  const obLinesRef = useRef([]);
  const posLinesRef = useRef([]);
  const markersPluginRef = useRef(null);
  const lastCandleRef = useRef(null);
  const isForex = symbol === "EURUSD";
  const [chartH, setChartH] = useState(getChartH);

  const toUnix = (iso) => Math.floor(new Date(iso).getTime() / 1000);

  // Create chart instance once
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: { background: { color: "#0d1421" }, textColor: "#9598a1" },
      grid: { vertLines: { color: "#1a2540" }, horzLines: { color: "#1a2540" } },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: "#1a2540" },
      timeScale: { borderColor: "#1a2540", timeVisible: true, secondsVisible: false },
      localization: {
        timeFormatter: (ts) =>
          new Date(ts * 1000).toLocaleTimeString("fr-FR", {
            hour: "2-digit", minute: "2-digit", timeZone: "Europe/Paris",
          }),
      },
      width: containerRef.current.clientWidth,
      height: getChartH(),
    });
    chartRef.current = chart;

    const priceFormat = { type: "price", precision: isForex ? 5 : 2, minMove: isForex ? 0.00001 : 0.01 };

    seriesRef.current.candle = chart.addSeries(CandlestickSeries, {
      upColor: "#16c784", downColor: "#ea3943",
      borderUpColor: "#16c784", borderDownColor: "#ea3943",
      wickUpColor: "#16c784", wickDownColor: "#ea3943",
      priceFormat,
    });
    seriesRef.current.ema9 = chart.addSeries(LineSeries, {
      color: "#f59e0b", lineWidth: 1, lastValueVisible: false, priceLineVisible: false,
    });
    seriesRef.current.ema21 = chart.addSeries(LineSeries, {
      color: "#3b82f6", lineWidth: 1, lastValueVisible: false, priceLineVisible: false,
    });
    seriesRef.current.ema200 = chart.addSeries(LineSeries, {
      color: "#c084fc", lineWidth: 1.5, lastValueVisible: false, priceLineVisible: false,
    });

    const obs = new ResizeObserver(() => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    obs.observe(containerRef.current);

    const onWinResize = () => {
      const h = getChartH();
      setChartH(h);
      chart.applyOptions({ height: h });
    };
    window.addEventListener("resize", onWinResize);

    return () => {
      obs.disconnect();
      window.removeEventListener("resize", onWinResize);
      if (markersPluginRef.current) { markersPluginRef.current.detach(); markersPluginRef.current = null; }
      chart.remove();
      chartRef.current = null;
      seriesRef.current = {};
    };
  }, [isForex]);

  // Update candle + EMA data + S/R + markers
  useEffect(() => {
    const s = seriesRef.current;
    if (!s.candle || !candles?.length) return;

    s.candle.setData(candles.map((c) => ({
      time: toUnix(c.time), open: c.open, high: c.high, low: c.low, close: c.close,
    })));
    s.ema9.setData(candles.map((c) => ({ time: toUnix(c.time), value: c.ema9 })));
    s.ema21.setData(candles.map((c) => ({ time: toUnix(c.time), value: c.ema21 })));
    s.ema200.setData(candles.map((c) => ({ time: toUnix(c.time), value: c.ema200 })));

    // Trade markers
    const marks = [];
    (markers || []).forEach((m) => {
      const t = toUnix(m.time);
      if (m.type === "entry") {
        marks.push({
          time: t,
          position: m.direction === "long" ? "belowBar" : "aboveBar",
          color: m.direction === "long" ? "#16c784" : "#ea3943",
          shape: m.direction === "long" ? "arrowUp" : "arrowDown",
          text: m.direction === "long" ? "L" : "S",
        });
      } else if (m.type === "exit") {
        marks.push({ time: t, position: "aboveBar", color: "#9598a1", shape: "circle", text: "X" });
      }
    });
    const sorted = marks.sort((a, b) => a.time - b.time);
    if (markersPluginRef.current) {
      markersPluginRef.current.setMarkers(sorted);
    } else {
      markersPluginRef.current = createSeriesMarkers(s.candle, sorted);
    }

    // S/R levels
    srLinesRef.current.forEach((pl) => s.candle.removePriceLine(pl));
    srLinesRef.current = [];
    (levels?.resistance || []).forEach((r) => {
      srLinesRef.current.push(s.candle.createPriceLine({ price: r, color: "#ea394355", lineWidth: 1, lineStyle: 2, axisLabelVisible: false }));
    });
    (levels?.support || []).forEach((sv) => {
      srLinesRef.current.push(s.candle.createPriceLine({ price: sv, color: "#16c78455", lineWidth: 1, lineStyle: 2, axisLabelVisible: false }));
    });

    // Order blocks — 2 price lines per zone (top + bottom)
    obLinesRef.current.forEach((pl) => s.candle.removePriceLine(pl));
    obLinesRef.current = [];
    (orderBlocks || []).forEach((ob) => {
      const color = ob.type === "bullish" ? "#16c784" : "#ea3943";
      obLinesRef.current.push(s.candle.createPriceLine({
        price: ob.high, color: color + "cc", lineWidth: 1, lineStyle: 0,
        axisLabelVisible: false, title: ob.type === "bullish" ? "OB↑" : "OB↓",
      }));
      obLinesRef.current.push(s.candle.createPriceLine({
        price: ob.low, color: color + "66", lineWidth: 1, lineStyle: 1,
        axisLabelVisible: false,
      }));
    });

    chartRef.current?.timeScale().scrollToRealTime();
    if (candles?.length) lastCandleRef.current = candles[candles.length - 1];
  }, [candles, markers, levels, orderBlocks]);

  // Anime la bougie courante avec le prix temps réel (WebSocket, ~2s).
  // Utilise toujours le timestamp du bar M5 courant (floor à 5 min) pour
  // ne pas animer une vieille bougie si les données OHLCV sont périmées.
  useEffect(() => {
    const s = seriesRef.current;
    const base = lastCandleRef.current;
    if (!s.candle || !base || !livePrice) return;
    const nowFloor5 = Math.floor(Date.now() / (5 * 60 * 1000)) * (5 * 60);
    const baseTime = toUnix(base.time);
    const liveTime = Math.max(baseTime, nowFloor5);
    const isNewBar = liveTime > baseTime;
    s.candle.update({
      time: liveTime,
      open: isNewBar ? livePrice : base.open,
      high: isNewBar ? livePrice : Math.max(base.high, livePrice),
      low: isNewBar ? livePrice : Math.min(base.low, livePrice),
      close: livePrice,
    });
  }, [livePrice]);

  // Position SL / TP lines
  useEffect(() => {
    const s = seriesRef.current;
    if (!s.candle) return;
    posLinesRef.current.forEach((pl) => s.candle.removePriceLine(pl));
    posLinesRef.current = [];
    if (position) {
      const add = (price, color, title, lineStyle = 0) => {
        if (!price) return;
        posLinesRef.current.push(s.candle.createPriceLine({ price, color, lineWidth: 1, lineStyle, axisLabelVisible: true, title }));
      };
      add(position.stop_loss, "#ea3943", "SL");
      add(position.entry, "#f0b429", "Entrée", 2);
      if (!position.tp1_done) add(position.take_profit1, "#16c784", "TP1");
      add(position.take_profit2, "#16c784", "TP2", 1);
    }
  }, [position]);

  return (
    <div style={{ position: "relative", height: chartH, borderRadius: 8, overflow: "hidden" }}>
      <div ref={containerRef} style={{ height: "100%", background: "#0d1421" }} />
      {!candles?.length && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center",
          justifyContent: "center", color: COLORS.sub, fontSize: 14, background: "#0d1421" }}>
          Chargement du graphique…
        </div>
      )}
    </div>
  );
}

/* ============================== gauges =================================== */
function AtrGauge({ atr, avg, min }) {
  const ratio = avg ? Math.min((atr || 0) / (avg * 2), 1) : 0;
  const ok = (atr || 0) >= (min || 0.8);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: COLORS.sub }}>
        <span>ATR M5</span>
        <span style={{ color: ok ? COLORS.green : COLORS.amber }}>
          {fmt(atr, 2)} {ok ? "✓" : "⚠ bas"}
        </span>
      </div>
      <div style={{ height: 8, background: "#1a2233", borderRadius: 4, marginTop: 4, position: "relative" }}>
        <div style={{ width: `${ratio * 100}%`, height: "100%", borderRadius: 4,
          background: ok ? COLORS.green : COLORS.amber, transition: "width .4s" }} />
        <div style={{ position: "absolute", left: `${(avg ? (min / (avg * 2)) : 0.4) * 100}%`,
          top: -2, height: 12, width: 2, background: COLORS.text }} />
      </div>
      <div style={{ fontSize: 11, color: COLORS.sub, marginTop: 3 }}>
        moyenne 50p: {fmt(avg, 2)} · seuil scalp: {fmt(min, 1)}
      </div>
    </div>
  );
}

function RsiBar({ label, value }) {
  const v = value ?? 50;
  const inZone = v >= 45 && v <= 55;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: COLORS.sub }}>
        <span>{label}</span>
        <span style={{ color: inZone ? COLORS.green : COLORS.text }}>{fmt(value, 1)}</span>
      </div>
      <div style={{ height: 8, background: "#1a2233", borderRadius: 4, marginTop: 4, position: "relative" }}>
        <div style={{ position: "absolute", left: "45%", width: "10%", height: "100%",
          background: COLORS.green, opacity: 0.18 }} />
        <div style={{ position: "absolute", left: `${v}%`, top: -2, height: 12, width: 3,
          background: inZone ? COLORS.green : COLORS.blue, borderRadius: 2, transition: "left .4s" }} />
      </div>
    </div>
  );
}

/* ============================ countdown ================================== */
function useCountdown(seconds) {
  const [t, setT] = useState(seconds ?? 0);
  useEffect(() => setT(seconds ?? 0), [seconds]);
  useEffect(() => {
    if (t <= 0) return;
    const id = setInterval(() => setT((x) => Math.max(0, x - 1)), 1000);
    return () => clearInterval(id);
  }, [t > 0]);
  return t;
}
const hms = (s) => {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
};

/* ----------------------------- auth helpers ----------------------------- */
function authHeaders() {
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function logout401(onLogout) {
  localStorage.removeItem("token");
  if (onLogout) onLogout();
  else window.location.reload();
}

/* ============================ Dashboard ================================== */
export default function Dashboard({ onLogout }) {
  const [activeMarket, setActiveMarket] = useState("XAUUSD");
  const [weightsOpen, setWeightsOpen] = useState(false);
  const [state, setState] = useState(null);
  const [chart, setChart] = useState(null);
  const [tf, setTf] = useState("M5");
  const [trades, setTrades] = useState({ trades: [], equity_curve: [] });
  const [tradesScope, setTradesScope] = useState("today");
  const [cleanupLoading, setCleanupLoading] = useState(false);
  const [cleanupResult, setCleanupResult] = useState(null);
  const [blockedHours, setBlockedHours] = useState([]);
  const [deletingTrade, setDeletingTrade] = useState(null);
  const [connected, setConnected] = useState(false);
  const [patternStats, setPatternStats] = useState({});
  const [correlations, setCorrelations] = useState({});
  const [newsFeed, setNewsFeed] = useState(null);
  const [pretrainStatus, setPretrainStatus]   = useState(null);
  const [pretrainLoading, setPretrainLoading] = useState(false);
  const [multiStatus, setMultiStatus]         = useState(null);
  const [multiLoading, setMultiLoading]       = useState(false);
  const [wfStatus, setWfStatus]               = useState(null);
  const [wfLoading, setWfLoading]             = useState(false);
  const [optunaStatus, setOptunaStatus]       = useState(null);
  const [optunaLoading, setOptunaLoading]     = useState(false);
  const [wfSplits, setWfSplits]               = useState(4);
  const [optunaTrials, setOptunaTrials]       = useState(30);
  const [pretrainTrades, setPretrainTrades]   = useState(null);
  const [pretrainFilter, setPretrainFilter]   = useState("losses");
  const [pretrainPage, setPretrainPage]       = useState(0);
  const [pretrainStats, setPretrainStats]     = useState(null);
  const [pretrainDiag, setPretrainDiag]       = useState(false);
  const [pretrainResetML, setPretrainResetML] = useState(false);
  const [pretrainCapital, setPretrainCapital] = useState(1000);
  const [pretrainRiskPct, setPretrainRiskPct] = useState(2.0);
  const [pretrainStart, setPretrainStart]     = useState(() => { const d = new Date(); d.setMonth(d.getMonth() - 6); return d.toISOString().slice(0, 10); });
  const [pretrainEnd, setPretrainEnd]         = useState(() => new Date().toISOString().slice(0, 10));
  const PRETRAIN_PAGE = 20;
  const [agentStatus, setAgentStatus] = useState(null);
  const [agentHistory, setAgentHistory] = useState([]);
  const [rlStatus, setRlStatus] = useState(null);
  const [rlHistory, setRlHistory] = useState([]);
  const [rlLoading, setRlLoading] = useState(false);
  const [settingsEdit, setSettingsEdit] = useState(false);
  const [settingsDraft, setSettingsDraft] = useState({});
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [fedData, setFedData] = useState(null);
  const [tradeReport, setTradeReport] = useState(null);
  const [reportError, setReportError] = useState(null);
  const [reportLlmOpen, setReportLlmOpen] = useState(false);
  const [aiReport, setAiReport] = useState(null);
  const [aiReportLoading, setAiReportLoading] = useState(false);
  const [aiReportError, setAiReportError] = useState(null);
  const [adaptiveRunning, setAdaptiveRunning] = useState(false);
  const [adaptiveResult, setAdaptiveResult] = useState(null);
  const [adaptiveError, setAdaptiveError] = useState(null);
  const beep = useBeep();
  const lastAlertTs = useRef(null);

  const pendingSettingsRef = useRef(null);

  /* WebSocket live state */
  useEffect(() => {
    let ws;
    let retry;
    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        retry = setTimeout(connect, 3000);
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "state") {
            const pending = pendingSettingsRef.current;
            if (pending && Date.now() < pending.until) {
              setState({ ...msg.data, settings: pending.settings });
            } else {
              pendingSettingsRef.current = null;
              setState(msg.data);
            }
          }
        } catch (e) {}
      };
    };
    connect();
    return () => {
      clearTimeout(retry);
      if (ws) ws.close();
    };
  }, []);

  /* Alert sounds on new entry/exit */
  useEffect(() => {
    const alerts = state?.alerts || [];
    if (alerts.length === 0) return;
    const latest = alerts[alerts.length - 1];
    if (latest.ts !== lastAlertTs.current) {
      if (lastAlertTs.current !== null) {
        if (latest.kind === "entry") beep(720, 0.18);
        else if (latest.kind === "exit") beep(440, 0.22);
        else if (latest.kind === "danger") beep(220, 0.4);
      }
      lastAlertTs.current = latest.ts;
    }
  }, [state, beep]);

  const mkt = state?.markets?.[activeMarket] || {};

  /* Chart polling */
  useEffect(() => {
    let active = true;
    const load = () =>
      fetch(`${API}/api/chart?tf=${tf}&symbol=${activeMarket}`, {
        headers: authHeaders(),
      })
        .then((r) => { if (r.status === 401) { logout401(onLogout); throw new Error("401"); } return r.json(); })
        .then((d) => active && setChart(d))
        .catch(() => {});
    load();
    const id = setInterval(load, 15000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [tf, activeMarket]);

  /* Trades polling */
  useEffect(() => {
    let active = true;
    const load = () =>
      fetch(`${API}/api/trades?scope=${tradesScope}`, { headers: authHeaders() })
        .then((r) => { if (r.status === 401) { logout401(onLogout); throw new Error("401"); } return r.json(); })
        .then((d) => active && setTrades(d))
        .catch(() => {});
    load();
    const id = setInterval(load, 10000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [tradesScope]);

  /* Heures bloquées — chargement initial */
  useEffect(() => {
    fetch(`${API}/api/strategy/blocked-hours`, { headers: authHeaders() })
      .then((r) => r.json())
      .then((d) => setBlockedHours(d.blocked_hours || []))
      .catch(() => {});
  }, []);

  const handleToggleHour = (h) => {
    fetch(`${API}/api/strategy/blocked-hours/${h}`, { method: "POST", headers: authHeaders() })
      .then((r) => r.json())
      .then((d) => setBlockedHours(d.blocked_hours || []))
      .catch(() => {});
  };

  /* Suppression manuelle d'un trade */
  const handleDeleteTrade = (id) => {
    if (!window.confirm(`Supprimer le trade #${id} de l'historique ?`)) return;
    setDeletingTrade(id);
    fetch(`${API}/api/trades/${id}`, { method: "DELETE", headers: authHeaders() })
      .then(() => {
        setTrades((prev) => ({ ...prev, trades: prev.trades.filter((t) => t.id !== id) }));
        setDeletingTrade(null);
      })
      .catch(() => setDeletingTrade(null));
  };

  /* Reset complet de l'historique */
  const handleResetHistory = () => {
    if (!window.confirm("⚠️ Supprimer TOUT l'historique des trades et remettre les stats à zéro ?\n\nCette action est IRRÉVERSIBLE.")) return;
    if (!window.confirm("Confirmation finale : supprimer tous les trades ?")) return;
    fetch(`${API}/api/admin/reset-history`, { method: "POST", headers: authHeaders() })
      .then((r) => r.json())
      .then((d) => alert(`✅ ${d.deleted} trade(s) supprimé(s). Historique remis à zéro.`))
      .catch(() => alert("Erreur lors du reset."));
  };

  /* Reset ML Gate (dormant jusqu'à 200 trades live) */
  const handleResetML = () => {
    fetch(`${API}/api/bot/reset_ml`, { method: "POST", headers: authHeaders() })
      .then((r) => r.json())
      .catch(() => {});
  };

  /* Reset compteur trades du jour */
  const handleResetDaily = () => {
    fetch(`${API}/api/risk/reset-daily`, { method: "POST", headers: authHeaders() })
      .then((r) => r.json())
      .catch(() => {});
  };

  /* Nettoyage doublons */
  const handleCleanupDuplicates = () => {
    if (!window.confirm("Supprimer tous les trades en double (même entrée, même minute) ? Cette action est irréversible.")) return;
    setCleanupLoading(true);
    setCleanupResult(null);
    fetch(`${API}/api/admin/cleanup-duplicates`, { method: "POST", headers: authHeaders() })
      .then((r) => r.json())
      .then((d) => { setCleanupResult(d); setCleanupLoading(false); })
      .catch(() => setCleanupLoading(false));
  };

  /* Trade report polling */
  useEffect(() => {
    const load = () =>
      fetch(`${API}/api/trades/report`, { headers: authHeaders() })
        .then((r) => {
          if (r.status === 401) { logout401(onLogout); throw new Error("401"); }
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        })
        .then((d) => { setTradeReport(d); setReportError(null); })
        .catch((e) => setReportError(e.message || "erreur"));
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  /* Pattern stats polling */
  useEffect(() => {
    const load = () =>
      fetch(`${API}/api/pattern-stats`, { headers: authHeaders() })
        .then((r) => { if (r.status === 401) { logout401(onLogout); throw new Error("401"); } return r.json(); })
        .then(setPatternStats)
        .catch(() => {});
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, []);

  /* Pré-entraînement — polling pendant l'exécution */
  useEffect(() => {
    let id = null;
    const poll = () =>
      fetch(`${API}/api/pretrain/status`, { headers: authHeaders() })
        .then(r => r.ok ? r.json() : null)
        .then(d => { if (d) setPretrainStatus(d); })
        .catch(() => {});
    poll();
    id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  /* Multi-périodes — polling constant (évite le stale closure React) */
  useEffect(() => {
    const poll = () =>
      fetch(`${API}/api/pretrain/multi`, { headers: authHeaders() })
        .then(r => r.ok ? r.json() : null)
        .then(d => { if (d) setMultiStatus(d); })
        .catch(() => {});
    poll();
    const id = setInterval(poll, 4000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const poll = () => Promise.all([
      fetch(`${API}/api/pretrain/walkforward`, { headers: authHeaders() })
        .then(r => r.ok ? r.json() : null).then(d => { if (d) setWfStatus(d); }).catch(() => {}),
      fetch(`${API}/api/optimize/bayesian`, { headers: authHeaders() })
        .then(r => r.ok ? r.json() : null).then(d => { if (d) setOptunaStatus(d); }).catch(() => {}),
    ]);
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  const setPretrainPeriod = (months) => {
    const end = new Date();
    const start = new Date(end);
    start.setMonth(start.getMonth() - months);
    setPretrainStart(start.toISOString().slice(0, 10));
    setPretrainEnd(end.toISOString().slice(0, 10));
  };

  const launchPretrain = (symbol) => {
    const sym = symbol || activeMarket;
    setPretrainLoading(true);
    setPretrainTrades(null);
    setPretrainStats(null);
    setPretrainDiag(false);
    setPretrainStatus({ running: true, pct: 0, bars_done: 0, bars_total: 0,
      trades: 0, status: "Démarrage…", error: null, last_result: null });
    fetch(`${API}/api/pretrain`, {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ start: pretrainStart, end: pretrainEnd, symbol: sym, reset: pretrainResetML, strategy_mode: strategyMode, capital: pretrainCapital, risk_pct: pretrainRiskPct }),
    })
      .then(r => r.json())
      .then(d => { if (d.progress) setPretrainStatus(d.progress); setPretrainLoading(false); })
      .catch(() => {
        setPretrainStatus({ running: false, status: "error", error: "Impossible de contacter le serveur" });
        setPretrainLoading(false);
      });
  };

  const launchWalkForward = () => {
    setWfLoading(true);
    setWfStatus({ running: true, window: 0, n_splits: wfSplits, result: null, error: null });
    fetch(`${API}/api/pretrain/walkforward`, {
      method: "POST", headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ start: pretrainStart, end: pretrainEnd, n_splits: wfSplits, symbol: activeMarket, capital: pretrainCapital, risk_pct: pretrainRiskPct }),
    }).then(() => setWfLoading(false)).catch(() => setWfLoading(false));
  };

  const launchOptunaOptimize = () => {
    setOptunaLoading(true);
    setOptunaStatus({ running: true, progress: 0, n_trials: optunaTrials, best_score: 0, result: null, error: null });
    fetch(`${API}/api/optimize/bayesian`, {
      method: "POST", headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ start: pretrainStart, end: pretrainEnd, n_trials: optunaTrials, n_splits: 3, symbol: activeMarket, capital: pretrainCapital, risk_pct: pretrainRiskPct }),
    }).then(() => setOptunaLoading(false)).catch(() => setOptunaLoading(false));
  };

  const launchMultiPretrain = () => {
    setMultiLoading(true);
    setMultiStatus({ running: true, current: 0, total: 3, results: [], error: null });
    fetch(`${API}/api/pretrain/multi`, {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: activeMarket, reset: pretrainResetML, strategy_mode: strategyMode, capital: pretrainCapital, risk_pct: pretrainRiskPct }),
    })
      .then(r => r.json())
      .then(() => setMultiLoading(false))
      .catch(() => { setMultiStatus({ running: false, error: "Erreur réseau" }); setMultiLoading(false); });
  };

  const loadPretrainTrades = (filter, page) => {
    const f = filter ?? pretrainFilter;
    const p = page  ?? pretrainPage;
    fetch(`${API}/api/pretrain/trades?filter=${f}&offset=${p * PRETRAIN_PAGE}&limit=${PRETRAIN_PAGE}`, { headers: authHeaders() })
      .then(r => r.json())
      .then(d => { setPretrainTrades(d); setPretrainFilter(f); setPretrainPage(p); })
      .catch(() => {});
  };

  const loadPretrainStats = () => {
    fetch(`${API}/api/pretrain/stats`, { headers: authHeaders() })
      .then(r => r.json())
      .then(d => { setPretrainStats(d); setPretrainDiag(true); })
      .catch(() => {});
  };

  /* Agent IA status + history polling — every 30 seconds */
  useEffect(() => {
    const load = () => {
      fetch(`${API}/api/agent`, { headers: authHeaders() })
        .then((r) => { if (r.status === 401) { logout401(onLogout); throw new Error("401"); } return r.json(); })
        .then(setAgentStatus)
        .catch(() => {});
      fetch(`${API}/api/agent/history`, { headers: authHeaders() })
        .then((r) => r.ok ? r.json() : [])
        .then(setAgentHistory)
        .catch(() => {});
    };
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, []);

  /* Correlations polling — every 5 minutes (daily data, slow-moving) */
  useEffect(() => {
    const load = () =>
      fetch(`${API}/api/correlations`, { headers: authHeaders() })
        .then((r) => { if (r.status === 401) { logout401(onLogout); throw new Error("401"); } return r.json(); })
        .then(setCorrelations)
        .catch(() => {});
    load();
    const id = setInterval(load, 300000);
    return () => clearInterval(id);
  }, []);

  /* Fed / Central bank polling — every 1 hour (FRED updates daily) */
  useEffect(() => {
    const load = () =>
      fetch(`${API}/api/fed`, { headers: authHeaders() })
        .then((r) => r.ok ? r.json() : null)
        .then((d) => d && setFedData(d))
        .catch(() => {});
    load();
    const id = setInterval(load, 3600000);
    return () => clearInterval(id);
  }, []);

  /* Finnhub news-feed polling — every 2 minutes */
  useEffect(() => {
    let active = true;
    const load = () =>
      fetch(`${API}/api/news-feed`, { headers: authHeaders() })
        .then((r) => { if (r.status === 401) { logout401(onLogout); throw new Error("401"); } return r.json(); })
        .then((d) => active && setNewsFeed(d))
        .catch(() => {});
    load();
    const id = setInterval(load, 120000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  /* RL agent status + history polling */
  useEffect(() => {
    let active = true;
    const load = () => {
      fetch(`${API}/api/rl?symbol=${activeMarket}`, { headers: authHeaders() })
        .then((r) => { if (r.status === 401) { logout401(onLogout); throw new Error(); } return r.json(); })
        .then((d) => active && setRlStatus(d))
        .catch(() => {});
      fetch(`${API}/api/rl/history?symbol=${activeMarket}`, { headers: authHeaders() })
        .then((r) => r.ok ? r.json() : [])
        .then((d) => active && setRlHistory(d))
        .catch(() => {});
    };
    load();
    const id = setInterval(load, 15000);
    return () => { active = false; clearInterval(id); };
  }, [activeMarket]);

  const trainRL = () => {
    if (!window.confirm("Lancer l'entraînement RL ? Ça prend 5-15 min en arrière-plan.")) return;
    setRlLoading(true);
    fetch(`${API}/api/rl/train?symbol=${activeMarket}`, {
      method: "POST", headers: authHeaders(),
    })
      .then((r) => r.json())
      .then((d) => alert(d.message || "Entraînement lancé"))
      .catch(() => alert("Erreur"))
      .finally(() => setRlLoading(false));
  };

  const openSettingsEdit = () => {
    setSettingsDraft({
      capital: state?.risk?.capital ?? "",
      risk_per_trade_pct: state?.risk?.risk_per_trade_pct ?? "",
      daily_stop_pct: state?.risk?.daily_stop_pct ?? "",
      max_trades_per_day: state?.risk?.max_trades_per_day ?? "",
    });
    setSettingsEdit(true);
  };

  const saveSettings = () => {
    const capital = parseFloat(settingsDraft.capital);
    const riskPct = parseFloat(settingsDraft.risk_per_trade_pct);
    const stopPct = parseFloat(settingsDraft.daily_stop_pct);
    const maxTrades = parseInt(settingsDraft.max_trades_per_day, 10);
    if ([capital, riskPct, stopPct].some(v => isNaN(v) || v <= 0) || isNaN(maxTrades) || maxTrades < 1) {
      alert("Valeurs invalides — vérifie que tous les champs sont remplis avec des nombres positifs.");
      return;
    }
    const payload = {
      capital,
      risk_per_trade_pct: riskPct,
      daily_stop_pct: stopPct,
      max_trades_per_day: maxTrades,
      confirm_risk_change: true,
    };
    setSettingsSaving(true);
    fetch(`${API}/api/settings`, {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then(() => { setSettingsEdit(false); })
      .catch(() => alert("Erreur lors de la sauvegarde"))
      .finally(() => setSettingsSaving(false));
  };

  const pos = mkt.position;
  const remaining = useCountdown(pos?.remaining_seconds);
  const newsCountdown = useCountdown(state?.news?.next_event_countdown_sec);

  const closeNow = () =>
    fetch(`${API}/api/close?symbol=${activeMarket}`, {
      method: "POST",
      headers: authHeaders(),
    }).then((r) => { if (r.status === 401) logout401(onLogout); });

  const toggleBot = () =>
    fetch(`${API}/api/bot/toggle`, {
      method: "POST",
      headers: authHeaders(),
    }).then((r) => { if (r.status === 401) logout401(onLogout); });

  const sessionFilterOn = state?.settings?.session_filter !== false;
  const toggle247 = () => {
    fetch(`${API}/api/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ session_filter: sessionFilterOn ? false : true }),
    }).then((r) => { if (r.status === 401) logout401(onLogout); });
  };

  // Strategy is fixed per symbol — EURUSD always uses B (ICT), others use A (EMA)
  const strategyMode = activeMarket === "EURUSD" ? "B" : "A";

  const switchLive = () => {
    if (state?.mode === "live") {
      fetch(`${API}/api/mode`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ mode: "paper" }),
      }).then((r) => { if (r.status === 401) logout401(onLogout); });
      return;
    }
    if (!window.confirm("⚠️ Passer en mode LIVE (argent réel) ?")) return;
    if (!window.confirm("CONFIRMATION FINALE : exécuter de vrais ordres ?")) return;
    fetch(`${API}/api/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ mode: "live", confirm: true, confirm_again: true }),
    }).then((r) => { if (r.status === 401) logout401(onLogout); });
  };

  const [testSignalLoading, setTestSignalLoading] = useState(false);
  const sendTestSignal = (direction) => {
    if (state?.mode !== "paper") { alert("Signal test disponible en paper mode uniquement."); return; }
    if (mkt.position) { alert("Une position est déjà ouverte."); return; }
    setTestSignalLoading(true);
    fetch(`${API}/api/test/signal?symbol=${activeMarket}&direction=${direction}`, {
      method: "POST", headers: authHeaders(),
    })
      .then((r) => r.json())
      .then((d) => alert(d.message || d.detail || JSON.stringify(d)))
      .catch(() => alert("Erreur"))
      .finally(() => setTestSignalLoading(false));
  };

  const statusColor =
    state?.bot_status === "ACTIF" ? COLORS.green
      : state?.bot_status === "BLOQUE" ? COLORS.red : COLORS.amber;

  return (
    <div className="dashboard-root" style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text,
      fontFamily: "'Inter', system-ui, sans-serif", padding: 16 }}>

      {/* ===== loading screen when no data yet ===== */}
      {!state && (
        <div style={{ position: "fixed", inset: 0, background: COLORS.bg,
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", gap: 16, zIndex: 10 }}>
          <div style={{ fontSize: 36 }}>🟡</div>
          <div style={{ fontSize: 18, color: COLORS.text }}>Scalping Bot</div>
          <div style={{ fontSize: 13, color: connected ? COLORS.amber : COLORS.red }}>
            {connected ? "⏳ Connexion établie, chargement des données…" : "⏳ Connexion au serveur…"}
          </div>
          <div style={{ fontSize: 11, color: COLORS.sub }}>
            {connected ? "Le backend calcule les indicateurs (peut prendre 10-20 s)" : "Reconnexion dans 3 s…"}
          </div>
        </div>
      )}

      {/* ===== header / tabs ===== */}
      <div className="header-bar" style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
        <h1 style={{ fontSize: 20, margin: 0, letterSpacing: 0.5 }}>
          🟡 <span style={{ color: COLORS.sub, fontWeight: 400 }}>Scalping Bot</span>
        </h1>
        <span style={{ fontSize: 12, color: connected ? COLORS.green : COLORS.red }}>
          ● {connected ? "connecté" : "déconnecté"}
        </span>
        <span style={{ fontSize: 12, color: state?.realtime?.connected ? COLORS.green : COLORS.grey }}>
          {state?.realtime?.connected ? "⚡ Temps réel" : "○ Polling"}
        </span>
        <span style={{ fontSize: 12, padding: "2px 8px", borderRadius: 4,
          background: state?.mode === "live" ? COLORS.red : COLORS.border,
          color: state?.mode === "live" ? "#fff" : COLORS.sub }}>
          {state?.mode === "live" ? "LIVE" : "PAPER"}
        </span>
        <div className="header-tabs" style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          {Object.keys(state?.markets || { XAUUSD: 1 }).map((sym) => (
            <button key={sym} onClick={() => setActiveMarket(sym)}
              style={tabBtn(activeMarket === sym)}>
              {sym === "XAUUSD" ? "XAU/USD" : sym === "EURUSD" ? "EUR/USD" : sym === "XAGUSD" ? "XAG/USD" : sym}
            </button>
          ))}
          <div style={{ width: 1, background: COLORS.border, margin: "0 4px" }} />
          <button
            onClick={() => { localStorage.removeItem("token"); if (onLogout) onLogout(); }}
            style={{
              background: "transparent",
              color: COLORS.sub,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 6,
              padding: "6px 14px",
              fontSize: 13,
              cursor: "pointer",
              fontWeight: 500,
            }}
            title="Déconnexion"
          >
            Déconnexion
          </button>
        </div>
      </div>

      <>
          {/* ===== top band ===== */}
          <div className="stat-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 14 }}>
            <Stat label="Biais du jour"
              value={mkt.bias || "—"}
              color={biasColor(mkt.bias)} big />
            <Stat label="Session"
              value={mkt.session || "—"}
              color={mkt.session?.includes("session") ? COLORS.grey : COLORS.blue} />
            <Stat label="P&L du jour"
              value={`${money(state?.day_pnl)} (${pct(state?.day_pnl_pct)})`}
              color={(state?.day_pnl || 0) >= 0 ? COLORS.green : COLORS.red} />
            <div style={panel()}>
              <div style={{ fontSize: 11, color: COLORS.sub, textTransform: "uppercase", letterSpacing: 0.5 }}>Trades du jour</div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
                <span style={{ fontSize: 18, fontWeight: 700, color: COLORS.text }}>
                  {state?.trades_today ?? 0} / {state?.max_trades_per_day ?? 4}
                </span>
                <button onClick={handleResetDaily} title="Resynchroniser le compteur depuis la DB"
                  style={{ fontSize: 11, padding: "1px 6px", background: "transparent",
                    border: `1px solid ${COLORS.border}`, color: COLORS.sub, borderRadius: 3, cursor: "pointer" }}>
                  ↺
                </button>
              </div>
            </div>
          </div>

          <div className="main-layout" style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 14 }}>
            {/* ===== main chart ===== */}
            <div className="dashboard-panel" style={panel()}>
              <div style={{ display: "flex", alignItems: "center", marginBottom: 10 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>Graphique</h3>
                <span style={{ marginLeft: 12, color: COLORS.sub, fontSize: 13 }}>
                  {fmt(mkt.price, activeMarket === "EURUSD" ? 5 : 2)} {activeMarket === "EURUSD" ? "" : "$"}
                </span>
                {mkt.data_provider === "synthetic" && (
                  <span
                    title={mkt.data_errors ? Object.entries(mkt.data_errors).map(([k,v]) => `${k}: ${v}`).join(" | ") : "Données simulées — le bot ne trade pas ce marché"}
                    style={{ marginLeft: 10, padding: "2px 8px", borderRadius: 4, fontSize: 10,
                      fontWeight: 600, background: COLORS.red + "22", color: COLORS.red, cursor: "help" }}>
                    ⚠ DONNÉES SIMULÉES
                  </span>
                )}
                {mkt.data_provider && mkt.data_provider !== "synthetic" && (
                  <span title={`Source: ${mkt.data_provider}`}
                    style={{ marginLeft: 10, padding: "2px 8px", borderRadius: 4, fontSize: 10,
                      fontWeight: 600, background: COLORS.green + "22", color: COLORS.green }}>
                    ● {mkt.data_provider}
                  </span>
                )}
                <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                  {["M5", "M15", "H1"].map((t) => (
                    <button key={t} onClick={() => setTf(t)} style={tabBtn(tf === t, true)}>{t}</button>
                  ))}
                </div>
              </div>
              <TvChart candles={chart?.candles} markers={chart?.markers} levels={chart?.levels}
                orderBlocks={chart?.order_blocks} position={pos} symbol={activeMarket}
                livePrice={mkt.price} />
              <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 11, color: COLORS.sub, flexWrap: "wrap" }}>
                <Legend c={COLORS.amber} t="EMA9" /><Legend c={COLORS.blue} t="EMA21" />
                <Legend c="#c084fc" t="EMA200" />
                <Legend c={COLORS.green} t="Support" /><Legend c={COLORS.red} t="Résistance" />
                <Legend c={COLORS.green + "cc"} t="OB haussier" /><Legend c={COLORS.red + "cc"} t="OB baissier" />
                <span>▲ entrée long · ▼ entrée short · ✕ sortie</span>
              </div>
              {chart?.order_blocks?.length > 0 && (
                <div style={{ marginTop: 6, fontSize: 11, color: COLORS.sub }}>
                  {chart.order_blocks.filter(ob => ob.type === "bullish").length} OB haussier(s) ·{" "}
                  {chart.order_blocks.filter(ob => ob.type === "bearish").length} OB baissier(s) détectés
                </div>
              )}
            </div>

            {/* ===== side panel ===== */}
            <div className="dashboard-panel" style={panel()}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>Statut bot</h3>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ padding: "3px 10px", borderRadius: 4, fontWeight: 600, fontSize: 12,
                    background: statusColor + "22", color: statusColor }}>
                    {state?.bot_status || "—"}
                  </span>
                  <span style={{ padding: "3px 8px", borderRadius: 4, fontWeight: 700, fontSize: 11,
                    background: strategyMode === "B" ? COLORS.blue + "33" : COLORS.amber + "22",
                    color: strategyMode === "B" ? COLORS.blue : COLORS.amber,
                    border: `1px solid ${strategyMode === "B" ? COLORS.blue : COLORS.amber}88`,
                    letterSpacing: "0.5px" }}>
                    Strat {strategyMode === "A" ? "A EMA" : "B ICT"}
                  </span>
                  {state?.bot_status === "BLOQUE" && (
                    <button onClick={() => {
                      if (!window.confirm("Réinitialiser la journée ? Les compteurs sont remis à zéro mais l'historique est conservé.")) return;
                      fetch(`${API}/api/reset-day`, { method: "POST", headers: authHeaders() })
                        .then(r => r.json())
                        .then(() => {})
                        .catch(() => alert("Erreur lors de la réinitialisation"));
                    }} style={{ padding: "3px 10px", borderRadius: 4, fontSize: 11, fontWeight: 600,
                      background: "#f59e0b22", color: COLORS.amber, border: `1px solid ${COLORS.amber}`,
                      cursor: "pointer" }}>
                      ↺ Restart
                    </button>
                  )}
                </div>
              </div>

              <RsiBar label="RSI M5" value={mkt.indicators?.rsi_m5} />
              <RsiBar label="RSI M15" value={mkt.indicators?.rsi_m15} />
              <div style={{ margin: "12px 0" }}>
                <AtrGauge atr={mkt.indicators?.atr_m5} avg={mkt.indicators?.atr_avg}
                  min={mkt.indicators?.atr_min} />
              </div>

              {/* risk summary — editable */}
              <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 10, fontSize: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: COLORS.sub }}>Paramètres de risque</span>
                  {!settingsEdit && (
                    <button onClick={openSettingsEdit} style={{ background: "none", border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "1px 7px", color: COLORS.sub, cursor: "pointer", fontSize: 10 }}>
                      ✏ Modifier
                    </button>
                  )}
                </div>
                {settingsEdit ? (
                  <div style={{ fontSize: 11 }}>
                    {[
                      { label: "Capital ($)", key: "capital", min: 100, max: 1000000 },
                      { label: "Risque / trade (%)", key: "risk_per_trade_pct", min: 0.1, max: 20 },
                      { label: "Stop journalier (%)", key: "daily_stop_pct", min: 0.5, max: 50 },
                      { label: "Trades max / jour", key: "max_trades_per_day", min: 1, max: 20 },
                    ].map(({ label, key, min, max }) => (
                      <div key={key} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                        <span style={{ color: COLORS.sub, flex: 1 }}>{label}</span>
                        <input
                          type="number" min={min} max={max} step="any"
                          value={settingsDraft[key] ?? ""}
                          onChange={(e) => setSettingsDraft(d => ({ ...d, [key]: e.target.value }))}
                          style={{ width: 70, background: COLORS.panelbg, border: `1px solid ${COLORS.border}`, borderRadius: 4, color: COLORS.text, padding: "2px 5px", fontSize: 11, textAlign: "right" }}
                        />
                      </div>
                    ))}
                    <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
                      <button onClick={saveSettings} disabled={settingsSaving}
                        style={{ flex: 1, background: COLORS.green, border: "none", borderRadius: 4, color: "#fff", padding: "4px 0", cursor: "pointer", fontSize: 11, opacity: settingsSaving ? 0.6 : 1 }}>
                        {settingsSaving ? "…" : "Sauvegarder"}
                      </button>
                      <button onClick={() => setSettingsEdit(false)}
                        style={{ flex: 1, background: "none", border: `1px solid ${COLORS.border}`, borderRadius: 4, color: COLORS.sub, padding: "4px 0", cursor: "pointer", fontSize: 11 }}>
                        Annuler
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <Row k="Capital" v={`$${fmtUSD(state?.risk?.capital, 2)}`} />
                    <Row k="Risque / trade" v={`${fmt(state?.risk?.risk_per_trade_pct, 1)}% · $${fmtUSD(state?.risk?.risk_amount_usd, 2)}`} />
                    <Row k="Stop journalier" v={`-$${fmtUSD(state?.risk?.daily_loss_limit_usd, 0)}`} />
                    <Row k="Trades max / jour" v={`${state?.risk?.max_trades_per_day ?? "—"}`} />
                  </>
                )}
              </div>

              {/* ---- heures bloquées ---- */}
              <div style={{ background: "#0a1020", borderRadius: 6, padding: "8px 10px", marginBottom: 10 }}>
                <div style={{ color: COLORS.sub, fontWeight: 600, fontSize: 11, marginBottom: 8 }}>
                  Heures bloquées CET
                  <span style={{ fontWeight: 400, marginLeft: 6 }}>
                    {blockedHours.length === 0 ? "aucune" : blockedHours.map(h => `${h}h`).join(", ")}
                  </span>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                  {Array.from({ length: 24 }, (_, h) => {
                    const isSession = (h >= 8 && h < 12) || (h >= 14 && h < 18);
                    const isBlocked = blockedHours.includes(h);
                    return (
                      <button key={h} onClick={() => handleToggleHour(h)}
                        title={isBlocked ? `Débloquer ${h}h` : `Bloquer ${h}h`}
                        style={{
                          width: 30, height: 24, fontSize: 10, borderRadius: 3, cursor: "pointer",
                          background: isBlocked ? COLORS.red : isSession ? "#1a2a1a" : "#0d1624",
                          color: isBlocked ? "#fff" : isSession ? COLORS.green : COLORS.sub,
                          border: `1px solid ${isBlocked ? COLORS.red : isSession ? COLORS.green : COLORS.border}`,
                          fontWeight: isBlocked ? 700 : 400,
                        }}>
                        {h}
                      </button>
                    );
                  })}
                </div>
                <div style={{ fontSize: 9, color: COLORS.sub, marginTop: 5 }}>
                  Vert = session active · Rouge = bloqué · Cliquer pour bloquer/débloquer
                </div>
              </div>

              {/* ---- trading conditions checklist ---- */}
              {(mkt.ict_conditions || mkt.conditions) && (
                <div style={{ background: "#0a1020", borderRadius: 6, padding: "8px 10px", marginBottom: 10, fontSize: 11 }}>
                  {mkt.ict_conditions ? (
                    /* ---- Strategy B (ICT / Order Blocks) ---- */
                    <>
                      <div style={{ color: COLORS.sub, fontWeight: 600, marginBottom: 6, fontSize: 11 }}>
                        Conditions d'entrée
                        {mkt.ict_conditions.blocking_reason ? (
                          <span style={{ marginLeft: 6, color: COLORS.amber, fontWeight: 400 }}>
                            — bloqué: {mkt.ict_conditions.blocking_reason.replace(/_/g, " ")}
                          </span>
                        ) : (
                          <span style={{ marginLeft: 6, color: COLORS.green, fontWeight: 400 }}>✓ prêt</span>
                        )}
                      </div>
                      {[
                        { label: "Biais H1", ok: mkt.ict_conditions.h1_bias !== "NEUTRE", val: mkt.ict_conditions.h1_bias || "NEUTRE" },
                        { label: `ADX H1 (≥20)`, ok: mkt.ict_conditions.adx_ok,
                          val: mkt.ict_conditions.adx_h1 != null ? `${mkt.ict_conditions.adx_h1.toFixed(1)} ${mkt.ict_conditions.adx_ok ? "✓" : "✗"}` : "—" },
                        { label: "OBs haussiers", ok: mkt.ict_conditions.ob_count_long > 0,
                          val: `${mkt.ict_conditions.ob_count_long ?? 0} détecté(s)` },
                        { label: "OBs baissiers", ok: mkt.ict_conditions.ob_count_short > 0,
                          val: `${mkt.ict_conditions.ob_count_short ?? 0} détecté(s)` },
                        { label: "Prix dans OB", ok: mkt.ict_conditions.in_ob_zone,
                          val: mkt.ict_conditions.in_ob_zone ? "✓ dans zone" : "✗ hors zone" },
                        { label: "Zone S/R H1", ok: mkt.ict_conditions.sr_active,
                          val: mkt.ict_conditions.sr_active ? `✓ ${mkt.ict_conditions.sr_zone || ""}` : "—" },
                      ].map(({ label, ok, val }) => (
                        <div key={label} style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                          <span style={{ color: COLORS.sub }}>{label}</span>
                          <span style={{ color: ok ? COLORS.green : ok === false ? COLORS.red : COLORS.grey, fontWeight: 500 }}>{val}</span>
                        </div>
                      ))}
                    </>
                  ) : (
                    /* ---- Strategy A (EMA / Patterns) ---- */
                    <>
                  <div style={{ color: COLORS.sub, fontWeight: 600, marginBottom: 6, fontSize: 11 }}>
                    Conditions d'entrée
                    {mkt.conditions.blocking_reason ? (
                      <span style={{ marginLeft: 6, color: COLORS.amber, fontWeight: 400 }}>
                        — bloqué: {mkt.conditions.blocking_reason.replace(/_/g, " ")}
                      </span>
                    ) : (
                      <span style={{ marginLeft: 6, color: COLORS.green, fontWeight: 400 }}>✓ prêt</span>
                    )}
                  </div>
                  {[
                    { label: "Biais H1 EMA50", ok: mkt.conditions.h1_bias !== "NEUTRE", val: mkt.conditions.h1_bias },
                    { label: mkt.conditions.h1_bias === "SHORT" ? "M15 EMA9<EMA21" : "M15 EMA9>EMA21", ok: mkt.conditions.m15_ema_aligned,
                      val: mkt.conditions.m15_ema_aligned == null ? "—"
                        : mkt.conditions.m15_ema_aligned ? "✓"
                        : (() => {
                            const gap = mkt.conditions.m15_ema_gap;
                            const tol = mkt.conditions.m15_ema_tol ?? 0;
                            const bias = mkt.conditions.h1_bias;
                            if (gap == null) return "✗";
                            const manque = bias === "SHORT" ? (gap - tol) : (-tol - gap);
                            return `✗ (gap ${gap > 0 ? "+" : ""}${gap.toFixed(3)}, manque ${manque.toFixed(3)})`;
                          })() },
                    { label: "M15 RSI dans zone", ok: mkt.conditions.m15_rsi_ok,
                      val: mkt.conditions.m15_rsi_ok == null ? "—" : (mkt.conditions.m15_rsi_ok ? "✓" : "✗") },
                    { label: "ATR M5", ok: mkt.conditions.atr_ok, val: mkt.conditions.atr_ok ? "✓" : "✗" },
                    { label: "EMA9 aligné M5", ok: mkt.conditions.ema9_aligned, val: mkt.conditions.ema9_aligned ? "✓" : "✗" },
                    { label: "Pattern", ok: mkt.conditions.patterns?.length > 0,
                      val: mkt.conditions.patterns?.length > 0 ? mkt.conditions.patterns.join(", ").replace(/_/g, " ") : "aucun" },
                  ].map(({ label, ok, val }) => (
                    <div key={label} style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                      <span style={{ color: COLORS.sub }}>{label}</span>
                      <span style={{ color: ok ? COLORS.green : COLORS.red, fontWeight: 500 }}>{val}</span>
                    </div>
                  ))}
                  {/* Pattern weight gate */}
                  {mkt.conditions.patterns?.length > 0 && (
                    <div style={{ marginTop: 4, borderTop: `1px solid ${COLORS.border}`, paddingTop: 4 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                        <span style={{ color: COLORS.sub }}>Poids total patterns</span>
                        <span style={{
                          color: mkt.conditions.weight_gate_ok ? COLORS.green : COLORS.red,
                          fontWeight: 600,
                        }}>
                          {mkt.conditions.pattern_weight_sum != null
                            ? `${mkt.conditions.pattern_weight_sum.toFixed(2)} / 1.0 ${mkt.conditions.weight_gate_ok ? "✓" : "✗"}`
                            : "—"}
                        </span>
                      </div>
                      {mkt.conditions.pattern_weight_detail && Object.entries(mkt.conditions.pattern_weight_detail).map(([p, w]) => (
                        <div key={p} style={{ display: "flex", justifyContent: "space-between", paddingLeft: 8, opacity: 0.8 }}>
                          <span style={{ color: COLORS.sub, fontSize: 10 }}>{p.replace(/_/g, " ")}</span>
                          <span style={{ fontSize: 10, color: w >= 1.0 ? COLORS.green : w >= 0.6 ? COLORS.amber : COLORS.red }}>
                            x{w.toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  {/* ML Gate probability */}
                  <div style={{ marginTop: 4, borderTop: `1px solid ${COLORS.border}`, paddingTop: 4 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ color: COLORS.sub }}>ML Gate</span>
                      {!mkt.conditions.ml_ready ? (
                        <span style={{ color: COLORS.grey, fontSize: 11 }}>
                          {mkt?.ml_gate?.n_samples != null
                            ? `apprentissage… ${mkt.ml_gate.n_samples}/${mkt.ml_gate.n_min}`
                            : "inactif"}
                        </span>
                      ) : (
                        <span style={{
                          color: mkt.conditions.ml_prob >= 0.55 ? COLORS.green
                               : mkt.conditions.ml_prob >= 0.45 ? COLORS.amber
                               : COLORS.red,
                          fontWeight: 600,
                        }}>
                          {mkt.conditions.ml_prob != null
                            ? `${(mkt.conditions.ml_prob * 100).toFixed(0)}% ${mkt.conditions.ml_prob >= (mkt?.ml_gate?.threshold ?? 0.45) ? "✓" : "✗"}`
                            : "—"}
                        </span>
                      )}
                    </div>
                    {/* Reset ML Gate */}
                    {mkt?.ml_gate?.n_samples > 0 && (
                      <div style={{ marginTop: 4 }}>
                        <button onClick={handleResetML}
                          title="Remet n_samples=0 en mémoire + DB — gate dormant jusqu'à 200 trades live"
                          style={{ fontSize: 10, padding: "1px 6px", background: "transparent",
                            border: `1px solid ${COLORS.red}`, color: COLORS.red,
                            borderRadius: 3, cursor: "pointer", width: "100%" }}>
                          ↺ Reset ML Gate ({mkt.ml_gate.n_samples} samples)
                        </button>
                      </div>
                    )}
                    {/* Série noire */}
                    {mkt?.ml_gate?.consecutive_losses >= 3 && (
                      <div style={{ display: "flex", justifyContent: "space-between", paddingLeft: 8, marginTop: 2 }}>
                        <span style={{ color: COLORS.red, fontSize: 10 }}>
                          ⚠ Série noire ({mkt.ml_gate.consecutive_losses} pertes)
                        </span>
                        <span style={{ color: COLORS.amber, fontSize: 10 }}>
                          seuil {(mkt.ml_gate.threshold * 100).toFixed(0)}%
                          {mkt.ml_gate.streak_boost > 0 && ` (+${(mkt.ml_gate.streak_boost * 100).toFixed(0)}%)`}
                        </span>
                      </div>
                    )}
                  </div>
                    </> /* fin Strategy A */
                  )} {/* fin ternaire ICT vs A */}
                  {/* Pré-entraînement — commun aux deux stratégies */}
                  <div style={{ marginTop: 6, borderTop: `1px solid ${COLORS.border}`, paddingTop: 6 }}>
                    {/* Sélecteur de période */}
                    {!pretrainStatus?.running && (
                      <div style={{ marginBottom: 6 }}>
                        <div style={{ display: "flex", gap: 3, marginBottom: 4 }}>
                          {[1, 3, 6].map(m => {
                            const s = new Date(); s.setMonth(s.getMonth() - m);
                            const active = pretrainStart === s.toISOString().slice(0, 10);
                            return (
                              <button key={m} onClick={() => setPretrainPeriod(m)}
                                style={{ flex: 1, fontSize: 10, padding: "2px 0", cursor: "pointer",
                                  borderRadius: 3, border: `1px solid ${COLORS.blue}`,
                                  background: active ? COLORS.blue + "44" : "transparent",
                                  color: COLORS.blue }}>
                                {m}M
                              </button>
                            );
                          })}
                        </div>
                        <div style={{ display: "flex", gap: 4 }}>
                          <input type="date" value={pretrainStart} onChange={e => setPretrainStart(e.target.value)}
                            style={{ flex: 1, fontSize: 10, background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                              borderRadius: 3, color: COLORS.text, padding: "2px 4px" }} />
                          <input type="date" value={pretrainEnd} onChange={e => setPretrainEnd(e.target.value)}
                            style={{ flex: 1, fontSize: 10, background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                              borderRadius: 3, color: COLORS.text, padding: "2px 4px" }} />
                        </div>
                        <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: 9, color: COLORS.sub, marginBottom: 2 }}>Capital ($)</div>
                            <input type="number" value={pretrainCapital} onChange={e => setPretrainCapital(parseFloat(e.target.value) || 1000)}
                              style={{ width: "100%", fontSize: 10, background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                                borderRadius: 3, color: COLORS.text, padding: "2px 4px", boxSizing: "border-box" }} />
                          </div>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: 9, color: COLORS.sub, marginBottom: 2 }}>Risque / trade (%)</div>
                            <input type="number" step="0.1" value={pretrainRiskPct} onChange={e => setPretrainRiskPct(parseFloat(e.target.value) || 2.0)}
                              style={{ width: "100%", fontSize: 10, background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                                borderRadius: 3, color: COLORS.text, padding: "2px 4px", boxSizing: "border-box" }} />
                          </div>
                        </div>
                        <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4, cursor: "pointer" }}>
                          <input type="checkbox" checked={pretrainResetML} onChange={e => setPretrainResetML(e.target.checked)}
                            style={{ accentColor: COLORS.amber }} />
                          <span style={{ fontSize: 10, color: pretrainResetML ? COLORS.amber : COLORS.sub }}>
                            Réinitialiser ML {pretrainResetML ? "(repart de zéro)" : "(cumule avec sessions précédentes)"}
                          </span>
                        </label>
                      </div>
                    )}
                    {pretrainStatus?.running ? (
                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span style={{ color: COLORS.amber, fontSize: 11 }}>Pré-entraînement en cours…</span>
                          <span style={{ color: COLORS.amber, fontSize: 11 }}>{pretrainStatus.pct ?? 0}%</span>
                        </div>
                        <div style={{ background: COLORS.border, borderRadius: 3, height: 4 }}>
                          <div style={{ background: COLORS.amber, width: `${pretrainStatus.pct ?? 0}%`, height: 4, borderRadius: 3, transition: "width 0.5s" }} />
                        </div>
                        <div style={{ color: COLORS.sub, fontSize: 10, marginTop: 2 }}>
                          {pretrainStatus.trades ?? 0} trades · {pretrainStatus.bars_done ?? 0}/{pretrainStatus.bars_total ?? 0} barres — {pretrainStatus.status}
                        </div>
                      </div>
                    ) : pretrainStatus?.status === "error" ? (
                      <div>
                        <div style={{ color: COLORS.red, fontSize: 11, marginBottom: 4 }}>
                          ✗ Erreur pré-entraînement
                        </div>
                        <div style={{ color: COLORS.sub, fontSize: 10, marginBottom: 6, wordBreak: "break-word" }}>
                          {pretrainStatus.error || "Erreur inconnue"}
                        </div>
                        <button
                          onClick={() => launchPretrain(activeMarket)}
                          disabled={pretrainLoading}
                          style={{ width: "100%", background: COLORS.red + "22",
                            border: `1px solid ${COLORS.red}`, borderRadius: 4,
                            color: COLORS.red, padding: "4px 0", cursor: "pointer", fontSize: 11 }}>
                          Réessayer
                        </button>
                      </div>
                    ) : pretrainStatus?.status === "done" && pretrainStatus.last_result ? (
                      <div>
                        <div style={{ color: COLORS.green, fontSize: 11, marginBottom: 4 }}>
                          ✓ Pré-entraînement terminé
                        </div>
                        {/* Ligne 1 : trades + WR + PF */}
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                          <span style={{ color: COLORS.sub }}>{pretrainStatus.last_result.n_trades} trades</span>
                          <span style={{ color: pretrainStatus.last_result.win_rate >= 0.45 ? COLORS.green : COLORS.red }}>
                            {(pretrainStatus.last_result.win_rate * 100).toFixed(0)}% WR
                          </span>
                          <span style={{ color: (pretrainStatus.last_result.profit_factor ?? 0) >= 1.0 ? COLORS.green : COLORS.red }}>
                            PF {(pretrainStatus.last_result.profit_factor ?? 0).toFixed(2)}
                          </span>
                        </div>
                        {/* Ligne 2 : avg win / avg loss */}
                        {pretrainStatus.last_result.avg_win != null && (
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 2 }}>
                            <span style={{ color: COLORS.sub }}>Moy. gain</span>
                            <span style={{ color: COLORS.green }}>+{pretrainStatus.last_result.avg_win.toFixed(2)}$</span>
                            <span style={{ color: COLORS.sub }}>Moy. perte</span>
                            <span style={{ color: COLORS.red }}>-{pretrainStatus.last_result.avg_loss.toFixed(2)}$</span>
                          </div>
                        )}
                        {/* Ligne 3 : net PnL */}
                        {pretrainStatus.last_result.net_pnl != null && (
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 4 }}>
                            <span style={{ color: COLORS.sub }}>Net PnL</span>
                            <span style={{ color: pretrainStatus.last_result.net_pnl >= 0 ? COLORS.green : COLORS.red, fontWeight: 600 }}>
                              {pretrainStatus.last_result.net_pnl >= 0 ? "+" : ""}{pretrainStatus.last_result.net_pnl.toFixed(2)}$
                            </span>
                            <span style={{ color: COLORS.sub }}>{pretrainStatus.last_result.period}</span>
                          </div>
                        )}
                        <div style={{ color: COLORS.sub, fontSize: 10, marginBottom: 4 }}>
                          ML: {pretrainStatus.last_result.ml_samples} échantillons
                        </div>
                        {/* Equity curve */}
                        {pretrainStatus.last_result.equity_curve?.length > 1 && (() => {
                          const curve = pretrainStatus.last_result.equity_curve;
                          const vals = curve.map(p => p.equity);
                          const minV = Math.min(...vals);
                          const maxV = Math.max(...vals);
                          const range = maxV - minV || 1;
                          const W = 220, H = 48;
                          const pts = curve.map((p, i) => {
                            const x = (i / (curve.length - 1)) * W;
                            const y = H - ((p.equity - minV) / range) * H;
                            return `${x},${y}`;
                          }).join(" ");
                          const isProfit = vals[vals.length - 1] >= vals[0];
                          const color = isProfit ? COLORS.green : COLORS.red;
                          return (
                            <div style={{ marginBottom: 6 }}>
                              <svg width={W} height={H} style={{ display: "block", width: "100%", height: H }}>
                                <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" />
                                <line x1="0" y1={H - ((10000 - minV) / range) * H} x2={W} y2={H - ((10000 - minV) / range) * H}
                                  stroke={COLORS.border} strokeWidth="1" strokeDasharray="3,3" />
                              </svg>
                            </div>
                          );
                        })()}
                        {/* Analyse trade par trade */}
                        <div style={{ marginTop: 6, borderTop: `1px solid ${COLORS.border}`, paddingTop: 6 }}>
                          {/* Filtres */}
                          <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
                            {["losses", "wins", "all"].map(f => (
                              <button key={f}
                                onClick={() => { loadPretrainTrades(f, 0); }}
                                style={{
                                  flex: 1, fontSize: 10, padding: "2px 0", cursor: "pointer", borderRadius: 3,
                                  border: `1px solid ${f === "losses" ? COLORS.red : f === "wins" ? COLORS.green : COLORS.sub}`,
                                  background: pretrainFilter === f && pretrainTrades
                                    ? (f === "losses" ? COLORS.red + "33" : f === "wins" ? COLORS.green + "33" : COLORS.sub + "33")
                                    : "transparent",
                                  color: f === "losses" ? COLORS.red : f === "wins" ? COLORS.green : COLORS.sub,
                                }}>
                                {f === "losses" ? "Pertes" : f === "wins" ? "Gains" : "Tous"}
                              </button>
                            ))}
                          </div>
                          {/* Table */}
                          {pretrainTrades && (
                            <>
                              <div style={{ overflowY: "auto", maxHeight: 260, fontSize: 10 }}>
                                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                                  <thead>
                                    <tr style={{ color: COLORS.sub, textAlign: "left" }}>
                                      <th style={{ paddingBottom: 2 }}>Date</th>
                                      <th>Sess</th>
                                      <th>Dir</th>
                                      <th>Sortie</th>
                                      <th style={{ textAlign: "right" }}>PnL</th>
                                      <th style={{ textAlign: "right" }}>MAE/R</th>
                                      <th style={{ textAlign: "right" }}>MFE/R</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {pretrainTrades.trades.map((t, idx) => (
                                      <tr key={idx}
                                        style={{ borderTop: `1px solid ${COLORS.border}`, color: t.won ? COLORS.green : COLORS.red }}
                                        title={t.patterns?.join(", ") || "—"}>
                                        <td style={{ paddingTop: 2, paddingBottom: 2, color: COLORS.sub }}>
                                          {t.entry_ts?.slice(5, 16).replace("T", " ")}
                                        </td>
                                        <td style={{ color: COLORS.sub }}>{t.session?.slice(0, 2)}</td>
                                        <td>{t.direction === "long" ? "↑" : "↓"}</td>
                                        <td style={{ color: COLORS.sub, fontSize: 9 }}>{t.exit_reason}</td>
                                        <td style={{ textAlign: "right" }}>{t.pnl >= 0 ? "+" : ""}{t.pnl?.toFixed(2)}</td>
                                        <td style={{ textAlign: "right", color: COLORS.sub }}>{t.mae_r?.toFixed(2)}</td>
                                        <td style={{ textAlign: "right", color: COLORS.sub }}>{t.mfe_r?.toFixed(2)}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                              {/* Pagination */}
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
                                <button onClick={() => loadPretrainTrades(pretrainFilter, pretrainPage - 1)}
                                  disabled={pretrainPage === 0}
                                  style={{ fontSize: 10, background: "transparent", border: `1px solid ${COLORS.border}`,
                                    color: COLORS.sub, borderRadius: 3, padding: "1px 6px", cursor: "pointer" }}>‹</button>
                                <span style={{ fontSize: 10, color: COLORS.sub }}>
                                  {pretrainPage * PRETRAIN_PAGE + 1}–{Math.min((pretrainPage + 1) * PRETRAIN_PAGE, pretrainTrades.total)} / {pretrainTrades.total}
                                </span>
                                <button onClick={() => loadPretrainTrades(pretrainFilter, pretrainPage + 1)}
                                  disabled={(pretrainPage + 1) * PRETRAIN_PAGE >= pretrainTrades.total}
                                  style={{ fontSize: 10, background: "transparent", border: `1px solid ${COLORS.border}`,
                                    color: COLORS.sub, borderRadius: 3, padding: "1px 6px", cursor: "pointer" }}>›</button>
                              </div>
                            </>
                          )}
                        </div>

                        {/* ── Diagnostics stratégie ── */}
                        <div style={{ marginTop: 6, borderTop: `1px solid ${COLORS.border}`, paddingTop: 6 }}>
                          <button
                            onClick={() => pretrainDiag ? setPretrainDiag(false) : loadPretrainStats()}
                            style={{ width: "100%", fontSize: 10, padding: "3px 0", cursor: "pointer",
                              background: pretrainDiag ? COLORS.amber + "22" : "transparent",
                              border: `1px solid ${COLORS.amber}`, borderRadius: 3, color: COLORS.amber }}>
                            {pretrainDiag ? "▲ Masquer le diagnostic" : "▼ Diagnostic stratégie"}
                          </button>

                          {pretrainDiag && pretrainStats && !pretrainStats.error && (() => {
                            const sd  = pretrainStats.by_session_dir || {};
                            const er  = pretrainStats.exit_reasons   || {};
                            const keys = ["London_long", "London_short", "NY_long", "NY_short"];
                            const erOrder = ["sl", "sl_after_tp1", "timeout", "tp1", "tp2"];
                            const erLabel = { sl: "SL direct", sl_after_tp1: "SL après TP1", timeout: "Timeout", tp1: "TP1 seul", tp2: "TP2 ✓" };
                            const sessLabel = { London_long: "Lo ↑", London_short: "Lo ↓", NY_long: "NY ↑", NY_short: "NY ↓" };

                            const cov = pretrainStats.data_coverage || {};
                            const covOk = cov.full_coverage !== false;

                            return (
                              <div style={{ marginTop: 6, fontSize: 10 }}>

                                {/* Couverture données */}
                                {cov.actual_start && (
                                  <div style={{
                                    marginBottom: 6, padding: "3px 6px", borderRadius: 3,
                                    background: covOk ? COLORS.green + "18" : COLORS.red + "18",
                                    border: `1px solid ${covOk ? COLORS.green : COLORS.red}44`,
                                    color: covOk ? COLORS.green : COLORS.red,
                                  }}>
                                    {covOk ? "✓" : "⚠"} Données : {cov.actual_start} → {cov.actual_end} ({cov.bars} bougies)
                                    {!covOk && <span style={{ color: COLORS.sub }}> — demandé depuis {cov.requested_start}</span>}
                                    {cov.provider_errors && Object.keys(cov.provider_errors).length > 0 && (
                                      <div style={{ color: COLORS.red, fontSize: 9, marginTop: 2 }}>
                                        Erreur provider : {Object.entries(cov.provider_errors).map(([k,v]) => `${k}: ${v}`).join(" | ")}
                                      </div>
                                    )}
                                  </div>
                                )}

                                {/* Near-wins & lucky wins */}
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, gap: 4 }}>
                                  <div style={{ flex: 1, background: COLORS.red + "18", borderRadius: 4, padding: "4px 6px", textAlign: "center" }}>
                                    <div style={{ color: COLORS.red, fontWeight: 600, fontSize: 13 }}>{pretrainStats.near_wins_pct}%</div>
                                    <div style={{ color: COLORS.sub }}>pertes "near-win"</div>
                                    <div style={{ color: COLORS.sub, fontSize: 9 }}>MFE ≥ 0.5R avant SL</div>
                                  </div>
                                  <div style={{ flex: 1, background: COLORS.amber + "18", borderRadius: 4, padding: "4px 6px", textAlign: "center" }}>
                                    <div style={{ color: COLORS.amber, fontWeight: 600, fontSize: 13 }}>{pretrainStats.lucky_wins_pct}%</div>
                                    <div style={{ color: COLORS.sub }}>gains "chanceux"</div>
                                    <div style={{ color: COLORS.sub, fontSize: 9 }}>MAE ≥ 0.5R avant TP</div>
                                  </div>
                                  <div style={{ flex: 1, background: COLORS.orange + "18", borderRadius: 4, padding: "4px 6px", textAlign: "center" }}>
                                    <div style={{ color: COLORS.orange || "#f97316", fontWeight: 600, fontSize: 13 }}>{pretrainStats.false_stops_pct ?? "—"}%</div>
                                    <div style={{ color: COLORS.sub }}>false stops</div>
                                    <div style={{ color: COLORS.sub, fontSize: 9 }}>SL → TP1 dans 10 bougies</div>
                                  </div>
                                  <div style={{ flex: 1, background: "#8b5cf622", borderRadius: 4, padding: "4px 6px", textAlign: "center" }}>
                                    <div style={{ color: "#8b5cf6", fontWeight: 600, fontSize: 13 }}>
                                      {pretrainStats.false_be_pct != null ? `${pretrainStats.false_be_pct}%` : "—"}
                                    </div>
                                    <div style={{ color: COLORS.sub }}>false BE</div>
                                    <div style={{ color: COLORS.sub, fontSize: 9 }}>SL-BE → TP2 dans 20 bougies ({pretrainStats.false_be_n ?? 0})</div>
                                  </div>
                                </div>

                                {/* Grille session × direction */}
                                <div style={{ color: COLORS.sub, marginBottom: 3 }}>WR par session / direction</div>
                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 3, marginBottom: 8 }}>
                                  {keys.map(k => {
                                    const d = sd[k];
                                    if (!d) return <div key={k} style={{ background: COLORS.border + "44", borderRadius: 3, padding: "3px 0", textAlign: "center", color: COLORS.sub }}>—</div>;
                                    const wr = d.wr * 100;
                                    const col = wr >= 35 ? COLORS.green : wr >= 25 ? COLORS.amber : COLORS.red;
                                    return (
                                      <div key={k} style={{ background: col + "22", borderRadius: 3, padding: "3px 4px", textAlign: "center", border: `1px solid ${col}44` }}>
                                        <div style={{ color: col, fontWeight: 600 }}>{wr.toFixed(0)}%</div>
                                        <div style={{ color: COLORS.sub, fontSize: 9 }}>{sessLabel[k]}</div>
                                        <div style={{ color: COLORS.sub, fontSize: 9 }}>{d.n} trades</div>
                                      </div>
                                    );
                                  })}
                                </div>

                                {/* Décomposition raisons de sortie */}
                                <div style={{ color: COLORS.sub, marginBottom: 3 }}>Raisons de sortie</div>
                                {erOrder.filter(k => er[k]).map(k => {
                                  const d = er[k];
                                  const barColor = k.startsWith("sl") ? COLORS.red : k === "timeout" ? COLORS.amber : COLORS.green;
                                  return (
                                    <div key={k} style={{ marginBottom: 3 }}>
                                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 1 }}>
                                        <span style={{ color: COLORS.sub }}>{erLabel[k] || k}</span>
                                        <span style={{ color: barColor }}>{d.pct}% ({d.count}) — moy {d.avg_pnl >= 0 ? "+" : ""}{d.avg_pnl}$</span>
                                      </div>
                                      <div style={{ background: COLORS.border, borderRadius: 2, height: 3 }}>
                                        <div style={{ background: barColor, width: `${d.pct}%`, height: 3, borderRadius: 2 }} />
                                      </div>
                                      <div style={{ color: COLORS.sub, fontSize: 9, marginTop: 1 }}>
                                        MAE moy {d.avg_mae_r}R · MFE moy {d.avg_mfe_r}R
                                      </div>
                                    </div>
                                  );
                                })}

                                {/* Diagnostic indicateurs : SL direct vs TP2 */}
                                {(() => {
                                  const diag = pretrainStats.indicator_diagnostic || {};
                                  const sl  = diag["SL_direct"] || {};
                                  const tp2 = diag["TP2"] || {};
                                  if (!sl.n || !tp2.n) return null;
                                  const rows = [
                                    { key: "rsi_m5",         label: "RSI M5"         },
                                    { key: "rsi_m15",        label: "RSI M15"        },
                                    { key: "adx_h1",         label: "ADX H1"         },
                                    { key: "atr",            label: "ATR (M5)"       },
                                    { key: "n_patterns",     label: "Nb patterns"    },
                                    { key: "ema9_dist_r",    label: "Dist EMA9 (R)"  },
                                    { key: "ema200_dist_r",  label: "Dist EMA200 H1" },
                                    { key: "vwap_above_pct", label: "Au-dessus VWAP %"},
                                    { key: "london_pct",     label: "London %"       },
                                    { key: "h1_rsi",         label: "RSI H1"         },
                                    { key: "body_ratio",     label: "Corps/ATR"      },
                                    { key: "h4_bias",        label: "Biais H4 (+1L/-1S)"},
                                  ];
                                  return (
                                    <div style={{ marginTop: 8 }}>
                                      <div style={{ color: COLORS.sub, marginBottom: 4 }}>Indicateurs : SL direct vs TP2</div>
                                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
                                        <thead>
                                          <tr>
                                            <th style={{ textAlign: "left", color: COLORS.sub, fontWeight: "normal", paddingBottom: 3 }}>Indicateur</th>
                                            <th style={{ textAlign: "right", color: COLORS.red, fontWeight: "normal" }}>SL dir ({sl.n})</th>
                                            <th style={{ textAlign: "right", color: COLORS.green, fontWeight: "normal" }}>TP2 ({tp2.n})</th>
                                            <th style={{ textAlign: "right", color: COLORS.amber, fontWeight: "normal" }}>Δ</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {rows.map(({ key, label }) => {
                                            const sv = sl[key];
                                            const tv = tp2[key];
                                            const delta = (sv != null && tv != null) ? sv - tv : null;
                                            const abs = Math.abs(delta ?? 0);
                                            const deltaCol = abs >= 3 ? COLORS.amber : COLORS.sub;
                                            return (
                                              <tr key={key}>
                                                <td style={{ color: COLORS.sub, paddingRight: 4, paddingTop: 2 }}>{label}</td>
                                                <td style={{ textAlign: "right", color: COLORS.red, paddingTop: 2 }}>{sv ?? "—"}</td>
                                                <td style={{ textAlign: "right", color: COLORS.green, paddingTop: 2 }}>{tv ?? "—"}</td>
                                                <td style={{ textAlign: "right", color: deltaCol, fontWeight: abs >= 3 ? 600 : "normal", paddingTop: 2 }}>
                                                  {delta != null ? `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}` : "—"}
                                                </td>
                                              </tr>
                                            );
                                          })}
                                        </tbody>
                                      </table>
                                      <div style={{ color: COLORS.sub, fontSize: 9, marginTop: 3 }}>
                                        Δ en orange (≥3) = indicateur discriminant → candidat à resserrer
                                      </div>
                                    </div>
                                  );
                                })()}

                                {/* WR par heure CET */}
                                {(() => {
                                  const wbh = pretrainStats.wr_by_hour || {};
                                  if (Object.keys(wbh).length === 0) return null;
                                  const hours = [8,9,10,11,12,13,14,15,16,17,18];
                                  const isLondon = h => h >= 8 && h < 12;
                                  const isNY     = h => h >= 14 && h < 18;
                                  return (
                                    <div style={{ marginTop: 10 }}>
                                      <div style={{ color: COLORS.sub, marginBottom: 5 }}>WR par heure CET</div>
                                      {hours.map(h => {
                                        const d = wbh[String(h)];
                                        const inSession = isLondon(h) || isNY(h);
                                        const wr = d ? d.wr * 100 : null;
                                        const barCol = !d ? COLORS.border
                                          : wr >= 40 ? COLORS.green
                                          : wr >= 30 ? COLORS.amber
                                          : COLORS.red;
                                        const rowBg = isLondon(h) ? "#3b82f611" : isNY(h) ? "#f9731611" : "transparent";
                                        const sessTag = isLondon(h) ? " Lo" : isNY(h) ? " NY" : "";
                                        return (
                                          <div key={h} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2, background: rowBg, borderRadius: 2, padding: "1px 3px" }}>
                                            <div style={{ width: 32, color: inSession ? COLORS.text : COLORS.sub, fontSize: 9, flexShrink: 0 }}>
                                              {h}h{sessTag}
                                            </div>
                                            <div style={{ flex: 1, background: COLORS.border + "44", borderRadius: 2, height: 10, overflow: "hidden" }}>
                                              {d && (
                                                <div style={{ width: `${Math.min(wr, 100)}%`, height: "100%", background: barCol, borderRadius: 2 }} />
                                              )}
                                            </div>
                                            <div style={{ width: 52, textAlign: "right", fontSize: 9, color: d ? barCol : COLORS.sub, flexShrink: 0 }}>
                                              {d ? `${wr.toFixed(0)}% (${d.n})` : "—"}
                                            </div>
                                          </div>
                                        );
                                      })}
                                      <div style={{ color: COLORS.sub, fontSize: 9, marginTop: 2 }}>
                                        Lo=London 8-12h · NY=14-18h · vert≥40% · orange30-40% · rouge&lt;30%
                                      </div>
                                    </div>
                                  );
                                })()}

                                {/* WR par session */}
                                {(() => {
                                  const wbs = pretrainStats.wr_by_session || {};
                                  const sessions = Object.entries(wbs);
                                  if (sessions.length === 0) return null;
                                  return (
                                    <div style={{ marginTop: 10 }}>
                                      <div style={{ color: COLORS.sub, marginBottom: 5 }}>WR par session</div>
                                      <div style={{ display: "flex", gap: 6 }}>
                                        {sessions.map(([sess, d]) => {
                                          const wr = d.wr * 100;
                                          const col = wr >= 55 ? COLORS.green : wr >= 45 ? COLORS.amber : COLORS.red;
                                          return (
                                            <div key={sess} style={{ flex: 1, textAlign: "center", background: COLORS.border + "55", borderRadius: 4, padding: "5px 4px" }}>
                                              <div style={{ fontSize: 9, color: COLORS.sub }}>{sess}</div>
                                              <div style={{ fontSize: 14, color: col, fontWeight: 600 }}>{wr.toFixed(0)}%</div>
                                              <div style={{ fontSize: 9, color: COLORS.sub }}>{d.n} trades</div>
                                            </div>
                                          );
                                        })}
                                      </div>
                                    </div>
                                  );
                                })()}

                                {/* WR par pattern */}
                                {(() => {
                                  const wbp = pretrainStats.wr_by_pattern || {};
                                  const entries = Object.entries(wbp);
                                  if (entries.length === 0) return null;
                                  const patLabels = {
                                    bullish_engulfing: "Engulfing haussier", bearish_engulfing: "Engulfing baissier",
                                    hammer: "Hammer", shooting_star: "Shooting star",
                                    pin_bar: "Pin bar", marubozu: "Marubozu",
                                    harami: "Harami haussier", bearish_harami: "Harami baissier",
                                    three_white_soldiers: "3 soldats blancs", three_black_crows: "3 corbeaux",
                                    tweezer_bottom: "Tweezer bottom", tweezer_top: "Tweezer top",
                                    piercing_line: "Piercing line", dark_cloud_cover: "Dark cloud",
                                    ema9_pullback: "EMA9 pullback", micro_breakout: "Micro breakout",
                                    doji_reversal: "Doji reversal", evening_star: "Evening star",
                                    near_order_block: "OB confluent", near_fvg: "FVG confluent",
                                  };
                                  return (
                                    <div style={{ marginTop: 10 }}>
                                      <div style={{ color: COLORS.sub, marginBottom: 5 }}>WR par pattern</div>
                                      {entries.map(([pat, d]) => {
                                        const wr = d.wr * 100;
                                        const col = wr >= 55 ? COLORS.green : wr >= 45 ? COLORS.amber : COLORS.red;
                                        return (
                                          <div key={pat} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                                            <div style={{ width: 120, fontSize: 9, color: COLORS.sub, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                              {patLabels[pat] || pat}
                                            </div>
                                            <div style={{ flex: 1, background: COLORS.border + "44", borderRadius: 2, height: 8, overflow: "hidden" }}>
                                              <div style={{ width: `${Math.min(wr, 100)}%`, height: "100%", background: col, borderRadius: 2 }} />
                                            </div>
                                            <div style={{ width: 62, textAlign: "right", fontSize: 9, color: col, flexShrink: 0 }}>
                                              {wr.toFixed(0)}% ({d.n})
                                            </div>
                                          </div>
                                        );
                                      })}
                                      <div style={{ fontSize: 9, color: COLORS.sub, marginTop: 2 }}>vert ≥55% · orange 45-55% · rouge &lt;45%</div>
                                    </div>
                                  );
                                })()}

                                {/* Rejets pipeline */}
                                {(() => {
                                  const rc = pretrainStats.rejection_counts || {};
                                  const entries = Object.entries(rc);
                                  if (entries.length === 0) return null;
                                  const total = entries.reduce((s, [, n]) => s + n, 0);
                                  const stageLabels = {
                                    timing: "Timing (lun/ven)", session: "Session (hors Lo/NY)",
                                    h1_neutre: "H1 neutre (EMA50≈200)", h1_ema200: "H1 EMA200 mauvais côté",
                                    h1_ema200_dist: "H1 dist EMA200 trop proche", h4_bias: "H4 biais contraire",
                                    m15: "M15 EMA/RSI", atr_min: "ATR trop bas",
                                    atr_max: "ATR trop haut", adx: "ADX H1 trop faible",
                                    ema9: "EMA9 M5 désalignée", rsi_m5: "RSI M5 faible",
                                    vwap: "VWAP mauvais côté", patterns: "Patterns insuffisants",
                                    body: "Corps bougie < 40%",
                                  };
                                  return (
                                    <div style={{ marginTop: 10 }}>
                                      <div style={{ color: COLORS.sub, marginBottom: 5 }}>Rejets pipeline (total {total.toLocaleString()})</div>
                                      {entries.map(([stage, n]) => {
                                        const pct = total > 0 ? n / total * 100 : 0;
                                        const barCol = pct >= 30 ? COLORS.red : pct >= 15 ? COLORS.amber : COLORS.green;
                                        return (
                                          <div key={stage} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                                            <div style={{ width: 140, fontSize: 9, color: COLORS.sub, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                              {stageLabels[stage] || stage}
                                            </div>
                                            <div style={{ flex: 1, background: COLORS.border + "44", borderRadius: 2, height: 8, overflow: "hidden" }}>
                                              <div style={{ width: `${Math.min(pct, 100)}%`, height: "100%", background: barCol, borderRadius: 2 }} />
                                            </div>
                                            <div style={{ width: 60, textAlign: "right", fontSize: 9, color: COLORS.text, flexShrink: 0 }}>
                                              {pct.toFixed(1)}% ({n.toLocaleString()})
                                            </div>
                                          </div>
                                        );
                                      })}
                                      <div style={{ fontSize: 9, color: COLORS.sub, marginTop: 2 }}>rouge ≥ 30% · orange 15-30% · vert &lt;15%</div>
                                    </div>
                                  );
                                })()}

                                {/* Profondeur des faux stops */}
                                {(() => {
                                  const sp = pretrainStats.false_stop_spike_stats;
                                  if (!sp || !sp.n) return null;
                                  const cov = sp.coverage || {};
                                  return (
                                    <div style={{ marginTop: 10 }}>
                                      <div style={{ color: COLORS.sub, marginBottom: 4 }}>Profondeur faux stops (ATR au-delà SL)</div>
                                      <div style={{ display: "flex", gap: 4, marginBottom: 5 }}>
                                        {[["Moy", sp.avg], ["p50", sp.p50], ["p80", sp.p80], ["p90", sp.p90]].map(([lbl, val]) => (
                                          <div key={lbl} style={{ flex: 1, textAlign: "center", background: COLORS.border + "55", borderRadius: 3, padding: "3px 2px" }}>
                                            <div style={{ fontSize: 8, color: COLORS.sub }}>{lbl}</div>
                                            <div style={{ fontSize: 10, color: COLORS.amber }}>{val}</div>
                                          </div>
                                        ))}
                                      </div>
                                      <div style={{ fontSize: 9, color: COLORS.sub, marginBottom: 3 }}>+X ATR sur SL → couverture faux stops</div>
                                      {Object.entries(cov).map(([thresh, pct]) => (
                                        <div key={thresh} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                                          <div style={{ width: 32, fontSize: 9, color: COLORS.sub, flexShrink: 0 }}>+{thresh}</div>
                                          <div style={{ flex: 1, background: COLORS.border + "44", borderRadius: 2, height: 8, overflow: "hidden" }}>
                                            <div style={{ width: `${Math.min(pct, 100)}%`, height: "100%", background: pct >= 50 ? COLORS.green : COLORS.amber, borderRadius: 2 }} />
                                          </div>
                                          <div style={{ width: 36, textAlign: "right", fontSize: 9, color: COLORS.text, flexShrink: 0 }}>{pct}%</div>
                                        </div>
                                      ))}
                                      <div style={{ fontSize: 9, color: COLORS.sub, marginTop: 2 }}>N={sp.n} faux stops · SL actuel=1.4×ATR</div>
                                    </div>
                                  );
                                })()}

                                {/* Faux stops % par heure CET */}
                                {(() => {
                                  const fsbh = pretrainStats.false_stop_by_hour || {};
                                  if (Object.keys(fsbh).length === 0) return null;
                                  const hours = [8,9,10,11,12,13,14,15,16,17,18];
                                  const isLondon = h => h >= 8 && h < 12;
                                  const isNY     = h => h >= 14 && h < 18;
                                  return (
                                    <div style={{ marginTop: 10 }}>
                                      <div style={{ color: COLORS.sub, marginBottom: 4 }}>% Faux stops par heure CET</div>
                                      {hours.map(h => {
                                        const d = fsbh[String(h)];
                                        if (!d) return null;
                                        const pct = d.pct_false;
                                        const barCol = pct >= 60 ? COLORS.red : pct >= 40 ? COLORS.amber : COLORS.green;
                                        const rowBg = isLondon(h) ? "#3b82f611" : isNY(h) ? "#f9731611" : "transparent";
                                        const sessTag = isLondon(h) ? " Lo" : isNY(h) ? " NY" : "";
                                        return (
                                          <div key={h} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2, background: rowBg, borderRadius: 2, padding: "1px 3px" }}>
                                            <div style={{ width: 32, color: COLORS.text, fontSize: 9, flexShrink: 0 }}>{h}h{sessTag}</div>
                                            <div style={{ flex: 1, background: COLORS.border + "44", borderRadius: 2, height: 10, overflow: "hidden" }}>
                                              <div style={{ width: `${Math.min(pct, 100)}%`, height: "100%", background: barCol, borderRadius: 2 }} />
                                            </div>
                                            <div style={{ width: 68, textAlign: "right", fontSize: 9, color: barCol, flexShrink: 0 }}>
                                              {pct}% ({d.n_false_stops}/{d.n_sl})
                                            </div>
                                          </div>
                                        );
                                      })}
                                      <div style={{ color: COLORS.sub, fontSize: 9, marginTop: 2 }}>
                                        % faux stops / SL directs · rouge≥60% · orange40-60% · vert&lt;40%
                                      </div>
                                    </div>
                                  );
                                })()}

                                {/* Faux stops % par pattern */}
                                {(() => {
                                  const fsbp = pretrainStats.false_stop_by_pattern || {};
                                  const entries = Object.entries(fsbp);
                                  if (entries.length === 0) return null;
                                  return (
                                    <div style={{ marginTop: 10 }}>
                                      <div style={{ color: COLORS.sub, marginBottom: 4 }}>% Faux stops par pattern</div>
                                      {entries.map(([pat, d]) => {
                                        const pct = d.pct_false;
                                        const barCol = pct >= 60 ? COLORS.red : pct >= 40 ? COLORS.amber : COLORS.green;
                                        return (
                                          <div key={pat} style={{ marginBottom: 3 }}>
                                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 1 }}>
                                              <span style={{ color: COLORS.sub, fontSize: 9 }}>{pat}</span>
                                              <span style={{ color: barCol, fontSize: 9 }}>{pct}% ({d.n_false_stops}/{d.n_sl})</span>
                                            </div>
                                            <div style={{ background: COLORS.border + "44", borderRadius: 2, height: 6 }}>
                                              <div style={{ width: `${Math.min(pct, 100)}%`, height: "100%", background: barCol, borderRadius: 2 }} />
                                            </div>
                                          </div>
                                        );
                                      })}
                                      <div style={{ color: COLORS.sub, fontSize: 9, marginTop: 2 }}>
                                        % faux stops parmi SL directs par pattern
                                      </div>
                                    </div>
                                  );
                                })()}

                                {/* Patterns dominants dans les pertes */}
                                {pretrainStats.top_patterns_losses?.length > 0 && (
                                  <div style={{ marginTop: 6 }}>
                                    <div style={{ color: COLORS.sub, marginBottom: 3 }}>Patterns fréquents dans les pertes</div>
                                    {pretrainStats.top_patterns_losses.map(([p, n]) => (
                                      <div key={p} style={{ display: "flex", justifyContent: "space-between", color: COLORS.red, marginBottom: 1 }}>
                                        <span style={{ color: COLORS.sub }}>{p}</span>
                                        <span>{n}×</span>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            );
                          })()}

                          {pretrainDiag && pretrainStats?.error && (
                            <div style={{ color: COLORS.sub, fontSize: 10, marginTop: 4 }}>{pretrainStats.error}</div>
                          )}
                        </div>

                        <button
                          onClick={() => launchPretrain(activeMarket)}
                          disabled={pretrainLoading}
                          style={{ marginTop: 6, width: "100%", background: COLORS.blue + "22",
                            border: `1px solid ${COLORS.blue}`, borderRadius: 4,
                            color: COLORS.blue, padding: "4px 0", cursor: "pointer", fontSize: 11 }}>
                          Relancer (6 mois)
                        </button>
                      </div>
                    ) : (
                      <div>
                        <div style={{ color: COLORS.sub, fontSize: 11, marginBottom: 6 }}>
                          Pré-entraîner le bot sur la période sélectionnée ({pretrainStart} → {pretrainEnd})
                        </div>
                        <button
                          onClick={() => launchPretrain(activeMarket)}
                          disabled={pretrainLoading || multiStatus?.running}
                          style={{ width: "100%", background: COLORS.blue + "33",
                            border: `1px solid ${COLORS.blue}`, borderRadius: 4,
                            color: COLORS.blue, padding: "6px 0", cursor: "pointer",
                            fontSize: 12, fontWeight: 600 }}>
                          {pretrainLoading ? "Lancement…" : "Pré-entraîner maintenant"}
                        </button>

                      </div>
                    )}

                    {/* Bouton multi-périodes — toujours visible */}
                    <button
                      onClick={launchMultiPretrain}
                      disabled={multiStatus?.running || pretrainLoading || pretrainStatus?.running}
                      style={{ width: "100%", marginTop: 6, background: COLORS.amber + "22",
                        border: `1px solid ${COLORS.amber}`, borderRadius: 4,
                        color: COLORS.amber, padding: "5px 0", cursor: "pointer", fontSize: 11 }}>
                      {multiStatus?.running
                        ? `⏳ P${multiStatus.current}/${multiStatus.total} en cours…`
                        : "Test 3 périodes (0-6M / 6-12M / 12-18M)"}
                    </button>

                    {/* Walk-forward anti-overfitting */}
                    <div style={{ marginTop: 8, borderTop: `1px solid ${COLORS.border}`, paddingTop: 8 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                        <span style={{ color: COLORS.sub, fontSize: 11, flex: 1 }}>Walk-forward (fenêtres)</span>
                        <select value={wfSplits} onChange={e => setWfSplits(parseInt(e.target.value))}
                          style={{ background: COLORS.panel, color: COLORS.text, border: `1px solid ${COLORS.border}`, borderRadius: 3, padding: "1px 4px", fontSize: 11 }}>
                          {[3,4,5,6].map(n => <option key={n} value={n}>{n}</option>)}
                        </select>
                      </div>
                      <button onClick={launchWalkForward}
                        disabled={wfStatus?.running || pretrainLoading || pretrainStatus?.running}
                        style={{ width: "100%", background: "#22c55e22",
                          border: `1px solid #22c55e`, borderRadius: 4,
                          color: "#22c55e", padding: "5px 0", cursor: "pointer", fontSize: 11 }}>
                        {wfStatus?.running ? `⏳ Walk-forward en cours…` : "Lancer Walk-Forward"}
                      </button>
                    </div>

                    {/* Optimisation Bayésienne (Optuna) */}
                    <div style={{ marginTop: 8, borderTop: `1px solid ${COLORS.border}`, paddingTop: 8 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                        <span style={{ color: COLORS.sub, fontSize: 11, flex: 1 }}>Optuna (essais)</span>
                        <select value={optunaTrials} onChange={e => setOptunaTrials(parseInt(e.target.value))}
                          style={{ background: COLORS.panel, color: COLORS.text, border: `1px solid ${COLORS.border}`, borderRadius: 3, padding: "1px 4px", fontSize: 11 }}>
                          {[10,20,30,50].map(n => <option key={n} value={n}>{n}</option>)}
                        </select>
                      </div>
                      <button onClick={launchOptunaOptimize}
                        disabled={optunaStatus?.running || pretrainLoading || pretrainStatus?.running}
                        style={{ width: "100%", background: "#a855f722",
                          border: `1px solid #a855f7`, borderRadius: 4,
                          color: "#a855f7", padding: "5px 0", cursor: "pointer", fontSize: 11 }}>
                        {optunaStatus?.running
                          ? `⏳ ${optunaStatus.progress}/${optunaStatus.n_trials} essais…`
                          : "Optimiser paramètres (Optuna)"}
                      </button>
                    </div>
                  </div>

                  {/* Adaptive thresholds */}
                  {mkt.conditions.adaptive && (
                    <div style={{ marginTop: 4, borderTop: `1px solid ${COLORS.border}`, paddingTop: 4 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                        <span style={{ color: COLORS.sub }}>Seuils adaptatifs</span>
                        <span style={{ color: mkt.conditions.adaptive.ready ? COLORS.green : COLORS.grey, fontSize: 11 }}>
                          {mkt.conditions.adaptive.ready
                            ? `${mkt.conditions.adaptive.n_wins}W / ${mkt.conditions.adaptive.n_losses}L (${mkt.conditions.adaptive.win_rate != null ? (mkt.conditions.adaptive.win_rate * 100).toFixed(0) + "%" : "—"})`
                            : `apprentissage… ${mkt.conditions.adaptive.n_total}/${mkt.conditions.adaptive.n_min}`}
                        </span>
                      </div>
                      {mkt.conditions.adaptive.ready && (
                        <>
                          {[
                            ["ATR min", mkt.conditions.adaptive.atr_min?.toFixed(3), mkt.conditions.adaptive.atr_min_default?.toFixed(3)],
                            ["EMA9 tol ×", mkt.conditions.adaptive.ema9_mult?.toFixed(2), "0.50"],
                            ["M15 tol ×",  mkt.conditions.adaptive.m15_mult?.toFixed(2),  "0.30"],
                          ].map(([label, val, def]) => (
                            <div key={label} style={{ display: "flex", justifyContent: "space-between", paddingLeft: 8, opacity: 0.85 }}>
                              <span style={{ color: COLORS.sub, fontSize: 10 }}>{label}</span>
                              <span style={{ fontSize: 10, color: COLORS.text }}>
                                {val}
                                <span style={{ color: COLORS.sub, marginLeft: 4 }}>(défaut {def})</span>
                              </span>
                            </div>
                          ))}
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* ---- test signal buttons (paper mode only) ---- */}
              {state?.mode === "paper" && !mkt.position && (
                <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 2, marginBottom: 10 }}>
                  <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 6 }}>
                    Test pipeline (paper uniquement)
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    <button onClick={() => sendTestSignal("long")} disabled={testSignalLoading}
                      style={{ flex: 1, background: COLORS.green + "22", border: `1px solid ${COLORS.green}`, borderRadius: 4,
                        color: COLORS.green, padding: "5px 0", cursor: "pointer", fontSize: 11, fontWeight: 600 }}>
                      ▲ Test LONG
                    </button>
                    <button onClick={() => sendTestSignal("short")} disabled={testSignalLoading}
                      style={{ flex: 1, background: COLORS.red + "22", border: `1px solid ${COLORS.red}`, borderRadius: 4,
                        color: COLORS.red, padding: "5px 0", cursor: "pointer", fontSize: 11, fontWeight: 600 }}>
                      ▼ Test SHORT
                    </button>
                  </div>
                </div>
              )}
              {/* macro indicators: 4-in-a-row on desktop, 2x2 on mobile */}
              <div className="macro-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, marginBottom: 4 }}>
                {state?.macro?.dxy && (
                  <div style={{ display: "flex", flexDirection: "column", padding: "4px 0" }}>
                    <span style={{ color: COLORS.sub, fontSize: 12 }}>DXY</span>
                    <span style={{ fontSize: 12 }}>
                      {state.macro.dxy?.toFixed(2)}
                      <span style={{ marginLeft: 4, color: state.macro.dxy_trend === "up" ? COLORS.red : state.macro.dxy_trend === "down" ? COLORS.green : COLORS.sub }}>
                        {state.macro.dxy_trend === "up" ? "▲" : state.macro.dxy_trend === "down" ? "▼" : "—"}
                      </span>
                    </span>
                  </div>
                )}
                {state?.macro?.vix !== undefined && state?.macro?.vix !== null && (
                  <div style={{ display: "flex", flexDirection: "column", padding: "4px 0" }}>
                    <span style={{ color: COLORS.sub, fontSize: 12 }}>VIX</span>
                    <span style={{ fontSize: 12, color: state.macro.vix > 25 ? COLORS.red : state.macro.vix > 15 ? COLORS.amber : COLORS.green }}>
                      {state.macro.vix?.toFixed(1)} {state.macro.vix_blocked ? "🛑" : ""}
                    </span>
                  </div>
                )}
                {state?.macro?.tnx && (
                  <div style={{ display: "flex", flexDirection: "column", padding: "4px 0" }}>
                    <span style={{ fontSize: 12, color: COLORS.sub }}>TNX 10Y</span>
                    <span style={{ fontSize: 12, color: state.macro.tnx_trend === "up" ? COLORS.red : state.macro.tnx_trend === "down" ? COLORS.green : COLORS.text }}>
                      {state.macro.tnx?.toFixed(2)}%
                      {state.macro.tnx_trend === "up" ? " ▲" : state.macro.tnx_trend === "down" ? " ▼" : ""}
                    </span>
                  </div>
                )}
                {state?.macro?.fear_greed !== undefined && state?.macro?.fear_greed !== null && (
                  <div style={{ display: "flex", flexDirection: "column", padding: "4px 0" }}>
                    <span style={{ fontSize: 12, color: COLORS.sub }}>Peur/Avidité</span>
                    <span style={{ fontSize: 12, color: state.macro.fear_greed < 25 ? COLORS.red : state.macro.fear_greed > 75 ? COLORS.green : COLORS.amber }}>
                      {state.macro.fear_greed}/100 {state.macro.fear_greed < 25 ? "😱" : state.macro.fear_greed > 75 ? "🤑" : "😐"}
                    </span>
                  </div>
                )}
              </div>

              {/* correlations */}
              {Object.keys(correlations).length > 0 && (
                <CorrelationsPanel data={correlations} />
              )}

              {/* Fed & Banques Centrales */}
              {fedData && (
                <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 6 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <span style={{ fontSize: 11, fontWeight: 600, color: COLORS.sub }}>Fed & Banques Centrales</span>
                    <span style={{
                      fontSize: 10, padding: "1px 6px", borderRadius: 3,
                      background: fedData.bias === "bullish" ? COLORS.green + "33" : fedData.bias === "bearish" ? COLORS.red + "33" : COLORS.border,
                      color: fedData.bias === "bullish" ? COLORS.green : fedData.bias === "bearish" ? COLORS.red : COLORS.sub,
                      fontWeight: 600,
                    }}>
                      {fedData.bias === "bullish" ? "Haussier or" : fedData.bias === "bearish" ? "Baissier or" : "Neutre"}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 8px" }}>
                    <span style={{ color: COLORS.sub }}>Taux Fed</span>
                    <span style={{ textAlign: "right" }}>
                      {fedData.fed?.fed_rate != null ? `${fedData.fed.fed_rate.toFixed(2)}%` : "—"}
                      <span style={{
                        marginLeft: 4,
                        color: fedData.fed?.fed_direction === "cutting" ? COLORS.green : fedData.fed?.fed_direction === "hiking" ? COLORS.red : COLORS.sub
                      }}>
                        {fedData.fed?.fed_direction === "cutting" ? "▼ Baisse" : fedData.fed?.fed_direction === "hiking" ? "▲ Hausse" : "— Stable"}
                      </span>
                    </span>
                    <span style={{ color: COLORS.sub }}>Taux réels</span>
                    <span style={{
                      textAlign: "right",
                      color: fedData.fed?.real_rate != null
                        ? (fedData.fed.real_rate < 0 ? COLORS.green : fedData.fed.real_rate > 1.5 ? COLORS.red : COLORS.text)
                        : COLORS.sub
                    }}>
                      {fedData.fed?.real_rate != null ? `${fedData.fed.real_rate.toFixed(2)}%` : "—"}
                    </span>
                    <span style={{ color: COLORS.sub }}>TNX 10Y</span>
                    <span style={{ textAlign: "right" }}>
                      {fedData.fed?.dgs10 != null ? `${fedData.fed.dgs10.toFixed(2)}%` : "—"}
                    </span>
                    <span style={{ color: COLORS.sub }}>Banques centrales</span>
                    <span style={{
                      textAlign: "right",
                      color: fedData.central_banks?.trend === "buying" ? COLORS.green : fedData.central_banks?.trend === "selling" ? COLORS.red : COLORS.sub
                    }}>
                      {fedData.central_banks?.trend === "buying" ? "Acheteuses" : fedData.central_banks?.trend === "selling" ? "Vendeuses" : "Neutre"}
                    </span>
                  </div>
                  {fedData.signals?.length > 0 && (
                    <div style={{ marginTop: 6 }}>
                      {fedData.signals.map((s, i) => (
                        <div key={i} style={{ fontSize: 10, color: COLORS.sub, marginBottom: 2 }}>· {s}</div>
                      ))}
                    </div>
                  )}
                  {fedData.fed?.source === "no_key" && (
                    <div style={{ fontSize: 10, color: COLORS.amber, marginTop: 4 }}>
                      ⚠ FRED_API_KEY manquante — à ajouter dans Railway Variables
                    </div>
                  )}
                  {fedData.fed?.source === "error" && (
                    <div style={{ fontSize: 10, color: COLORS.sub, marginTop: 4 }}>
                      ○ FRED indisponible (clé présente, erreur réseau temporaire)
                    </div>
                  )}
                </div>
              )}

              {/* news */}
              <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 6 }}>
                <div style={{ fontSize: 12, color: COLORS.sub, marginBottom: 4 }}>Prochaine news majeure</div>
                {state?.news?.next_event ? (
                  <div>
                    <div style={{ fontSize: 13 }}>{state.news.next_event.title}</div>
                    <div style={{ fontSize: 12, color: COLORS.amber }}>
                      ⏱ dans {hms(newsCountdown)} {state.news.blocked ? "· 🛑 BLOQUÉ" : ""}
                    </div>
                  </div>
                ) : newsFeed?.upcoming_events?.length > 0 ? (
                  <div>
                    <div style={{ fontSize: 13 }}>{newsFeed.upcoming_events[0].event}</div>
                    <div style={{ fontSize: 11, color: COLORS.sub }}>
                      {newsFeed.upcoming_events[0].time} UTC · {newsFeed.upcoming_events[0].currency}
                      {newsFeed.upcoming_events[0].impact === "high" && (
                        <span style={{ color: COLORS.red, marginLeft: 4 }}>● fort impact</span>
                      )}
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: 13, color: COLORS.sub }}>Aucune news imminente</div>
                )}
              </div>

              {/* pattern weights */}
              {Object.keys(patternStats).length > 0 && (
                <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <button
                      onClick={() => setWeightsOpen((o) => !o)}
                      style={{ background: "none", border: "none", padding: 0, cursor: "pointer",
                        fontSize: 12, color: COLORS.sub, display: "flex", gap: 6, alignItems: "center" }}>
                      <span>Poids des patterns</span>
                      <span>{weightsOpen ? "▲" : "▼"}</span>
                    </button>
                  </div>
                  {weightsOpen && Object.entries(patternStats)
                    .filter(([, s]) => s.trades >= 1)
                    .sort((a, b) => b[1].weight - a[1].weight)
                    .map(([name, s]) => (
                      <div key={name} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
                        <span style={{ color: COLORS.text }}>{name.replace(/_/g, " ")}</span>
                        <span style={{ color: COLORS.sub }}>{s.trades}t {s.win_rate}%</span>
                        <span style={{ color: s.weight > 1.2 ? COLORS.green : s.weight < 0.8 ? COLORS.red : COLORS.sub, fontWeight: "bold" }}>
                          x{fmt(s.weight, 2)}
                        </span>
                      </div>
                    ))
                  }
                </div>
              )}

              {/* Agent IA */}
              {agentStatus && (
                <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: COLORS.sub }}>Agent IA</div>
                  <Row k="Statut" v={agentStatus.running ? "🟢 Actif" : "⚫ Arrêté"} />
                  <Row k="Dernier run" v={agentStatus.last_run || "—"} />
                  <Row k="Prochain run" v={agentStatus.next_run || "—"} />
                  <Row k="Sharpe actuel" v={fmt(agentStatus.current_sharpe, 3)} />
                  {agentStatus.last_improvement && (
                    <Row k="Dernière amélio." v={`+${fmt(agentStatus.last_improvement, 1)}%`}
                         color={COLORS.green} />
                  )}
                  {agentStatus.params_applied && (
                    <div style={{ fontSize: 11, color: COLORS.green, marginTop: 4 }}>
                      ✓ Params mis à jour automatiquement
                    </div>
                  )}
                  {agentHistory.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 4, fontWeight: 600 }}>
                        Historique des runs ({agentHistory.length})
                      </div>
                      <div style={{ maxHeight: 140, overflowY: "auto", fontSize: 10 }}>
                        {agentHistory.slice(0, 20).map((r, i) => (
                          <div key={i} style={{
                            display: "flex", justifyContent: "space-between",
                            padding: "3px 0", borderBottom: `1px solid ${COLORS.border}`,
                            color: r.applied ? COLORS.green : COLORS.sub
                          }}>
                            <span style={{ color: COLORS.sub }}>
                              {new Date(r.timestamp).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}
                            </span>
                            <span>{r.symbol}</span>
                            <span style={{ color: r.improvement_pct >= 10 ? COLORS.green : COLORS.amber }}>
                              {r.improvement_pct >= 0 ? "+" : ""}{r.improvement_pct?.toFixed(1)}%
                            </span>
                            <span style={{ color: r.applied ? COLORS.green : COLORS.red }}>
                              {r.applied ? "✓" : "—"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
                <button onClick={toggleBot} className="btn-action" style={{ ...tabBtn(false), flex: 1 }}>
                  Pause / Reprise
                </button>
                <button onClick={switchLive}
                  style={{ ...tabBtn(false), flex: 1,
                    borderColor: state?.mode === "live" ? COLORS.red : COLORS.border,
                    color: state?.mode === "live" ? COLORS.red : COLORS.text }}>
                  {state?.mode === "live" ? "→ Paper" : "→ Live"}
                </button>
                <button onClick={toggle247}
                  style={{ ...tabBtn(false), flex: 1,
                    background: !sessionFilterOn ? COLORS.green : "transparent",
                    borderColor: !sessionFilterOn ? COLORS.green : COLORS.border,
                    color: !sessionFilterOn ? "#fff" : COLORS.text }}>
                  {!sessionFilterOn ? "24/7 ON" : "Sessions"}
                </button>
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
                <div style={{ flex: 1, padding: "5px 10px", borderRadius: 4, fontSize: 12,
                  background: strategyMode === "B" ? COLORS.blue + "22" : COLORS.amber + "22",
                  border: `1px solid ${strategyMode === "B" ? COLORS.blue : COLORS.amber}55`,
                  color: strategyMode === "B" ? COLORS.blue : COLORS.amber, textAlign: "center" }}>
                  {activeMarket === "EURUSD" ? "EUR/USD → ICT / Order Blocks (B)" : "XAU/USD → EMA / Patterns (A)"}
                </div>
              </div>

              {/* RL Agent panel */}
              <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 10 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: COLORS.sub, marginBottom: 6 }}>
                  Agent RL (Deep Learning)
                </div>
                {rlStatus ? (
                  <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                      <span>Statut</span>
                      <span style={{ color: rlStatus.training ? COLORS.amber : rlStatus.paper_running ? COLORS.blue : COLORS.sub }}>
                        {rlStatus.training ? "⏳ Entraînement…" : rlStatus.paper_running ? "📊 Paper trading" : rlStatus.status}
                      </span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                      <span>Modèle</span>
                      <span style={{ color: rlStatus.model_exists ? COLORS.green : COLORS.red }}>
                        {rlStatus.model_exists ? "✓ Disponible" : "✗ Non entraîné"}
                      </span>
                    </div>
                    {rlStatus.paper_running && (
                      <>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                          <span>Capital paper</span>
                          <span>${fmt(rlStatus.paper_capital, 2)}</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                          <span>P&L paper</span>
                          <span style={{ color: (rlStatus.paper_pnl||0) >= 0 ? COLORS.green : COLORS.red }}>
                            {money(rlStatus.paper_pnl)}
                          </span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                          <span>Trades</span>
                          <span>{rlStatus.paper_trades}</span>
                        </div>
                      </>
                    )}
                    {rlStatus.promoted && (
                      <div style={{ color: COLORS.green, fontSize: 11, marginTop: 4 }}>
                        🚀 Promu — prêt pour le live
                      </div>
                    )}
                    {rlStatus.val_metrics && (
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                        <span>Sharpe validation</span>
                        <span style={{ color: rlStatus.val_metrics.sharpe_ratio >= 0.5 ? COLORS.green : COLORS.amber }}>
                          {fmt(rlStatus.val_metrics.sharpe_ratio, 2)}
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 8 }}>Non initialisé</div>
                )}
                <button
                  onClick={trainRL}
                  disabled={rlLoading || rlStatus?.training}
                  style={{ ...tabBtn(false), width: "100%",
                    background: rlStatus?.training ? COLORS.amber + "33" : "transparent",
                    borderColor: rlStatus?.promoted ? COLORS.green : COLORS.blue,
                    color: rlStatus?.promoted ? COLORS.green : COLORS.blue,
                    opacity: (rlLoading || rlStatus?.training) ? 0.6 : 1 }}>
                  {rlStatus?.training ? "⏳ Entraînement en cours…" : rlStatus?.model_exists ? "🔄 Ré-entraîner l'IA" : "🧠 Entraîner l'IA"}
                </button>
                {rlHistory.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 4, fontWeight: 600 }}>
                      Historique entraînements ({rlHistory.length})
                    </div>
                    <div style={{ maxHeight: 120, overflowY: "auto", fontSize: 10 }}>
                      {rlHistory.slice(0, 10).map((r, i) => (
                        <div key={i} style={{
                          display: "flex", justifyContent: "space-between",
                          padding: "3px 0", borderBottom: `1px solid ${COLORS.border}`,
                        }}>
                          <span style={{ color: COLORS.sub }}>
                            {new Date(r.timestamp).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}
                          </span>
                          <span style={{ color: r.passed_gate ? COLORS.green : COLORS.amber }}>
                            Sharpe {r.val_sharpe?.toFixed(2) ?? "—"}
                          </span>
                          <span style={{ color: r.passed_gate ? COLORS.green : COLORS.red }}>
                            {r.passed_gate ? "✓ OK" : "✗ Rejeté"}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ===== Robustesse multi-périodes ===== */}
          {(multiStatus?.running || multiStatus?.results?.length > 0) && (
            <div className="section-gap" style={{ marginTop: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>Robustesse multi-périodes</h3>
                {multiStatus?.running && (
                  <span style={{ fontSize: 11, color: COLORS.amber }}>
                    ⏳ Période {multiStatus.current}/{multiStatus.total} en cours…
                  </span>
                )}
              </div>
              <div className="multi-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
                {(multiStatus?.results?.length > 0 ? multiStatus.results : [{},{},{}]).map((r, i) => {
                  const isLoading = multiStatus?.running && i >= (multiStatus?.results?.length ?? 0);
                  const isEmpty   = !r.label;
                  const pf   = r.profit_factor ?? 0;
                  const wr   = r.win_rate ?? 0;
                  const pfColor = pf >= 1.25 ? COLORS.green : pf >= 1.0 ? COLORS.amber : COLORS.red;
                  const wrColor = wr >= 0.52 ? COLORS.green : wr >= 0.45 ? COLORS.amber : COLORS.red;
                  const curve = r.equity_curve ?? [];
                  const W = 260, H = 60;
                  const vals = curve.map(p => p.equity);
                  const minV = Math.min(...vals);
                  const maxV = Math.max(...vals);
                  const range = maxV - minV || 1;
                  const pts = vals.map((v, j) =>
                    `${(j / Math.max(vals.length - 1, 1)) * W},${H - ((v - minV) / range) * H}`
                  ).join(" ");
                  const finalColor = vals.length > 1
                    ? (vals[vals.length - 1] >= vals[0] ? COLORS.green : COLORS.red)
                    : COLORS.sub;
                  return (
                    <div key={i} className="dashboard-panel" style={{
                      ...panel(),
                      opacity: isLoading || isEmpty ? 0.4 : 1,
                      minHeight: 160,
                    }}>
                      {isEmpty || isLoading ? (
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 140, color: COLORS.sub, fontSize: 12 }}>
                          {isLoading ? "⏳ En cours…" : `Période ${i + 1}`}
                        </div>
                      ) : (
                        <>
                          <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 8 }}>{r.label}</div>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, marginBottom: 10 }}>
                            <div style={{ textAlign: "center" }}>
                              <div style={{ fontSize: 20, fontWeight: 700, color: pfColor }}>{pf.toFixed(2)}</div>
                              <div style={{ fontSize: 10, color: COLORS.sub }}>PF</div>
                            </div>
                            <div style={{ textAlign: "center" }}>
                              <div style={{ fontSize: 20, fontWeight: 700, color: wrColor }}>{(wr * 100).toFixed(0)}%</div>
                              <div style={{ fontSize: 10, color: COLORS.sub }}>WR</div>
                            </div>
                            <div style={{ textAlign: "center" }}>
                              <div style={{ fontSize: 20, fontWeight: 700, color: COLORS.text }}>{r.n_trades}</div>
                              <div style={{ fontSize: 10, color: COLORS.sub }}>trades</div>
                            </div>
                          </div>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, marginBottom: 10, fontSize: 11 }}>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                              <span style={{ color: COLORS.sub }}>Net PnL</span>
                              <span style={{ color: r.net_pnl >= 0 ? COLORS.green : COLORS.red, fontWeight: 600 }}>
                                {r.net_pnl >= 0 ? "+" : ""}{r.net_pnl?.toFixed(0)}$
                              </span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                              <span style={{ color: COLORS.sub }}>SL direct</span>
                              <span style={{ color: r.sl_direct_pct <= 30 ? COLORS.green : COLORS.amber }}>
                                {r.sl_direct_pct}%
                              </span>
                            </div>
                          </div>
                          {curve.length > 1 && (
                            <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", marginBottom: 8 }}>
                              <polyline points={pts} fill="none" stroke={finalColor} strokeWidth="1.5" />
                            </svg>
                          )}

                          {/* Mini-diagnostic indicateurs */}
                          {r.diag_sl != null && r.diag_tp2 != null && (
                            <DiagSection sl={r.diag_sl} tp={r.diag_tp2} wrByHour={r.wr_by_hour} />
                          )}
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ===== Walk-forward résultats ===== */}
          {(wfStatus?.running || wfStatus?.result) && (
            <div className="section-gap" style={{ marginTop: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>Walk-Forward ({wfStatus?.n_splits ?? wfSplits} fenêtres)</h3>
                {wfStatus?.running && <span style={{ fontSize: 11, color: COLORS.amber }}>⏳ En cours…</span>}
                {wfStatus?.result && (() => {
                  const r = wfStatus.result;
                  const robustColor = r.is_robust ? COLORS.green : r.avg_pf > 1.0 ? COLORS.amber : COLORS.red;
                  return (
                    <span style={{ fontSize: 11, color: robustColor, fontWeight: 600 }}>
                      {r.is_robust ? "✓ Robuste" : "⚠ Fragile"} · PF moy {r.avg_pf?.toFixed(2)} ± {r.std_pf?.toFixed(2)}
                      · {r.pct_profitable?.toFixed(0)}% fenêtres rentables
                    </span>
                  );
                })()}
              </div>
              {wfStatus?.result && (
                <div style={{ display: "grid", gridTemplateColumns: `repeat(${wfStatus.result.windows?.length ?? wfSplits}, 1fr)`, gap: 10 }}>
                  {(wfStatus.result.windows ?? []).map((w, i) => {
                    const pf = w.profit_factor ?? 0;
                    const wr = w.win_rate ?? 0;
                    const pfColor = pf >= 1.25 ? COLORS.green : pf >= 1.0 ? COLORS.amber : COLORS.red;
                    return (
                      <div key={i} style={{ ...panel(), padding: 10 }}>
                        {w.error ? (
                          <div style={{ fontSize: 10, color: COLORS.red }}>{w.error}</div>
                        ) : (
                          <>
                            <div style={{ fontSize: 10, color: COLORS.sub, marginBottom: 6 }}>F{w.window} · {w.period?.slice(0, 10)} →</div>
                            <div style={{ fontSize: 20, fontWeight: 700, color: pfColor, textAlign: "center" }}>{pf.toFixed(2)}</div>
                            <div style={{ fontSize: 10, color: COLORS.sub, textAlign: "center", marginBottom: 4 }}>PF</div>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10 }}>
                              <span style={{ color: COLORS.sub }}>WR</span>
                              <span style={{ color: wr >= 0.5 ? COLORS.green : COLORS.amber }}>{(wr * 100).toFixed(0)}%</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10 }}>
                              <span style={{ color: COLORS.sub }}>Trades</span>
                              <span>{w.n_trades}</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10 }}>
                              <span style={{ color: COLORS.sub }}>SL dir.</span>
                              <span style={{ color: (w.sl_direct_pct ?? 0) <= 30 ? COLORS.green : COLORS.amber }}>{w.sl_direct_pct}%</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10 }}>
                              <span style={{ color: COLORS.sub }}>Net</span>
                              <span style={{ color: w.net_pnl >= 0 ? COLORS.green : COLORS.red }}>{w.net_pnl >= 0 ? "+" : ""}{w.net_pnl?.toFixed(0)}$</span>
                            </div>
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* ===== Optuna résultats ===== */}
          {(optunaStatus?.running || optunaStatus?.result) && (
            <div className="section-gap" style={{ marginTop: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>Optimisation Bayésienne (Optuna)</h3>
                {optunaStatus?.running && (
                  <span style={{ fontSize: 11, color: "#a855f7" }}>
                    ⏳ {optunaStatus.progress}/{optunaStatus.n_trials} essais · meilleur score {optunaStatus.best_score?.toFixed(3)}
                  </span>
                )}
              </div>
              {optunaStatus?.result && (() => {
                const r = optunaStatus.result;
                if (r.error) return <div style={{ color: COLORS.red, fontSize: 12 }}>{r.error}</div>;
                return (
                  <div style={{ ...panel(), padding: 12 }}>
                    <div style={{ marginBottom: 10 }}>
                      <span style={{ fontSize: 12, color: COLORS.sub }}>Meilleurs paramètres ({r.n_completed}/{r.n_trials} essais · score {r.best_score?.toFixed(3)})</span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                      {Object.entries(r.best_params ?? {}).map(([k, v]) => (
                        <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, borderBottom: `1px solid ${COLORS.border}`, paddingBottom: 3 }}>
                          <span style={{ color: COLORS.sub }}>{k}</span>
                          <span style={{ fontWeight: 600, color: "#a855f7" }}>{typeof v === "number" ? v.toFixed(1) : v}</span>
                        </div>
                      ))}
                    </div>
                    {r.top_trials?.length > 0 && (
                      <div style={{ marginTop: 10 }}>
                        <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 4 }}>Top essais</div>
                        {r.top_trials.slice(0, 5).map((t, i) => (
                          <div key={i} style={{ fontSize: 10, color: COLORS.sub, display: "flex", gap: 6, marginBottom: 2 }}>
                            <span style={{ color: "#a855f7", fontWeight: 600 }}>#{t.trial}</span>
                            <span style={{ color: COLORS.text }}>score {t.score?.toFixed(3)}</span>
                            {Object.entries(t.params ?? {}).map(([k, v]) => (
                              <span key={k}>{k.replace("RSI_M5_", "").replace("_", "").toLowerCase()}={typeof v === "number" ? v.toFixed(1) : v}</span>
                            ))}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          )}

          {/* ===== active trade ===== */}
          {pos && (() => {
            const cs = activeMarket === "EURUSD" ? 100000 : 100;
            const sign = pos.direction === "long" ? 1 : -1;
            // TP1 ferme 50% de la position, TP2 ferme les 50% restants → gains cumulatifs
            const gainTp1 = sign * (pos.take_profit1 - pos.entry) * (pos.volume * 0.5) * cs;
            const gainTp2 = gainTp1 + sign * (pos.take_profit2 - pos.entry) * (pos.volume * 0.5) * cs;
            const dp = activeMarket === "EURUSD" ? 5 : 2;

            // P&L et progress bars en temps réel via le tick live (mkt.price)
            const liveP = mkt.price || pos.price || pos.entry;
            const livePnl = pos.unrealised_pnl + (liveP - (pos.price || pos.entry)) * sign * cs * (pos.remaining || pos.volume);
            const d1 = Math.abs(pos.take_profit1 - pos.entry) || 1e-9;
            const d2 = Math.abs(pos.take_profit2 - pos.entry) || 1e-9;
            const liveProg1 = pos.tp1_done ? 1.0 : Math.max(-1, Math.min(1.5, sign * (liveP - pos.entry) / d1));
            const liveProg2 = Math.max(-1, Math.min(1.5, sign * (liveP - pos.entry) / d2));

            return (
              <div className="dashboard-panel section-gap" style={{ ...panel(), marginTop: 14, borderColor: pos.direction === "long" ? COLORS.green : COLORS.red }}>
                <div className="active-trade-row" style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
                  <span style={{ fontWeight: 700, fontSize: 16,
                    color: pos.direction === "long" ? COLORS.green : COLORS.red }}>
                    {pos.direction === "long" ? "▲ LONG" : "▼ SHORT"} · {pos.volume} lots
                  </span>
                  <Row k="Entrée" v={fmt(pos.entry, dp)} inline />
                  <Row k="SL" v={fmt(pos.stop_loss, dp)} inline />
                  <Row k="TP1" v={fmt(pos.take_profit1, dp)} inline />
                  <Row k="TP2" v={fmt(pos.take_profit2, dp)} inline />
                  {pos.risk_amount > 0 && (
                    <span style={{ fontSize: 12, color: COLORS.sub }}>
                      Mise: <span style={{ color: COLORS.amber }}>{money(pos.risk_amount)}</span>
                    </span>
                  )}
                  <span style={{ fontSize: 12, color: COLORS.sub }}>
                    Gain pot:{" "}
                    <span style={{ color: COLORS.green }}>
                      {money(gainTp1)} (TP1) · {money(gainTp2)} (TP2)
                    </span>
                  </span>
                  <span style={{ fontWeight: 700,
                    color: livePnl >= 0 ? COLORS.green : COLORS.red }}>
                    {money(livePnl)}
                  </span>
                  <span style={{ marginLeft: "auto", fontSize: 13, color: remaining < 300 ? COLORS.amber : COLORS.sub }}>
                    ⏱ {hms(remaining)} / 45:00
                  </span>
                  <button onClick={closeNow} className="btn-action"
                    style={{ ...tabBtn(false), borderColor: COLORS.red, color: COLORS.red, fontWeight: 700 }}>
                    FERMER MAINTENANT
                  </button>
                </div>
                <div style={{ marginTop: 12 }}>
                  <ProgressBar label="TP1 (50%)" value={liveProg1} done={pos.tp1_done} />
                  <ProgressBar label="TP2 (50%)" value={liveProg2} />
                </div>
              </div>
            );
          })()}

          {/* ===== history + equity ===== */}
          <div className="history-layout section-gap" style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 14, marginTop: 14 }}>
            <div className="dashboard-panel" style={panel()}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>
                  {tradesScope === "today" ? "Historique du jour" : "Tout l'historique"}
                </h3>
                <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                  <button onClick={() => setTradesScope("today")} style={{ ...tabBtn(tradesScope === "today"), fontSize: 11, padding: "3px 8px" }}>Aujourd'hui</button>
                  <button onClick={() => setTradesScope("all")} style={{ ...tabBtn(tradesScope === "all"), fontSize: 11, padding: "3px 8px" }}>Tout</button>
                  <button onClick={handleCleanupDuplicates} disabled={cleanupLoading}
                    title="Supprimer les trades en double (même entrée, même minute)"
                    style={{ fontSize: 10, padding: "3px 7px", background: "transparent", border: `1px solid ${COLORS.red}`, color: COLORS.red, borderRadius: 4, cursor: "pointer", opacity: cleanupLoading ? 0.5 : 1 }}>
                    {cleanupLoading ? "…" : "🗑 Doublons"}
                  </button>
                  <button onClick={handleResetHistory}
                    title="Supprimer TOUT l'historique et remettre les stats à zéro"
                    style={{ fontSize: 10, padding: "3px 7px", background: "transparent", border: `1px solid #ff4444`, color: "#ff4444", borderRadius: 4, cursor: "pointer", fontWeight: 700 }}>
                    ⚠ Reset
                  </button>
                  {cleanupResult && (
                    <span style={{ fontSize: 10, color: cleanupResult.deleted > 0 ? COLORS.green : COLORS.sub }}>
                      {cleanupResult.deleted > 0 ? `${cleanupResult.deleted} supprimé(s)` : "Aucun doublon"}
                    </span>
                  )}
                </div>
              </div>
              <div className="table-scroll" style={{ maxHeight: 240, overflowY: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, minWidth: 520 }}>
                  <thead>
                    <tr style={{ color: COLORS.sub, textAlign: "left" }}>
                      <th style={th}>{tradesScope === "all" ? "Date / Heure" : "Heure"}</th>
                      <th style={th}>Actif</th>
                      <th style={th}>Dir</th>
                      <th style={th}>Entrée</th>
                      <th style={th}>Sortie</th>
                      <th style={th}>Mise</th>
                      <th style={th}>Gain pot.</th>
                      <th style={th}>Résultat</th>
                      <th style={{ ...th, width: 20 }}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {(trades.trades || []).filter((t) => t.status === "closed").reverse().map((t) => {
                      const cs = t.symbol === "EURUSD" ? 100000 : 100;
                      const sign = t.direction === "long" ? 1 : -1;
                      const gTp1 = t.take_profit1 && t.entry_price && t.volume
                        ? sign * (t.take_profit1 - t.entry_price) * t.volume * cs : null;
                      const gTp2 = t.take_profit2 && t.entry_price && t.volume
                        ? sign * (t.take_profit2 - t.entry_price) * t.volume * cs : null;
                      const dp = t.symbol === "EURUSD" ? 5 : 2;
                      return (
                        <tr key={t.id} style={{ borderTop: `1px solid ${COLORS.border}` }}>
                          <td style={td}>
                            {tradesScope === "all" ? (
                              <span>
                                <span style={{ color: COLORS.sub, fontSize: 10 }}>{t.date_cet || new Date(t.entry_time).toLocaleDateString("fr-FR")} </span>
                                {fmtLocalTime(t.entry_time)}
                              </span>
                            ) : fmtLocalTime(t.entry_time)}
                          </td>
                          <td style={{ ...td, color: COLORS.sub, fontSize: 11 }}>
                            {t.symbol || "XAUUSD"}
                          </td>
                          <td style={{ ...td, color: t.direction === "long" ? COLORS.green : COLORS.red }}>
                            {t.direction === "long" ? "LONG" : "SHORT"}
                          </td>
                          <td style={td}>{fmt(t.entry_price, dp)}</td>
                          <td style={td}>{fmt(t.exit_price, dp)}</td>
                          <td style={{ ...td, color: COLORS.amber }}>
                            {t.risk_amount ? money(t.risk_amount) : "—"}
                          </td>
                          <td style={{ ...td, fontSize: 11 }}>
                            {gTp1 != null ? (
                              <span title={`TP1: ${money(gTp1)} · TP2: ${money(gTp2)}`}>
                                <span style={{ color: COLORS.green }}>{money(gTp1)}</span>
                                {gTp2 != null && (
                                  <span style={{ color: COLORS.sub }}> / {money(gTp2)}</span>
                                )}
                              </span>
                            ) : "—"}
                          </td>
                          <td style={{ ...td, fontWeight: 600, color: (t.pnl || 0) >= 0 ? COLORS.green : COLORS.red }}>
                            {money(t.pnl)}
                          </td>
                          <td style={td}>
                            <button onClick={() => handleDeleteTrade(t.id)}
                              disabled={deletingTrade === t.id}
                              title="Supprimer ce trade"
                              style={{ background: "transparent", border: "none", color: COLORS.red, cursor: "pointer", fontSize: 13, padding: "0 2px", opacity: deletingTrade === t.id ? 0.4 : 0.6, lineHeight: 1 }}>
                              ✕
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                    {(trades.trades || []).filter((t) => t.status === "closed").length === 0 && (
                      <tr><td style={{ ...td, color: COLORS.sub }} colSpan={8}>Aucun trade clôturé aujourd'hui</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="dashboard-panel" style={panel()}>
              <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>Équité intraday</h3>
              <div className="chart-container" style={{ height: 210 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={(trades.equity_curve || []).map((p, i) => ({
                  i, equity: p.equity, t: fmtLocalTime(p.ts),
                }))}>
                  <defs>
                    <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={COLORS.green} stopOpacity={0.5} />
                      <stop offset="100%" stopColor={COLORS.green} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
                  <XAxis dataKey="t" stroke={COLORS.sub} fontSize={11} />
                  <YAxis stroke={COLORS.sub} fontSize={11} domain={["auto", "auto"]} />
                  <Tooltip contentStyle={{ background: COLORS.panel, border: `1px solid ${COLORS.border}` }} />
                  <Area type="monotone" dataKey="equity" stroke={COLORS.green} fill="url(#eq)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
          </div>

          {/* ===== rapport historique ===== */}
          <div className="dashboard-panel section-gap" style={{ ...panel(), marginTop: 14 }}>
              <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>Rapport historique</h3>
              {reportError && (
                <div style={{ fontSize: 12, color: COLORS.red, marginBottom: 8 }}>
                  Erreur : {reportError}
                </div>
              )}
              {!tradeReport && !reportError && (
                <div style={{ fontSize: 12, color: COLORS.sub }}>Chargement...</div>
              )}
              {tradeReport && (!tradeReport.stats || tradeReport.stats.total === 0) && (
                <div style={{ fontSize: 12, color: COLORS.sub }}>Aucun trade clôturé en base.</div>
              )}
          {tradeReport && tradeReport.stats && tradeReport.stats.total > 0 && (
            <div>

              {/* Stats globales */}
              <div className="report-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 12 }}>
                {[
                  { label: "Trades", value: tradeReport.stats.total },
                  { label: "WR", value: `${tradeReport.stats.win_rate}%`, color: tradeReport.stats.win_rate >= 50 ? COLORS.green : COLORS.amber },
                  { label: "PF", value: tradeReport.stats.profit_factor, color: tradeReport.stats.profit_factor >= 1 ? COLORS.green : COLORS.red },
                  { label: "PnL total", value: `${tradeReport.stats.total_pnl > 0 ? "+" : ""}${tradeReport.stats.total_pnl}$`, color: tradeReport.stats.total_pnl >= 0 ? COLORS.green : COLORS.red },
                  { label: "Gains", value: tradeReport.stats.wins },
                  { label: "Pertes", value: tradeReport.stats.losses },
                  { label: "Gain moy.", value: `+${tradeReport.stats.avg_win}$`, color: COLORS.green },
                  { label: "Perte moy.", value: `${tradeReport.stats.avg_loss}$`, color: COLORS.red },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ background: COLORS.bg, borderRadius: 6, padding: "6px 10px" }}>
                    <div style={{ fontSize: 10, color: COLORS.sub, marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: color || COLORS.text }}>{value}</div>
                  </div>
                ))}
              </div>

              {/* WR par heure CET */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 6 }}>WR par heure CET</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  {Object.entries(tradeReport.by_hour || {}).map(([h, v]) => {
                    const hi = parseInt(h);
                    const isLondon = hi >= 8 && hi < 12;
                    const isNY = hi >= 14 && hi < 18;
                    const barColor = v.wr >= 50 ? COLORS.green : v.wr >= 40 ? COLORS.amber : COLORS.red;
                    const sessionLabel = isLondon ? "Lo" : isNY ? "NY" : "—";
                    const sessionBg = isLondon ? "rgba(59,130,246,0.07)" : isNY ? "rgba(245,158,11,0.07)" : "transparent";
                    return (
                      <div key={h} style={{ display: "flex", alignItems: "center", gap: 6, background: sessionBg, borderRadius: 4, padding: "2px 4px" }}>
                        <span style={{ fontSize: 11, color: COLORS.sub, width: 32, flexShrink: 0 }}>{h}h</span>
                        <span style={{ fontSize: 10, color: COLORS.sub, width: 20, flexShrink: 0 }}>{sessionLabel}</span>
                        <div style={{ flex: 1, background: COLORS.border, borderRadius: 3, height: 8, overflow: "hidden" }}>
                          <div style={{ width: `${v.wr}%`, background: barColor, height: "100%", borderRadius: 3 }} />
                        </div>
                        <span style={{ fontSize: 11, color: barColor, width: 38, textAlign: "right", flexShrink: 0 }}>{v.wr}%</span>
                        <span style={{ fontSize: 10, color: COLORS.sub, width: 30, flexShrink: 0 }}>n={v.n}</span>
                        <span style={{ fontSize: 10, color: v.pnl >= 0 ? COLORS.green : COLORS.red, width: 50, textAlign: "right", flexShrink: 0 }}>
                          {v.pnl > 0 ? "+" : ""}{v.pnl}$
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* WR par session et direction */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
                <div>
                  <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 4 }}>Par session</div>
                  {Object.entries(tradeReport.by_session || {}).map(([s, v]) => (
                    <div key={s} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 2 }}>
                      <span style={{ color: COLORS.text }}>{s}</span>
                      <span>
                        <span style={{ color: v.wr >= 50 ? COLORS.green : COLORS.amber }}>{v.wr}%</span>
                        <span style={{ color: COLORS.sub }}> ({v.n})</span>
                      </span>
                    </div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 4 }}>Par direction</div>
                  {Object.entries(tradeReport.by_direction || {}).map(([d, v]) => (
                    <div key={d} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 2 }}>
                      <span style={{ color: d === "long" ? COLORS.green : COLORS.red }}>{d.toUpperCase()}</span>
                      <span>
                        <span style={{ color: v.wr >= 50 ? COLORS.green : COLORS.amber }}>{v.wr}%</span>
                        <span style={{ color: COLORS.sub }}> ({v.n})</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Résumé LLM collapsible */}
              <div>
                <button
                  onClick={() => setReportLlmOpen(o => !o)}
                  style={{ background: "none", border: `1px solid ${COLORS.border}`, color: COLORS.sub, fontSize: 11, padding: "3px 8px", borderRadius: 4, cursor: "pointer" }}
                >
                  {reportLlmOpen ? "▲ Masquer" : "▼ Résumé agent IA"}
                </button>
                {reportLlmOpen && (
                  <pre style={{ marginTop: 8, fontSize: 10, color: COLORS.sub, background: COLORS.bg, padding: 10, borderRadius: 6, overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                    {tradeReport.llm_summary}
                  </pre>
                )}
              </div>
            </div>
          )}
          </div>

          {/* ===== agent adaptatif autonome ===== */}
          <div className="dashboard-panel section-gap" style={{ ...panel(), marginTop: 14 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <h3 style={{ margin: 0, fontSize: 14 }}>Agent Adaptatif — Contrôle autonome</h3>
              <button
                onClick={() => {
                  setAdaptiveRunning(true);
                  setAdaptiveError(null);
                  fetch(`${API}/api/adaptive-agent/run`, { method: "POST", headers: authHeaders() })
                    .then((r) => { if (!r.ok) return r.json().then(d => { throw new Error(d.detail || `HTTP ${r.status}`); }); return r.json(); })
                    .then((d) => { setAdaptiveResult(d); setAdaptiveRunning(false); })
                    .catch((e) => { setAdaptiveError(e.message); setAdaptiveRunning(false); });
                }}
                disabled={adaptiveRunning}
                style={{ background: "#22c55e", color: "#000", border: "none", borderRadius: 6, padding: "5px 12px", fontSize: 12, fontWeight: 600, cursor: adaptiveRunning ? "wait" : "pointer", opacity: adaptiveRunning ? 0.7 : 1 }}
              >
                {adaptiveRunning ? "Analyse en cours..." : "Lancer l'agent"}
              </button>
            </div>
            {adaptiveError && (
              <div style={{ fontSize: 12, color: COLORS.red, marginBottom: 8 }}>Erreur : {adaptiveError}</div>
            )}
            {!adaptiveResult && !adaptiveRunning && !adaptiveError && (
              <div style={{ fontSize: 12, color: COLORS.sub }}>
                L'agent analyse les stats de trades et ajuste automatiquement les paramètres (heures bloquées, RSI, ADX). Tourne aussi automatiquement toutes les 6h hors session.
              </div>
            )}
            {adaptiveResult && (
              <div>
                <div style={{ fontSize: 10, color: COLORS.sub, marginBottom: 6 }}>
                  Run le {new Date(adaptiveResult.timestamp).toLocaleString("fr-FR")} · {adaptiveResult.trades_analyzed} trades analysés
                </div>
                {adaptiveResult.skipped && (
                  <div style={{ fontSize: 12, color: COLORS.sub }}>{adaptiveResult.skipped}</div>
                )}
                {adaptiveResult.analysis && (
                  <div style={{ fontSize: 12, color: COLORS.text, marginBottom: 8, lineHeight: 1.5 }}>
                    {adaptiveResult.analysis}
                  </div>
                )}
                {(adaptiveResult.actions_taken || []).length > 0 ? (
                  <div>
                    <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 4 }}>Actions appliquées :</div>
                    {(adaptiveResult.actions_taken || []).map((a, i) => (
                      <div key={i} style={{ fontSize: 12, color: "#22c55e", fontFamily: "monospace" }}>✓ {a}</div>
                    ))}
                  </div>
                ) : (
                  !adaptiveResult.skipped && <div style={{ fontSize: 12, color: COLORS.sub }}>Aucune action nécessaire.</div>
                )}
              </div>
            )}
            {/* Historique des runs automatiques depuis le state WebSocket */}
            {(state?.adaptive?.history || []).length > 0 && (
              <div style={{ marginTop: 12, borderTop: `1px solid ${COLORS.border}`, paddingTop: 8 }}>
                <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 4 }}>Historique automatique :</div>
                {(state.adaptive.history).slice().reverse().slice(0, 3).map((run, i) => (
                  <div key={i} style={{ fontSize: 11, marginBottom: 6 }}>
                    <span style={{ color: COLORS.sub }}>{new Date(run.timestamp).toLocaleString("fr-FR")} — </span>
                    {(run.actions_taken || []).length > 0
                      ? <span style={{ color: "#22c55e" }}>{run.actions_taken.join(" · ")}</span>
                      : <span style={{ color: COLORS.sub }}>aucune action</span>
                    }
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ===== agent IA analyse ===== */}
          <div className="dashboard-panel section-gap" style={{ ...panel(), marginTop: 14 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <h3 style={{ margin: 0, fontSize: 14 }}>Agent IA — Analyse du bot</h3>
              <button
                onClick={() => {
                  setAiReportLoading(true);
                  setAiReportError(null);
                  fetch(`${API}/api/ai-report`, { method: "POST", headers: authHeaders() })
                    .then((r) => { if (!r.ok) return r.json().then(d => { throw new Error(d.detail || `HTTP ${r.status}`); }); return r.json(); })
                    .then((d) => { setAiReport(d); setAiReportLoading(false); })
                    .catch((e) => { setAiReportError(e.message); setAiReportLoading(false); });
                }}
                disabled={aiReportLoading}
                style={{ background: COLORS.amber, color: "#000", border: "none", borderRadius: 6, padding: "5px 12px", fontSize: 12, fontWeight: 600, cursor: aiReportLoading ? "wait" : "pointer", opacity: aiReportLoading ? 0.7 : 1 }}
              >
                {aiReportLoading ? "Analyse en cours..." : "Demander un rapport"}
              </button>
            </div>
            {aiReportError && (
              <div style={{ fontSize: 12, color: COLORS.red, marginBottom: 8 }}>Erreur : {aiReportError}</div>
            )}
            {!aiReport && !aiReportLoading && !aiReportError && (
              <div style={{ fontSize: 12, color: COLORS.sub }}>
                Cliquez sur "Demander un rapport" pour obtenir une analyse IA de la situation du bot et des recommandations.
              </div>
            )}
            {aiReport && (
              <div>
                <div style={{ fontSize: 10, color: COLORS.sub, marginBottom: 8 }}>
                  Généré le {new Date(aiReport.generated_at).toLocaleString("fr-FR")} · {aiReport.trades_total} trades analysés
                </div>
                <div style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
                  {aiReport.report}
                </div>
              </div>
            )}
          </div>

          {/* ===== alerts feed ===== */}
          <div className="dashboard-panel alerts-panel" style={{ ...panel(), marginTop: 14 }}>
            <h3 style={{ margin: "0 0 8px", fontSize: 14 }}>Alertes</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 130, overflowY: "auto" }}>
              {(state?.alerts || []).slice().reverse().map((a, i) => (
                <div key={i} style={{ fontSize: 12, color: alertColor(a.kind) }}>
                  <span style={{ color: COLORS.sub }}>{fmtLocalTime(a.ts, true)} </span>
                  {a.message}
                </div>
              ))}
              {(state?.alerts || []).length === 0 && (
                <span style={{ fontSize: 12, color: COLORS.sub }}>Aucune alerte</span>
              )}
            </div>
          </div>

          {/* ===== actualités forex (Finnhub) ===== */}
          {newsFeed?.latest_news?.length > 0 && (
            <div className="dashboard-panel section-gap" style={{ ...panel(), marginTop: 14 }}>
              <h3 style={{ margin: "0 0 8px", fontSize: 14 }}>Actualités</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {newsFeed.latest_news.slice(0, 5).map((item, i) => {
                  const ts = item.datetime
                    ? new Date(item.datetime * 1000).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })
                    : "";
                  const headline = item.headline.length > 80
                    ? item.headline.slice(0, 79) + "…"
                    : item.headline;
                  return (
                    <div key={i} style={{ fontSize: 11, lineHeight: 1.4 }}>
                      <span style={{ color: COLORS.sub, marginRight: 6 }}>{ts}</span>
                      {item.url ? (
                        <a href={item.url} target="_blank" rel="noopener noreferrer"
                          style={{ color: COLORS.text, textDecoration: "none" }}
                          onMouseEnter={(e) => e.target.style.color = COLORS.blue}
                          onMouseLeave={(e) => e.target.style.color = COLORS.text}>
                          {headline}
                        </a>
                      ) : (
                        <span style={{ color: COLORS.text }}>{headline}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
      </>
    </div>
  );
}

/* ---------------------------- small UI bits ----------------------------- */
const alertColor = (k) =>
  k === "entry" ? COLORS.green : k === "exit" ? COLORS.blue
    : k === "danger" ? COLORS.red : k === "warn" ? COLORS.amber : COLORS.text;

const panel = () => ({
  background: COLORS.panel, border: `1px solid ${COLORS.border}`,
  borderRadius: 10, padding: 14,
});
const tabBtn = (active, small) => ({
  background: active ? COLORS.blue : "transparent",
  color: active ? "#fff" : COLORS.text,
  border: `1px solid ${active ? COLORS.blue : COLORS.border}`,
  borderRadius: 6, padding: small ? "3px 10px" : "6px 14px",
  fontSize: small ? 12 : 13, cursor: "pointer", fontWeight: 500,
});
const th = { padding: "6px 8px", fontWeight: 500 };
const td = { padding: "6px 8px" };

/* ============================ correlations panel ========================= */
function CorrelationsPanel({ data }) {
  const entries = Object.entries(data);
  return (
    <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 8, marginTop: 8 }}>
      <div style={{ fontSize: 11, color: COLORS.sub, textTransform: "uppercase",
        letterSpacing: 0.5, marginBottom: 6 }}>
        Corrélations XAU/USD
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5, maxHeight: 180, overflowY: "auto" }}>
        {entries.map(([name, info]) => {
          const c = info?.correlation ?? null;
          const trend = info?.trend;
          const barColor =
            c === null ? COLORS.grey
            : c > 0.3 ? COLORS.green
            : c < -0.3 ? COLORS.red
            : COLORS.grey;
          const valStr =
            c === null ? "—"
            : (c >= 0 ? "+" : "") + c.toFixed(2);
          const trendArrow =
            trend === "strengthening" ? "▲"
            : trend === "weakening" ? "▼"
            : "";
          /* bar fills proportionally: c in [-1, 1] mapped to [0%, 100%] from centre */
          const barWidth = c === null ? 0 : Math.abs(c) * 50; // max 50% of half
          const barLeft = c === null ? "50%" : c >= 0 ? "50%" : `${50 - Math.abs(c) * 50}%`;
          return (
            <div key={name} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
              {/* Asset name */}
              <span style={{ width: 64, color: COLORS.sub, flexShrink: 0, overflow: "hidden",
                textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={name}>
                {name}
              </span>
              {/* Bar track */}
              <div style={{ flex: 1, height: 6, background: "#1a2233", borderRadius: 3, position: "relative" }}>
                {/* centre marker */}
                <div style={{ position: "absolute", left: "50%", top: 0, width: 1, height: "100%",
                  background: COLORS.border }} />
                {/* coloured fill */}
                {c !== null && (
                  <div style={{
                    position: "absolute",
                    left: barLeft,
                    width: `${barWidth}%`,
                    height: "100%",
                    borderRadius: 3,
                    background: barColor,
                    transition: "width .4s",
                  }} />
                )}
              </div>
              {/* Value + trend */}
              <span style={{ width: 46, textAlign: "right", color: barColor, fontWeight: 600, flexShrink: 0 }}>
                {valStr}
                {trendArrow && (
                  <span style={{ fontSize: 9, marginLeft: 2, color: COLORS.sub }}>{trendArrow}</span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Stat({ label, value, color, big }) {
  return (
    <div style={panel()}>
      <div style={{ fontSize: 11, color: COLORS.sub, textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: big ? 24 : 18, fontWeight: 700, color, marginTop: 4 }}>{value}</div>
    </div>
  );
}
function DiagSection({ sl, tp, wrByHour }) {
  const diagRows = [
    ["RSI M5",  sl.rsi_m5,  tp.rsi_m5,  1],
    ["ADX H1",  sl.adx_h1,  tp.adx_h1,  1],
    ["ATR",     sl.atr,     tp.atr,     2],
    ["London%", sl.london_pct, tp.london_pct, 0],
  ];
  const hours = wrByHour
    ? Object.entries(wrByHour).map(([h, v]) => ({ h: parseInt(h), ...v })).sort((a, b) => a.wr - b.wr)
    : [];
  return (
    <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 6, marginTop: 6 }}>
      <div style={{ fontSize: 10, color: COLORS.sub, marginBottom: 4 }}>
        Diag — SL direct ({sl.n ?? 0}) vs TP2 ({tp.n ?? 0})
      </div>
      <table style={{ width: "100%", fontSize: 10, borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <td style={{ color: COLORS.sub, paddingBottom: 2 }}></td>
            <td style={{ color: COLORS.red,   textAlign: "right", paddingBottom: 2 }}>SL dir</td>
            <td style={{ color: COLORS.green, textAlign: "right", paddingBottom: 2 }}>TP2</td>
            <td style={{ color: COLORS.amber, textAlign: "right", paddingBottom: 2 }}>Δ</td>
          </tr>
        </thead>
        <tbody>
          {diagRows.map(([lbl, sv, tv, dec]) => {
            const fmt = (v) => v != null ? Number(v).toFixed(dec) + (lbl === "London%" ? "%" : "") : "—";
            const svF = fmt(sv);
            const tvF = fmt(tv);
            const delta = (sv != null && tv != null) ? Number(sv) - Number(tv) : null;
            const dColor = delta != null && Math.abs(delta) >= 3 ? COLORS.amber : COLORS.sub;
            return (
              <tr key={lbl}>
                <td style={{ color: COLORS.sub }}>{lbl}</td>
                <td style={{ textAlign: "right", color: COLORS.text }}>{svF}</td>
                <td style={{ textAlign: "right", color: COLORS.text }}>{tvF}</td>
                <td style={{ textAlign: "right", color: dColor }}>
                  {delta != null ? (delta >= 0 ? "+" : "") + delta.toFixed(dec) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {hours.length > 0 && (
        <div style={{ marginTop: 6 }}>
          <div style={{ fontSize: 10, color: COLORS.sub, marginBottom: 3 }}>WR / heure CET</div>
          {hours.map(({ h, n, wr: w }) => {
            const pct = Math.round(w * 100);
            const bc = pct >= 50 ? COLORS.green : pct >= 40 ? COLORS.amber : COLORS.red;
            return (
              <div key={h} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                <span style={{ width: 22, fontSize: 9, color: COLORS.sub, flexShrink: 0 }}>{h}h</span>
                <div style={{ flex: 1, background: COLORS.border, borderRadius: 2, height: 5 }}>
                  <div style={{ width: `${pct}%`, background: bc, height: 5, borderRadius: 2 }} />
                </div>
                <span style={{ width: 40, fontSize: 9, color: bc, textAlign: "right", flexShrink: 0 }}>
                  {pct}% <span style={{ color: COLORS.sub }}>({n})</span>
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
function Row({ k, v, inline, color }) {
  if (inline)
    return (
      <span style={{ fontSize: 12, color: COLORS.sub }}>
        {k}: <span style={{ color: color || COLORS.text }}>{v}</span>
      </span>
    );
  return (
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
      <span style={{ color: COLORS.sub }}>{k}</span>
      <span style={{ color: color || COLORS.text }}>{v}</span>
    </div>
  );
}
function Legend({ c, t }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      <span style={{ width: 10, height: 2, background: c, display: "inline-block" }} /> {t}
    </span>
  );
}
function ProgressBar({ label, value, done }) {
  const v = done ? 1 : Math.max(0, Math.min(value ?? 0, 1));
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: COLORS.sub }}>
        <span>{label} {done ? "✓" : ""}</span>
        <span>{(v * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: 6, background: "#1a2233", borderRadius: 3, marginTop: 2 }}>
        <div style={{ width: `${v * 100}%`, height: "100%", borderRadius: 3,
          background: done ? COLORS.green : COLORS.blue, transition: "width .5s" }} />
      </div>
    </div>
  );
}
