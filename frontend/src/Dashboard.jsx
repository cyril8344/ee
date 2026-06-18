import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  ComposedChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid, Area, AreaChart, Bar, BarChart, Cell,
} from "recharts";
import BacktestPanel from "./BacktestPanel";

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

/* ============================= candlestick =============================== */
function Candles({ candles, markers, levels }) {
  if (!candles || candles.length === 0)
    return <div style={{ color: COLORS.sub, padding: 40 }}>Chargement du graphique…</div>;

  const W = 100; // logical, scaled by ResponsiveContainer via SVG viewBox
  const prices = candles.flatMap((c) => [c.high, c.low]);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const pad = (max - min) * 0.08 || 1;
  const lo = min - pad;
  const hi = max + pad;
  const height = 420;
  const width = Math.max(candles.length * 9, 600);
  const cw = width / candles.length;
  const bw = cw * 0.6;

  const y = (p) => height - ((p - lo) / (hi - lo)) * height;

  const markerByTime = {};
  (markers || []).forEach((m) => {
    markerByTime[m.time] = m;
  });

  return (
    <div style={{ overflowX: "auto", background: COLORS.panel2, borderRadius: 8 }}>
      <svg width={width} height={height + 30} style={{ display: "block" }}>
        {/* support / resistance */}
        {(levels?.resistance || []).map((r, i) => (
          <line key={"r" + i} x1={0} x2={width} y1={y(r)} y2={y(r)}
            stroke={COLORS.red} strokeOpacity={0.18} strokeDasharray="4 4" />
        ))}
        {(levels?.support || []).map((s, i) => (
          <line key={"s" + i} x1={0} x2={width} y1={y(s)} y2={y(s)}
            stroke={COLORS.green} strokeOpacity={0.18} strokeDasharray="4 4" />
        ))}

        {/* EMA lines */}
        {["ema9", "ema21", "ema200"].map((key, idx) => {
          const stroke = [COLORS.amber, COLORS.blue, "#c084fc"][idx];
          const d = candles
            .map((c, i) => `${i === 0 ? "M" : "L"} ${i * cw + cw / 2} ${y(c[key])}`)
            .join(" ");
          return <path key={key} d={d} fill="none" stroke={stroke} strokeWidth={1.3} opacity={0.9} />;
        })}

        {/* candles */}
        {candles.map((c, i) => {
          const x = i * cw + cw / 2;
          const up = c.close >= c.open;
          const color = up ? COLORS.candleUp : COLORS.candleDown;
          const yo = y(c.open);
          const yc = y(c.close);
          const bodyTop = Math.min(yo, yc);
          const bodyH = Math.max(Math.abs(yc - yo), 1);
          const signal = markerByTime[c.time];
          return (
            <g key={i}>
              <line x1={x} x2={x} y1={y(c.high)} y2={y(c.low)} stroke={color} strokeWidth={1} />
              <rect x={x - bw / 2} y={bodyTop} width={bw} height={bodyH} fill={color} />
              {signal && signal.type === "entry" && (
                <text x={x} y={signal.direction === "long" ? y(c.low) + 16 : y(c.high) - 8}
                  fontSize="14" textAnchor="middle"
                  fill={signal.direction === "long" ? COLORS.green : COLORS.red}>
                  {signal.direction === "long" ? "▲" : "▼"}
                </text>
              )}
              {signal && signal.type === "exit" && (
                <text x={x} y={y(c.high) - 8} fontSize="13" textAnchor="middle" fill={COLORS.sub}>✕</text>
              )}
            </g>
          );
        })}
      </svg>
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
  const [tab, setTab] = useState("live");
  const [activeMarket, setActiveMarket] = useState("XAUUSD");
  const [weightsOpen, setWeightsOpen] = useState(false);
  const [state, setState] = useState(null);
  const [chart, setChart] = useState(null);
  const [tf, setTf] = useState("M5");
  const [trades, setTrades] = useState({ trades: [], equity_curve: [] });
  const [tradesScope, setTradesScope] = useState("today");
  const [connected, setConnected] = useState(false);
  const [patternStats, setPatternStats] = useState({});
  const [correlations, setCorrelations] = useState({});
  const [newsFeed, setNewsFeed] = useState(null);
  const [agentStatus, setAgentStatus] = useState(null);
  const [agentHistory, setAgentHistory] = useState([]);
  const [rlStatus, setRlStatus] = useState(null);
  const [rlHistory, setRlHistory] = useState([]);
  const [rlLoading, setRlLoading] = useState(false);
  const [settingsEdit, setSettingsEdit] = useState(false);
  const [settingsDraft, setSettingsDraft] = useState({});
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [fedData, setFedData] = useState(null);
  const beep = useBeep();
  const lastAlertTs = useRef(null);

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
          if (msg.type === "state") setState(msg.data);
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
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text,
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
              {sym === "XAUUSD" ? "XAU/USD" : sym === "EURUSD" ? "EUR/USD" : sym}
            </button>
          ))}
          <div style={{ width: 1, background: COLORS.border, margin: "0 4px" }} />
          {["live", "backtest"].map((t) => (
            <button key={t} onClick={() => setTab(t)} style={tabBtn(tab === t)}>
              {t === "live" ? "Live" : "Backtest"}
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

      {tab === "backtest" ? (
        <BacktestPanel api={API} />
      ) : (
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
            <Stat label="Trades du jour"
              value={`${state?.trades_today ?? 0} / ${state?.max_trades_per_day ?? 4}`}
              color={COLORS.text} />
          </div>

          <div className="main-layout" style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 14 }}>
            {/* ===== main chart ===== */}
            <div className="dashboard-panel" style={panel()}>
              <div style={{ display: "flex", alignItems: "center", marginBottom: 10 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>Graphique</h3>
                <span style={{ marginLeft: 12, color: COLORS.sub, fontSize: 13 }}>
                  {fmt(mkt.price, activeMarket === "EURUSD" ? 5 : 2)} {activeMarket === "EURUSD" ? "" : "$"}
                </span>
                <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                  {["M5", "M15", "H1"].map((t) => (
                    <button key={t} onClick={() => setTf(t)} style={tabBtn(tf === t, true)}>{t}</button>
                  ))}
                </div>
              </div>
              <Candles candles={chart?.candles} markers={chart?.markers} levels={chart?.levels} />
              <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 11, color: COLORS.sub }}>
                <Legend c={COLORS.amber} t="EMA9" /><Legend c={COLORS.blue} t="EMA21" />
                <Legend c="#c084fc" t="EMA200" />
                <Legend c={COLORS.green} t="Support" /><Legend c={COLORS.red} t="Résistance" />
                <span>▲ entrée long · ▼ entrée short · ✕ sortie</span>
              </div>
            </div>

            {/* ===== side panel ===== */}
            <div className="dashboard-panel" style={panel()}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>Statut bot</h3>
                <span style={{ padding: "3px 10px", borderRadius: 4, fontWeight: 600, fontSize: 12,
                  background: statusColor + "22", color: statusColor }}>
                  {state?.bot_status || "—"}
                </span>
              </div>

              <RsiBar label="RSI M5" value={mkt.indicators?.rsi_m5} />
              <RsiBar label="RSI M15" value={mkt.indicators?.rsi_m15} />
              <div style={{ margin: "12px 0" }}>
                <AtrGauge atr={mkt.indicators?.atr_m5} avg={mkt.indicators?.atr_avg}
                  min={mkt.indicators?.atr_min} />
              </div>

              {/* ---- trading conditions checklist ---- */}
              {mkt.conditions && (
                <div style={{ background: "#0a1020", borderRadius: 6, padding: "8px 10px", marginBottom: 10, fontSize: 11 }}>
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
                    { label: "Biais H1 EMA200", ok: mkt.conditions.h1_bias !== "NEUTRE", val: mkt.conditions.h1_bias },
                    { label: "M15 EMA9/RSI", ok: mkt.conditions.m15_confirmed, val: mkt.conditions.m15_confirmed ? "✓" : "✗" },
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
                  {fedData.fed?.source === "unavailable" && (
                    <div style={{ fontSize: 10, color: COLORS.amber, marginTop: 4 }}>
                      ⚠ Clé FRED_API_KEY manquante — ajouter dans Railway Variables
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
                      { label: "Risque / trade (%)", key: "risk_per_trade_pct", min: 0.1, max: 5 },
                      { label: "Stop journalier (%)", key: "daily_stop_pct", min: 0.5, max: 10 },
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
                    <Row k="Capital" v={`$${fmt(state?.risk?.capital, 2)}`} />
                    <Row k="Risque / trade" v={`${fmt(state?.risk?.risk_per_trade_pct, 1)}% · $${fmt(state?.risk?.risk_amount_usd, 0)}`} />
                    <Row k="Stop journalier" v={`-$${fmt(state?.risk?.daily_loss_limit_usd, 0)}`} />
                    <Row k="Trades max / jour" v={`${state?.risk?.max_trades_per_day ?? "—"}`} />
                  </>
                )}
              </div>

              {/* pattern weights */}
              {Object.keys(patternStats).length > 0 && (
                <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 10 }}>
                  <button
                    onClick={() => setWeightsOpen((o) => !o)}
                    style={{ background: "none", border: "none", padding: 0, cursor: "pointer",
                      fontSize: 12, color: COLORS.sub, marginBottom: 6, textAlign: "left",
                      width: "100%", display: "flex", justifyContent: "space-between" }}>
                    <span>Poids des patterns</span>
                    <span>{weightsOpen ? "▲" : "▼"}</span>
                  </button>
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
                <button onClick={toggleBot} style={{ ...tabBtn(false), flex: 1 }}>
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

          {/* ===== active trade ===== */}
          {pos && (
            <div className="dashboard-panel" style={{ ...panel(), marginTop: 14, borderColor: pos.direction === "long" ? COLORS.green : COLORS.red }}>
              <div className="active-trade-row" style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
                <span style={{ fontWeight: 700, fontSize: 16,
                  color: pos.direction === "long" ? COLORS.green : COLORS.red }}>
                  {pos.direction === "long" ? "▲ LONG" : "▼ SHORT"} · {pos.volume} lots
                </span>
                <Row k="Entrée" v={fmt(pos.entry, 2)} inline />
                <Row k="SL" v={fmt(pos.stop_loss, 2)} inline />
                <Row k="TP1" v={fmt(pos.take_profit1, 2)} inline />
                <Row k="TP2" v={fmt(pos.take_profit2, 2)} inline />
                <span style={{ fontWeight: 700,
                  color: pos.unrealised_pnl >= 0 ? COLORS.green : COLORS.red }}>
                  {money(pos.unrealised_pnl)}
                </span>
                <span style={{ marginLeft: "auto", fontSize: 13, color: remaining < 300 ? COLORS.amber : COLORS.sub }}>
                  ⏱ {hms(remaining)} / 45:00
                </span>
                <button onClick={closeNow}
                  style={{ ...tabBtn(false), borderColor: COLORS.red, color: COLORS.red, fontWeight: 700 }}>
                  FERMER MAINTENANT
                </button>
              </div>
              <div style={{ marginTop: 12 }}>
                <ProgressBar label="TP1 (60%)" value={pos.progress_tp1} done={pos.tp1_done} />
                <ProgressBar label="TP2 (40%)" value={pos.progress_tp2} />
              </div>
            </div>
          )}

          {/* ===== history + equity ===== */}
          <div className="history-layout" style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 14, marginTop: 14 }}>
            <div className="dashboard-panel" style={panel()}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>
                  {tradesScope === "today" ? "Historique du jour" : "Tout l'historique"}
                </h3>
                <div style={{ display: "flex", gap: 4 }}>
                  <button onClick={() => setTradesScope("today")} style={{ ...tabBtn(tradesScope === "today"), fontSize: 11, padding: "3px 8px" }}>Aujourd'hui</button>
                  <button onClick={() => setTradesScope("all")} style={{ ...tabBtn(tradesScope === "all"), fontSize: 11, padding: "3px 8px" }}>Tout</button>
                </div>
              </div>
              <div style={{ maxHeight: 240, overflowY: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ color: COLORS.sub, textAlign: "left" }}>
                      <th style={th}>Heure</th><th style={th}>Dir</th><th style={th}>Entrée</th>
                      <th style={th}>Sortie</th>
                      <th className="hide-mobile" style={th}>Durée</th>
                      <th style={th}>Résultat</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(trades.trades || []).filter((t) => t.status === "closed").reverse().map((t) => (
                      <tr key={t.id} style={{ borderTop: `1px solid ${COLORS.border}` }}>
                        <td style={td}>{fmtLocalTime(t.entry_time)}</td>
                        <td style={{ ...td, color: t.direction === "long" ? COLORS.green : COLORS.red }}>
                          {t.direction === "long" ? "LONG" : "SHORT"}
                        </td>
                        <td style={td}>{fmt(t.entry_price, 2)}</td>
                        <td style={td}>{fmt(t.exit_price, 2)}</td>
                        <td className="hide-mobile" style={td}>{fmt(t.duration_min, 0)}m</td>
                        <td style={{ ...td, fontWeight: 600, color: (t.pnl || 0) >= 0 ? COLORS.green : COLORS.red }}>
                          {money(t.pnl)}
                        </td>
                      </tr>
                    ))}
                    {(trades.trades || []).filter((t) => t.status === "closed").length === 0 && (
                      <tr><td style={{ ...td, color: COLORS.sub }} colSpan={6}>Aucun trade clôturé aujourd'hui</td></tr>
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
            <div className="dashboard-panel" style={{ ...panel(), marginTop: 14 }}>
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
      )}
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
  const v = Math.max(0, Math.min(value ?? 0, 1));
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
