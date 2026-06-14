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

const API = import.meta?.env?.VITE_API_URL || "http://localhost:8000";
const WS_URL = API.replace(/^http/, "ws") + "/ws";

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

/* ============================ Dashboard ================================== */
export default function Dashboard() {
  const [tab, setTab] = useState("live");
  const [state, setState] = useState(null);
  const [chart, setChart] = useState(null);
  const [tf, setTf] = useState("M5");
  const [trades, setTrades] = useState({ trades: [], equity_curve: [] });
  const [connected, setConnected] = useState(false);
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

  /* Chart polling */
  useEffect(() => {
    let active = true;
    const load = () =>
      fetch(`${API}/api/chart?tf=${tf}`)
        .then((r) => r.json())
        .then((d) => active && setChart(d))
        .catch(() => {});
    load();
    const id = setInterval(load, 15000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [tf]);

  /* Trades polling */
  useEffect(() => {
    let active = true;
    const load = () =>
      fetch(`${API}/api/trades?scope=today`)
        .then((r) => r.json())
        .then((d) => active && setTrades(d))
        .catch(() => {});
    load();
    const id = setInterval(load, 10000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const pos = state?.position;
  const remaining = useCountdown(pos?.remaining_seconds);
  const newsCountdown = useCountdown(state?.news?.next_event_countdown_sec);

  const closeNow = () => fetch(`${API}/api/close`, { method: "POST" });
  const toggleBot = () => fetch(`${API}/api/bot/toggle`, { method: "POST" });

  const switchLive = () => {
    if (state?.mode === "live") {
      fetch(`${API}/api/mode`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "paper" }),
      });
      return;
    }
    if (!window.confirm("⚠️ Passer en mode LIVE (argent réel) ?")) return;
    if (!window.confirm("CONFIRMATION FINALE : exécuter de vrais ordres ?")) return;
    fetch(`${API}/api/mode`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "live", confirm: true, confirm_again: true }),
    });
  };

  const statusColor =
    state?.bot_status === "ACTIF" ? COLORS.green
      : state?.bot_status === "BLOQUE" ? COLORS.red : COLORS.amber;

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text,
      fontFamily: "'Inter', system-ui, sans-serif", padding: 16 }}>
      {/* ===== header / tabs ===== */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16 }}>
        <h1 style={{ fontSize: 20, margin: 0, letterSpacing: 0.5 }}>
          🟡 XAU/USD <span style={{ color: COLORS.sub, fontWeight: 400 }}>Scalping Bot</span>
        </h1>
        <span style={{ fontSize: 12, color: connected ? COLORS.green : COLORS.red }}>
          ● {connected ? "connecté" : "déconnecté"}
        </span>
        <span style={{ fontSize: 12, padding: "2px 8px", borderRadius: 4,
          background: state?.mode === "live" ? COLORS.red : COLORS.border,
          color: state?.mode === "live" ? "#fff" : COLORS.sub }}>
          {state?.mode === "live" ? "LIVE" : "PAPER"}
        </span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          {["live", "backtest"].map((t) => (
            <button key={t} onClick={() => setTab(t)} style={tabBtn(tab === t)}>
              {t === "live" ? "Live" : "Backtest"}
            </button>
          ))}
        </div>
      </div>

      {tab === "backtest" ? (
        <BacktestPanel api={API} />
      ) : (
        <>
          {/* ===== top band ===== */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 14 }}>
            <Stat label="Biais du jour"
              value={state?.bias || "—"}
              color={biasColor(state?.bias)} big />
            <Stat label="Session"
              value={state?.session || "—"}
              color={state?.session?.includes("session") ? COLORS.grey : COLORS.blue} />
            <Stat label="P&L du jour"
              value={`${money(state?.day_pnl)} (${pct(state?.day_pnl_pct)})`}
              color={(state?.day_pnl || 0) >= 0 ? COLORS.green : COLORS.red} />
            <Stat label="Trades du jour"
              value={`${state?.trades_today ?? 0} / ${state?.max_trades_per_day ?? 4}`}
              color={COLORS.text} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 14 }}>
            {/* ===== main chart ===== */}
            <div style={panel()}>
              <div style={{ display: "flex", alignItems: "center", marginBottom: 10 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>Graphique</h3>
                <span style={{ marginLeft: 12, color: COLORS.sub, fontSize: 13 }}>
                  {fmt(state?.price, 2)} $
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
            <div style={panel()}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h3 style={{ margin: 0, fontSize: 14 }}>Statut bot</h3>
                <span style={{ padding: "3px 10px", borderRadius: 4, fontWeight: 600, fontSize: 12,
                  background: statusColor + "22", color: statusColor }}>
                  {state?.bot_status || "—"}
                </span>
              </div>

              <RsiBar label="RSI M5" value={state?.indicators?.rsi_m5} />
              <RsiBar label="RSI M15" value={state?.indicators?.rsi_m15} />
              <div style={{ margin: "12px 0" }}>
                <AtrGauge atr={state?.indicators?.atr_m5} avg={state?.indicators?.atr_avg}
                  min={state?.indicators?.atr_min} />
              </div>

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
                ) : (
                  <div style={{ fontSize: 13, color: COLORS.sub }}>Aucune news imminente</div>
                )}
              </div>

              {/* risk summary */}
              <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 10, fontSize: 12 }}>
                <Row k="Capital" v={`$${fmt(state?.risk?.capital, 2)}`} />
                <Row k="Risque / trade" v={`${fmt(state?.risk?.risk_per_trade_pct, 1)}% · $${fmt(state?.risk?.risk_amount_usd, 0)}`} />
                <Row k="Stop journalier" v={`-$${fmt(state?.risk?.daily_loss_limit_usd, 0)}`} />
              </div>

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
              </div>
            </div>
          </div>

          {/* ===== active trade ===== */}
          {pos && (
            <div style={{ ...panel(), marginTop: 14, borderColor: pos.direction === "long" ? COLORS.green : COLORS.red }}>
              <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
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
          <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 14, marginTop: 14 }}>
            <div style={panel()}>
              <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>Historique du jour</h3>
              <div style={{ maxHeight: 240, overflowY: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ color: COLORS.sub, textAlign: "left" }}>
                      <th style={th}>Heure</th><th style={th}>Dir</th><th style={th}>Entrée</th>
                      <th style={th}>Sortie</th><th style={th}>Durée</th><th style={th}>Résultat</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(trades.trades || []).filter((t) => t.status === "closed").reverse().map((t) => (
                      <tr key={t.id} style={{ borderTop: `1px solid ${COLORS.border}` }}>
                        <td style={td}>{(t.entry_time || "").slice(11, 16)}</td>
                        <td style={{ ...td, color: t.direction === "long" ? COLORS.green : COLORS.red }}>
                          {t.direction === "long" ? "LONG" : "SHORT"}
                        </td>
                        <td style={td}>{fmt(t.entry_price, 2)}</td>
                        <td style={td}>{fmt(t.exit_price, 2)}</td>
                        <td style={td}>{fmt(t.duration_min, 0)}m</td>
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

            <div style={panel()}>
              <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>Équité intraday</h3>
              <ResponsiveContainer width="100%" height={210}>
                <AreaChart data={(trades.equity_curve || []).map((p, i) => ({
                  i, equity: p.equity, t: (p.ts || "").slice(11, 16),
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

          {/* ===== alerts feed ===== */}
          <div style={{ ...panel(), marginTop: 14 }}>
            <h3 style={{ margin: "0 0 8px", fontSize: 14 }}>Alertes</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 130, overflowY: "auto" }}>
              {(state?.alerts || []).slice().reverse().map((a, i) => (
                <div key={i} style={{ fontSize: 12, color: alertColor(a.kind) }}>
                  <span style={{ color: COLORS.sub }}>{(a.ts || "").slice(11, 19)} </span>
                  {a.message}
                </div>
              ))}
              {(state?.alerts || []).length === 0 && (
                <span style={{ fontSize: 12, color: COLORS.sub }}>Aucune alerte</span>
              )}
            </div>
          </div>
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

function Stat({ label, value, color, big }) {
  return (
    <div style={panel()}>
      <div style={{ fontSize: 11, color: COLORS.sub, textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: big ? 24 : 18, fontWeight: 700, color, marginTop: 4 }}>{value}</div>
    </div>
  );
}
function Row({ k, v, inline }) {
  if (inline)
    return (
      <span style={{ fontSize: 12, color: COLORS.sub }}>
        {k}: <span style={{ color: COLORS.text }}>{v}</span>
      </span>
    );
  return (
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
      <span style={{ color: COLORS.sub }}>{k}</span>
      <span>{v}</span>
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
