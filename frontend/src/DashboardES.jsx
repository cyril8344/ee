import React, { useState, useEffect, useRef } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell,
} from "recharts";

/* ============================================================================
 * ES (S&P 500 E-mini) — Dashboard Order Flow
 * Dashboard indépendant — même niveau que XAU :
 *   - Multi-TF (M5 + H1 bias)
 *   - VWAP / ADX / bad_hours / body_ratio / RSI 48/52
 *   - Walk-forward validation
 *   - Table de diagnostic par sortie / direction / heure
 * ========================================================================== */

const API = import.meta?.env?.VITE_API_URL || "";

const C = {
  bg:     "#080c14",
  panel:  "#0f1520",
  panel2: "#0a1018",
  border: "#1e2a3a",
  text:   "#e5e7eb",
  sub:    "#8b95a7",
  green:  "#16c784",
  red:    "#ea3943",
  grey:   "#6b7280",
  blue:   "#3b82f6",
  amber:  "#f59e0b",
  purple: "#a855f7",
  teal:   "#14b8a6",
};

/* ── Utils ───────────────────────────────────────────────────────────────── */
const fmt    = (n, d = 2) => n == null || isNaN(n) ? "—" : Number(n).toFixed(d);
const fmtUSD = (n) =>
  n == null || isNaN(n) ? "—" :
  (n >= 0 ? "+" : "−") + "$" + Number(Math.abs(n)).toLocaleString("en-US", { maximumFractionDigits: 0 });
const fmtTime = (iso) => {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("fr-FR", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  } catch { return iso.slice(0, 16); }
};

function apiGet(path, token) {
  return fetch(API + path, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
    .then(r => r.ok ? r.json() : null);
}
function apiPost(path, token, body) {
  return fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify(body),
  }).then(r => r.ok ? r.json() : null);
}

/* ── Styles partagés ─────────────────────────────────────────────────────── */
const S = {
  root:    { background: C.bg, color: C.text, minHeight: "100vh", fontFamily: "'Inter',system-ui,sans-serif", fontSize: 14 },
  header:  { background: C.panel, borderBottom: `1px solid ${C.border}`, padding: "12px 20px", display: "flex", alignItems: "center", gap: 14 },
  section: { background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: 18, marginBottom: 14 },
  h2:      { fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 },
  input:   { background: C.panel2, border: `1px solid ${C.border}`, borderRadius: 6, color: C.text, padding: "5px 8px", fontSize: 13, width: "100%", boxSizing: "border-box" },
  btn:     { padding: "7px 16px", borderRadius: 7, border: "none", fontWeight: 600, cursor: "pointer", fontSize: 13 },
  kpi:     { background: C.panel2, borderRadius: 8, padding: "12px 14px", border: `1px solid ${C.border}`, textAlign: "center" },
};

/* ── KPI Tile ────────────────────────────────────────────────────────────── */
function KpiTile({ label, value, color, sub }) {
  return (
    <div style={S.kpi}>
      <div style={{ color: C.sub, fontSize: 10, marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: color || C.text }}>{value}</div>
      {sub && <div style={{ color: C.sub, fontSize: 10, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

/* ── Progress bar ────────────────────────────────────────────────────────── */
function ProgressBar({ pct, status, color }) {
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ color: C.sub, fontSize: 11 }}>{status}</span>
        <span style={{ color: C.text, fontSize: 11, fontWeight: 600 }}>{pct}%</span>
      </div>
      <div style={{ height: 5, background: C.panel2, borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color || C.blue, borderRadius: 3, transition: "width 0.3s" }} />
      </div>
    </div>
  );
}

/* ── Param row (numérique) ───────────────────────────────────────────────── */
function ParamNum({ label, k, settings, onChange, step = 1, min }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 7 }}>
      <label style={{ color: C.sub, fontSize: 12, flex: 1 }}>{label}</label>
      <input type="number" step={step} min={min} value={settings[k] ?? ""}
        onChange={e => onChange(k, step < 1 ? parseFloat(e.target.value) : parseFloat(e.target.value))}
        style={{ ...S.input, width: 80, textAlign: "right" }} />
    </div>
  );
}

/* ── Toggle row ──────────────────────────────────────────────────────────── */
function ParamToggle({ label, k, settings, onChange }) {
  const on = !!settings[k];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 7 }}>
      <label style={{ color: C.sub, fontSize: 12, flex: 1 }}>{label}</label>
      <button onClick={() => onChange(k, on ? 0 : 1)}
        style={{ ...S.btn, padding: "3px 12px", fontSize: 12,
                 background: on ? C.green : C.grey, color: "#000", opacity: on ? 1 : 0.7 }}>
        {on ? "ON" : "OFF"}
      </button>
    </div>
  );
}

/* ── Equity curve ─────────────────────────────────────────────────────────── */
function EquityCurve({ data }) {
  if (!data || data.length < 2) return (
    <div style={{ height: 160, display: "flex", alignItems: "center", justifyContent: "center", color: C.sub, fontSize: 12 }}>
      Aucune donnée
    </div>
  );
  const col = data[data.length - 1]?.equity >= data[0]?.equity ? C.green : C.red;
  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={data} margin={{ top: 6, right: 4, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="esEqG" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={col} stopOpacity={0.3} />
            <stop offset="95%" stopColor={col} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <XAxis dataKey="ts" hide />
        <YAxis domain={["auto", "auto"]} hide />
        <Tooltip contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 11 }}
          formatter={v => ["$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 0 }), "Equity"]}
          labelFormatter={() => ""} />
        <Area type="monotone" dataKey="equity" stroke={col} strokeWidth={2} fill="url(#esEqG)" dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ── Heatmap horaire ─────────────────────────────────────────────────────── */
function HourlyChart({ data }) {
  if (!data || Object.keys(data).length === 0) return null;
  const entries = Object.entries(data).map(([h, v]) => ({
    h: parseInt(h), trades: v.trades, wins: v.wins,
    wr: v.trades > 0 ? Math.round(v.wins / v.trades * 100) : 0,
  }));
  return (
    <ResponsiveContainer width="100%" height={90}>
      <BarChart data={entries} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
        <XAxis dataKey="h" tickFormatter={h => `${h}h`} tick={{ fill: C.sub, fontSize: 10 }} />
        <YAxis hide />
        <Tooltip contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 11 }}
          formatter={(_, __, { payload: p }) => [`${p.wins}W/${p.trades - p.wins}L  (${p.wr}% WR)`, "Trades"]}
          labelFormatter={h => `${h}h ET`} />
        <Bar dataKey="trades" radius={[3, 3, 0, 0]}>
          {entries.map((e, i) => (
            <Cell key={i} fill={e.wr >= 55 ? C.green : e.wr >= 45 ? C.amber : e.wr > 0 ? C.red : C.grey} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ── Table diagnostic ────────────────────────────────────────────────────── */
function DiagTable({ data, title, keyLabel }) {
  if (!data || Object.keys(data).length === 0) return null;
  const rows = Object.entries(data).sort((a, b) => b[1].n - a[1].n);
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ color: C.sub, fontSize: 11, fontWeight: 600, textTransform: "uppercase",
                    letterSpacing: "0.06em", marginBottom: 6 }}>{title}</div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr>
              {[keyLabel, "N", "WR%", "RSI", "ATR", "ADX", "Vol/moy", "MAE-R", "MFE-R"].map(h => (
                <th key={h} style={{ padding: "4px 8px", textAlign: "left", color: C.sub,
                                      borderBottom: `1px solid ${C.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k} style={{ borderBottom: `1px solid ${C.border}` }}>
                <td style={{ padding: "4px 8px", fontWeight: 600,
                              color: v.wr >= 55 ? C.green : v.wr >= 45 ? C.amber : C.red }}>
                  {k === "sl" ? "SL" : k === "tp2" ? "TP2" : k === "sl_after_tp1" ? "SL/BE" :
                   k === "timeout" ? "TO" : k === "timeout_tp1" ? "TO+TP1" : k === "tp_direct" ? "TP dir" : k}
                </td>
                <td style={{ padding: "4px 8px" }}>{v.n}</td>
                <td style={{ padding: "4px 8px", color: v.wr >= 55 ? C.green : v.wr >= 45 ? C.amber : C.red, fontWeight: 600 }}>{v.wr}%</td>
                <td style={{ padding: "4px 8px" }}>{fmt(v.rsi, 1)}</td>
                <td style={{ padding: "4px 8px" }}>{fmt(v.atr, 2)}</td>
                <td style={{ padding: "4px 8px" }}>{fmt(v.adx, 1)}</td>
                <td style={{ padding: "4px 8px" }}>{fmt(v.vol_ratio, 2)}</td>
                <td style={{ padding: "4px 8px", color: C.red }}>{fmt(v.mae_r, 3)}</td>
                <td style={{ padding: "4px 8px", color: C.green }}>{fmt(v.mfe_r, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Trade table ──────────────────────────────────────────────────────────── */
function TradeTable({ trades }) {
  if (!trades || trades.length === 0) return (
    <div style={{ color: C.sub, fontSize: 12, padding: "16px 0", textAlign: "center" }}>
      Aucun trade
    </div>
  );
  const EXIT_COLOR = { tp2: C.green, tp_direct: C.green, sl_after_tp1: C.amber, timeout_tp1: C.amber, sl: C.red, timeout: C.sub };
  const EXIT_LABEL = { tp2: "TP2", tp_direct: "TP↑", sl_after_tp1: "SL/BE", timeout_tp1: "TO+TP1", sl: "SL", timeout: "TO" };
  return (
    <div style={{ overflowX: "auto", maxHeight: 340, overflowY: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
        <thead>
          <tr>
            {["Entrée", "Sens", "Prix", "Sortie", "Raison", "P&L", "Contrats", "Vol/moy", "RSI", "ADX"].map(h => (
              <th key={h} style={{ padding: "5px 8px", textAlign: "left", color: C.sub,
                                    borderBottom: `1px solid ${C.border}`, position: "sticky",
                                    top: 0, background: C.panel }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...trades].reverse().map((t, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${C.border}`, background: i % 2 ? "rgba(255,255,255,0.02)" : "transparent" }}>
              <td style={{ padding: "4px 8px" }}>{fmtTime(t.entry_ts)}</td>
              <td style={{ padding: "4px 8px", fontWeight: 600, color: t.direction === "long" ? C.green : C.red }}>
                {t.direction?.toUpperCase()}
              </td>
              <td style={{ padding: "4px 8px" }}>{fmt(t.entry, 2)}</td>
              <td style={{ padding: "4px 8px" }}>{fmt(t.exit_price, 2)}</td>
              <td style={{ padding: "4px 8px", fontWeight: 600, color: EXIT_COLOR[t.exit_reason] || C.sub }}>
                {EXIT_LABEL[t.exit_reason] || t.exit_reason}
              </td>
              <td style={{ padding: "4px 8px", fontWeight: 600, color: t.pnl >= 0 ? C.green : C.red }}>
                {fmtUSD(t.pnl)}
              </td>
              <td style={{ padding: "4px 8px" }}>{t.contracts}</td>
              <td style={{ padding: "4px 8px" }}>{fmt(t.vol_ratio, 2)}</td>
              <td style={{ padding: "4px 8px" }}>{fmt(t.rsi, 1)}</td>
              <td style={{ padding: "4px 8px" }}>{fmt(t.adx, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Walk-forward results ─────────────────────────────────────────────────── */
function WFResults({ wf }) {
  if (!wf?.result) return null;
  const r = wf.result;
  const robust = r.robust;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8, marginBottom: 12 }}>
        <KpiTile label="PF moyen" value={fmt(r.mean_pf, 2)}
                 color={r.mean_pf >= 1.15 ? C.green : r.mean_pf >= 1.0 ? C.amber : C.red} />
        <KpiTile label="Std PF" value={fmt(r.std_pf, 2)}
                 color={r.std_pf <= 0.20 ? C.green : r.std_pf <= 0.30 ? C.amber : C.red} />
        <KpiTile label="Fenêtres profitables" value={`${r.pct_profitable}%`}
                 color={r.pct_profitable >= 75 ? C.green : C.red} sub="objectif ≥ 75%" />
        <KpiTile label="Robuste" value={robust ? "OUI" : "NON"}
                 color={robust ? C.green : C.red} sub={robust ? "Critères OK" : "Curve-fit?"} />
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr>
              {["Fenêtre", "Période", "Trades", "WR %", "PF", "P&L", "SL direct"].map(h => (
                <th key={h} style={{ padding: "4px 10px", textAlign: "left", color: C.sub,
                                      borderBottom: `1px solid ${C.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {r.windows.map((w, i) => {
              const pfCol = w.profit_factor >= 1.15 ? C.green : w.profit_factor >= 1.0 ? C.amber : C.red;
              return (
                <tr key={i} style={{ borderBottom: `1px solid ${C.border}`,
                                      background: i % 2 ? "rgba(255,255,255,0.02)" : "transparent" }}>
                  <td style={{ padding: "4px 10px", color: C.blue, fontWeight: 600 }}>#{w.window}</td>
                  <td style={{ padding: "4px 10px", color: C.sub }}>{w.start?.slice(0, 10)} → {w.end?.slice(0, 10)}</td>
                  <td style={{ padding: "4px 10px" }}>{w.n_trades}</td>
                  <td style={{ padding: "4px 10px", color: w.win_rate >= 50 ? C.green : C.amber }}>{fmt(w.win_rate, 1)}%</td>
                  <td style={{ padding: "4px 10px", color: pfCol, fontWeight: 600 }}>{fmt(w.profit_factor, 2)}</td>
                  <td style={{ padding: "4px 10px", color: w.total_pnl >= 0 ? C.green : C.red }}>{fmtUSD(w.total_pnl)}</td>
                  <td style={{ padding: "4px 10px", color: w.sl_direct_pct <= 32 ? C.green : w.sl_direct_pct <= 38 ? C.amber : C.red }}>
                    {fmt(w.sl_direct_pct, 1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ============================================================================
 * Main component
 * ========================================================================== */
export default function DashboardES({ onBack, token }) {
  const [settings,       setSettings]       = useState(null);
  const [settingsDirty,  setSettingsDirty]  = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);

  // Prétrain
  const [startDate, setStartDate] = useState("2024-01-01");
  const [endDate,   setEndDate]   = useState("2024-12-31");
  const [capital,   setCapital]   = useState(50000);
  const [riskPct,   setRiskPct]   = useState(1);
  const [progress,  setProgress]  = useState(null);
  const [result,    setResult]    = useState(null);
  const [polling,   setPolling]   = useState(false);

  // Walk-forward
  const [wfSplits,  setWfSplits]  = useState(4);
  const [wfRunning, setWfRunning] = useState(false);
  const [wfData,    setWfData]    = useState(null);
  const [wfPolling, setWfPolling] = useState(false);

  const pollRef   = useRef(null);
  const wfPollRef = useRef(null);

  /* ── Fetch initial ─────────────────────────────────────────────────────── */
  useEffect(() => {
    apiGet("/api/es/settings", token).then(d => { if (d) setSettings(d); });
    apiGet("/api/es/pretrain/result", token).then(d => { if (d?.ok) setResult(d); });
    apiGet("/api/es/pretrain/walkforward", token).then(d => {
      if (d?.result) setWfData(d);
    });
  }, [token]);

  /* ── Poll prétrain ─────────────────────────────────────────────────────── */
  useEffect(() => {
    if (!polling) return;
    pollRef.current = setInterval(async () => {
      const p = await apiGet("/api/es/pretrain/status", token);
      if (!p) return;
      setProgress(p);
      if (!p.running && (p.status === "done" || p.status === "error")) {
        clearInterval(pollRef.current);
        setPolling(false);
        if (p.status === "done" && p.last_result) setResult(p.last_result);
      }
    }, 1400);
    return () => clearInterval(pollRef.current);
  }, [polling, token]);

  /* ── Poll walk-forward ─────────────────────────────────────────────────── */
  useEffect(() => {
    if (!wfPolling) return;
    wfPollRef.current = setInterval(async () => {
      const d = await apiGet("/api/es/pretrain/walkforward", token);
      if (!d) return;
      setWfData(d);
      if (!d.running) {
        clearInterval(wfPollRef.current);
        setWfPolling(false);
        setWfRunning(false);
      }
    }, 2000);
    return () => clearInterval(wfPollRef.current);
  }, [wfPolling, token]);

  /* ── Handlers ──────────────────────────────────────────────────────────── */
  const handleParam = (k, v) => {
    setSettings(s => ({ ...s, [k]: v }));
    setSettingsDirty(true);
  };

  const handleBadHours = (raw) => {
    try {
      const parsed = raw.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n));
      handleParam("bad_hours_et", parsed);
    } catch { /* ignore */ }
  };

  const saveSettings = async () => {
    setSavingSettings(true);
    await apiPost("/api/es/settings", token, settings);
    setSavingSettings(false);
    setSettingsDirty(false);
  };

  const launchPretrain = async () => {
    if (!settings) return;
    setResult(null);
    setProgress({ running: true, pct: 0, status: "Lancement…", trades: 0, wins: 0 });
    await apiPost("/api/es/pretrain", token, {
      start: startDate, end: endDate,
      capital: Number(capital), risk_pct: Number(riskPct),
      params: settings,
    });
    setPolling(true);
  };

  const launchWF = async () => {
    setWfRunning(true);
    setWfData({ running: true, window: 0, n_splits: wfSplits, result: null });
    await apiPost("/api/es/pretrain/walkforward", token, {
      start: startDate, end: endDate,
      n_splits: Number(wfSplits),
      capital: Number(capital), risk_pct: Number(riskPct),
      params: settings,
    });
    setWfPolling(true);
  };

  /* ── Métriques ─────────────────────────────────────────────────────────── */
  const r       = result;
  const running = progress?.running;

  const wrColor  = !r ? C.grey : r.win_rate  >= 52   ? C.green : r.win_rate  >= 46   ? C.amber : C.red;
  const pfColor  = !r ? C.grey : r.profit_factor >= 1.15 ? C.green : r.profit_factor >= 1.0 ? C.amber : C.red;
  const slColor  = !r ? C.grey : r.sl_direct_pct <= 32   ? C.green : r.sl_direct_pct <= 38 ? C.amber : C.red;
  const pnlColor = !r ? C.grey : r.total_pnl >= 0 ? C.green : C.red;

  const g = (n) => ({ display: "grid", gridTemplateColumns: `repeat(${n},1fr)`, gap: 8 });

  if (!settings) return (
    <div style={{ ...S.root, display: "flex", alignItems: "center", justifyContent: "center", color: C.sub }}>
      Chargement…
    </div>
  );

  return (
    <div style={S.root}>
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <div style={S.header}>
        <button onClick={onBack}
          style={{ ...S.btn, background: "transparent", border: `1px solid ${C.border}`, color: C.sub, padding: "4px 10px", fontSize: 11 }}>
          ← XAU/USD
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 17, fontWeight: 700 }}>ES</span>
          <span style={{ color: C.sub, fontSize: 12 }}>S&P 500 E-mini — Order Flow</span>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          {running && <span style={{ color: C.amber, fontSize: 11 }}>⬤ Prétrain en cours</span>}
          {wfRunning && <span style={{ color: C.purple, fontSize: 11 }}>⬤ Walk-forward {wfData?.window || 0}/{wfSplits}</span>}
        </div>
      </div>

      {/* ── Content ──────────────────────────────────────────────────────── */}
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "16px 14px" }}>

        {/* Info NinjaTrader */}
        <div style={{ ...S.section, borderLeft: `3px solid ${C.blue}`, background: "rgba(59,130,246,0.04)", marginBottom: 14 }}>
          <div style={{ color: C.blue, fontWeight: 600, fontSize: 12, marginBottom: 4 }}>
            DOMScanner NinjaTrader — intégration live via POST /api/es/dom
          </div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontSize: 11 }}>
            {[["Tick", "0.25 pts"], ["Valeur tick", "$12.50/ctr"], ["Point", "$50/ctr"],
              ["Session RTH", "9h30–16h ET"], ["Timeout", "9 bougies (45 min)"]].map(([l, v]) => (
              <span key={l} style={{ background: C.panel2, padding: "3px 8px", borderRadius: 5 }}>
                <span style={{ color: C.sub }}>{l}: </span><b>{v}</b>
              </span>
            ))}
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "310px 1fr", gap: 14 }}>

          {/* ── Colonne gauche : paramètres ──────────────────────────────── */}
          <div>
            {/* Tendance */}
            <div style={S.section}>
              <div style={S.h2}>Filtres tendance</div>
              <ParamNum label="EMA rapide"      k="ema_fast"   settings={settings} onChange={handleParam} />
              <ParamNum label="EMA lente"       k="ema_slow"   settings={settings} onChange={handleParam} />
              <ParamNum label="EMA tendance"    k="ema_trend"  settings={settings} onChange={handleParam} />
              <ParamNum label="RSI min LONG"    k="rsi_long"   settings={settings} onChange={handleParam} />
              <ParamNum label="RSI max SHORT"   k="rsi_short"  settings={settings} onChange={handleParam} />
              <ParamNum label="ATR min (pts)"   k="atr_min_pts" settings={settings} onChange={handleParam} step={0.1} />
              <ParamNum label="ADX min"         k="adx_min"    settings={settings} onChange={handleParam} />
            </div>

            {/* Filtres avancés */}
            <div style={S.section}>
              <div style={S.h2}>Filtres avancés</div>
              <ParamToggle label="VWAP alignment"   k="vwap_filter" settings={settings} onChange={handleParam} />
              <ParamToggle label="H1 bias (EMA200)" k="h1_filter"   settings={settings} onChange={handleParam} />
              <div style={{ marginTop: 4, marginBottom: 7 }}>
                <div style={{ color: C.sub, fontSize: 12, marginBottom: 3 }}>Heures bloquées ET (ex: 10,11)</div>
                <input
                  type="text"
                  defaultValue={(settings.bad_hours_et || []).join(",")}
                  onBlur={e => handleBadHours(e.target.value)}
                  style={S.input}
                  placeholder="10"
                />
              </div>
            </div>

            {/* Volume absorption */}
            <div style={S.section}>
              <div style={S.h2}>Volume absorption</div>
              <ParamNum label="Multiplicateur vol" k="vol_multiplier"  settings={settings} onChange={handleParam} step={0.1} />
              <ParamNum label="Lookback (barres)"  k="vol_lookback"   settings={settings} onChange={handleParam} />
              <ParamNum label="Close % range LONG ≥" k="close_pct_long"  settings={settings} onChange={handleParam} step={0.05} />
              <ParamNum label="Close % range SHORT ≤" k="close_pct_short" settings={settings} onChange={handleParam} step={0.05} />
              <ParamNum label="Corps/ATR min"       k="body_ratio_min" settings={settings} onChange={handleParam} step={0.05} />
            </div>

            {/* SL/TP */}
            <div style={S.section}>
              <div style={S.h2}>SL / TP (ticks)</div>
              <ParamNum label="Stop Loss"  k="sl_ticks"  settings={settings} onChange={handleParam} />
              <ParamNum label="TP1"        k="tp1_ticks" settings={settings} onChange={handleParam} />
              <ParamNum label="TP2"        k="tp2_ticks" settings={settings} onChange={handleParam} />
              <div style={{ marginTop: 8, padding: "7px 10px", background: C.panel2, borderRadius: 5,
                            fontSize: 11, color: C.sub, lineHeight: 1.8 }}>
                SL {settings.sl_ticks}t = {(settings.sl_ticks * 0.25).toFixed(2)}pts
                  = <b style={{ color: C.red }}>${(settings.sl_ticks * 12.5).toFixed(0)}/ctr</b><br />
                TP1 {settings.tp1_ticks}t = {(settings.tp1_ticks * 0.25).toFixed(2)}pts
                  = <b style={{ color: C.green }}>${(settings.tp1_ticks * 12.5).toFixed(0)}/ctr</b><br />
                TP2 {settings.tp2_ticks}t = {(settings.tp2_ticks * 0.25).toFixed(2)}pts
                  = <b style={{ color: C.green }}>${(settings.tp2_ticks * 12.5).toFixed(0)}/ctr</b><br />
                R:R = 1 : {(settings.tp2_ticks / settings.sl_ticks).toFixed(2)}
              </div>
            </div>

            {/* Session */}
            <div style={S.section}>
              <div style={S.h2}>Session (heure ET)</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {[["Ouv. h", "session_open_h"], ["Ouv. m", "session_open_m"],
                  ["Ferm. h", "session_close_h"], ["Ferm. m", "session_close_m"]].map(([l, k]) => (
                  <div key={k}>
                    <div style={{ color: C.sub, fontSize: 10, marginBottom: 2 }}>{l}</div>
                    <input type="number" value={settings[k] ?? ""}
                           onChange={e => handleParam(k, parseInt(e.target.value))}
                           style={S.input} />
                  </div>
                ))}
              </div>
            </div>

            <button onClick={saveSettings} disabled={!settingsDirty || savingSettings}
              style={{ ...S.btn, background: settingsDirty ? C.blue : C.grey, color: "#fff",
                       width: "100%", opacity: settingsDirty ? 1 : 0.5, marginBottom: 6 }}>
              {savingSettings ? "Enregistrement…" : settingsDirty ? "Enregistrer" : "Enregistré"}
            </button>
          </div>

          {/* ── Colonne droite ───────────────────────────────────────────── */}
          <div>
            {/* Prétrain config */}
            <div style={S.section}>
              <div style={S.h2}>Prétrain ES — données ES=F (yfinance + H1 multi-TF)</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginBottom: 10 }}>
                {[["Début", "date", startDate, setStartDate],
                  ["Fin",   "date", endDate,   setEndDate]].map(([l, t, v, set]) => (
                  <div key={l}>
                    <div style={{ color: C.sub, fontSize: 10, marginBottom: 2 }}>{l}</div>
                    <input type={t} value={v} onChange={e => set(e.target.value)} style={S.input} />
                  </div>
                ))}
                <div>
                  <div style={{ color: C.sub, fontSize: 10, marginBottom: 2 }}>Capital ($)</div>
                  <input type="number" value={capital} onChange={e => setCapital(Number(e.target.value))} style={S.input} />
                </div>
                <div>
                  <div style={{ color: C.sub, fontSize: 10, marginBottom: 2 }}>Risque %</div>
                  <input type="number" value={riskPct} step={0.1} min={0.1} max={5}
                         onChange={e => setRiskPct(Number(e.target.value))} style={S.input} />
                </div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={launchPretrain} disabled={running || wfRunning}
                  style={{ ...S.btn, background: running ? C.grey : C.green, color: "#000",
                           opacity: (running || wfRunning) ? 0.5 : 1, flex: 2 }}>
                  {running ? "En cours…" : "Lancer le prétrain ES"}
                </button>
                <div style={{ display: "flex", gap: 6, alignItems: "center", flex: 1 }}>
                  <div style={{ color: C.sub, fontSize: 11, whiteSpace: "nowrap" }}>Walk-forward</div>
                  <input type="number" value={wfSplits} min={2} max={8}
                         onChange={e => setWfSplits(Number(e.target.value))}
                         style={{ ...S.input, width: 50, textAlign: "center" }} />
                  <span style={{ color: C.sub, fontSize: 11 }}>fen.</span>
                  <button onClick={launchWF} disabled={running || wfRunning}
                    style={{ ...S.btn, background: wfRunning ? C.grey : C.purple, color: "#fff",
                             opacity: (running || wfRunning) ? 0.5 : 1, whiteSpace: "nowrap" }}>
                    {wfRunning ? `${wfData?.window || 0}/${wfSplits}` : "WF →"}
                  </button>
                </div>
              </div>
              {progress && (
                <ProgressBar pct={progress.pct || 0}
                  status={`${progress.status || ""}  —  ${progress.trades || 0} trades  ${progress.wins || 0}W`} />
              )}
              {wfRunning && wfData && (
                <ProgressBar pct={Math.round((wfData.window || 0) / wfSplits * 100)}
                  status={`Walk-forward — fenêtre ${wfData.window}/${wfSplits}`}
                  color={C.purple} />
              )}
            </div>

            {/* Walk-forward results */}
            {wfData?.result && (
              <div style={S.section}>
                <div style={S.h2}>Walk-Forward — validation OOS ({wfData.result.windows?.length} fenêtres)</div>
                <WFResults wf={wfData} />
                <div style={{ marginTop: 8, padding: "7px 10px", background: C.panel2, borderRadius: 5, fontSize: 11, color: C.sub }}>
                  Critères robustesse : PF moyen &gt; 1.0 ET std_pf &lt; 0.30 ET ≥ 75% fenêtres profitables.
                  {wfData.result.robust
                    ? <span style={{ color: C.green, marginLeft: 6, fontWeight: 600 }}>✓ Stratégie validée OOS</span>
                    : <span style={{ color: C.red,   marginLeft: 6, fontWeight: 600 }}>✗ Ajuster les paramètres</span>}
                </div>
              </div>
            )}

            {/* KPI résultats */}
            {r && (
              <>
                <div style={S.section}>
                  <div style={S.h2}>Résultats prétrain
                    {r.data_start && <span style={{ color: C.sub, marginLeft: 8, fontWeight: 400 }}>
                      {r.data_start} → {r.data_end} ({r.bars_total?.toLocaleString()} barres M5)
                    </span>}
                  </div>

                  <div style={{ ...g(5), marginBottom: 10 }}>
                    <KpiTile label="Trades"        value={r.n_trades}  sub={`${r.n_wins}W / ${r.n_losses}L`} />
                    <KpiTile label="Win Rate"      value={`${fmt(r.win_rate, 1)}%`}    color={wrColor} />
                    <KpiTile label="Profit Factor" value={fmt(r.profit_factor, 2)}      color={pfColor} />
                    <KpiTile label="P&L total"     value={fmtUSD(r.total_pnl)}          color={pnlColor} />
                    <KpiTile label="SL direct %"   value={`${fmt(r.sl_direct_pct, 1)}%`} color={slColor} sub="obj < 32%" />
                  </div>
                  <div style={{ ...g(4), marginBottom: 10 }}>
                    <KpiTile label="Drawdown max"  value={`$${Number(r.max_dd || 0).toFixed(0)}`}  color={C.amber} sub={`${fmt(r.max_dd_pct, 1)}%`} />
                    <KpiTile label="Gain moyen"    value={`$${fmt(r.avg_win, 0)}`}   color={C.green} />
                    <KpiTile label="Perte moyenne" value={`$${fmt(r.avg_loss, 0)}`}  color={C.red} />
                    <KpiTile label="TP2 %"         value={`${fmt(r.tp2_pct, 1)}%`}   color={C.purple} />
                  </div>

                  {/* Direction */}
                  <div style={{ ...g(2), marginBottom: 10 }}>
                    <div style={{ ...S.kpi, borderLeft: `3px solid ${C.green}` }}>
                      <div style={{ fontSize: 10, color: C.sub, marginBottom: 3 }}>LONG</div>
                      <div style={{ fontSize: 16, fontWeight: 700 }}>{r.long_trades} trades</div>
                      <div style={{ color: C.green, fontSize: 11 }}>
                        {r.long_trades > 0 ? Math.round(r.long_wins / r.long_trades * 100) : 0}% WR ({r.long_wins}W)
                      </div>
                    </div>
                    <div style={{ ...S.kpi, borderLeft: `3px solid ${C.red}` }}>
                      <div style={{ fontSize: 10, color: C.sub, marginBottom: 3 }}>SHORT</div>
                      <div style={{ fontSize: 16, fontWeight: 700 }}>{r.short_trades} trades</div>
                      <div style={{ color: C.red, fontSize: 11 }}>
                        {r.short_trades > 0 ? Math.round(r.short_wins / r.short_trades * 100) : 0}% WR ({r.short_wins}W)
                      </div>
                    </div>
                  </div>

                  {/* Equity curve */}
                  <EquityCurve data={r.equity_curve} />
                </div>

                {/* Heatmap horaire */}
                {r.hourly && Object.keys(r.hourly).length > 0 && (
                  <div style={S.section}>
                    <div style={S.h2}>Répartition horaire (ET) — vert ≥ 55% WR · orange 45–55% · rouge &lt; 45%</div>
                    <HourlyChart data={r.hourly} />
                  </div>
                )}

                {/* Diagnostic tables */}
                {(r.diag_by_exit || r.diag_by_bias) && (
                  <div style={S.section}>
                    <div style={S.h2}>Diagnostic — indicateurs moyens par groupe</div>
                    {r.diag_by_exit  && <DiagTable data={r.diag_by_exit}  title="Par sortie"    keyLabel="Raison"    />}
                    {r.diag_by_bias  && <DiagTable data={r.diag_by_bias}  title="Par direction" keyLabel="Direction" />}
                    {r.diag_by_hour  && <DiagTable data={r.diag_by_hour}  title="Par heure ET"  keyLabel="Heure"     />}
                  </div>
                )}

                {/* Table trades */}
                {r.trades && (
                  <div style={S.section}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                      <div style={S.h2}>Derniers trades</div>
                      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                        {[["tp2","TP2",C.green],["tp_direct","TP↑",C.green],["timeout_tp1","TO+TP1",C.amber],
                          ["sl_after_tp1","SL/BE",C.amber],["sl","SL",C.red],["timeout","TO",C.sub]].map(([k, l, col]) => {
                          const n = r.trades.filter(t => t.exit_reason === k).length;
                          return (
                            <span key={k} style={{ background: C.panel2, borderRadius: 5, padding: "2px 8px", fontSize: 10 }}>
                              <span style={{ color: col, fontWeight: 600 }}>{l}</span>
                              <span style={{ color: C.sub, marginLeft: 4 }}>{n} ({r.trades.length > 0 ? Math.round(n / r.trades.length * 100) : 0}%)</span>
                            </span>
                          );
                        })}
                      </div>
                    </div>
                    <TradeTable trades={r.trades} />
                  </div>
                )}
              </>
            )}

            {!r && !running && (
              <div style={{ ...S.section, textAlign: "center", color: C.sub, padding: 36 }}>
                <div style={{ fontSize: 28, marginBottom: 8 }}>📊</div>
                <div style={{ fontSize: 13 }}>Configure les paramètres et lance un prétrain</div>
                <div style={{ fontSize: 11, marginTop: 4 }}>Multi-TF M5+H1 · VWAP · ADX · bad_hours · walk-forward</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
