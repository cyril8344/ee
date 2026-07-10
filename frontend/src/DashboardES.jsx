import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, CartesianGrid,
} from "recharts";

/* ============================================================================
 * ES (S&P 500 E-mini) — Dashboard Order Flow
 * Dashboard indépendant pour la stratégie order flow ES.
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
};

const fmt  = (n, d = 2) =>
  n == null || isNaN(n) ? "—" : Number(n).toFixed(d);
const fmtUSD = (n) =>
  n == null || isNaN(n) ? "—" :
  (n >= 0 ? "+" : "") + "$" + Number(Math.abs(n)).toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 });
const fmtTime = (iso) => {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("fr-FR", {
      month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit",
    });
  } catch { return iso.slice(0, 16); }
};

function apiGet(path, token) {
  return fetch(API + path, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  }).then(r => r.ok ? r.json() : null);
}
function apiPost(path, token, body) {
  return fetch(API + path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
  }).then(r => r.ok ? r.json() : null);
}

/* ── Helpers UI ────────────────────────────────────────────────────────────── */
const S = {
  root: {
    background: C.bg, color: C.text, minHeight: "100vh",
    fontFamily: "'Inter', system-ui, sans-serif", fontSize: 14,
  },
  header: {
    background: C.panel, borderBottom: `1px solid ${C.border}`,
    padding: "12px 20px", display: "flex", alignItems: "center", gap: 16,
  },
  section: {
    background: C.panel, border: `1px solid ${C.border}`,
    borderRadius: 10, padding: 20, marginBottom: 16,
  },
  h2: { fontSize: 13, fontWeight: 700, color: C.sub, textTransform: "uppercase",
        letterSpacing: "0.08em", marginBottom: 14 },
  input: {
    background: C.panel2, border: `1px solid ${C.border}`,
    borderRadius: 6, color: C.text, padding: "6px 10px", fontSize: 13,
    width: "100%", boxSizing: "border-box",
  },
  btn: {
    padding: "8px 18px", borderRadius: 7, border: "none",
    fontWeight: 600, cursor: "pointer", fontSize: 13,
  },
  kpi: {
    background: C.panel2, borderRadius: 8, padding: "14px 18px",
    border: `1px solid ${C.border}`, textAlign: "center",
  },
};

/* ── KPI Tile ────────────────────────────────────────────────────────────── */
function KpiTile({ label, value, color, sub }) {
  return (
    <div style={S.kpi}>
      <div style={{ color: C.sub, fontSize: 11, marginBottom: 4, textTransform: "uppercase",
                    letterSpacing: "0.06em" }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || C.text }}>{value}</div>
      {sub && <div style={{ color: C.sub, fontSize: 11, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

/* ── Progress bar ────────────────────────────────────────────────────────── */
function ProgressBar({ pct, status }) {
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: C.sub, fontSize: 12 }}>{status}</span>
        <span style={{ color: C.text, fontSize: 12, fontWeight: 600 }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: C.panel2, borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: C.blue,
                      borderRadius: 3, transition: "width 0.3s" }} />
      </div>
    </div>
  );
}

/* ── Paramètre numérique inline ───────────────────────────────────────────── */
function ParamRow({ label, k, settings, onChange }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
      <label style={{ color: C.sub, fontSize: 12, flex: 1, minWidth: 140 }}>{label}</label>
      <input
        type="number"
        value={settings[k] ?? ""}
        onChange={e => onChange(k, parseFloat(e.target.value))}
        style={{ ...S.input, width: 90, textAlign: "right" }}
      />
    </div>
  );
}

/* ── Equity curve ─────────────────────────────────────────────────────────── */
function EquityCurve({ data }) {
  if (!data || data.length < 2) return (
    <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center",
                  color: C.sub, fontSize: 13 }}>
      Aucune donnée — lancer un prétrain
    </div>
  );
  const start = data[0]?.equity || 0;
  const end   = data[data.length - 1]?.equity || 0;
  const col   = end >= start ? C.green : C.red;
  return (
    <ResponsiveContainer width="100%" height={180}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="esEqGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={col} stopOpacity={0.3} />
            <stop offset="95%" stopColor={col} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <XAxis dataKey="ts" hide />
        <YAxis domain={["auto", "auto"]} hide />
        <Tooltip
          contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6 }}
          formatter={(v) => ["$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 0 }), "Equity"]}
          labelFormatter={() => ""}
        />
        <Area type="monotone" dataKey="equity" stroke={col} strokeWidth={2}
              fill="url(#esEqGrad)" dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ── Heatmap horaire (ET) ─────────────────────────────────────────────────── */
function HourlyHeatmap({ data }) {
  if (!data || Object.keys(data).length === 0) return null;
  const entries = Object.entries(data).map(([h, v]) => ({
    h: parseInt(h), ...v,
    wr: v.trades > 0 ? Math.round(v.wins / v.trades * 100) : 0,
  }));
  const maxT = Math.max(...entries.map(e => e.trades), 1);
  return (
    <ResponsiveContainer width="100%" height={100}>
      <BarChart data={entries} margin={{ top: 8, right: 0, bottom: 4, left: 0 }}>
        <XAxis dataKey="h" tickFormatter={h => `${h}h`}
               tick={{ fill: C.sub, fontSize: 10 }} />
        <YAxis hide />
        <Tooltip
          contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6 }}
          formatter={(v, _, { payload }) =>
            [`${payload.wins}W / ${payload.trades - payload.wins}L (${payload.wr}% WR)`, "Trades"]}
          labelFormatter={h => `${h}h ET`}
        />
        <Bar dataKey="trades" radius={[3, 3, 0, 0]}>
          {entries.map((e, i) => (
            <Cell key={i} fill={e.wr >= 55 ? C.green : e.wr >= 45 ? C.amber : e.wr > 0 ? C.red : C.grey} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ── Table trades ─────────────────────────────────────────────────────────── */
function TradeTable({ trades }) {
  if (!trades || trades.length === 0) return (
    <div style={{ color: C.sub, fontSize: 13, padding: "20px 0", textAlign: "center" }}>
      Aucun trade — lancer un prétrain
    </div>
  );
  const cols = [
    { k: "entry_ts",    label: "Entrée",     render: (v) => fmtTime(v) },
    { k: "direction",   label: "Sens",       render: (v) => (
      <span style={{ color: v === "long" ? C.green : C.red, fontWeight: 600 }}>
        {v?.toUpperCase()}
      </span>
    )},
    { k: "entry",      label: "Entrée $",   render: (v) => fmt(v, 2) },
    { k: "exit_price", label: "Sortie $",   render: (v) => fmt(v, 2) },
    { k: "exit_reason", label: "Raison",    render: (v) => {
      const col = v === "tp2" ? C.green : v === "sl" ? C.red : v === "sl_after_tp1" ? C.amber : C.sub;
      const lbl = v === "tp2" ? "TP2" : v === "sl" ? "SL" : v === "sl_after_tp1" ? "SL/BE" :
                  v === "timeout" ? "TO" : v === "timeout_tp1" ? "TO+TP1" : v || "?";
      return <span style={{ color: col, fontWeight: 600 }}>{lbl}</span>;
    }},
    { k: "pnl",        label: "P&L",        render: (v) => (
      <span style={{ color: v >= 0 ? C.green : C.red, fontWeight: 600 }}>
        {fmtUSD(v)}
      </span>
    )},
    { k: "contracts",  label: "Contrats",   render: (v) => v },
    { k: "vol_ratio",  label: "Vol/moy",    render: (v) => fmt(v, 2) },
    { k: "rsi",        label: "RSI",        render: (v) => fmt(v, 1) },
  ];

  return (
    <div style={{ overflowX: "auto", maxHeight: 380, overflowY: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            {cols.map(c => (
              <th key={c.k} style={{ padding: "6px 10px", textAlign: "left",
                                     color: C.sub, borderBottom: `1px solid ${C.border}`,
                                     position: "sticky", top: 0, background: C.panel }}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...trades].reverse().map((t, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${C.border}`,
                                  background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.02)" }}>
              {cols.map(c => (
                <td key={c.k} style={{ padding: "5px 10px" }}>
                  {c.render ? c.render(t[c.k]) : (t[c.k] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ============================================================================
 * Main component
 * ========================================================================== */
export default function DashboardES({ onBack, token }) {
  const [settings, setSettings] = useState(null);
  const [settingsDirty, setSettingsDirty] = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);

  const [startDate, setStartDate] = useState("2024-01-01");
  const [endDate,   setEndDate]   = useState("2024-12-31");
  const [capital,   setCapital]   = useState(50000);
  const [riskPct,   setRiskPct]   = useState(1);

  const [progress, setProgress] = useState(null);
  const [result,   setResult]   = useState(null);
  const [polling,  setPolling]  = useState(false);

  const pollRef = useRef(null);

  /* ── Fetch settings ──────────────────────────────────────────────────────── */
  useEffect(() => {
    apiGet("/api/es/settings", token).then(d => {
      if (d) setSettings(d);
    });
    // On startup, also check if a result exists
    apiGet("/api/es/pretrain/result", token).then(d => {
      if (d && d.ok) setResult(d);
    });
  }, [token]);

  /* ── Poll progress while running ─────────────────────────────────────────── */
  useEffect(() => {
    if (!polling) return;
    pollRef.current = setInterval(async () => {
      const prog = await apiGet("/api/es/pretrain/status", token);
      if (!prog) return;
      setProgress(prog);
      if (!prog.running && (prog.status === "done" || prog.status === "error")) {
        clearInterval(pollRef.current);
        setPolling(false);
        if (prog.status === "done" && prog.last_result) {
          setResult(prog.last_result);
        }
      }
    }, 1200);
    return () => clearInterval(pollRef.current);
  }, [polling, token]);

  /* ── Handlers ─────────────────────────────────────────────────────────────── */
  const handleParamChange = (k, v) => {
    setSettings(s => ({ ...s, [k]: v }));
    setSettingsDirty(true);
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
    const res = await apiPost("/api/es/pretrain", token, {
      start: startDate, end: endDate,
      capital: Number(capital), risk_pct: Number(riskPct),
      params: settings,
    });
    if (res) {
      setPolling(true);
    }
  };

  /* ── Métriques ────────────────────────────────────────────────────────────── */
  const r = result;
  const running = progress?.running;

  const wrColor = !r ? C.grey :
    r.win_rate >= 52 ? C.green : r.win_rate >= 46 ? C.amber : C.red;
  const pfColor = !r ? C.grey :
    r.profit_factor >= 1.15 ? C.green : r.profit_factor >= 1.0 ? C.amber : C.red;
  const slColor = !r ? C.grey :
    r.sl_direct_pct <= 32 ? C.green : r.sl_direct_pct <= 38 ? C.amber : C.red;
  const pnlColor = !r ? C.grey : r.total_pnl >= 0 ? C.green : C.red;

  const grid = (cols) => ({
    display: "grid",
    gridTemplateColumns: `repeat(${cols}, 1fr)`,
    gap: 10,
  });

  if (!settings) {
    return (
      <div style={{ ...S.root, display: "flex", alignItems: "center",
                    justifyContent: "center", color: C.sub }}>
        Chargement…
      </div>
    );
  }

  return (
    <div style={S.root}>
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div style={S.header}>
        <button onClick={onBack}
          style={{ ...S.btn, background: "transparent", border: `1px solid ${C.border}`,
                   color: C.sub, padding: "5px 12px", fontSize: 12 }}>
          ← XAU/USD
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 18, fontWeight: 700 }}>ES</span>
          <span style={{ color: C.sub, fontSize: 13 }}>S&P 500 E-mini — Stratégie Order Flow</span>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%",
                         background: running ? C.amber : C.grey,
                         display: "inline-block" }} />
          <span style={{ color: C.sub, fontSize: 12 }}>
            {running ? "Prétrain en cours…" : "En veille"}
          </span>
        </div>
      </div>

      {/* ── Content ─────────────────────────────────────────────────────── */}
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "20px 16px" }}>

        {/* ── Info NinjaTrader ──────────────────────────────────────────── */}
        <div style={{ ...S.section, borderLeft: `3px solid ${C.blue}`,
                      background: "rgba(59,130,246,0.05)", marginBottom: 16 }}>
          <div style={{ color: C.blue, fontWeight: 600, marginBottom: 6, fontSize: 13 }}>
            DOMScanner NinjaTrader — Intégration live
          </div>
          <div style={{ color: C.sub, fontSize: 12, lineHeight: 1.6 }}>
            Le DOMScanner.cs détecte l'absorption réelle sur le carnet d'ordres ES et envoie
            un signal via <code style={{ background: C.panel2, padding: "1px 5px", borderRadius: 3 }}>
            POST /api/es/dom</code>.{" "}
            En backtest, un proxy volume (volume &gt; 2× moyenne + close directionnelle) remplace ce signal.
          </div>
          <div style={{ display: "flex", gap: 16, marginTop: 10, flexWrap: "wrap" }}>
            {[
              { label: "Tick ES", val: "0.25 pts" },
              { label: "Valeur tick", val: "$12.50 / contrat" },
              { label: "Valeur point", val: "$50 / contrat" },
              { label: "Session RTH", val: "9h30 – 16h00 ET" },
            ].map(({ label, val }) => (
              <div key={label} style={{ background: C.panel2, borderRadius: 6,
                                        padding: "6px 12px", fontSize: 12 }}>
                <span style={{ color: C.sub }}>{label}: </span>
                <span style={{ fontWeight: 600 }}>{val}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: 16 }}>

          {/* ── Colonne gauche : paramètres ─────────────────────────────── */}
          <div>
            {/* Tendance */}
            <div style={S.section}>
              <div style={S.h2}>Filtres tendance</div>
              {[
                ["EMA rapide", "ema_fast"],
                ["EMA lente", "ema_slow"],
                ["EMA tendance", "ema_trend"],
                ["RSI seuil LONG (min)", "rsi_long"],
                ["RSI seuil SHORT (max)", "rsi_short"],
                ["ATR minimum (points)", "atr_min_pts"],
              ].map(([label, k]) => (
                <ParamRow key={k} label={label} k={k}
                          settings={settings} onChange={handleParamChange} />
              ))}
            </div>

            {/* Volume absorption */}
            <div style={S.section}>
              <div style={S.h2}>Volume absorption</div>
              {[
                ["Multiplicateur volume", "vol_multiplier"],
                ["Lookback (barres)", "vol_lookback"],
                ["Close % range LONG (≥)", "close_pct_long"],
                ["Close % range SHORT (≤)", "close_pct_short"],
              ].map(([label, k]) => (
                <ParamRow key={k} label={label} k={k}
                          settings={settings} onChange={handleParamChange} />
              ))}
            </div>

            {/* SL/TP */}
            <div style={S.section}>
              <div style={S.h2}>SL / TP (ticks)</div>
              {[
                ["Stop Loss (ticks)", "sl_ticks"],
                ["TP1 (ticks)", "tp1_ticks"],
                ["TP2 (ticks)", "tp2_ticks"],
              ].map(([label, k]) => (
                <ParamRow key={k} label={label} k={k}
                          settings={settings} onChange={handleParamChange} />
              ))}
              <div style={{ marginTop: 10, padding: "8px 12px", background: C.panel2,
                            borderRadius: 6, fontSize: 11, color: C.sub, lineHeight: 1.7 }}>
                SL {settings.sl_ticks} ticks = {(settings.sl_ticks * 0.25).toFixed(2)} pts
                = ${(settings.sl_ticks * 12.5).toFixed(0)}/contrat<br />
                TP1 {settings.tp1_ticks} ticks = {(settings.tp1_ticks * 0.25).toFixed(2)} pts
                = ${(settings.tp1_ticks * 12.5).toFixed(0)}/contrat<br />
                TP2 {settings.tp2_ticks} ticks = {(settings.tp2_ticks * 0.25).toFixed(2)} pts
                = ${(settings.tp2_ticks * 12.5).toFixed(0)}/contrat
              </div>
            </div>

            {/* Session */}
            <div style={S.section}>
              <div style={S.h2}>Session (heure ET)</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {[
                  ["Ouvert h", "session_open_h"],
                  ["Ouvert m", "session_open_m"],
                  ["Fermé h", "session_close_h"],
                  ["Fermé m", "session_close_m"],
                ].map(([label, k]) => (
                  <div key={k}>
                    <div style={{ color: C.sub, fontSize: 11, marginBottom: 3 }}>{label}</div>
                    <input type="number" value={settings[k] ?? ""}
                           onChange={e => handleParamChange(k, parseInt(e.target.value))}
                           style={{ ...S.input, textAlign: "right" }} />
                  </div>
                ))}
              </div>
            </div>

            <button onClick={saveSettings} disabled={!settingsDirty || savingSettings}
              style={{ ...S.btn, background: settingsDirty ? C.blue : C.grey,
                       color: "#fff", width: "100%", opacity: settingsDirty ? 1 : 0.5 }}>
              {savingSettings ? "Enregistrement…" : settingsDirty ? "Enregistrer les paramètres" : "Paramètres enregistrés"}
            </button>
          </div>

          {/* ── Colonne droite : prétrain + résultats ────────────────────── */}
          <div>
            {/* Prétrain config */}
            <div style={S.section}>
              <div style={S.h2}>Prétrain ES — Données historiques ES=F</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
                <div>
                  <div style={{ color: C.sub, fontSize: 11, marginBottom: 3 }}>Début</div>
                  <input type="date" value={startDate}
                         onChange={e => setStartDate(e.target.value)}
                         style={S.input} />
                </div>
                <div>
                  <div style={{ color: C.sub, fontSize: 11, marginBottom: 3 }}>Fin</div>
                  <input type="date" value={endDate}
                         onChange={e => setEndDate(e.target.value)}
                         style={S.input} />
                </div>
                <div>
                  <div style={{ color: C.sub, fontSize: 11, marginBottom: 3 }}>Capital ($)</div>
                  <input type="number" value={capital}
                         onChange={e => setCapital(Number(e.target.value))}
                         style={S.input} />
                </div>
                <div>
                  <div style={{ color: C.sub, fontSize: 11, marginBottom: 3 }}>Risque %</div>
                  <input type="number" value={riskPct} min={0.1} max={5} step={0.1}
                         onChange={e => setRiskPct(Number(e.target.value))}
                         style={S.input} />
                </div>
              </div>
              <button onClick={launchPretrain} disabled={running}
                style={{ ...S.btn, background: running ? C.grey : C.green, color: "#000",
                         opacity: running ? 0.6 : 1 }}>
                {running ? "En cours…" : "Lancer le prétrain ES"}
              </button>
              {progress && (
                <ProgressBar
                  pct={progress.pct || 0}
                  status={`${progress.status || ""} — ${progress.trades || 0} trades détectés`}
                />
              )}
            </div>

            {/* Résultats KPI */}
            {r && (
              <>
                <div style={S.section}>
                  <div style={S.h2}>Résultats prétrain</div>
                  <div style={{ ...grid(5), marginBottom: 12 }}>
                    <KpiTile label="Trades" value={r.n_trades}
                             sub={`${r.n_wins}W / ${r.n_losses}L`} />
                    <KpiTile label="Win Rate" value={`${fmt(r.win_rate, 1)}%`}
                             color={wrColor} />
                    <KpiTile label="Profit Factor" value={fmt(r.profit_factor, 2)}
                             color={pfColor} />
                    <KpiTile label="P&L total" value={fmtUSD(r.total_pnl)}
                             color={pnlColor} />
                    <KpiTile label="SL direct %" value={`${fmt(r.sl_direct_pct, 1)}%`}
                             color={slColor} sub="objectif < 32%" />
                  </div>
                  <div style={{ ...grid(4), marginBottom: 12 }}>
                    <KpiTile label="Drawdown max" value={`$${Number(r.max_dd || 0).toFixed(0)}`}
                             sub={`${fmt(r.max_dd_pct, 1)}%`} color={C.amber} />
                    <KpiTile label="Gain moyen" value={`$${fmt(r.avg_win, 0)}`} color={C.green} />
                    <KpiTile label="Perte moyenne" value={`$${fmt(r.avg_loss, 0)}`} color={C.red} />
                    <KpiTile label="TP2 %" value={`${fmt(r.tp2_pct, 1)}%`} color={C.purple} />
                  </div>

                  {/* Direction */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
                    <div style={{ ...S.kpi, borderLeft: `3px solid ${C.green}` }}>
                      <div style={{ fontSize: 11, color: C.sub, marginBottom: 4 }}>LONG</div>
                      <div style={{ fontSize: 17, fontWeight: 700 }}>{r.long_trades} trades</div>
                      <div style={{ color: C.green, fontSize: 12 }}>
                        {r.long_trades > 0 ? Math.round(r.long_wins / r.long_trades * 100) : 0}% WR
                        ({r.long_wins}W)
                      </div>
                    </div>
                    <div style={{ ...S.kpi, borderLeft: `3px solid ${C.red}` }}>
                      <div style={{ fontSize: 11, color: C.sub, marginBottom: 4 }}>SHORT</div>
                      <div style={{ fontSize: 17, fontWeight: 700 }}>{r.short_trades} trades</div>
                      <div style={{ color: C.red, fontSize: 12 }}>
                        {r.short_trades > 0 ? Math.round(r.short_wins / r.short_trades * 100) : 0}% WR
                        ({r.short_wins}W)
                      </div>
                    </div>
                  </div>

                  {/* Période */}
                  {r.data_start && (
                    <div style={{ color: C.sub, fontSize: 11, marginBottom: 4 }}>
                      Données : {r.data_start} → {r.data_end} ({r.bars_total?.toLocaleString()} barres M5)
                    </div>
                  )}

                  {/* Equity curve */}
                  <div style={{ marginTop: 8 }}>
                    <div style={{ color: C.sub, fontSize: 11, marginBottom: 6 }}>Equity curve</div>
                    <EquityCurve data={r.equity_curve} />
                  </div>
                </div>

                {/* Heatmap horaire ET */}
                {r.hourly && Object.keys(r.hourly).length > 0 && (
                  <div style={S.section}>
                    <div style={S.h2}>Répartition horaire (ET)</div>
                    <HourlyHeatmap data={r.hourly} />
                    <div style={{ color: C.sub, fontSize: 11, marginTop: 4 }}>
                      Couleur : vert ≥ 55% WR · orange 45–55% · rouge &lt; 45%
                    </div>
                  </div>
                )}

                {/* Table des exits */}
                {r.trades && (
                  <div style={S.section}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                      <div style={S.h2}>Derniers trades</div>
                      <div style={{ fontSize: 12, color: C.sub }}>
                        (affichage 200 derniers)
                      </div>
                    </div>

                    {/* Mini stats sorties */}
                    {(() => {
                      const tl = r.trades;
                      const tot = tl.length;
                      if (tot === 0) return null;
                      const byReason = tl.reduce((acc, t) => {
                        acc[t.exit_reason] = (acc[t.exit_reason] || 0) + 1;
                        return acc;
                      }, {});
                      return (
                        <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                          {[
                            { k: "tp2",           label: "TP2",    col: C.green },
                            { k: "tp_direct",     label: "TP dir", col: C.green },
                            { k: "timeout_tp1",   label: "TO+TP1", col: C.amber },
                            { k: "sl_after_tp1",  label: "SL/BE",  col: C.amber },
                            { k: "sl",            label: "SL",     col: C.red },
                            { k: "timeout",       label: "TO",     col: C.sub },
                          ].map(({ k, label, col }) => (
                            <div key={k} style={{ background: C.panel2, borderRadius: 5,
                                                   padding: "4px 10px", fontSize: 11 }}>
                              <span style={{ color: col, fontWeight: 600 }}>{label}</span>
                              <span style={{ color: C.sub, marginLeft: 6 }}>
                                {byReason[k] || 0} ({tot > 0 ? Math.round((byReason[k] || 0) / tot * 100) : 0}%)
                              </span>
                            </div>
                          ))}
                        </div>
                      );
                    })()}

                    <TradeTable trades={r.trades} />
                  </div>
                )}
              </>
            )}

            {/* Placeholder si pas encore de résultat */}
            {!r && !running && (
              <div style={{ ...S.section, textAlign: "center", color: C.sub, padding: 40 }}>
                <div style={{ fontSize: 32, marginBottom: 10 }}>📊</div>
                <div style={{ fontSize: 14 }}>Configure les paramètres et lance un prétrain</div>
                <div style={{ fontSize: 12, marginTop: 6 }}>
                  Les données ES=F seront téléchargées via yfinance
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
