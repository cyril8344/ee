import React, { useState } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, BarChart, Bar, Cell,
} from "recharts";

/* ============================================================================
 * Backtest panel — XAU/USD M5 scalping
 * ==========================================================================*/

const COLORS = {
  bg: "#0a0e17", panel: "#121826", panel2: "#0f1420", border: "#1f2937",
  text: "#e5e7eb", sub: "#8b95a7", green: "#16c784", red: "#ea3943",
  blue: "#3b82f6", amber: "#f59e0b", grey: "#6b7280",
};

const fmt = (n, d = 2) =>
  n === null || n === undefined || isNaN(n) ? "—" : Number(n).toFixed(d);
const money = (n) =>
  n === null || n === undefined ? "—" : (n >= 0 ? "+$" : "-$") + Math.abs(n).toFixed(2);

function defaultDates() {
  const end = new Date();
  const start = new Date();
  start.setMonth(start.getMonth() - 6);
  return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
}

/* ============================================================================
 * Optimiser panel
 * ==========================================================================*/
function OptimizerPanel({ api }) {
  const d = defaultDates();
  const [form, setForm] = useState({
    start: d.start, end: d.end, symbol: "XAUUSD",
    capital: 10000, risk_pct: 1.0,
    spread_pips: 0.3, slippage_pips: 0.1,
    max_trades_per_day: 4, daily_stop_pct: 2.0,
  });
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [applyStatus, setApplyStatus] = useState({});

  const update = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const run = async () => {
    setLoading(true); setErr(null); setRes(null); setApplyStatus({});
    try {
      const r = await fetch(`${api}/api/optimize`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          capital: Number(form.capital), risk_pct: Number(form.risk_pct),
          spread_pips: Number(form.spread_pips), slippage_pips: Number(form.slippage_pips),
          max_trades_per_day: Number(form.max_trades_per_day),
          daily_stop_pct: Number(form.daily_stop_pct),
        }),
      });
      const data = await r.json();
      if (data.detail) setErr(data.detail);
      else setRes(data);
    } catch (e) {
      setErr("Échec de la requête optimisation: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const applyParams = async (idx, params) => {
    setApplyStatus((s) => ({ ...s, [idx]: "loading" }));
    try {
      const r = await fetch(`${api}/api/optimize/apply`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          adx_min: params.adx_min,
          rsi_low: params.rsi_low,
          rsi_high: params.rsi_high,
          sl_atr_mult: params.sl_atr_mult,
          sr_proximity: params.sr_proximity,
        }),
      });
      if (r.ok) setApplyStatus((s) => ({ ...s, [idx]: "ok" }));
      else setApplyStatus((s) => ({ ...s, [idx]: "err" }));
    } catch {
      setApplyStatus((s) => ({ ...s, [idx]: "err" }));
    }
  };

  return (
    <div>
      {/* Parameters */}
      <div style={panel()}>
        <h3 style={{ margin: "0 0 12px", fontSize: 14 }}>Paramètres de l'optimisation (Grid Search)</h3>
        <div style={{ fontSize: 12, color: COLORS.sub, marginBottom: 12 }}>
          Grille de recherche : ADX_MIN × RSI_LOW × RSI_HIGH × SL_ATR_MULT × SR_PROXIMITY —{" "}
          <strong style={{ color: COLORS.amber }}>108 combinaisons</strong>, optimise le Sharpe × Profit Factor.
          Peut prendre 2 à 5 minutes selon la période.
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
          <Field label="Date début"><input type="date" value={form.start}
            onChange={(e) => update("start", e.target.value)} style={inp} /></Field>
          <Field label="Date fin"><input type="date" value={form.end}
            onChange={(e) => update("end", e.target.value)} style={inp} /></Field>
          <Field label="Capital ($)"><input type="number" value={form.capital}
            onChange={(e) => update("capital", e.target.value)} style={inp} /></Field>
          <Field label="Symbole">
            <select value={form.symbol} onChange={(e) => update("symbol", e.target.value)} style={inp}>
              <option value="XAUUSD">XAU/USD</option>
              <option value="EURUSD">EUR/USD</option>
            </select>
          </Field>
          <Field label="Risque / trade (%)"><input type="number" step="0.1" value={form.risk_pct}
            onChange={(e) => update("risk_pct", e.target.value)} style={inp} /></Field>
          <Field label="Spread (pips)"><input type="number" step="0.1" value={form.spread_pips}
            onChange={(e) => update("spread_pips", e.target.value)} style={inp} /></Field>
          <Field label="Slippage (pips)"><input type="number" step="0.1" value={form.slippage_pips}
            onChange={(e) => update("slippage_pips", e.target.value)} style={inp} /></Field>
          <Field label="Max trades / jour"><input type="number" value={form.max_trades_per_day}
            onChange={(e) => update("max_trades_per_day", e.target.value)} style={inp} /></Field>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 14 }}>
          <button onClick={run} disabled={loading} style={{
            background: loading ? COLORS.amber : COLORS.blue, color: "#fff", border: "none",
            borderRadius: 6, padding: "9px 22px", fontSize: 14, fontWeight: 600,
            cursor: loading ? "wait" : "pointer", opacity: 1, display: "flex", alignItems: "center", gap: 8,
          }}>
            {loading && (
              <span style={{
                display: "inline-block", width: 14, height: 14, border: "2px solid #fff3",
                borderTop: "2px solid #fff", borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
              }} />
            )}
            {loading ? "Optimisation en cours…" : "Lancer l'optimisation"}
          </button>
          {loading && (
            <span style={{ fontSize: 12, color: COLORS.amber }}>
              Calcul de 108 combinaisons, veuillez patienter…
            </span>
          )}
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>

      {err && (
        <div style={{ ...panel(), marginTop: 14, borderColor: COLORS.red, color: COLORS.red }}>
          {err}
        </div>
      )}

      {res && (
        <>
          {/* Summary */}
          <div style={{ ...panel(), marginTop: 14 }}>
            <div style={{ display: "flex", gap: 24, fontSize: 13 }}>
              <span>Combinaisons totales : <strong>{res.total_combinations}</strong></span>
              <span>Testées (min 10 trades) : <strong>{res.tested}</strong></span>
              {res.best && (
                <span style={{ color: COLORS.green }}>
                  Meilleur score : Sharpe {fmt(res.best.sharpe)} × PF {fmt(res.best.profit_factor)}
                </span>
              )}
            </div>
          </div>

          {/* Top 5 results table */}
          {res.top_results && res.top_results.length > 0 && (
            <div style={{ ...panel(), marginTop: 14 }}>
              <h3 style={{ margin: "0 0 12px", fontSize: 14 }}>Top 5 combinaisons</h3>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ color: COLORS.sub, textAlign: "left" }}>
                      <th style={th}>#</th>
                      <th style={th}>ADX min</th>
                      <th style={th}>RSI low</th>
                      <th style={th}>RSI high</th>
                      <th style={th}>SL ATR mult</th>
                      <th style={th}>SR prox</th>
                      <th style={th}>Sharpe</th>
                      <th style={th}>Profit Factor</th>
                      <th style={th}>Winrate</th>
                      <th style={th}>Trades</th>
                      <th style={th}>Net P&L %</th>
                      <th style={th}>Max DD %</th>
                      <th style={th}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {res.top_results.map((row, idx) => {
                      const status = applyStatus[idx];
                      return (
                        <tr key={idx} style={{
                          borderTop: `1px solid ${COLORS.border}`,
                          background: idx === 0 ? "rgba(59,130,246,0.06)" : "transparent",
                        }}>
                          <td style={{ ...td, color: idx === 0 ? COLORS.blue : COLORS.sub }}>
                            {idx === 0 ? "★" : idx + 1}
                          </td>
                          <td style={td}>{fmt(row.params.adx_min, 0)}</td>
                          <td style={td}>{fmt(row.params.rsi_low, 0)}</td>
                          <td style={td}>{fmt(row.params.rsi_high, 0)}</td>
                          <td style={td}>{fmt(row.params.sl_atr_mult, 1)}</td>
                          <td style={td}>{fmt(row.params.sr_proximity, 1)}</td>
                          <td style={{ ...td, color: row.sharpe >= 1 ? COLORS.green : row.sharpe >= 0 ? COLORS.amber : COLORS.red, fontWeight: 600 }}>
                            {fmt(row.sharpe)}
                          </td>
                          <td style={{ ...td, color: row.profit_factor >= 1 ? COLORS.green : COLORS.red }}>
                            {fmt(row.profit_factor)}
                          </td>
                          <td style={td}>{fmt(row.winrate, 1)}%</td>
                          <td style={td}>{row.trades}</td>
                          <td style={{ ...td, color: row.net_profit_pct >= 0 ? COLORS.green : COLORS.red }}>
                            {fmt(row.net_profit_pct, 2)}%
                          </td>
                          <td style={{ ...td, color: COLORS.red }}>
                            -{fmt(row.max_drawdown_pct, 1)}%
                          </td>
                          <td style={td}>
                            <button
                              onClick={() => applyParams(idx, row.params)}
                              disabled={status === "loading" || status === "ok"}
                              style={{
                                background: status === "ok" ? COLORS.green : status === "err" ? COLORS.red : COLORS.blue,
                                color: "#fff", border: "none", borderRadius: 4,
                                padding: "4px 10px", fontSize: 11, fontWeight: 600,
                                cursor: status === "loading" || status === "ok" ? "default" : "pointer",
                              }}>
                              {status === "loading" ? "…" : status === "ok" ? "Appliqué ✓" : status === "err" ? "Erreur" : "Appliquer"}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div style={{ fontSize: 11, color: COLORS.sub, marginTop: 8 }}>
                Score = Sharpe × Profit Factor · "Appliquer" met à jour les paramètres en mémoire (non persistés, redémarrage les réinitialise)
              </div>
            </div>
          )}

          {res.top_results && res.top_results.length === 0 && (
            <div style={{ ...panel(), marginTop: 14, color: COLORS.amber }}>
              Aucune combinaison n'a généré suffisamment de trades (min 10) sur cette période.
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function BacktestPanel({ api }) {
  const d = defaultDates();
  const [subTab, setSubTab] = useState("backtest");
  const [form, setForm] = useState({
    start: d.start, end: d.end, capital: 10000, risk_pct: 1.0,
    spread_pips: 0.3, slippage_pips: 0.1, max_trades_per_day: 4, daily_stop_pct: 2.0,
    symbol: "XAUUSD",
  });
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);

  const update = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const subTabBtn = (active) => ({
    background: active ? COLORS.blue : "transparent",
    color: active ? "#fff" : COLORS.text,
    border: `1px solid ${active ? COLORS.blue : COLORS.border}`,
    borderRadius: 6, padding: "6px 16px", fontSize: 13,
    cursor: "pointer", fontWeight: 500,
  });

  const run = async () => {
    setLoading(true); setErr(null); setRes(null);
    try {
      const r = await fetch(`${api}/api/backtest`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          capital: Number(form.capital), risk_pct: Number(form.risk_pct),
          spread_pips: Number(form.spread_pips), slippage_pips: Number(form.slippage_pips),
          max_trades_per_day: Number(form.max_trades_per_day),
          daily_stop_pct: Number(form.daily_stop_pct),
          symbol: form.symbol || "XAUUSD",
        }),
      });
      const data = await r.json();
      if (data.error) setErr(data.error);
      else setRes(data);
    } catch (e) {
      setErr("Échec de la requête backtest: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const s = res?.summary;
  const heat = res?.heatmap || [];
  const maxAbsPnl = Math.max(1, ...heat.map((h) => Math.abs(h.pnl)));

  return (
    <div>
      {/* ===== sub-tab navigation ===== */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button onClick={() => setSubTab("backtest")} style={subTabBtn(subTab === "backtest")}>
          Backtest
        </button>
        <button onClick={() => setSubTab("optimizer")} style={subTabBtn(subTab === "optimizer")}>
          Optimiseur
        </button>
      </div>

      {subTab === "optimizer" && <OptimizerPanel api={api} />}

      {subTab === "backtest" && <>
      {/* ===== parameters ===== */}
      <div style={panel()}>
        <h3 style={{ margin: "0 0 12px", fontSize: 14 }}>Paramètres du backtest (M5)</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
          <Field label="Date début"><input type="date" value={form.start}
            onChange={(e) => update("start", e.target.value)} style={inp} /></Field>
          <Field label="Date fin"><input type="date" value={form.end}
            onChange={(e) => update("end", e.target.value)} style={inp} /></Field>
          <Field label="Capital ($)"><input type="number" value={form.capital}
            onChange={(e) => update("capital", e.target.value)} style={inp} /></Field>
          <Field label="Risque / trade (%)"><input type="number" step="0.1" value={form.risk_pct}
            onChange={(e) => update("risk_pct", e.target.value)} style={inp} /></Field>
          <Field label="Spread (pips)"><input type="number" step="0.1" value={form.spread_pips}
            onChange={(e) => update("spread_pips", e.target.value)} style={inp} /></Field>
          <Field label="Slippage (pips)"><input type="number" step="0.1" value={form.slippage_pips}
            onChange={(e) => update("slippage_pips", e.target.value)} style={inp} /></Field>
          <Field label="Max trades / jour"><input type="number" value={form.max_trades_per_day}
            onChange={(e) => update("max_trades_per_day", e.target.value)} style={inp} /></Field>
          <Field label="Stop journalier (%)"><input type="number" step="0.1" value={form.daily_stop_pct}
            onChange={(e) => update("daily_stop_pct", e.target.value)} style={inp} /></Field>
          <Field label="Symbole">
            <select value={form.symbol || "XAUUSD"} onChange={(e) => update("symbol", e.target.value)}
              style={inp}>
              <option value="XAUUSD">XAU/USD</option>
              <option value="EURUSD">EUR/USD</option>
            </select>
          </Field>
        </div>
        <button onClick={run} disabled={loading} style={{
          marginTop: 14, background: COLORS.blue, color: "#fff", border: "none",
          borderRadius: 6, padding: "9px 22px", fontSize: 14, fontWeight: 600,
          cursor: loading ? "wait" : "pointer", opacity: loading ? 0.7 : 1,
        }}>
          {loading ? "Calcul en cours…" : "▶ Lancer le backtest"}
        </button>
        <span style={{ marginLeft: 12, fontSize: 12, color: COLORS.sub }}>
          Données : yfinance (GC=F) · repli synthétique hors-ligne
        </span>
      </div>

      {err && (
        <div style={{ ...panel(), marginTop: 14, borderColor: COLORS.red, color: COLORS.red }}>
          ⚠ {err}
        </div>
      )}

      {s && (
        <>
          {/* ===== KPI cards ===== */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 12, marginTop: 14 }}>
            <Kpi label="Net P&L" value={money(s.net_profit)} sub={`${fmt(s.net_profit_pct)}%`}
              color={s.net_profit >= 0 ? COLORS.green : COLORS.red} />
            <Kpi label="Winrate" value={`${fmt(s.winrate, 1)}%`} sub={`${s.wins}W / ${s.losses}L`} />
            <Kpi label="Profit Factor" value={s.profit_factor == null ? "∞" : fmt(s.profit_factor, 2)}
              color={(s.profit_factor || 0) >= 1 ? COLORS.green : COLORS.red} />
            <Kpi label="Sharpe Ratio" value={s.sharpe_ratio !== undefined ? fmt(s.sharpe_ratio, 2) : "—"}
              color={(s.sharpe_ratio || 0) >= 1 ? COLORS.green : (s.sharpe_ratio || 0) >= 0 ? COLORS.amber : COLORS.red} />
            <Kpi label="Max Drawdown" value={`${fmt(s.max_drawdown_pct, 1)}%`}
              sub={money(s.max_drawdown_usd)} color={COLORS.red} />
            <Kpi label="Trades" value={s.trades} sub={`exp. ${money(s.expectancy)}`} />
            <Kpi label="Équité finale" value={`$${fmt(s.final_equity, 0)}`} />
          </div>

          {/* ===== equity curve ===== */}
          <div style={{ ...panel(), marginTop: 14 }}>
            <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>Courbe d'équité</h3>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={(res.equity_curve || []).map((p, i) => ({ i, equity: p.equity }))}>
                <defs>
                  <linearGradient id="eqbt" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={COLORS.blue} stopOpacity={0.5} />
                    <stop offset="100%" stopColor={COLORS.blue} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
                <XAxis dataKey="i" stroke={COLORS.sub} fontSize={11} />
                <YAxis stroke={COLORS.sub} fontSize={11} domain={["auto", "auto"]} />
                <Tooltip contentStyle={{ background: COLORS.panel, border: `1px solid ${COLORS.border}` }} />
                <Area type="monotone" dataKey="equity" stroke={COLORS.blue} fill="url(#eqbt)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
            {/* ===== by session ===== */}
            <div style={panel()}>
              <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>Winrate par session</h3>
              {["London", "NewYork"].map((sess) => {
                const v = res.by_session?.[sess] || {};
                return (
                  <div key={sess} style={{ marginBottom: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                      <span>{sess === "NewYork" ? "New York" : "London"}</span>
                      <span style={{ color: COLORS.sub }}>
                        {v.trades || 0} trades · {money(v.pnl)}
                      </span>
                    </div>
                    <div style={{ height: 10, background: "#1a2233", borderRadius: 5, marginTop: 4 }}>
                      <div style={{ width: `${v.winrate || 0}%`, height: "100%", borderRadius: 5,
                        background: (v.winrate || 0) >= 50 ? COLORS.green : COLORS.amber }} />
                    </div>
                    <div style={{ fontSize: 11, color: COLORS.sub, marginTop: 2 }}>
                      Winrate {fmt(v.winrate, 1)}%
                    </div>
                  </div>
                );
              })}
              <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 10, marginTop: 6, fontSize: 12 }}>
                <Row k="Durée moy. gagnants" v={`${fmt(res.duration?.avg_win_min, 1)} min`} />
                <Row k="Durée moy. perdants" v={`${fmt(res.duration?.avg_loss_min, 1)} min`} />
                <Row k="Gain moyen" v={money(s.avg_win)} />
                <Row k="Perte moyenne" v={money(s.avg_loss)} />
                {res.best_hour && <Row k="Meilleure heure (UTC)" v={`${res.best_hour.hour}h · ${money(res.best_hour.pnl)}`} />}
                {res.worst_hour && <Row k="Pire heure (UTC)" v={`${res.worst_hour.hour}h · ${money(res.worst_hour.pnl)}`} />}
              </div>
            </div>

            {/* ===== hourly heatmap ===== */}
            <div style={panel()}>
              <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>Heatmap horaire (P&L par heure UTC)</h3>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {Array.from({ length: 24 }).map((_, h) => {
                  const cell = heat.find((x) => x.hour === h);
                  const pnl = cell?.pnl || 0;
                  const intensity = Math.min(Math.abs(pnl) / maxAbsPnl, 1);
                  const bg = !cell ? "#161c2a"
                    : pnl >= 0
                      ? `rgba(22,199,132,${0.15 + intensity * 0.7})`
                      : `rgba(234,57,67,${0.15 + intensity * 0.7})`;
                  return (
                    <div key={h} title={cell ? `${h}h: ${money(pnl)} · ${cell.trades} trades · WR ${fmt(cell.winrate,0)}%` : `${h}h: aucun trade`}
                      style={{ width: 38, height: 44, background: bg, borderRadius: 4,
                        display: "flex", flexDirection: "column", alignItems: "center",
                        justifyContent: "center", fontSize: 10, color: COLORS.text,
                        border: `1px solid ${COLORS.border}` }}>
                      <span style={{ color: COLORS.sub }}>{h}h</span>
                      <span>{cell ? cell.trades : ""}</span>
                    </div>
                  );
                })}
              </div>
              <div style={{ fontSize: 11, color: COLORS.sub, marginTop: 10 }}>
                Vert = heures profitables · Rouge = heures perdantes · chiffre = nb de trades
              </div>
            </div>
          </div>

          {/* ===== trade list ===== */}
          <div style={{ ...panel(), marginTop: 14 }}>
            <h3 style={{ margin: "0 0 10px", fontSize: 14 }}>Détail des trades ({(res.trades || []).length})</h3>
            <div style={{ maxHeight: 280, overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ color: COLORS.sub, textAlign: "left", position: "sticky", top: 0, background: COLORS.panel }}>
                    <th style={th}>Entrée</th><th style={th}>Session</th><th style={th}>Dir</th>
                    <th style={th}>Prix</th><th style={th}>Sortie</th><th style={th}>Durée</th>
                    <th style={th}>Raison</th><th style={th}>P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {(res.trades || []).map((t, i) => (
                    <tr key={i} style={{ borderTop: `1px solid ${COLORS.border}` }}>
                      <td style={td}>{t.entry_time.slice(5, 16).replace("T", " ")}</td>
                      <td style={td}>{t.session === "NewYork" ? "NY" : "LDN"}</td>
                      <td style={{ ...td, color: t.direction === "long" ? COLORS.green : COLORS.red }}>
                        {t.direction === "long" ? "LONG" : "SHORT"}
                      </td>
                      <td style={td}>{fmt(t.entry, 2)}</td>
                      <td style={td}>{fmt(t.exit, 2)}</td>
                      <td style={td}>{fmt(t.duration_min, 0)}m</td>
                      <td style={{ ...td, color: COLORS.sub }}>{t.exit_reason}</td>
                      <td style={{ ...td, fontWeight: 600, color: t.pnl >= 0 ? COLORS.green : COLORS.red }}>
                        {money(t.pnl)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
      </>}
    </div>
  );
}

/* ---------------------------- UI helpers -------------------------------- */
const panel = () => ({
  background: COLORS.panel, border: `1px solid ${COLORS.border}`,
  borderRadius: 10, padding: 14,
});
const inp = {
  width: "100%", background: COLORS.panel2, border: `1px solid ${COLORS.border}`,
  borderRadius: 6, color: COLORS.text, padding: "7px 9px", fontSize: 13, boxSizing: "border-box",
};
const th = { padding: "6px 8px", fontWeight: 500 };
const td = { padding: "6px 8px" };

function Field({ label, children }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: COLORS.sub, marginBottom: 4 }}>{label}</div>
      {children}
    </div>
  );
}
function Kpi({ label, value, sub, color }) {
  return (
    <div style={panel()}>
      <div style={{ fontSize: 11, color: COLORS.sub, textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: color || COLORS.text, marginTop: 4 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: COLORS.sub, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}
function Row({ k, v }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
      <span style={{ color: COLORS.sub }}>{k}</span>
      <span>{v}</span>
    </div>
  );
}
