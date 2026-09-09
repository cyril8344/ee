import React, { useRef, useState } from "react";

/**
 * Scanner SMC — page dédiée, séparée du dashboard de trading.
 *
 * On envoie une capture d'écran de graphique (MT5, TradingView, n'importe quoi
 * avec des bougies colorées) et on récupère la même image avec les Order Blocks
 * dessinés dessus. Aucune donnée de marché n'est consultée : tout vient des
 * pixels, côté serveur (backend/smc/chart_image.py).
 *
 * La calibration se fait en touchant DEUX repères de prix sur l'image et en
 * saisissant leur valeur. Elle est facultative : sans elle, les zones sont
 * visibles mais pas chiffrées. Deux prix seuls ne suffiraient pas — une première
 * version prenait « prix du haut / prix du bas » en supposant qu'ils tombaient
 * sur les extrémités des bougies, et sortait 79 282 pour une zone réellement à
 * 78 840. Des prix plausibles mais faux sont pires que pas de prix.
 */

const API = import.meta?.env?.VITE_API_URL || "";

const COLORS = {
  bg: "#0a0e17", panel: "#121826", border: "#1f2937", text: "#e5e7eb",
  sub: "#8b95a7", green: "#16c784", red: "#ea3943", blue: "#3b82f6",
  amber: "#f59e0b",
};

function authHeaders() {
  const t = localStorage.getItem("token");
  return t ? { Authorization: `Bearer ${t}` } : {};
}

const panel = () => ({
  background: COLORS.panel,
  border: `1px solid ${COLORS.border}`,
  borderRadius: 10,
  padding: 14,
});

export default function ScannerPage({ onBack }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Points de calibration : {y (pixel image), price}
  const [points, setPoints] = useState([]);
  const [priceDraft, setPriceDraft] = useState("");
  const imgRef = useRef(null);

  const reset = (f) => {
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
    setResult(null);
    setError(null);
    setPoints([]);
    setPriceDraft("");
  };

  // Un clic sur l'aperçu enregistre une position. On convertit tout de suite en
  // pixels de l'image d'origine : la même capture affichée sur deux écrans n'a
  // pas la même taille, et le serveur raisonne en pixels natifs.
  const handleClick = (e) => {
    if (points.length >= 2 || !imgRef.current) return;
    const r = imgRef.current.getBoundingClientRect();
    const yAffiche = e.clientY - r.top;
    const yNatif = (yAffiche / r.height) * imgRef.current.naturalHeight;
    setPoints([...points, { y: Math.round(yNatif), price: null }]);
  };

  const setPriceOfLast = () => {
    const v = parseFloat(priceDraft.replace(",", "."));
    if (!isFinite(v)) return;
    const next = [...points];
    const i = next.findIndex((p) => p.price == null);
    if (i < 0) return;
    next[i].price = v;
    setPoints(next);
    setPriceDraft("");
  };

  const pretsACalibrer = points.length === 2 && points.every((p) => p.price != null);

  const analyser = () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const fd = new FormData();
    fd.append("file", file);
    if (pretsACalibrer) {
      fd.append("cal_y1", String(points[0].y));
      fd.append("cal_price1", String(points[0].price));
      fd.append("cal_y2", String(points[1].y));
      fd.append("cal_price2", String(points[1].price));
    }
    fetch(`${API}/api/smc/chart-image`, { method: "POST", headers: authHeaders(), body: fd })
      .then((r) => r.json().then((d) => ({ status: r.status, d })))
      .then(({ status, d }) => {
        if (status >= 400) throw new Error(d.detail || `HTTP ${status}`);
        setResult(d);
        setLoading(false);
      })
      .catch((e) => { setError(e.message); setLoading(false); });
  };

  const enAttenteDePrix = points.length > 0 && points.some((p) => p.price == null);

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text,
      fontFamily: "system-ui, -apple-system, sans-serif", padding: 12 }}>

      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
        <button onClick={onBack}
          style={{ background: "transparent", border: `1px solid ${COLORS.border}`,
            color: COLORS.sub, borderRadius: 6, padding: "4px 10px", cursor: "pointer" }}>
          ← Bot
        </button>
        <h2 style={{ margin: 0, fontSize: 17 }}>Scanner SMC — capture → Order Blocks</h2>
      </div>

      <div className="dashboard-panel" style={panel()}>
        <div style={{ fontSize: 12, color: COLORS.sub, marginBottom: 10 }}>
          Envoie une capture de graphique en bougies. Les Order Blocks sont détectés
          <b> depuis les pixels</b> — aucune donnée de marché n'est utilisée, et rien
          n'est tradé. Les thèmes en noir et blanc ne marchent pas : sans couleur, rien
          ne distingue une bougie haussière d'une baissière.
        </div>

        <input type="file" accept="image/*"
          onChange={(e) => reset(e.target.files?.[0] || null)}
          style={{ fontSize: 12, color: COLORS.sub, marginBottom: 10 }} />

        {preview && (
          <>
            <div style={{ fontSize: 12, color: COLORS.sub, margin: "8px 0 6px" }}>
              <b style={{ color: COLORS.text }}>Calibration (facultative)</b> — touche un
              repère de prix sur l'image, saisis sa valeur, recommence une deuxième fois.
              Sans ça, les zones sont dessinées mais pas chiffrées.
            </div>

            <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
              <img ref={imgRef} src={preview} alt="capture" onClick={handleClick}
                style={{ maxWidth: "100%", display: "block", cursor: points.length < 2 ? "crosshair" : "default",
                  border: `1px solid ${COLORS.border}`, borderRadius: 6 }} />
              {imgRef.current && points.map((p, i) => (
                <div key={i} style={{
                  position: "absolute", left: 0, right: 0,
                  top: `${(p.y / imgRef.current.naturalHeight) * 100}%`,
                  height: 0, borderTop: `2px dashed ${COLORS.amber}`, pointerEvents: "none",
                }}>
                  <span style={{ position: "absolute", right: 2, top: -16, fontSize: 11,
                    color: COLORS.amber, background: "rgba(0,0,0,.6)", padding: "0 4px" }}>
                    {p.price != null ? p.price : `point ${i + 1} — saisis le prix`}
                  </span>
                </div>
              ))}
            </div>

            {enAttenteDePrix && (
              <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 8 }}>
                <input value={priceDraft} onChange={(e) => setPriceDraft(e.target.value)}
                  inputMode="decimal" placeholder="prix de ce repère"
                  onKeyDown={(e) => e.key === "Enter" && setPriceOfLast()}
                  style={{ fontSize: 13, background: COLORS.bg, color: COLORS.text,
                    border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "4px 8px", width: 160 }} />
                <button onClick={setPriceOfLast}
                  style={{ fontSize: 12, background: COLORS.blue, color: "#fff", border: "none",
                    borderRadius: 4, padding: "5px 10px", cursor: "pointer" }}>
                  Valider
                </button>
              </div>
            )}

            <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 10 }}>
              <button onClick={analyser} disabled={loading}
                style={{ background: COLORS.amber, color: "#000", border: "none", borderRadius: 6,
                  padding: "6px 14px", fontSize: 13, fontWeight: 600,
                  cursor: loading ? "wait" : "pointer" }}>
                {loading ? "Analyse…" : "Détecter les Order Blocks"}
              </button>
              {points.length > 0 && (
                <button onClick={() => setPoints([])}
                  style={{ fontSize: 11, background: "transparent", color: COLORS.sub,
                    border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "4px 8px", cursor: "pointer" }}>
                  Effacer les repères
                </button>
              )}
              <span style={{ fontSize: 11, color: pretsACalibrer ? COLORS.green : COLORS.sub }}>
                {pretsACalibrer ? "✓ calibré — les zones sortiront en prix" : "non calibré — zones en pixels"}
              </span>
            </div>
          </>
        )}

        {error && <div style={{ fontSize: 12, color: COLORS.red, marginTop: 10 }}>Erreur : {error}</div>}
      </div>

      {result && !result.ok && (
        <div className="dashboard-panel" style={{ ...panel(), marginTop: 14 }}>
          <div style={{ fontSize: 13, color: COLORS.amber }}>{result.error}</div>
        </div>
      )}

      {result?.ok && (
        <div className="dashboard-panel" style={{ ...panel(), marginTop: 14 }}>
          <div style={{ fontSize: 12, color: COLORS.sub, marginBottom: 8 }}>
            {result.candles} bougies lues · pas {result.pitch} px · périodicité {result.periodicity}
          </div>

          <img src={result.image} alt="annotée"
            style={{ maxWidth: "100%", display: "block", border: `1px solid ${COLORS.border}`, borderRadius: 6 }} />

          <div style={{ fontSize: 11, color: COLORS.sub, margin: "8px 0" }}>
            Trait épais = zone encore vivante · trait pâle = déjà mitigée (le prix y est
            revenu, il n'y a plus de retest à en attendre).
          </div>

          {result.note && (
            <div style={{ fontSize: 11, color: COLORS.amber, marginBottom: 8 }}>{result.note}</div>
          )}

          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", color: COLORS.sub, fontWeight: "normal" }}>Zone</th>
                <th style={{ textAlign: "left", color: COLORS.sub, fontWeight: "normal" }}>État</th>
                <th style={{ textAlign: "right", color: COLORS.sub, fontWeight: "normal" }}>
                  {result.calibrated ? "Bas → Haut" : "Bas → Haut (pixels)"}
                </th>
                <th style={{ textAlign: "right", color: COLORS.sub, fontWeight: "normal" }}>Impulsion</th>
              </tr>
            </thead>
            <tbody>
              {result.zones.map((z, i) => (
                <tr key={i}>
                  <td style={{ paddingTop: 3, color: z.type === "haussier" ? COLORS.green : COLORS.red }}>
                    {z.type === "haussier" ? "▲" : "▼"} {z.type}
                  </td>
                  <td style={{ paddingTop: 3, color: z.state === "active" ? COLORS.text : COLORS.sub }}>
                    {z.state === "active" ? "vivante" : z.state === "mitigated" ? "mitigée" : "périmée"}
                  </td>
                  <td style={{ paddingTop: 3, textAlign: "right", color: COLORS.text }}>
                    {result.calibrated
                      ? `${z.low} → ${z.high}`
                      : `${z.low_px} → ${z.high_px}`}
                  </td>
                  <td style={{ paddingTop: 3, textAlign: "right", color: COLORS.sub }}>
                    {z.displacement_atr}×ATR
                  </td>
                </tr>
              ))}
              {result.zones.length === 0 && (
                <tr><td colSpan={4} style={{ color: COLORS.sub, paddingTop: 6 }}>
                  Aucun Order Block sur cette image — pas de cassure de structure avec une
                  impulsion suffisante.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
