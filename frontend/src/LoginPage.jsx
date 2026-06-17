import React, { useState } from "react";

/* ============================================================================
 * LoginPage — XAU/USD Scalping Bot
 * Dark theme matching the main dashboard.
 * On success stores the JWT in localStorage under "token".
 * ==========================================================================*/

const API = import.meta?.env?.VITE_API_URL || "";

const COLORS = {
  bg: "#0a0e17",
  panel: "#121826",
  border: "#1f2937",
  text: "#e5e7eb",
  sub: "#8b95a7",
  green: "#16c784",
  red: "#ea3943",
  blue: "#3b82f6",
  amber: "#f59e0b",
};

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API}/api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || "Identifiants invalides");
        return;
      }

      localStorage.setItem("token", data.access_token);
      if (onLogin) onLogin(data.access_token);
    } catch (err) {
      setError("Impossible de joindre le serveur: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        background: COLORS.bg,
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'Inter', system-ui, sans-serif",
        color: COLORS.text,
      }}
    >
      <div
        style={{
          background: COLORS.panel,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12,
          padding: "40px 36px",
          width: "100%",
          maxWidth: 380,
          boxShadow: "0 8px 40px rgba(0,0,0,0.5)",
        }}
      >
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>🟡</div>
          <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0, letterSpacing: 0.5 }}>
            Scalping Bot
          </h1>
          <p style={{ fontSize: 13, color: COLORS.sub, marginTop: 6 }}>
            XAU/USD · EUR/USD
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 16 }}>
            <label
              style={{
                display: "block",
                fontSize: 12,
                color: COLORS.sub,
                marginBottom: 6,
                textTransform: "uppercase",
                letterSpacing: 0.5,
              }}
            >
              Utilisateur
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
              style={{
                width: "100%",
                background: "#0a0e17",
                border: `1px solid ${COLORS.border}`,
                borderRadius: 6,
                color: COLORS.text,
                padding: "10px 12px",
                fontSize: 14,
                boxSizing: "border-box",
                outline: "none",
              }}
              onFocus={(e) => (e.target.style.borderColor = COLORS.blue)}
              onBlur={(e) => (e.target.style.borderColor = COLORS.border)}
            />
          </div>

          <div style={{ marginBottom: 24 }}>
            <label
              style={{
                display: "block",
                fontSize: 12,
                color: COLORS.sub,
                marginBottom: 6,
                textTransform: "uppercase",
                letterSpacing: 0.5,
              }}
            >
              Mot de passe
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
              style={{
                width: "100%",
                background: "#0a0e17",
                border: `1px solid ${COLORS.border}`,
                borderRadius: 6,
                color: COLORS.text,
                padding: "10px 12px",
                fontSize: 14,
                boxSizing: "border-box",
                outline: "none",
              }}
              onFocus={(e) => (e.target.style.borderColor = COLORS.blue)}
              onBlur={(e) => (e.target.style.borderColor = COLORS.border)}
            />
          </div>

          {/* Error message */}
          {error && (
            <div
              style={{
                background: "rgba(234,57,67,0.12)",
                border: `1px solid ${COLORS.red}`,
                borderRadius: 6,
                padding: "10px 12px",
                fontSize: 13,
                color: COLORS.red,
                marginBottom: 16,
              }}
            >
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{
              width: "100%",
              background: loading ? COLORS.amber : COLORS.blue,
              color: "#fff",
              border: "none",
              borderRadius: 6,
              padding: "11px 0",
              fontSize: 14,
              fontWeight: 600,
              cursor: loading ? "wait" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              transition: "background 0.2s",
            }}
          >
            {loading && (
              <span
                style={{
                  display: "inline-block",
                  width: 14,
                  height: 14,
                  border: "2px solid rgba(255,255,255,0.3)",
                  borderTop: "2px solid #fff",
                  borderRadius: "50%",
                  animation: "spin 0.8s linear infinite",
                }}
              />
            )}
            {loading ? "Connexion…" : "Se connecter"}
          </button>

          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </form>

        <p
          style={{
            fontSize: 11,
            color: COLORS.sub,
            textAlign: "center",
            marginTop: 24,
            marginBottom: 0,
          }}
        >
          Accès restreint — usage interne uniquement
        </p>
      </div>
    </div>
  );
}
