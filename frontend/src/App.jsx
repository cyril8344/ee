import React, { useState, useEffect } from "react";
import LoginPage from "./LoginPage";
import Dashboard from "./Dashboard";
import DashboardES from "./DashboardES";

/* Error boundary — prevents a single render error from blacking out the
 * entire app. Shows a readable message + reload button instead. */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ background: "#0a0e17", color: "#e5e7eb", minHeight: "100vh",
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", gap: 16, fontFamily: "system-ui, sans-serif", padding: 24 }}>
          <div style={{ fontSize: 36 }}>⚠️</div>
          <div style={{ fontSize: 18 }}>Une erreur d'affichage est survenue</div>
          <div style={{ fontSize: 12, color: "#8b95a7", maxWidth: 520, textAlign: "center" }}>
            {String(this.state.error?.message || this.state.error)}
          </div>
          <button onClick={() => window.location.reload()}
            style={{ marginTop: 8, padding: "8px 18px", borderRadius: 6, cursor: "pointer",
              background: "#3b82f6", color: "#fff", border: "none", fontSize: 14 }}>
            Recharger
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ============================================================================
 * App — top-level routing between LoginPage and Dashboard.
 *
 * If a valid token exists in localStorage → show Dashboard.
 * Otherwise → show LoginPage.
 * The Dashboard receives an `onLogout` prop that clears the token and forces
 * a return to the login screen.
 * ==========================================================================*/

export default function App() {
  const [token,   setToken]   = useState(() => localStorage.getItem("token") || null);
  const [page,    setPage]    = useState("xau"); // "xau" | "es"

  // Sync state if localStorage changes in another tab
  useEffect(() => {
    const handle = (e) => {
      if (e.key === "token") {
        setToken(e.newValue || null);
      }
    };
    window.addEventListener("storage", handle);
    return () => window.removeEventListener("storage", handle);
  }, []);

  const handleLogin = (newToken) => {
    setToken(newToken);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    setToken(null);
  };

  if (!token) {
    return <LoginPage onLogin={handleLogin} />;
  }

  if (page === "es") {
    return (
      <ErrorBoundary>
        <DashboardES token={token} onBack={() => setPage("xau")} />
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary>
      <Dashboard onLogout={handleLogout} onNavigateES={() => setPage("es")} />
    </ErrorBoundary>
  );
}
