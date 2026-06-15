import React, { useState, useEffect } from "react";
import LoginPage from "./LoginPage";
import Dashboard from "./Dashboard";

/* ============================================================================
 * App — top-level routing between LoginPage and Dashboard.
 *
 * If a valid token exists in localStorage → show Dashboard.
 * Otherwise → show LoginPage.
 * The Dashboard receives an `onLogout` prop that clears the token and forces
 * a return to the login screen.
 * ==========================================================================*/

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("token") || null);

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

  return <Dashboard onLogout={handleLogout} />;
}
