import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// By default the dashboard talks to the backend through the Vite dev-server
// proxy (same origin). Set VITE_API_URL to point at a remote backend.
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
