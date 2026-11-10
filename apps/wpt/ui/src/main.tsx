import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// the design system's tokens before its components' styles, then this interface's layout
import "@workers-intelligence-commons/tokens/tokens.css";
import "@workers-intelligence-commons/components/styles.css";
import { App } from "./app";
import "./style.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
