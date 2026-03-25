/* ------------------------------------------------------------------ */
/* Validation summary panel for the workflow canvas                     */
/* ------------------------------------------------------------------ */

import type { ValidationIssue } from "./validation";

interface ValidationPanelProps {
  issues: ValidationIssue[];
  open: boolean;
  onToggle: () => void;
  onNavigateToNode: (nodeId: string) => void;
}

const severityIcon: Record<string, string> = {
  error: "\u26d4",   // ⛔
  warning: "\u26a0", // ⚠
};

const severityColor: Record<string, string> = {
  error: "#ef4444",
  warning: "#f59e0b",
};

export default function ValidationPanel({
  issues,
  open,
  onToggle,
  onNavigateToNode,
}: ValidationPanelProps) {
  const errorCount = issues.filter((i) => i.severity === "error").length;
  const warningCount = issues.filter((i) => i.severity === "warning").length;
  const total = issues.length;

  return (
    <div
      className="nodrag nopan nowheel"
      style={{
        position: "absolute",
        bottom: 12,
        left: 12,
        zIndex: 20,
        fontFamily: "'Bricolage Grotesque', sans-serif",
      }}
    >
      {/* Toggle button */}
      <button
        onClick={onToggle}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "6px 12px",
          borderRadius: open ? "8px 8px 0 0" : 8,
          border: "1px solid var(--zen-subtle, #e0ddd0)",
          borderBottom: open ? "none" : undefined,
          background: "var(--zen-paper, #f2f0e3)",
          color: total > 0 ? (errorCount > 0 ? "#ef4444" : "#f59e0b") : "var(--zen-green, #63f78b)",
          cursor: "pointer",
          fontSize: 12,
          fontWeight: 600,
          fontFamily: "inherit",
          boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
        }}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          {total > 0 ? (
            <>
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </>
          ) : (
            <>
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </>
          )}
        </svg>
        {total === 0 ? "Valid" : `${total} issue${total !== 1 ? "s" : ""}`}
        {errorCount > 0 && (
          <span style={{ fontSize: 10, color: "#ef4444" }}>
            {errorCount} error{errorCount !== 1 ? "s" : ""}
          </span>
        )}
        {warningCount > 0 && (
          <span style={{ fontSize: 10, color: "#f59e0b" }}>
            {warningCount} warning{warningCount !== 1 ? "s" : ""}
          </span>
        )}
      </button>

      {/* Issue list */}
      {open && (
        <div
          style={{
            background: "var(--zen-paper, #f2f0e3)",
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            borderRadius: "0 8px 8px 8px",
            maxHeight: 240,
            overflowY: "auto",
            boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
            minWidth: 260,
          }}
        >
          {issues.length === 0 ? (
            <div
              style={{
                padding: "12px 16px",
                fontSize: 12,
                color: "var(--zen-muted, #999)",
                textAlign: "center",
              }}
            >
              No issues found
            </div>
          ) : (
            <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
              {issues.map((issue, i) => (
                <li
                  key={`${issue.nodeId}-${i}`}
                  onClick={() => onNavigateToNode(issue.nodeId)}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 8,
                    padding: "8px 12px",
                    borderBottom: i < issues.length - 1 ? "1px solid var(--zen-subtle, #e0ddd0)" : undefined,
                    cursor: "pointer",
                    fontSize: 12,
                    color: "var(--zen-dark, #2e2e2e)",
                    transition: "background 100ms",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLLIElement).style.background = "var(--zen-subtle, #e0ddd0)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLLIElement).style.background = "transparent";
                  }}
                >
                  <span style={{ color: severityColor[issue.severity], flexShrink: 0, fontSize: 13 }}>
                    {severityIcon[issue.severity]}
                  </span>
                  <span style={{ lineHeight: 1.4 }}>{issue.message}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
