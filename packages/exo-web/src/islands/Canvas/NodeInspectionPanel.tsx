import { useState, useEffect, useCallback } from "react";
import type { Node } from "@xyflow/react";
import { NODE_CATEGORIES } from "./NodeSidebar";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface NodeExecutionData {
  id: string;
  run_id: string;
  node_id: string;
  status: string;
  input_json: Record<string, unknown> | null;
  output_json: Record<string, unknown> | null;
  logs_text: string | null;
  token_usage_json: Record<string, unknown> | null;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

interface NodeInspectionPanelProps {
  node: Node | null;
  workflowId: string;
  runId: string;
  onClose: () => void;
}

type TabId = "input" | "output" | "logs" | "timing";

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

function getNodeTypeInfo(nodeType: string) {
  for (const cat of NODE_CATEGORIES) {
    const found = cat.types.find((t) => t.id === nodeType);
    if (found) return { label: found.label, category: cat.label, color: cat.color, icon: found.icon };
  }
  return { label: nodeType, category: "Unknown", color: "#999", icon: null };
}

function formatDuration(startedAt: string | null, completedAt: string | null): string {
  if (!startedAt || !completedAt) return "—";
  const start = new Date(startedAt).getTime();
  const end = new Date(completedAt).getTime();
  const ms = end - start;
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  const mins = Math.floor(ms / 60000);
  const secs = ((ms % 60000) / 1000).toFixed(1);
  return `${mins}m ${secs}s`;
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

/* ------------------------------------------------------------------ */
/* JSON viewer                                                          */
/* ------------------------------------------------------------------ */

function JsonBlock({ data, label }: { data: unknown; label?: string }) {
  const [copied, setCopied] = useState(false);
  const text = JSON.stringify(data, null, 2);

  const copy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [text]);

  if (data === null || data === undefined) {
    return (
      <div style={{ fontSize: 12, color: "var(--zen-muted, #999)", fontStyle: "italic", padding: "8px 0" }}>
        No {label || "data"} available
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={copy}
        title="Copy to clipboard"
        style={{
          position: "absolute",
          top: 6,
          right: 6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: 26,
          height: 26,
          border: "1px solid var(--zen-subtle, #e0ddd0)",
          borderRadius: 6,
          background: "var(--zen-paper, #f2f0e3)",
          color: copied ? "var(--zen-green, #63f78b)" : "var(--zen-muted, #999)",
          cursor: "pointer",
          fontSize: 11,
          zIndex: 1,
        }}
      >
        {copied ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
        )}
      </button>
      <pre
        style={{
          margin: 0,
          padding: "10px 12px",
          fontSize: 11,
          fontFamily: "monospace",
          lineHeight: 1.5,
          background: "var(--zen-subtle, #e0ddd0)",
          borderRadius: 8,
          overflowX: "auto",
          overflowY: "auto",
          maxHeight: 300,
          color: "var(--zen-dark, #2e2e2e)",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {text}
      </pre>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab content renderers                                                */
/* ------------------------------------------------------------------ */

function InputTab({ data }: { data: NodeExecutionData }) {
  return (
    <div>
      <SectionLabel>Data received from upstream nodes</SectionLabel>
      <JsonBlock data={data.input_json} label="input data" />
    </div>
  );
}

function OutputTab({ data }: { data: NodeExecutionData }) {
  return (
    <div>
      {data.error ? (
        <>
          <SectionLabel>Error</SectionLabel>
          <div
            style={{
              padding: "10px 12px",
              fontSize: 12,
              fontFamily: "monospace",
              lineHeight: 1.5,
              background: "#fef2f2",
              borderRadius: 8,
              color: "#ef4444",
              border: "1px solid #fecaca",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {data.error}
          </div>
        </>
      ) : (
        <>
          <SectionLabel>Data produced by this node</SectionLabel>
          <JsonBlock data={data.output_json} label="output data" />
        </>
      )}
    </div>
  );
}

function LogsTab({ data, nodeType }: { data: NodeExecutionData; nodeType: string }) {
  const isLlm = nodeType === "llm_call";
  const isCode = nodeType === "code_python" || nodeType === "code_javascript";

  return (
    <div>
      {isLlm && (
        <>
          <SectionLabel>Prompt & Response</SectionLabel>
          {data.input_json && (data.input_json as Record<string, unknown>).prompt && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--zen-muted, #999)", marginBottom: 4 }}>PROMPT</div>
              <div
                style={{
                  padding: "10px 12px",
                  fontSize: 12,
                  lineHeight: 1.5,
                  background: "var(--zen-subtle, #e0ddd0)",
                  borderRadius: 8,
                  color: "var(--zen-dark, #2e2e2e)",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  maxHeight: 200,
                  overflowY: "auto",
                }}
              >
                {String((data.input_json as Record<string, unknown>).prompt)}
              </div>
            </div>
          )}
          {data.output_json && (data.output_json as Record<string, unknown>).response && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--zen-muted, #999)", marginBottom: 4 }}>RESPONSE</div>
              <div
                style={{
                  padding: "10px 12px",
                  fontSize: 12,
                  lineHeight: 1.5,
                  background: "var(--zen-subtle, #e0ddd0)",
                  borderRadius: 8,
                  color: "var(--zen-dark, #2e2e2e)",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  maxHeight: 200,
                  overflowY: "auto",
                }}
              >
                {String((data.output_json as Record<string, unknown>).response)}
              </div>
            </div>
          )}
        </>
      )}

      {isCode && (
        <>
          <SectionLabel>stdout / stderr</SectionLabel>
        </>
      )}

      <SectionLabel>{isLlm || isCode ? "Log messages" : "Log output"}</SectionLabel>
      {data.logs_text ? (
        <pre
          style={{
            margin: 0,
            padding: "10px 12px",
            fontSize: 11,
            fontFamily: "monospace",
            lineHeight: 1.5,
            background: "var(--zen-subtle, #e0ddd0)",
            borderRadius: 8,
            overflowX: "auto",
            overflowY: "auto",
            maxHeight: 300,
            color: "var(--zen-dark, #2e2e2e)",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {data.logs_text}
        </pre>
      ) : (
        <div style={{ fontSize: 12, color: "var(--zen-muted, #999)", fontStyle: "italic", padding: "8px 0" }}>
          No log output
        </div>
      )}
    </div>
  );
}

function TimingTab({ data }: { data: NodeExecutionData }) {
  const tokenUsage = data.token_usage_json as Record<string, unknown> | null;

  return (
    <div>
      <SectionLabel>Execution timing</SectionLabel>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <InfoRow label="Start time" value={formatTimestamp(data.started_at)} />
        <InfoRow label="End time" value={formatTimestamp(data.completed_at)} />
        <InfoRow label="Duration" value={formatDuration(data.started_at, data.completed_at)} />
        <InfoRow
          label="Status"
          value={data.status}
          valueColor={
            data.status === "completed"
              ? "var(--zen-green, #63f78b)"
              : data.status === "failed"
                ? "#ef4444"
                : "var(--zen-coral, #F76F53)"
          }
        />
      </div>

      {tokenUsage && Object.keys(tokenUsage).length > 0 && (
        <>
          <div style={{ height: 1, background: "var(--zen-subtle, #e0ddd0)", margin: "16px 0" }} />
          <SectionLabel>Token usage</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {tokenUsage.prompt_tokens != null && (
              <InfoRow label="Prompt tokens" value={String(tokenUsage.prompt_tokens)} />
            )}
            {tokenUsage.completion_tokens != null && (
              <InfoRow label="Completion tokens" value={String(tokenUsage.completion_tokens)} />
            )}
            {tokenUsage.total_tokens != null && (
              <InfoRow label="Total tokens" value={String(tokenUsage.total_tokens)} />
            )}
            {/* Render any other token fields dynamically */}
            {Object.entries(tokenUsage)
              .filter(([k]) => !["prompt_tokens", "completion_tokens", "total_tokens"].includes(k))
              .map(([k, v]) => (
                <InfoRow key={k} label={k.replace(/_/g, " ")} value={String(v)} />
              ))}
          </div>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Shared sub-components                                                */
/* ------------------------------------------------------------------ */

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 11,
        fontWeight: 600,
        color: "var(--zen-muted, #999)",
        marginBottom: 8,
        textTransform: "uppercase",
        letterSpacing: "0.05em",
      }}
    >
      {children}
    </div>
  );
}

function InfoRow({
  label,
  value,
  valueColor,
}: {
  label: string;
  value: string;
  valueColor?: string;
}) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: 12, color: "var(--zen-muted, #999)" }}>{label}</span>
      <span
        style={{
          fontSize: 12,
          fontWeight: 500,
          fontFamily: "monospace",
          color: valueColor || "var(--zen-dark, #2e2e2e)",
          textTransform: "capitalize",
        }}
      >
        {value}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tabs                                                                 */
/* ------------------------------------------------------------------ */

const TABS: { id: TabId; label: string }[] = [
  { id: "input", label: "Input" },
  { id: "output", label: "Output" },
  { id: "logs", label: "Logs" },
  { id: "timing", label: "Timing" },
];

/* ------------------------------------------------------------------ */
/* Panel component                                                      */
/* ------------------------------------------------------------------ */

export default function NodeInspectionPanel({
  node,
  workflowId,
  runId,
  onClose,
}: NodeInspectionPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>("output");
  const [data, setData] = useState<NodeExecutionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /* Fetch node execution data when node changes */
  useEffect(() => {
    if (!node || !workflowId || !runId) {
      setData(null);
      return;
    }

    setLoading(true);
    setError(null);

    fetch(`/api/v1/workflows/${workflowId}/runs/${runId}/nodes/${node.id}`)
      .then((res) => {
        if (!res.ok) throw new Error(res.status === 404 ? "No execution data for this node" : "Failed to load");
        return res.json();
      })
      .then((d) => {
        setData(d as NodeExecutionData);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setData(null);
        setLoading(false);
      });
  }, [node, workflowId, runId]);

  /* Handle Escape key to close */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") {
          (e.target as HTMLElement).blur();
          return;
        }
        onClose();
      }
    };
    if (node) {
      window.addEventListener("keydown", handler);
      return () => window.removeEventListener("keydown", handler);
    }
  }, [node, onClose]);

  if (!node) return null;

  const nodeType = (node.data.nodeType as string) || "default";
  const typeInfo = getNodeTypeInfo(nodeType);

  return (
    <div
      className="nodrag nopan nowheel"
      style={{
        position: "absolute",
        top: 8,
        right: 8,
        bottom: 8,
        width: 340,
        zIndex: 10,
        display: "flex",
        flexDirection: "column",
        background: "var(--zen-paper, #f2f0e3)",
        border: "1px solid var(--zen-subtle, #e0ddd0)",
        borderRadius: 12,
        boxShadow: "0 2px 12px rgba(0,0,0,0.12)",
        overflow: "hidden",
        animation: "slideInRight 200ms ease-out",
        fontFamily: "'Bricolage Grotesque', sans-serif",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 14px 10px",
          borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, minWidth: 0 }}>
          <span
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 28,
              height: 28,
              borderRadius: 6,
              background: `${typeInfo.color}20`,
              color: typeInfo.color,
              flexShrink: 0,
            }}
          >
            {typeInfo.icon}
          </span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: "var(--zen-dark, #2e2e2e)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {(node.data.label as string) || typeInfo.label}
            </div>
            <div
              style={{
                fontSize: 10,
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: typeInfo.color,
                lineHeight: 1.2,
              }}
            >
              Inspection
            </div>
          </div>
        </div>

        <button
          onClick={onClose}
          title="Close panel (Esc)"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 26,
            height: 26,
            border: "none",
            borderRadius: 6,
            background: "transparent",
            color: "var(--zen-muted, #999)",
            cursor: "pointer",
            flexShrink: 0,
          }}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Tab bar */}
      <div
        style={{
          display: "flex",
          borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
          padding: "0 8px",
          gap: 0,
        }}
      >
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              flex: 1,
              padding: "8px 4px",
              fontSize: 11,
              fontWeight: 600,
              fontFamily: "'Bricolage Grotesque', sans-serif",
              border: "none",
              borderBottom: activeTab === tab.id ? `2px solid ${typeInfo.color}` : "2px solid transparent",
              background: "transparent",
              color: activeTab === tab.id ? "var(--zen-dark, #2e2e2e)" : "var(--zen-muted, #999)",
              cursor: "pointer",
              transition: "color 150ms, border-color 150ms",
              textTransform: "uppercase",
              letterSpacing: "0.03em",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Body */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "14px",
        }}
      >
        {loading ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "40px 0" }}>
            <div
              style={{
                width: 24,
                height: 24,
                border: "2px solid var(--zen-subtle, #e0ddd0)",
                borderTopColor: typeInfo.color,
                borderRadius: "50%",
                animation: "inspectionSpin 0.6s linear infinite",
              }}
            />
          </div>
        ) : error ? (
          <div
            style={{
              padding: "20px 12px",
              textAlign: "center",
              fontSize: 12,
              color: "var(--zen-muted, #999)",
            }}
          >
            <div style={{ marginBottom: 8, fontSize: 20, opacity: 0.5 }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block" }}>
                <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>
            {error}
          </div>
        ) : data ? (
          <>
            {activeTab === "input" && <InputTab data={data} />}
            {activeTab === "output" && <OutputTab data={data} />}
            {activeTab === "logs" && <LogsTab data={data} nodeType={nodeType} />}
            {activeTab === "timing" && <TimingTab data={data} />}
          </>
        ) : null}
      </div>

      {/* Inline keyframes */}
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        @keyframes inspectionSpin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
