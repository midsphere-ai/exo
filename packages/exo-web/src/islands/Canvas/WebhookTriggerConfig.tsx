import { useState, useCallback, useEffect } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface WebhookData {
  webhook_config_id?: string;
  webhook_enabled?: boolean;
}

interface WebhookConfigEntry {
  id: string;
  workflow_id: string;
  hook_id: string;
  url_token: string;
  webhook_url: string;
  enabled: boolean;
  request_log: RequestLogEntry[];
  created_at: string;
  updated_at: string;
}

interface RequestLogEntry {
  timestamp: string;
  payload_preview: string;
  status: string;
  response_status: number;
}

interface WebhookTriggerConfigProps {
  data: WebhookData;
  onChange: (updates: Partial<WebhookData>) => void;
  nodeId: string;
  workflowId?: string;
}

/* ------------------------------------------------------------------ */
/* Shared styles                                                        */
/* ------------------------------------------------------------------ */

const labelStyle: React.CSSProperties = {
  display: "block",
  fontSize: 11,
  fontWeight: 600,
  color: "var(--zen-muted, #999)",
  marginBottom: 4,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const btnStyle: React.CSSProperties = {
  padding: "6px 12px",
  fontSize: 11,
  fontWeight: 600,
  border: "1px solid var(--zen-subtle, #e0ddd0)",
  borderRadius: 6,
  background: "var(--zen-paper, #f2f0e3)",
  color: "var(--zen-dark, #2e2e2e)",
  cursor: "pointer",
  transition: "background 150ms",
};

/* ------------------------------------------------------------------ */
/* Status badge                                                         */
/* ------------------------------------------------------------------ */

const LOG_STATUS_STYLES: Record<string, { bg: string; color: string; label: string }> = {
  triggered: { bg: "#22c55e20", color: "#22c55e", label: "Triggered" },
  queued: { bg: "#6287f520", color: "#6287f5", label: "Queued" },
  rejected_disabled: { bg: "#ef444420", color: "#ef4444", label: "Disabled" },
  rejected_no_workflow: { bg: "#ef444420", color: "#ef4444", label: "Not Found" },
  rejected_empty: { bg: "#f59e0b20", color: "#f59e0b", label: "Empty" },
};

/* ------------------------------------------------------------------ */
/* Component                                                            */
/* ------------------------------------------------------------------ */

export default function WebhookTriggerConfig({
  data,
  onChange,
  nodeId,
  workflowId,
}: WebhookTriggerConfigProps) {
  const [webhook, setWebhook] = useState<WebhookConfigEntry | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [requestLog, setRequestLog] = useState<RequestLogEntry[]>([]);
  const [logLoading, setLogLoading] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const [expandedLogIndex, setExpandedLogIndex] = useState<number | null>(null);

  /* Load or create webhook config for this workflow */
  const loadWebhook = useCallback(async () => {
    if (!workflowId) return;
    setLoading(true);
    try {
      // Check if we already have a webhook_config_id stored on the node
      if (data.webhook_config_id) {
        const res = await fetch(`/api/v1/webhook-configs/${data.webhook_config_id}`);
        if (res.ok) {
          const wh: WebhookConfigEntry = await res.json();
          setWebhook(wh);
          setLoading(false);
          return;
        }
      }

      // Otherwise list webhooks for this workflow and use the first one
      const res = await fetch(
        `/api/v1/webhook-configs?workflow_id=${encodeURIComponent(workflowId)}`,
      );
      if (!res.ok) {
        setLoading(false);
        return;
      }
      const list: WebhookConfigEntry[] = await res.json();
      if (list.length > 0) {
        setWebhook(list[0]);
        onChange({ webhook_config_id: list[0].id });
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, [workflowId, data.webhook_config_id, onChange]);

  useEffect(() => {
    loadWebhook();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workflowId]);

  /* Create webhook */
  const handleCreate = useCallback(async () => {
    if (!workflowId) return;
    setLoading(true);
    try {
      const hookId = `hook-${nodeId}`;
      const res = await fetch("/api/v1/webhook-configs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow_id: workflowId, hook_id: hookId }),
      });
      if (res.ok) {
        const wh: WebhookConfigEntry = await res.json();
        setWebhook(wh);
        onChange({ webhook_config_id: wh.id, webhook_enabled: true });
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, [workflowId, nodeId, onChange]);

  /* Toggle enabled */
  const handleToggle = useCallback(async () => {
    if (!webhook) return;
    try {
      const res = await fetch(`/api/v1/webhook-configs/${webhook.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !webhook.enabled }),
      });
      if (res.ok) {
        const updated: WebhookConfigEntry = await res.json();
        setWebhook(updated);
        onChange({ webhook_enabled: updated.enabled });
      }
    } catch {
      // ignore
    }
  }, [webhook, onChange]);

  /* Regenerate URL token */
  const handleRegenerate = useCallback(async () => {
    if (!webhook) return;
    try {
      const res = await fetch(`/api/v1/webhook-configs/${webhook.id}/regenerate`, {
        method: "POST",
      });
      if (res.ok) {
        const updated: WebhookConfigEntry = await res.json();
        setWebhook(updated);
      }
    } catch {
      // ignore
    }
  }, [webhook]);

  /* Copy URL */
  const handleCopy = useCallback(() => {
    if (!webhook) return;
    const fullUrl = `${window.location.origin}${webhook.webhook_url}`;
    navigator.clipboard.writeText(fullUrl).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [webhook]);

  /* Load request log */
  const handleLoadLog = useCallback(async () => {
    if (!webhook) return;
    setLogLoading(true);
    setShowLog(true);
    try {
      const res = await fetch(`/api/v1/webhook-configs/${webhook.id}/request-log?limit=20`);
      if (res.ok) {
        const log: RequestLogEntry[] = await res.json();
        setRequestLog(log);
      }
    } catch {
      // ignore
    } finally {
      setLogLoading(false);
    }
  }, [webhook]);

  if (loading) {
    return (
      <div style={{ fontSize: 12, color: "var(--zen-muted, #999)", padding: "8px 0" }}>
        Loading webhook configuration...
      </div>
    );
  }

  if (!workflowId) {
    return (
      <div
        style={{
          padding: "12px",
          fontSize: 12,
          color: "var(--zen-muted, #999)",
          border: "1px dashed var(--zen-subtle, #e0ddd0)",
          borderRadius: 8,
          textAlign: "center",
        }}
      >
        Save the workflow first to configure webhooks
      </div>
    );
  }

  /* No webhook yet — show create button */
  if (!webhook) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div
          style={{
            padding: "12px",
            background: "var(--zen-subtle, #e0ddd0)",
            borderRadius: 8,
            fontSize: 11,
            color: "var(--zen-muted, #999)",
            lineHeight: 1.5,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4, color: "var(--zen-dark, #2e2e2e)" }}>
            Webhook Trigger
          </div>
          Create a webhook URL to trigger this workflow from external services.
          Incoming POST requests will start a new workflow run.
        </div>
        <button
          onClick={handleCreate}
          style={{
            ...btnStyle,
            background: "var(--zen-coral, #F76F53)",
            color: "#fff",
            border: "none",
            padding: "8px 16px",
          }}
        >
          Create Webhook URL
        </button>
      </div>
    );
  }

  /* Webhook exists — show URL and controls */
  const fullUrl = `${window.location.origin}${webhook.webhook_url}`;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Webhook URL */}
      <div>
        <label style={labelStyle}>Webhook URL</label>
        <div
          style={{
            display: "flex",
            alignItems: "stretch",
            gap: 4,
          }}
        >
          <div
            style={{
              flex: 1,
              padding: "7px 10px",
              fontSize: 11,
              fontFamily: "monospace",
              border: "1px solid var(--zen-subtle, #e0ddd0)",
              borderRadius: 6,
              background: "var(--zen-subtle, #e0ddd0)",
              color: "var(--zen-dark, #2e2e2e)",
              wordBreak: "break-all",
              lineHeight: 1.4,
            }}
          >
            {fullUrl}
          </div>
          <button
            onClick={handleCopy}
            title="Copy URL"
            style={{
              ...btnStyle,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 36,
              padding: 0,
              flexShrink: 0,
              background: copied ? "#22c55e20" : "var(--zen-paper, #f2f0e3)",
              borderColor: copied ? "#22c55e" : "var(--zen-subtle, #e0ddd0)",
              color: copied ? "#22c55e" : "var(--zen-dark, #2e2e2e)",
            }}
          >
            {copied ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Enabled toggle */}
      <div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div>
            <label style={{ ...labelStyle, marginBottom: 0 }}>Enabled</label>
            <div style={{ fontSize: 10, color: "var(--zen-muted, #999)", marginTop: 2 }}>
              {webhook.enabled ? "Accepting requests" : "Requests will be rejected"}
            </div>
          </div>
          <button
            onClick={handleToggle}
            style={{
              position: "relative",
              width: 36,
              height: 20,
              borderRadius: 10,
              border: "none",
              cursor: "pointer",
              background: webhook.enabled
                ? "var(--zen-blue, #6287f5)"
                : "var(--zen-subtle, #e0ddd0)",
              transition: "background 200ms",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 2,
                left: webhook.enabled ? 18 : 2,
                width: 16,
                height: 16,
                borderRadius: "50%",
                background: "#fff",
                boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
                transition: "left 200ms",
              }}
            />
          </button>
        </div>
      </div>

      {/* Regenerate button */}
      <div>
        <button
          onClick={handleRegenerate}
          style={{
            ...btnStyle,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.13-9.36L23 10" />
          </svg>
          Regenerate URL
        </button>
        <div style={{ marginTop: 4, fontSize: 10, color: "var(--zen-muted, #999)", lineHeight: 1.4 }}>
          Generates a new URL. The old URL will stop working.
        </div>
      </div>

      {/* Info box */}
      <div
        style={{
          padding: "10px 12px",
          background: "var(--zen-subtle, #e0ddd0)",
          borderRadius: 8,
          fontSize: 11,
          color: "var(--zen-muted, #999)",
          lineHeight: 1.5,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 4, color: "var(--zen-dark, #2e2e2e)" }}>
          How It Works
        </div>
        Send a POST request to the webhook URL with a JSON body. The payload will be
        available as the output of this trigger node (as JSON data).
      </div>

      {/* Request log section */}
      <div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
          <label style={{ ...labelStyle, marginBottom: 0 }}>Request Log</label>
          <button
            onClick={showLog ? () => setShowLog(false) : handleLoadLog}
            style={{
              ...btnStyle,
              fontSize: 10,
              padding: "3px 8px",
            }}
          >
            {showLog ? "Hide" : "View Log"}
          </button>
        </div>

        {showLog && (
          logLoading ? (
            <div style={{ fontSize: 12, color: "var(--zen-muted, #999)", padding: "8px 0" }}>
              Loading...
            </div>
          ) : requestLog.length === 0 ? (
            <div
              style={{
                fontSize: 12,
                color: "var(--zen-muted, #999)",
                padding: "12px",
                textAlign: "center",
                border: "1px dashed var(--zen-subtle, #e0ddd0)",
                borderRadius: 8,
              }}
            >
              No requests received yet
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {requestLog.map((entry, i) => {
                const st = LOG_STATUS_STYLES[entry.status] || { bg: "#99999920", color: "#999", label: entry.status };
                const expanded = expandedLogIndex === i;
                return (
                  <div key={i}>
                    <button
                      onClick={() => setExpandedLogIndex(expanded ? null : i)}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        width: "100%",
                        padding: "6px 8px",
                        borderRadius: 6,
                        border: "1px solid var(--zen-subtle, #e0ddd0)",
                        background: expanded ? "var(--zen-subtle, #e0ddd0)" : "transparent",
                        cursor: "pointer",
                        fontSize: 11,
                        textAlign: "left",
                      }}
                    >
                      <span
                        style={{
                          padding: "2px 6px",
                          borderRadius: 4,
                          background: st.bg,
                          color: st.color,
                          fontWeight: 600,
                          fontSize: 10,
                          flexShrink: 0,
                        }}
                      >
                        {st.label}
                      </span>
                      <span style={{ flex: 1, color: "var(--zen-muted, #999)", fontSize: 10 }}>
                        {entry.timestamp}
                      </span>
                      <span style={{ color: "var(--zen-muted, #999)", fontSize: 10, flexShrink: 0 }}>
                        {entry.response_status}
                      </span>
                    </button>
                    {expanded && entry.payload_preview && (
                      <div
                        style={{
                          margin: "4px 0 4px 8px",
                          padding: "8px",
                          fontSize: 10,
                          fontFamily: "monospace",
                          background: "var(--zen-subtle, #e0ddd0)",
                          borderRadius: 6,
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-all",
                          maxHeight: 120,
                          overflow: "auto",
                          color: "var(--zen-dark, #2e2e2e)",
                          lineHeight: 1.4,
                        }}
                      >
                        {entry.payload_preview}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )
        )}
      </div>
    </div>
  );
}
