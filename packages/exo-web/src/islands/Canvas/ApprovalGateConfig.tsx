import { useState, useCallback, useEffect } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface ApprovalGateData {
  timeout_minutes?: number;
  approval_message?: string;
}

interface ApprovalGateConfigProps {
  data: ApprovalGateData;
  onChange: (updates: Partial<ApprovalGateData>) => void;
  nodeId: string;
}

interface ApprovalHistoryEntry {
  id: string;
  run_id: string;
  node_id: string;
  status: string;
  timeout_minutes: number;
  comment: string | null;
  requested_at: string;
  responded_at: string | null;
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

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "8px 10px",
  fontSize: 13,
  border: "1px solid var(--zen-subtle, #e0ddd0)",
  borderRadius: 8,
  background: "var(--zen-paper, #f2f0e3)",
  color: "var(--zen-dark, #2e2e2e)",
  outline: "none",
  boxSizing: "border-box" as const,
  transition: "border-color 150ms",
};

const focusHandlers = {
  onFocus: (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    e.currentTarget.style.borderColor = "var(--zen-coral, #F76F53)";
  },
  onBlur: (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    e.currentTarget.style.borderColor = "var(--zen-subtle, #e0ddd0)";
  },
};

/* ------------------------------------------------------------------ */
/* Status badge helper                                                  */
/* ------------------------------------------------------------------ */

const STATUS_STYLES: Record<string, { bg: string; color: string; label: string }> = {
  approved: { bg: "#22c55e20", color: "#22c55e", label: "Approved" },
  rejected: { bg: "#ef444420", color: "#ef4444", label: "Rejected" },
  timed_out: { bg: "#f59e0b20", color: "#f59e0b", label: "Timed Out" },
  pending: { bg: "#6287f520", color: "#6287f5", label: "Pending" },
};

/* ------------------------------------------------------------------ */
/* Component                                                            */
/* ------------------------------------------------------------------ */

export default function ApprovalGateConfig({ data, onChange, nodeId }: ApprovalGateConfigProps) {
  const [history, setHistory] = useState<ApprovalHistoryEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  const handleTimeoutChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = parseInt(e.target.value, 10);
      onChange({ timeout_minutes: Number.isNaN(val) ? 60 : Math.max(1, Math.min(1440, val)) });
    },
    [onChange],
  );

  const handleMessageChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange({ approval_message: e.target.value });
    },
    [onChange],
  );

  /* Fetch approval history for this node */
  useEffect(() => {
    if (!nodeId) return;
    setHistoryLoading(true);
    fetch(`/api/v1/approvals/history?node_id=${encodeURIComponent(nodeId)}&limit=10`)
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: ApprovalHistoryEntry[]) => setHistory(rows))
      .catch(() => setHistory([]))
      .finally(() => setHistoryLoading(false));
  }, [nodeId]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Timeout */}
      <div>
        <label style={labelStyle}>Timeout (minutes)</label>
        <input
          type="number"
          min={1}
          max={1440}
          value={data.timeout_minutes ?? 60}
          onChange={handleTimeoutChange}
          style={inputStyle}
          {...focusHandlers}
        />
        <div
          style={{
            marginTop: 4,
            fontSize: 10,
            color: "var(--zen-muted, #999)",
            lineHeight: 1.4,
          }}
        >
          Auto-rejects if not responded within this time (1–1440 min)
        </div>
      </div>

      {/* Approval message */}
      <div>
        <label style={labelStyle}>Approval Message</label>
        <textarea
          value={data.approval_message || ""}
          onChange={handleMessageChange}
          placeholder="Optional context shown to approver..."
          rows={3}
          style={{
            ...inputStyle,
            resize: "vertical",
            minHeight: 50,
            lineHeight: 1.5,
          }}
          {...focusHandlers}
        />
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
        When execution reaches this node, the workflow pauses and waits for manual approval.
        A notification badge appears in the top bar. Approve or reject from the approvals panel.
      </div>

      {/* Approval history */}
      <div>
        <label style={labelStyle}>Approval History</label>
        {historyLoading ? (
          <div style={{ fontSize: 12, color: "var(--zen-muted, #999)", padding: "8px 0" }}>
            Loading...
          </div>
        ) : history.length === 0 ? (
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
            No approval history yet
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {history.map((entry) => {
              const st = STATUS_STYLES[entry.status] || STATUS_STYLES.pending;
              return (
                <div
                  key={entry.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "6px 8px",
                    borderRadius: 6,
                    border: "1px solid var(--zen-subtle, #e0ddd0)",
                    fontSize: 11,
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
                  <span style={{ flex: 1, color: "var(--zen-muted, #999)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {entry.comment || "—"}
                  </span>
                  <span style={{ color: "var(--zen-muted, #999)", fontSize: 10, flexShrink: 0 }}>
                    {entry.responded_at
                      ? new Date(entry.responded_at + "Z").toLocaleDateString()
                      : new Date(entry.requested_at + "Z").toLocaleDateString()}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
