/**
 * RunHistoryPanel — execution history sidebar for the workflow canvas.
 *
 * Features:
 *  - Paginated run list with status, timestamp, duration, trigger info
 *  - Filters: status, date range, trigger type
 *  - Click a run to load node statuses onto the canvas
 *  - Replay: re-run with same inputs
 *  - Replay with modifications: edit inputs before replaying
 *  - Comparison: select two runs to diff side-by-side
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface RunSummary {
  id: string;
  workflow_id: string;
  status: string;
  trigger_type: string | null;
  input_json: Record<string, unknown> | null;
  started_at: string | null;
  completed_at: string | null;
  step_count: number | null;
  total_tokens: number | null;
  total_cost: number | null;
  error: string | null;
  created_at: string;
}

interface NodeExecution {
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

interface RunDetail extends RunSummary {
  node_executions: NodeExecution[];
}

interface Checkpoint {
  id: string;
  run_id: string;
  name: string;
  step_number: number;
  state_blob: unknown;
  created_at: string;
}

interface CheckpointDiff {
  checkpoint_a: Checkpoint;
  checkpoint_b: Checkpoint;
  added_keys: string[];
  removed_keys: string[];
  changed_keys: string[];
  unchanged_keys: string[];
}

type NodeExecStatus = "running" | "completed" | "failed";

interface RunHistoryPanelProps {
  workflowId: string;
  open: boolean;
  onToggle: () => void;
  /** Callback: set node statuses on the canvas from a past run */
  onLoadRunState: (nodeStatuses: Record<string, NodeExecStatus>, runId: string) => void;
  /** Callback: start a new run (replay) */
  onReplay: (inputJson?: Record<string, unknown>) => void;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const STATUS_COLORS: Record<string, string> = {
  completed: "var(--zen-green, #63f78b)",
  failed: "#ef4444",
  cancelled: "var(--zen-muted, #999)",
  running: "var(--zen-coral, #F76F53)",
  pending: "var(--zen-blue, #6287f5)",
};

function formatDuration(startedAt: string | null, completedAt: string | null): string {
  if (!startedAt) return "-";
  const start = new Date(startedAt).getTime();
  const end = completedAt ? new Date(completedAt).getTime() : Date.now();
  const seconds = Math.round((end - start) / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${minutes}m ${secs}s`;
}

function formatTimestamp(ts: string): string {
  const d = new Date(ts);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/* ------------------------------------------------------------------ */
/* Main panel                                                          */
/* ------------------------------------------------------------------ */

export default function RunHistoryPanel({
  workflowId,
  open,
  onToggle,
  onLoadRunState,
  onReplay,
}: RunHistoryPanelProps) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [offset, setOffset] = useState(0);
  const limit = 15;

  // Filters
  const [statusFilter, setStatusFilter] = useState("");
  const [triggerFilter, setTriggerFilter] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  // Selected run for detail view
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Checkpoints for selected run
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [checkpointsTotal, setCheckpointsTotal] = useState(0);
  const [checkpointsLoading, setCheckpointsLoading] = useState(false);
  const [savingCheckpoint, setSavingCheckpoint] = useState(false);
  const [checkpointDiff, setCheckpointDiff] = useState<CheckpointDiff | null>(null);
  const [diffLoading, setDiffLoading] = useState(false);
  const [diffIds, setDiffIds] = useState<[string | null, string | null]>([null, null]);
  const [autoCheckpointInterval, setAutoCheckpointInterval] = useState(5);

  // Replay with modifications
  const [replayOpen, setReplayOpen] = useState(false);
  const [replayInputText, setReplayInputText] = useState("{}");
  const [replayParseError, setReplayParseError] = useState<string | null>(null);

  // Comparison mode
  const [compareMode, setCompareMode] = useState(false);
  const [compareRunIds, setCompareRunIds] = useState<[string | null, string | null]>([null, null]);
  const [compareDetails, setCompareDetails] = useState<[RunDetail | null, RunDetail | null]>([null, null]);
  const [compareLoading, setCompareLoading] = useState(false);

  // Fetch run history
  const fetchRuns = useCallback(async () => {
    setLoading(true);
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (statusFilter) params.set("status", statusFilter);
    if (triggerFilter) params.set("trigger_type", triggerFilter);
    if (startDate) params.set("start_date", startDate);
    if (endDate) params.set("end_date", endDate);

    try {
      const res = await fetch(`/api/v1/workflows/${workflowId}/runs?${params}`);
      if (res.ok) {
        const data = await res.json();
        setRuns(data.runs);
        setTotal(data.total);
      }
    } finally {
      setLoading(false);
    }
  }, [workflowId, offset, statusFilter, triggerFilter, startDate, endDate]);

  // Fetch on mount and when filters change
  useEffect(() => {
    if (!open) return;
    setOffset(0);
  }, [open, statusFilter, triggerFilter, startDate, endDate]);

  useEffect(() => {
    if (!open) return;
    fetchRuns();
  }, [open, fetchRuns]);

  // Auto-refresh every 10 seconds when panel is open
  const refreshRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (!open) {
      if (refreshRef.current) clearInterval(refreshRef.current);
      return;
    }
    refreshRef.current = setInterval(fetchRuns, 10000);
    return () => {
      if (refreshRef.current) clearInterval(refreshRef.current);
    };
  }, [open, fetchRuns]);

  // Fetch run detail
  const fetchRunDetail = useCallback(async (runId: string) => {
    setDetailLoading(true);
    try {
      const res = await fetch(`/api/v1/workflows/${workflowId}/runs/${runId}`);
      if (res.ok) {
        const data: RunDetail = await res.json();
        setRunDetail(data);
      }
    } finally {
      setDetailLoading(false);
    }
  }, [workflowId]);

  // When a run is selected, load detail
  useEffect(() => {
    if (selectedRunId) fetchRunDetail(selectedRunId);
  }, [selectedRunId, fetchRunDetail]);

  // Handle clicking a run — load it onto the canvas
  const handleLoadRun = useCallback((run: RunDetail) => {
    const statuses: Record<string, NodeExecStatus> = {};
    for (const ne of run.node_executions) {
      if (ne.status === "completed") statuses[ne.node_id] = "completed";
      else if (ne.status === "failed") statuses[ne.node_id] = "failed";
    }
    onLoadRunState(statuses, run.id);
  }, [onLoadRunState]);

  // Replay with same inputs
  const handleReplaySame = useCallback(() => {
    if (!runDetail) return;
    onReplay(runDetail.input_json ?? undefined);
  }, [runDetail, onReplay]);

  // Replay with modified inputs
  const handleReplayModified = useCallback(() => {
    try {
      const parsed = JSON.parse(replayInputText);
      setReplayParseError(null);
      setReplayOpen(false);
      onReplay(parsed);
    } catch (e) {
      setReplayParseError((e as Error).message);
    }
  }, [replayInputText, onReplay]);

  const openReplayEditor = useCallback(() => {
    if (runDetail?.input_json) {
      setReplayInputText(JSON.stringify(runDetail.input_json, null, 2));
    } else {
      setReplayInputText("{}");
    }
    setReplayParseError(null);
    setReplayOpen(true);
  }, [runDetail]);

  // Comparison mode
  const toggleCompare = useCallback((runId: string) => {
    setCompareRunIds(([a, b]) => {
      if (a === runId) return [b, null];
      if (b === runId) return [a, null];
      if (a === null) return [runId, b];
      if (b === null) return [a, runId];
      return [runId, b]; // replace first
    });
  }, []);

  // Fetch comparison details when both are selected
  useEffect(() => {
    if (!compareMode) return;
    const [a, b] = compareRunIds;
    if (!a || !b) {
      setCompareDetails([null, null]);
      return;
    }

    setCompareLoading(true);
    Promise.all([
      fetch(`/api/v1/workflows/${workflowId}/runs/${a}`).then((r) => r.ok ? r.json() : null),
      fetch(`/api/v1/workflows/${workflowId}/runs/${b}`).then((r) => r.ok ? r.json() : null),
    ]).then(([da, db]) => {
      setCompareDetails([da, db]);
    }).finally(() => {
      setCompareLoading(false);
    });
  }, [compareMode, compareRunIds, workflowId]);

  // Fetch checkpoints when a run is selected
  const fetchCheckpoints = useCallback(async (runId: string) => {
    setCheckpointsLoading(true);
    try {
      const res = await fetch(`/api/v1/runs/${runId}/checkpoints?limit=50`);
      if (res.ok) {
        const data = await res.json();
        setCheckpoints(data.checkpoints);
        setCheckpointsTotal(data.total);
      }
    } finally {
      setCheckpointsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedRunId) {
      fetchCheckpoints(selectedRunId);
      setCheckpointDiff(null);
      setDiffIds([null, null]);
    } else {
      setCheckpoints([]);
      setCheckpointsTotal(0);
    }
  }, [selectedRunId, fetchCheckpoints]);

  const handleSaveCheckpoint = useCallback(async () => {
    if (!selectedRunId || !runDetail) return;
    setSavingCheckpoint(true);
    try {
      const res = await fetch(`/api/v1/runs/${selectedRunId}/checkpoints`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `Manual checkpoint @ step ${runDetail.step_count ?? 0}`,
          step_number: runDetail.step_count ?? 0,
          state_blob: { status: runDetail.status, step_count: runDetail.step_count, total_tokens: runDetail.total_tokens },
        }),
      });
      if (res.ok) {
        fetchCheckpoints(selectedRunId);
      }
    } finally {
      setSavingCheckpoint(false);
    }
  }, [selectedRunId, runDetail, fetchCheckpoints]);

  const handleRestoreCheckpoint = useCallback(async (cpId: string) => {
    if (!selectedRunId) return;
    const res = await fetch(`/api/v1/runs/${selectedRunId}/checkpoints/${cpId}/restore`, { method: "POST" });
    if (res.ok) {
      fetchRunDetail(selectedRunId);
    }
  }, [selectedRunId, fetchRunDetail]);

  const handleDeleteCheckpoint = useCallback(async (cpId: string) => {
    if (!selectedRunId) return;
    const res = await fetch(`/api/v1/runs/${selectedRunId}/checkpoints/${cpId}`, { method: "DELETE" });
    if (res.ok || res.status === 204) {
      fetchCheckpoints(selectedRunId);
    }
  }, [selectedRunId, fetchCheckpoints]);

  const handleDiffCheckpoints = useCallback(async (idA: string, idB: string) => {
    if (!selectedRunId) return;
    setDiffLoading(true);
    try {
      const res = await fetch(`/api/v1/runs/${selectedRunId}/checkpoints/diff?a=${idA}&b=${idB}`);
      if (res.ok) {
        const data: CheckpointDiff = await res.json();
        setCheckpointDiff(data);
      }
    } finally {
      setDiffLoading(false);
    }
  }, [selectedRunId]);

  const toggleDiffSelection = useCallback((cpId: string) => {
    setDiffIds(([a, b]) => {
      if (a === cpId) return [b, null];
      if (b === cpId) return [a, null];
      if (a === null) return [cpId, b];
      if (b === null) return [a, cpId];
      return [cpId, b];
    });
  }, []);

  // Trigger diff when both IDs are selected
  useEffect(() => {
    const [a, b] = diffIds;
    if (a && b) {
      handleDiffCheckpoints(a, b);
    } else {
      setCheckpointDiff(null);
    }
  }, [diffIds, handleDiffCheckpoints]);

  // Pagination
  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  if (!open) return null;

  return (
    <div
      className="nodrag nopan nowheel"
      style={{
        position: "absolute",
        right: 0,
        top: 0,
        bottom: 0,
        width: compareMode && compareDetails[0] && compareDetails[1] ? 640 : 340,
        zIndex: 20,
        display: "flex",
        flexDirection: "column",
        background: "var(--zen-paper, #f2f0e3)",
        borderLeft: "1px solid var(--zen-subtle, #e0ddd0)",
        boxShadow: "-2px 0 12px rgba(0,0,0,0.06)",
        fontFamily: "'Bricolage Grotesque', sans-serif",
        overflow: "hidden",
        transition: "width 200ms ease",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 14px",
          borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
          </svg>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--zen-dark, #2e2e2e)" }}>
            Run History
          </span>
          <span style={{ fontSize: 11, color: "var(--zen-muted, #999)" }}>
            ({total})
          </span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <button
            onClick={() => {
              setCompareMode((v) => !v);
              setCompareRunIds([null, null]);
              setCompareDetails([null, null]);
            }}
            title="Compare Runs"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 28,
              height: 28,
              border: compareMode ? "1px solid var(--zen-coral, #F76F53)" : "1px solid transparent",
              borderRadius: 6,
              background: compareMode ? "rgba(247, 111, 83, 0.1)" : "transparent",
              color: compareMode ? "var(--zen-coral, #F76F53)" : "var(--zen-muted, #999)",
              cursor: "pointer",
              fontSize: 11,
              fontWeight: 600,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" />
            </svg>
          </button>
          <button
            onClick={fetchRuns}
            title="Refresh"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 28,
              height: 28,
              border: "none",
              borderRadius: 6,
              background: "transparent",
              color: "var(--zen-muted, #999)",
              cursor: "pointer",
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10" /><path d="M20.49 15a9 9 0 1 1-2.13-9.36L23 10" />
            </svg>
          </button>
          <button
            onClick={onToggle}
            title="Close History"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 28,
              height: 28,
              border: "none",
              borderRadius: 6,
              background: "transparent",
              color: "var(--zen-muted, #999)",
              cursor: "pointer",
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
      </div>

      {/* Filters */}
      <div style={{ padding: "8px 14px", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)", display: "flex", flexWrap: "wrap", gap: 6, flexShrink: 0 }}>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          style={filterSelectStyle}
        >
          <option value="">All statuses</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="cancelled">Cancelled</option>
          <option value="running">Running</option>
          <option value="pending">Pending</option>
        </select>
        <select
          value={triggerFilter}
          onChange={(e) => setTriggerFilter(e.target.value)}
          style={filterSelectStyle}
        >
          <option value="">All triggers</option>
          <option value="manual">Manual</option>
          <option value="scheduled">Scheduled</option>
          <option value="webhook">Webhook</option>
          <option value="api">API</option>
        </select>
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          placeholder="From"
          title="Start date"
          style={filterInputStyle}
        />
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          placeholder="To"
          title="End date"
          style={filterInputStyle}
        />
      </div>

      {/* Content area */}
      {compareMode && compareDetails[0] && compareDetails[1] ? (
        <ComparisonView
          runA={compareDetails[0]}
          runB={compareDetails[1]}
          loading={compareLoading}
        />
      ) : selectedRunId && runDetail ? (
        <RunDetailView
          run={runDetail}
          loading={detailLoading}
          onBack={() => {
            setSelectedRunId(null);
            setRunDetail(null);
          }}
          onLoadRun={() => handleLoadRun(runDetail)}
          onReplaySame={handleReplaySame}
          onReplayModified={openReplayEditor}
          checkpoints={checkpoints}
          checkpointsTotal={checkpointsTotal}
          checkpointsLoading={checkpointsLoading}
          savingCheckpoint={savingCheckpoint}
          onSaveCheckpoint={handleSaveCheckpoint}
          onRestoreCheckpoint={handleRestoreCheckpoint}
          onDeleteCheckpoint={handleDeleteCheckpoint}
          diffIds={diffIds}
          onToggleDiffSelection={toggleDiffSelection}
          checkpointDiff={checkpointDiff}
          diffLoading={diffLoading}
          onClearDiff={() => { setDiffIds([null, null]); setCheckpointDiff(null); }}
          autoCheckpointInterval={autoCheckpointInterval}
          onAutoCheckpointIntervalChange={setAutoCheckpointInterval}
        />
      ) : (
        <div style={{ flex: 1, overflow: "auto" }}>
          {loading ? (
            <div style={{ padding: 20, textAlign: "center", color: "var(--zen-muted, #999)", fontSize: 12 }}>
              Loading...
            </div>
          ) : runs.length === 0 ? (
            <div style={{ padding: 20, textAlign: "center", color: "var(--zen-muted, #999)", fontSize: 12 }}>
              No runs found
            </div>
          ) : (
            <>
              {runs.map((run) => (
                <RunRow
                  key={run.id}
                  run={run}
                  compareMode={compareMode}
                  isCompareSelected={compareRunIds.includes(run.id)}
                  onSelect={() => {
                    if (compareMode) {
                      toggleCompare(run.id);
                    } else {
                      setSelectedRunId(run.id);
                    }
                  }}
                />
              ))}

              {/* Pagination */}
              {totalPages > 1 && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 8,
                    padding: "10px 14px",
                    borderTop: "1px solid var(--zen-subtle, #e0ddd0)",
                  }}
                >
                  <button
                    onClick={() => setOffset(Math.max(0, offset - limit))}
                    disabled={offset === 0}
                    style={paginationBtnStyle}
                  >
                    Prev
                  </button>
                  <span style={{ fontSize: 11, color: "var(--zen-muted, #999)" }}>
                    {currentPage} / {totalPages}
                  </span>
                  <button
                    onClick={() => setOffset(offset + limit)}
                    disabled={currentPage >= totalPages}
                    style={paginationBtnStyle}
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Replay modification modal */}
      {replayOpen && (
        <ReplayInputEditor
          inputText={replayInputText}
          parseError={replayParseError}
          onChange={(text) => {
            setReplayInputText(text);
            setReplayParseError(null);
          }}
          onRun={handleReplayModified}
          onClose={() => setReplayOpen(false)}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Shared styles                                                       */
/* ------------------------------------------------------------------ */

const filterSelectStyle: React.CSSProperties = {
  fontSize: 11,
  fontFamily: "'Bricolage Grotesque', sans-serif",
  padding: "4px 6px",
  borderRadius: 6,
  border: "1px solid var(--zen-subtle, #e0ddd0)",
  background: "var(--zen-paper, #f2f0e3)",
  color: "var(--zen-dark, #2e2e2e)",
  outline: "none",
  cursor: "pointer",
};

const filterInputStyle: React.CSSProperties = {
  fontSize: 11,
  fontFamily: "'Bricolage Grotesque', sans-serif",
  padding: "4px 6px",
  borderRadius: 6,
  border: "1px solid var(--zen-subtle, #e0ddd0)",
  background: "var(--zen-paper, #f2f0e3)",
  color: "var(--zen-dark, #2e2e2e)",
  outline: "none",
  width: 110,
};

const cpActionBtnStyle: React.CSSProperties = {
  fontSize: 10,
  fontWeight: 500,
  fontFamily: "'Bricolage Grotesque', sans-serif",
  padding: "3px 8px",
  borderRadius: 5,
  border: "1px solid var(--zen-subtle, #e0ddd0)",
  background: "transparent",
  color: "var(--zen-dark, #2e2e2e)",
  cursor: "pointer",
};

const paginationBtnStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 500,
  fontFamily: "'Bricolage Grotesque', sans-serif",
  padding: "4px 10px",
  borderRadius: 6,
  border: "1px solid var(--zen-subtle, #e0ddd0)",
  background: "transparent",
  color: "var(--zen-dark, #2e2e2e)",
  cursor: "pointer",
};

/* ------------------------------------------------------------------ */
/* RunRow — single run in the list                                     */
/* ------------------------------------------------------------------ */

function RunRow({
  run,
  compareMode,
  isCompareSelected,
  onSelect,
}: {
  run: RunSummary;
  compareMode: boolean;
  isCompareSelected: boolean;
  onSelect: () => void;
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onClick={onSelect}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 14px",
        borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
        cursor: "pointer",
        background: hovered
          ? "rgba(0,0,0,0.03)"
          : isCompareSelected
            ? "rgba(247, 111, 83, 0.06)"
            : "transparent",
        transition: "background 100ms",
      }}
    >
      {/* Compare checkbox */}
      {compareMode && (
        <div
          style={{
            width: 16,
            height: 16,
            borderRadius: 4,
            border: isCompareSelected
              ? "2px solid var(--zen-coral, #F76F53)"
              : "2px solid var(--zen-muted, #ccc)",
            background: isCompareSelected ? "var(--zen-coral, #F76F53)" : "transparent",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          {isCompareSelected && (
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          )}
        </div>
      )}

      {/* Status dot */}
      <div
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: STATUS_COLORS[run.status] ?? "var(--zen-muted, #999)",
          flexShrink: 0,
        }}
      />

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: "var(--zen-dark, #2e2e2e)", textTransform: "capitalize" }}>
            {run.status}
          </span>
          {run.trigger_type && (
            <span
              style={{
                fontSize: 9,
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                padding: "1px 5px",
                borderRadius: 4,
                background: "var(--zen-subtle, #e0ddd0)",
                color: "var(--zen-muted, #999)",
              }}
            >
              {run.trigger_type}
            </span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 2 }}>
          <span style={{ fontSize: 10, color: "var(--zen-muted, #999)" }}>
            {formatTimestamp(run.created_at)}
          </span>
          <span style={{ fontSize: 10, color: "var(--zen-muted, #999)" }}>
            {formatDuration(run.started_at, run.completed_at)}
          </span>
          {run.step_count != null && (
            <span style={{ fontSize: 10, color: "var(--zen-muted, #999)" }}>
              {run.step_count} steps
            </span>
          )}
        </div>
      </div>

      {/* Chevron */}
      {!compareMode && (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--zen-muted, #999)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="9 18 15 12 9 6" />
        </svg>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* RunDetailView — full detail when a run is selected                  */
/* ------------------------------------------------------------------ */

function RunDetailView({
  run,
  loading,
  onBack,
  onLoadRun,
  onReplaySame,
  onReplayModified,
  checkpoints,
  checkpointsTotal,
  checkpointsLoading,
  savingCheckpoint,
  onSaveCheckpoint,
  onRestoreCheckpoint,
  onDeleteCheckpoint,
  diffIds,
  onToggleDiffSelection,
  checkpointDiff,
  diffLoading,
  onClearDiff,
  autoCheckpointInterval,
  onAutoCheckpointIntervalChange,
}: {
  run: RunDetail;
  loading: boolean;
  onBack: () => void;
  onLoadRun: () => void;
  onReplaySame: () => void;
  onReplayModified: () => void;
  checkpoints: Checkpoint[];
  checkpointsTotal: number;
  checkpointsLoading: boolean;
  savingCheckpoint: boolean;
  onSaveCheckpoint: () => void;
  onRestoreCheckpoint: (cpId: string) => void;
  onDeleteCheckpoint: (cpId: string) => void;
  diffIds: [string | null, string | null];
  onToggleDiffSelection: (cpId: string) => void;
  checkpointDiff: CheckpointDiff | null;
  diffLoading: boolean;
  onClearDiff: () => void;
  autoCheckpointInterval: number;
  onAutoCheckpointIntervalChange: (val: number) => void;
}) {
  if (loading) {
    return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--zen-muted, #999)", fontSize: 12 }}>
        Loading...
      </div>
    );
  }

  return (
    <div style={{ flex: 1, overflow: "auto" }}>
      {/* Back button + header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "10px 14px",
          borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
        }}
      >
        <button
          onClick={onBack}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 26,
            height: 26,
            border: "none",
            borderRadius: 6,
            background: "transparent",
            color: "var(--zen-dark, #2e2e2e)",
            cursor: "pointer",
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--zen-dark, #2e2e2e)", textTransform: "capitalize" }}>
            {run.status}
          </div>
          <div style={{ fontSize: 10, color: "var(--zen-muted, #999)" }}>
            {run.id.slice(0, 8)}
          </div>
        </div>
      </div>

      {/* Run summary */}
      <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <MetricItem label="Status" value={run.status} color={STATUS_COLORS[run.status]} />
          <MetricItem label="Duration" value={formatDuration(run.started_at, run.completed_at)} />
          <MetricItem label="Steps" value={String(run.step_count ?? 0)} />
          <MetricItem label="Tokens" value={String(run.total_tokens ?? 0)} />
          <MetricItem label="Cost" value={run.total_cost != null ? `$${run.total_cost.toFixed(4)}` : "-"} />
          <MetricItem label="Trigger" value={run.trigger_type ?? "manual"} />
        </div>
        {run.error && (
          <div style={{ marginTop: 8, padding: "6px 8px", borderRadius: 6, background: "rgba(239, 68, 68, 0.1)", fontSize: 11, color: "#ef4444", fontFamily: "monospace", wordBreak: "break-all" }}>
            {run.error}
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div style={{ display: "flex", gap: 6, padding: "10px 14px", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)", flexWrap: "wrap" }}>
        <ActionButton label="Load on Canvas" onClick={onLoadRun} icon={loadIcon} />
        <ActionButton label="Replay" onClick={onReplaySame} icon={replayIcon} />
        <ActionButton label="Replay Modified" onClick={onReplayModified} icon={editIcon} />
      </div>

      {/* Checkpoints section */}
      <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "var(--zen-muted, #999)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Checkpoints ({checkpointsTotal})
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {diffIds[0] || diffIds[1] ? (
              <button
                onClick={onClearDiff}
                style={{ ...cpActionBtnStyle, color: "var(--zen-coral, #F76F53)", borderColor: "var(--zen-coral, #F76F53)" }}
              >
                Clear Diff
              </button>
            ) : null}
            <button
              onClick={onSaveCheckpoint}
              disabled={savingCheckpoint || run.status !== "running"}
              title={run.status !== "running" ? "Can only save checkpoints during active execution" : "Save checkpoint"}
              style={{
                ...cpActionBtnStyle,
                opacity: run.status !== "running" ? 0.4 : 1,
                cursor: run.status !== "running" ? "not-allowed" : "pointer",
              }}
            >
              {savingCheckpoint ? "Saving..." : "Save Checkpoint"}
            </button>
          </div>
        </div>

        {/* Auto-checkpoint interval */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
          <span style={{ fontSize: 10, color: "var(--zen-muted, #999)" }}>Auto-checkpoint every</span>
          <input
            type="number"
            min={0}
            max={100}
            value={autoCheckpointInterval}
            onChange={(e) => onAutoCheckpointIntervalChange(Math.max(0, parseInt(e.target.value) || 0))}
            style={{
              width: 42,
              fontSize: 11,
              fontFamily: "'Bricolage Grotesque', sans-serif",
              padding: "2px 4px",
              borderRadius: 4,
              border: "1px solid var(--zen-subtle, #e0ddd0)",
              background: "var(--zen-paper, #f2f0e3)",
              color: "var(--zen-dark, #2e2e2e)",
              textAlign: "center",
              outline: "none",
            }}
          />
          <span style={{ fontSize: 10, color: "var(--zen-muted, #999)" }}>steps</span>
          {autoCheckpointInterval === 0 && (
            <span style={{ fontSize: 9, color: "var(--zen-coral, #F76F53)" }}>(disabled)</span>
          )}
        </div>

        {checkpointsLoading ? (
          <div style={{ fontSize: 11, color: "var(--zen-muted, #999)", padding: "8px 0" }}>Loading checkpoints...</div>
        ) : checkpoints.length === 0 ? (
          <div style={{ fontSize: 11, color: "var(--zen-muted, #999)", padding: "8px 0", fontStyle: "italic" }}>No checkpoints saved</div>
        ) : (
          <>
            <div style={{ fontSize: 9, color: "var(--zen-muted, #999)", marginBottom: 4 }}>
              Click two checkpoints to compare state
            </div>
            {checkpoints.map((cp) => (
              <CheckpointRow
                key={cp.id}
                cp={cp}
                isSelected={diffIds.includes(cp.id)}
                onToggleDiff={() => onToggleDiffSelection(cp.id)}
                onRestore={() => onRestoreCheckpoint(cp.id)}
                onDelete={() => onDeleteCheckpoint(cp.id)}
              />
            ))}
          </>
        )}

        {/* Checkpoint diff result */}
        {diffLoading && (
          <div style={{ fontSize: 11, color: "var(--zen-muted, #999)", padding: "8px 0" }}>Computing diff...</div>
        )}
        {checkpointDiff && !diffLoading && (
          <CheckpointDiffView diff={checkpointDiff} />
        )}
      </div>

      {/* Node executions */}
      <div style={{ padding: "10px 14px" }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "var(--zen-muted, #999)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Node Executions ({run.node_executions.length})
        </div>
        {run.node_executions.map((ne) => (
          <NodeExecutionRow key={ne.id} ne={ne} />
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* MetricItem                                                          */
/* ------------------------------------------------------------------ */

function MetricItem({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div style={{ fontSize: 9, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--zen-muted, #999)" }}>
        {label}
      </div>
      <div style={{ fontSize: 12, fontWeight: 600, color: color ?? "var(--zen-dark, #2e2e2e)", textTransform: "capitalize" }}>
        {value}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* ActionButton                                                        */
/* ------------------------------------------------------------------ */

function ActionButton({ label, onClick, icon }: { label: string; onClick: () => void; icon: React.ReactNode }) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 5,
        padding: "5px 10px",
        fontSize: 11,
        fontWeight: 500,
        fontFamily: "'Bricolage Grotesque', sans-serif",
        borderRadius: 6,
        border: "1px solid var(--zen-subtle, #e0ddd0)",
        background: hovered ? "var(--zen-subtle, #e0ddd0)" : "transparent",
        color: "var(--zen-dark, #2e2e2e)",
        cursor: "pointer",
        transition: "background 100ms",
      }}
    >
      {icon}
      {label}
    </button>
  );
}

const loadIcon = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" />
  </svg>
);

const replayIcon = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="5 3 19 12 5 21 5 3" />
  </svg>
);

const editIcon = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
  </svg>
);

/* ------------------------------------------------------------------ */
/* NodeExecutionRow                                                    */
/* ------------------------------------------------------------------ */

function NodeExecutionRow({ ne }: { ne: NodeExecution }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      style={{
        marginBottom: 4,
        borderRadius: 6,
        border: "1px solid var(--zen-subtle, #e0ddd0)",
        overflow: "hidden",
      }}
    >
      <div
        onClick={() => setExpanded((v) => !v)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 10px",
          cursor: "pointer",
          background: "transparent",
        }}
      >
        <div
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: STATUS_COLORS[ne.status] ?? "var(--zen-muted, #999)",
            flexShrink: 0,
          }}
        />
        <span style={{ flex: 1, fontSize: 11, fontWeight: 500, color: "var(--zen-dark, #2e2e2e)" }}>
          {ne.node_id}
        </span>
        <span style={{ fontSize: 10, color: "var(--zen-muted, #999)", textTransform: "capitalize" }}>
          {ne.status}
        </span>
        <svg
          width="10"
          height="10"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--zen-muted, #999)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transform: expanded ? "rotate(180deg)" : "rotate(0)", transition: "transform 150ms" }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>

      {expanded && (
        <div style={{ padding: "0 10px 10px", fontSize: 11 }}>
          {ne.input_json && (
            <JsonSection label="Input" data={ne.input_json} />
          )}
          {ne.output_json && (
            <JsonSection label="Output" data={ne.output_json} />
          )}
          {ne.logs_text && (
            <div style={{ marginTop: 6 }}>
              <div style={{ fontSize: 9, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--zen-muted, #999)", marginBottom: 3 }}>
                Logs
              </div>
              <pre style={{ padding: "6px 8px", borderRadius: 4, background: "var(--zen-subtle, #e0ddd0)", margin: 0, fontSize: 10, fontFamily: "monospace", whiteSpace: "pre-wrap", wordBreak: "break-all", maxHeight: 120, overflow: "auto" }}>
                {ne.logs_text}
              </pre>
            </div>
          )}
          {ne.token_usage_json && (
            <JsonSection label="Token Usage" data={ne.token_usage_json} />
          )}
          {ne.error && (
            <div style={{ marginTop: 6, padding: "6px 8px", borderRadius: 4, background: "rgba(239, 68, 68, 0.1)", fontSize: 10, color: "#ef4444", fontFamily: "monospace", wordBreak: "break-all" }}>
              {ne.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* JsonSection — expandable JSON viewer                                */
/* ------------------------------------------------------------------ */

function JsonSection({ label, data }: { label: string; data: Record<string, unknown> }) {
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ fontSize: 9, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--zen-muted, #999)", marginBottom: 3 }}>
        {label}
      </div>
      <pre
        style={{
          padding: "6px 8px",
          borderRadius: 4,
          background: "var(--zen-subtle, #e0ddd0)",
          margin: 0,
          fontSize: 10,
          fontFamily: "monospace",
          whiteSpace: "pre-wrap",
          wordBreak: "break-all",
          maxHeight: 120,
          overflow: "auto",
        }}
      >
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* CheckpointRow                                                       */
/* ------------------------------------------------------------------ */

function CheckpointRow({
  cp,
  isSelected,
  onToggleDiff,
  onRestore,
  onDelete,
}: {
  cp: Checkpoint;
  isSelected: boolean;
  onToggleDiff: () => void;
  onRestore: () => void;
  onDelete: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const blobSize = useMemo(() => {
    const str = typeof cp.state_blob === "string" ? cp.state_blob : JSON.stringify(cp.state_blob);
    const bytes = new Blob([str]).size;
    if (bytes < 1024) return `${bytes} B`;
    return `${(bytes / 1024).toFixed(1)} KB`;
  }, [cp.state_blob]);

  return (
    <div
      onClick={onToggleDiff}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "7px 8px",
        marginBottom: 3,
        borderRadius: 6,
        border: isSelected ? "1px solid var(--zen-blue, #6287f5)" : "1px solid var(--zen-subtle, #e0ddd0)",
        background: isSelected
          ? "rgba(98, 135, 245, 0.06)"
          : hovered
            ? "rgba(0,0,0,0.02)"
            : "transparent",
        cursor: "pointer",
        transition: "background 100ms, border-color 100ms",
      }}
    >
      {/* Diff selection indicator */}
      <div
        style={{
          width: 14,
          height: 14,
          borderRadius: 3,
          border: isSelected ? "2px solid var(--zen-blue, #6287f5)" : "2px solid var(--zen-muted, #ccc)",
          background: isSelected ? "var(--zen-blue, #6287f5)" : "transparent",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        {isSelected && (
          <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        )}
      </div>

      {/* Checkpoint info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 500, color: "var(--zen-dark, #2e2e2e)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {cp.name}
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 1 }}>
          <span style={{ fontSize: 9, color: "var(--zen-muted, #999)" }}>
            Step {cp.step_number}
          </span>
          <span style={{ fontSize: 9, color: "var(--zen-muted, #999)" }}>
            {blobSize}
          </span>
          <span style={{ fontSize: 9, color: "var(--zen-muted, #999)" }}>
            {formatTimestamp(cp.created_at)}
          </span>
        </div>
      </div>

      {/* Action buttons */}
      {hovered && (
        <div style={{ display: "flex", gap: 2, flexShrink: 0 }} onClick={(e) => e.stopPropagation()}>
          <button
            onClick={onRestore}
            title="Restore from this checkpoint"
            style={{ ...cpSmallBtnStyle, color: "var(--zen-green, #63f78b)" }}
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
            </svg>
          </button>
          <button
            onClick={onDelete}
            title="Delete this checkpoint"
            style={{ ...cpSmallBtnStyle, color: "#ef4444" }}
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}

const cpSmallBtnStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  width: 22,
  height: 22,
  border: "none",
  borderRadius: 4,
  background: "transparent",
  cursor: "pointer",
};

/* ------------------------------------------------------------------ */
/* CheckpointDiffView                                                  */
/* ------------------------------------------------------------------ */

function CheckpointDiffView({ diff }: { diff: CheckpointDiff }) {
  const hasChanges = diff.added_keys.length > 0 || diff.removed_keys.length > 0 || diff.changed_keys.length > 0;

  return (
    <div
      style={{
        marginTop: 8,
        padding: "8px 10px",
        borderRadius: 6,
        border: "1px solid var(--zen-blue, #6287f5)",
        background: "rgba(98, 135, 245, 0.04)",
      }}
    >
      <div style={{ fontSize: 10, fontWeight: 600, color: "var(--zen-blue, #6287f5)", marginBottom: 6 }}>
        Diff: {diff.checkpoint_a.name} vs {diff.checkpoint_b.name}
      </div>

      {!hasChanges ? (
        <div style={{ fontSize: 10, color: "var(--zen-muted, #999)", fontStyle: "italic" }}>
          No state differences
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {diff.added_keys.length > 0 && (
            <DiffKeyList label="Added" keys={diff.added_keys} color="var(--zen-green, #63f78b)" />
          )}
          {diff.removed_keys.length > 0 && (
            <DiffKeyList label="Removed" keys={diff.removed_keys} color="#ef4444" />
          )}
          {diff.changed_keys.length > 0 && (
            <DiffKeyList label="Changed" keys={diff.changed_keys} color="var(--zen-coral, #F76F53)" />
          )}
          <div style={{ fontSize: 9, color: "var(--zen-muted, #999)" }}>
            {diff.unchanged_keys.length} unchanged key{diff.unchanged_keys.length !== 1 ? "s" : ""}
          </div>
        </div>
      )}
    </div>
  );
}

function DiffKeyList({ label, keys, color }: { label: string; keys: string[]; color: string }) {
  return (
    <div>
      <span style={{ fontSize: 9, fontWeight: 600, color, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label} ({keys.length})
      </span>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginTop: 2 }}>
        {keys.map((k) => (
          <span
            key={k}
            style={{
              fontSize: 9,
              fontFamily: "monospace",
              padding: "1px 5px",
              borderRadius: 3,
              background: "var(--zen-subtle, #e0ddd0)",
              color: "var(--zen-dark, #2e2e2e)",
            }}
          >
            {k}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* ComparisonView — side-by-side diff of two runs                      */
/* ------------------------------------------------------------------ */

function ComparisonView({
  runA,
  runB,
  loading,
}: {
  runA: RunDetail;
  runB: RunDetail;
  loading: boolean;
}) {
  // Group node executions by node_id for comparison
  const nodeIdsA = useMemo(() => new Map(runA.node_executions.map((ne) => [ne.node_id, ne])), [runA]);
  const nodeIdsB = useMemo(() => new Map(runB.node_executions.map((ne) => [ne.node_id, ne])), [runB]);
  const allNodeIds = useMemo(() => {
    const set = new Set([...nodeIdsA.keys(), ...nodeIdsB.keys()]);
    return [...set];
  }, [nodeIdsA, nodeIdsB]);

  if (loading) {
    return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--zen-muted, #999)", fontSize: 12 }}>
        Loading comparison...
      </div>
    );
  }

  return (
    <div style={{ flex: 1, overflow: "auto" }}>
      {/* Header row */}
      <div style={{ display: "flex", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)" }}>
        <div style={{ flex: 1, padding: "10px 14px", borderRight: "1px solid var(--zen-subtle, #e0ddd0)" }}>
          <ComparisonRunHeader run={runA} label="Run A" />
        </div>
        <div style={{ flex: 1, padding: "10px 14px" }}>
          <ComparisonRunHeader run={runB} label="Run B" />
        </div>
      </div>

      {/* Summary comparison */}
      <div style={{ display: "flex", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)" }}>
        <div style={{ flex: 1, padding: "8px 14px", borderRight: "1px solid var(--zen-subtle, #e0ddd0)" }}>
          <ComparisonSummary run={runA} />
        </div>
        <div style={{ flex: 1, padding: "8px 14px" }}>
          <ComparisonSummary run={runB} />
        </div>
      </div>

      {/* Node-by-node comparison */}
      <div style={{ padding: "10px 14px" }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "var(--zen-muted, #999)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Node Comparison
        </div>
        {allNodeIds.map((nodeId) => {
          const a = nodeIdsA.get(nodeId);
          const b = nodeIdsB.get(nodeId);
          return (
            <ComparisonNodeRow key={nodeId} nodeId={nodeId} a={a ?? null} b={b ?? null} />
          );
        })}
      </div>
    </div>
  );
}

function ComparisonRunHeader({ run, label }: { run: RunDetail; label: string }) {
  return (
    <div>
      <div style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--zen-coral, #F76F53)" }}>
        {label}
      </div>
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--zen-dark, #2e2e2e)" }}>
        {run.id.slice(0, 8)}
      </div>
      <div style={{ fontSize: 10, color: "var(--zen-muted, #999)" }}>
        {formatTimestamp(run.created_at)}
      </div>
    </div>
  );
}

function ComparisonSummary({ run }: { run: RunDetail }) {
  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
      <CompactMetric label="Status" value={run.status} color={STATUS_COLORS[run.status]} />
      <CompactMetric label="Steps" value={String(run.step_count ?? 0)} />
      <CompactMetric label="Tokens" value={String(run.total_tokens ?? 0)} />
      <CompactMetric label="Duration" value={formatDuration(run.started_at, run.completed_at)} />
    </div>
  );
}

function CompactMetric({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div style={{ fontSize: 8, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--zen-muted, #999)" }}>
        {label}
      </div>
      <div style={{ fontSize: 11, fontWeight: 600, color: color ?? "var(--zen-dark, #2e2e2e)", textTransform: "capitalize" }}>
        {value}
      </div>
    </div>
  );
}

function ComparisonNodeRow({
  nodeId,
  a,
  b,
}: {
  nodeId: string;
  a: NodeExecution | null;
  b: NodeExecution | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const isDifferent = a?.status !== b?.status ||
    JSON.stringify(a?.output_json) !== JSON.stringify(b?.output_json);

  return (
    <div
      style={{
        marginBottom: 4,
        borderRadius: 6,
        border: isDifferent
          ? "1px solid var(--zen-coral, #F76F53)"
          : "1px solid var(--zen-subtle, #e0ddd0)",
        overflow: "hidden",
      }}
    >
      <div
        onClick={() => setExpanded((v) => !v)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 10px",
          cursor: "pointer",
        }}
      >
        <span style={{ flex: 1, fontSize: 11, fontWeight: 500, color: "var(--zen-dark, #2e2e2e)" }}>
          {nodeId}
        </span>
        {isDifferent && (
          <span style={{ fontSize: 9, fontWeight: 600, color: "var(--zen-coral, #F76F53)", textTransform: "uppercase" }}>
            diff
          </span>
        )}
        <div style={{ display: "flex", gap: 4 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: a ? (STATUS_COLORS[a.status] ?? "#999") : "#ddd" }} />
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: b ? (STATUS_COLORS[b.status] ?? "#999") : "#ddd" }} />
        </div>
        <svg
          width="10"
          height="10"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--zen-muted, #999)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transform: expanded ? "rotate(180deg)" : "rotate(0)", transition: "transform 150ms" }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>

      {expanded && (
        <div style={{ display: "flex", borderTop: "1px solid var(--zen-subtle, #e0ddd0)" }}>
          <div style={{ flex: 1, padding: "8px 10px", borderRight: "1px solid var(--zen-subtle, #e0ddd0)", fontSize: 10 }}>
            {a ? (
              <>
                <div style={{ fontSize: 9, fontWeight: 600, color: "var(--zen-muted, #999)", marginBottom: 4 }}>
                  Run A — {a.status}
                </div>
                {a.output_json && (
                  <pre style={{ padding: "4px 6px", borderRadius: 4, background: "var(--zen-subtle, #e0ddd0)", margin: 0, fontSize: 9, fontFamily: "monospace", whiteSpace: "pre-wrap", wordBreak: "break-all", maxHeight: 100, overflow: "auto" }}>
                    {JSON.stringify(a.output_json, null, 2)}
                  </pre>
                )}
                {a.error && <div style={{ color: "#ef4444", marginTop: 4, fontSize: 9 }}>{a.error}</div>}
              </>
            ) : (
              <div style={{ color: "var(--zen-muted, #999)", fontStyle: "italic" }}>Not executed</div>
            )}
          </div>
          <div style={{ flex: 1, padding: "8px 10px", fontSize: 10 }}>
            {b ? (
              <>
                <div style={{ fontSize: 9, fontWeight: 600, color: "var(--zen-muted, #999)", marginBottom: 4 }}>
                  Run B — {b.status}
                </div>
                {b.output_json && (
                  <pre style={{ padding: "4px 6px", borderRadius: 4, background: "var(--zen-subtle, #e0ddd0)", margin: 0, fontSize: 9, fontFamily: "monospace", whiteSpace: "pre-wrap", wordBreak: "break-all", maxHeight: 100, overflow: "auto" }}>
                    {JSON.stringify(b.output_json, null, 2)}
                  </pre>
                )}
                {b.error && <div style={{ color: "#ef4444", marginTop: 4, fontSize: 9 }}>{b.error}</div>}
              </>
            ) : (
              <div style={{ color: "var(--zen-muted, #999)", fontStyle: "italic" }}>Not executed</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* ReplayInputEditor — modal for editing inputs before replay          */
/* ------------------------------------------------------------------ */

function ReplayInputEditor({
  inputText,
  parseError,
  onChange,
  onRun,
  onClose,
}: {
  inputText: string;
  parseError: string | null;
  onChange: (text: string) => void;
  onRun: () => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onRun();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose, onRun]);

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: "rgba(0,0,0,0.3)",
        zIndex: 30,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 420,
          maxHeight: "80%",
          display: "flex",
          flexDirection: "column",
          background: "var(--zen-paper, #f2f0e3)",
          border: "1px solid var(--zen-subtle, #e0ddd0)",
          borderRadius: 12,
          boxShadow: "0 8px 32px rgba(0,0,0,0.18)",
          overflow: "hidden",
          fontFamily: "'Bricolage Grotesque', sans-serif",
        }}
      >
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 14px", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)" }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--zen-dark, #2e2e2e)" }}>
              Replay with Modified Inputs
            </div>
            <div style={{ fontSize: 10, color: "var(--zen-muted, #999)" }}>
              Edit workflow inputs before replaying
            </div>
          </div>
          <button
            onClick={onClose}
            style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 26, height: 26, border: "none", borderRadius: 6, background: "transparent", color: "var(--zen-muted, #999)", cursor: "pointer" }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Editor */}
        <div style={{ flex: 1, padding: "12px 14px", overflow: "auto" }}>
          <textarea
            value={inputText}
            onChange={(e) => onChange(e.target.value)}
            spellCheck={false}
            style={{
              width: "100%",
              minHeight: 200,
              padding: "10px 12px",
              fontSize: 12,
              fontFamily: "monospace",
              lineHeight: 1.5,
              background: "var(--zen-subtle, #e0ddd0)",
              border: parseError ? "1px solid #ef4444" : "1px solid transparent",
              borderRadius: 8,
              color: "var(--zen-dark, #2e2e2e)",
              resize: "vertical",
              outline: "none",
            }}
          />
          {parseError && (
            <div style={{ fontSize: 11, color: "#ef4444", marginTop: 4 }}>{parseError}</div>
          )}
        </div>

        {/* Footer */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, padding: "10px 14px", borderTop: "1px solid var(--zen-subtle, #e0ddd0)" }}>
          <button
            onClick={onClose}
            style={{ padding: "6px 14px", fontSize: 12, fontWeight: 500, fontFamily: "'Bricolage Grotesque', sans-serif", borderRadius: 6, border: "1px solid var(--zen-subtle, #e0ddd0)", background: "transparent", color: "var(--zen-dark, #2e2e2e)", cursor: "pointer" }}
          >
            Cancel
          </button>
          <button
            onClick={onRun}
            style={{ padding: "6px 14px", fontSize: 12, fontWeight: 600, fontFamily: "'Bricolage Grotesque', sans-serif", borderRadius: 6, border: "1px solid var(--zen-coral, #F76F53)", background: "var(--zen-coral, #F76F53)", color: "#fff", cursor: "pointer" }}
          >
            Replay
          </button>
        </div>
      </div>
    </div>
  );
}
