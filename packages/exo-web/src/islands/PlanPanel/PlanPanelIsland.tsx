/**
 * PlanPanelIsland — Autonomous mode plan management UI.
 *
 * Features:
 *  - Plan display: numbered step list with status icons (pending/running/completed/failed)
 *  - Plan modification: add, remove, reorder steps mid-execution
 *  - Verification results: pass/fail with explanation per step
 *  - Plan history: view previous plan versions
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface PlanStep {
  id: string;
  plan_id: string;
  step_number: number;
  description: string;
  dependencies_json: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  executor_output: string;
  verifier_result: string;
  verifier_passed: boolean | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
}

interface Plan {
  id: string;
  agent_id: string;
  goal: string;
  version: number;
  status: "active" | "superseded" | "completed" | "failed";
  user_id: string;
  created_at: string;
  updated_at: string;
  steps: PlanStep[];
}

type ViewMode = "current" | "history";

interface PlanPanelIslandProps {
  agentId: string;
}

/* ------------------------------------------------------------------ */
/* Status styling                                                      */
/* ------------------------------------------------------------------ */

const STATUS_COLORS: Record<string, string> = {
  pending: "var(--zen-muted, #999)",
  running: "var(--zen-coral, #F76F53)",
  completed: "var(--zen-green, #63f78b)",
  failed: "#ef4444",
  skipped: "var(--zen-muted, #999)",
};

const STATUS_BG: Record<string, string> = {
  pending: "var(--zen-subtle, #e0ddd0)",
  running: "rgba(247, 111, 83, 0.12)",
  completed: "rgba(99, 247, 139, 0.12)",
  failed: "rgba(239, 68, 68, 0.12)",
  skipped: "var(--zen-subtle, #e0ddd0)",
};

const PLAN_STATUS_LABELS: Record<string, string> = {
  active: "Active",
  superseded: "Superseded",
  completed: "Completed",
  failed: "Failed",
};

/* ------------------------------------------------------------------ */
/* SVG Icons                                                           */
/* ------------------------------------------------------------------ */

function StatusIcon({ status }: { status: string }) {
  const color = STATUS_COLORS[status] || "#999";

  switch (status) {
    case "completed":
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
          <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      );
    case "running":
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1.5s linear infinite" }}>
          <path d="M21 12a9 9 0 1 1-6.219-8.56" />
        </svg>
      );
    case "failed":
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="15" y1="9" x2="9" y2="15" />
          <line x1="9" y1="9" x2="15" y2="15" />
        </svg>
      );
    case "skipped":
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="8" y1="12" x2="16" y2="12" />
        </svg>
      );
    default: // pending
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 16 14" />
        </svg>
      );
  }
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function formatTimestamp(ts: string | null): string {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function parseDeps(depsJson: string): number[] {
  try {
    return JSON.parse(depsJson) as number[];
  } catch {
    return [];
  }
}

/* ------------------------------------------------------------------ */
/* StepRow                                                             */
/* ------------------------------------------------------------------ */

interface StepRowProps {
  step: PlanStep;
  expanded: boolean;
  onToggle: () => void;
  onDelete: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  isFirst: boolean;
  isLast: boolean;
  planStatus: string;
}

function StepRow({ step, expanded, onToggle, onDelete, onMoveUp, onMoveDown, isFirst, isLast, planStatus }: StepRowProps) {
  const deps = parseDeps(step.dependencies_json);
  const canModify = planStatus === "active" && step.status === "pending";

  return (
    <li
      style={{
        borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
      }}
    >
      {/* Step header row */}
      <div
        onClick={onToggle}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 14px",
          cursor: "pointer",
          transition: "background 100ms",
          background: expanded ? "var(--zen-subtle, #e0ddd0)" : "transparent",
        }}
        onMouseEnter={(e) => {
          if (!expanded) (e.currentTarget as HTMLDivElement).style.background = "rgba(0,0,0,0.02)";
        }}
        onMouseLeave={(e) => {
          if (!expanded) (e.currentTarget as HTMLDivElement).style.background = "transparent";
        }}
      >
        {/* Step number */}
        <span
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 24,
            height: 24,
            borderRadius: "50%",
            fontSize: 11,
            fontWeight: 700,
            color: STATUS_COLORS[step.status],
            background: STATUS_BG[step.status],
            flexShrink: 0,
          }}
        >
          {step.step_number}
        </span>

        {/* Status icon */}
        <span style={{ flexShrink: 0 }}>
          <StatusIcon status={step.status} />
        </span>

        {/* Description */}
        <span style={{ flex: 1, fontSize: 13, color: "var(--zen-dark, #2e2e2e)", lineHeight: 1.4 }}>
          {step.description}
        </span>

        {/* Status badge */}
        <span
          style={{
            fontSize: 10,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            color: STATUS_COLORS[step.status],
            flexShrink: 0,
          }}
        >
          {step.status}
        </span>

        {/* Expand chevron */}
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--zen-muted, #999)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transform: expanded ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 150ms", flexShrink: 0 }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div style={{ padding: "0 14px 12px", fontSize: 12, color: "var(--zen-dark, #2e2e2e)" }}>
          {/* Dependencies */}
          {deps.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <span style={{ fontWeight: 600, color: "var(--zen-muted, #999)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Dependencies:
              </span>{" "}
              <span style={{ fontSize: 12 }}>Steps {deps.join(", ")}</span>
            </div>
          )}

          {/* Timing */}
          {step.started_at && (
            <div style={{ marginBottom: 8, display: "flex", gap: 16, fontSize: 11, color: "var(--zen-muted, #999)" }}>
              <span>Started: {formatTimestamp(step.started_at)}</span>
              {step.completed_at && <span>Completed: {formatTimestamp(step.completed_at)}</span>}
            </div>
          )}

          {/* Executor output */}
          {step.executor_output && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontWeight: 600, color: "var(--zen-muted, #999)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>
                Executor Output
              </div>
              <div
                style={{
                  background: "rgba(0,0,0,0.03)",
                  borderRadius: 6,
                  padding: "8px 10px",
                  fontSize: 12,
                  lineHeight: 1.5,
                  whiteSpace: "pre-wrap",
                  maxHeight: 200,
                  overflowY: "auto",
                }}
              >
                {step.executor_output}
              </div>
            </div>
          )}

          {/* Verification result */}
          {step.verifier_result && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontWeight: 600, color: "var(--zen-muted, #999)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4, display: "flex", alignItems: "center", gap: 6 }}>
                Verification
                {step.verifier_passed !== null && (
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 3,
                      padding: "1px 6px",
                      borderRadius: 4,
                      fontSize: 10,
                      fontWeight: 700,
                      color: step.verifier_passed ? "var(--zen-green, #63f78b)" : "#ef4444",
                      background: step.verifier_passed ? "rgba(99, 247, 139, 0.12)" : "rgba(239, 68, 68, 0.12)",
                    }}
                  >
                    {step.verifier_passed ? "PASS" : "FAIL"}
                  </span>
                )}
              </div>
              <div
                style={{
                  background: "rgba(0,0,0,0.03)",
                  borderRadius: 6,
                  padding: "8px 10px",
                  fontSize: 12,
                  lineHeight: 1.5,
                  whiteSpace: "pre-wrap",
                  maxHeight: 150,
                  overflowY: "auto",
                }}
              >
                {step.verifier_result}
              </div>
            </div>
          )}

          {/* Modification buttons */}
          {canModify && (
            <div style={{ display: "flex", gap: 6, marginTop: 8, borderTop: "1px solid var(--zen-subtle, #e0ddd0)", paddingTop: 8 }}>
              <button
                onClick={(e) => { e.stopPropagation(); onMoveUp(); }}
                disabled={isFirst}
                style={{
                  display: "flex", alignItems: "center", gap: 3,
                  padding: "3px 8px", borderRadius: 4, border: "1px solid var(--zen-subtle, #e0ddd0)",
                  background: "var(--zen-paper, #f2f0e3)", fontSize: 11, color: "var(--zen-muted, #999)",
                  cursor: isFirst ? "not-allowed" : "pointer", opacity: isFirst ? 0.4 : 1,
                }}
                title="Move up"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="18 15 12 9 6 15" />
                </svg>
                Up
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onMoveDown(); }}
                disabled={isLast}
                style={{
                  display: "flex", alignItems: "center", gap: 3,
                  padding: "3px 8px", borderRadius: 4, border: "1px solid var(--zen-subtle, #e0ddd0)",
                  background: "var(--zen-paper, #f2f0e3)", fontSize: 11, color: "var(--zen-muted, #999)",
                  cursor: isLast ? "not-allowed" : "pointer", opacity: isLast ? 0.4 : 1,
                }}
                title="Move down"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="6 9 12 15 18 9" />
                </svg>
                Down
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(); }}
                style={{
                  display: "flex", alignItems: "center", gap: 3,
                  padding: "3px 8px", borderRadius: 4, border: "1px solid rgba(239, 68, 68, 0.3)",
                  background: "var(--zen-paper, #f2f0e3)", fontSize: 11, color: "#ef4444",
                  cursor: "pointer", marginLeft: "auto",
                }}
                title="Remove step"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
                Remove
              </button>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

/* ------------------------------------------------------------------ */
/* AddStepForm                                                         */
/* ------------------------------------------------------------------ */

function AddStepForm({ onAdd, onCancel }: { onAdd: (desc: string) => void; onCancel: () => void }) {
  const [desc, setDesc] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)", background: "rgba(98, 135, 245, 0.04)" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <input
          ref={inputRef}
          type="text"
          value={desc}
          onChange={(e) => setDesc(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && desc.trim()) onAdd(desc.trim());
            if (e.key === "Escape") onCancel();
          }}
          placeholder="Describe this step..."
          style={{
            flex: 1, padding: "6px 10px", borderRadius: 6,
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            background: "var(--zen-paper, #f2f0e3)",
            fontSize: 13, color: "var(--zen-dark, #2e2e2e)",
            outline: "none",
          }}
        />
        <button
          onClick={() => desc.trim() && onAdd(desc.trim())}
          disabled={!desc.trim()}
          style={{
            padding: "5px 12px", borderRadius: 6, border: "none",
            background: "var(--zen-coral, #F76F53)", color: "white",
            fontSize: 12, fontWeight: 600, cursor: desc.trim() ? "pointer" : "not-allowed",
            opacity: desc.trim() ? 1 : 0.5,
          }}
        >
          Add
        </button>
        <button
          onClick={onCancel}
          style={{
            padding: "5px 10px", borderRadius: 6,
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            background: "var(--zen-paper, #f2f0e3)",
            fontSize: 12, color: "var(--zen-muted, #999)", cursor: "pointer",
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* PlanHistoryItem                                                     */
/* ------------------------------------------------------------------ */

function PlanHistoryItem({ plan, isActive, onSelect }: { plan: Plan; isActive: boolean; onSelect: () => void }) {
  const completedSteps = plan.steps.filter((s) => s.status === "completed").length;
  const totalSteps = plan.steps.length;

  return (
    <div
      onClick={onSelect}
      style={{
        padding: "10px 14px",
        borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
        cursor: "pointer",
        background: isActive ? "rgba(98, 135, 245, 0.06)" : "transparent",
        transition: "background 100ms",
      }}
      onMouseEnter={(e) => {
        if (!isActive) (e.currentTarget as HTMLDivElement).style.background = "rgba(0,0,0,0.02)";
      }}
      onMouseLeave={(e) => {
        if (!isActive) (e.currentTarget as HTMLDivElement).style.background = "transparent";
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--zen-dark, #2e2e2e)" }}>
            v{plan.version}
          </span>
          <span
            style={{
              fontSize: 10, fontWeight: 600, textTransform: "uppercase",
              padding: "1px 6px", borderRadius: 4, letterSpacing: "0.05em",
              color: plan.status === "active" ? "var(--zen-green, #63f78b)" : plan.status === "completed" ? "var(--zen-blue, #6287f5)" : "var(--zen-muted, #999)",
              background: plan.status === "active" ? "rgba(99, 247, 139, 0.12)" : plan.status === "completed" ? "rgba(98, 135, 245, 0.1)" : "rgba(0,0,0,0.04)",
            }}
          >
            {PLAN_STATUS_LABELS[plan.status] || plan.status}
          </span>
        </div>
        <span style={{ fontSize: 11, color: "var(--zen-muted, #999)" }}>
          {completedSteps}/{totalSteps} steps
        </span>
      </div>
      <div style={{ fontSize: 12, color: "var(--zen-muted, #999)", lineHeight: 1.4, marginBottom: 2 }}>
        {plan.goal.length > 100 ? plan.goal.slice(0, 100) + "..." : plan.goal}
      </div>
      <div style={{ fontSize: 10, color: "var(--zen-muted, #999)" }}>
        {formatTimestamp(plan.created_at)}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* ProgressBar                                                         */
/* ------------------------------------------------------------------ */

function ProgressBar({ steps }: { steps: PlanStep[] }) {
  const total = steps.length;
  if (total === 0) return null;

  const completed = steps.filter((s) => s.status === "completed").length;
  const failed = steps.filter((s) => s.status === "failed").length;
  const running = steps.filter((s) => s.status === "running").length;
  const pct = total > 0 ? Math.round(((completed + failed) / total) * 100) : 0;

  return (
    <div style={{ padding: "0 14px 10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 11, color: "var(--zen-muted, #999)" }}>
        <span>{completed} completed{failed > 0 ? `, ${failed} failed` : ""}{running > 0 ? `, ${running} running` : ""}</span>
        <span>{pct}%</span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: "var(--zen-subtle, #e0ddd0)", overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            borderRadius: 2,
            background: failed > 0 ? "#ef4444" : "var(--zen-green, #63f78b)",
            transition: "width 300ms ease",
          }}
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main component                                                      */
/* ------------------------------------------------------------------ */

export default function PlanPanelIsland({ agentId }: PlanPanelIslandProps) {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [allPlans, setAllPlans] = useState<Plan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("current");
  const [expandedStepId, setExpandedStepId] = useState<string | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [selectedHistoryPlanId, setSelectedHistoryPlanId] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  // ---- API helpers ----

  const fetchActivePlan = useCallback(async () => {
    try {
      const resp = await fetch(`/api/v1/agents/${agentId}/plans/active`);
      if (resp.status === 404) {
        setPlan(null);
        return;
      }
      if (!resp.ok) throw new Error("Failed to load plan");
      const data: Plan = await resp.json();
      setPlan(data);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [agentId]);

  const fetchAllPlans = useCallback(async () => {
    try {
      const resp = await fetch(`/api/v1/agents/${agentId}/plans`);
      if (!resp.ok) return;
      const data: Plan[] = await resp.json();
      setAllPlans(data);
    } catch {
      // Non-critical
    }
  }, [agentId]);

  const fetchPlanById = useCallback(async (planId: string) => {
    try {
      const resp = await fetch(`/api/v1/agents/${agentId}/plans/${planId}`);
      if (!resp.ok) return null;
      return (await resp.json()) as Plan;
    } catch {
      return null;
    }
  }, [agentId]);

  // ---- Lifecycle ----

  useEffect(() => {
    fetchActivePlan();
    fetchAllPlans();

    // Poll for updates every 3 seconds when plan is active
    pollRef.current = window.setInterval(() => {
      fetchActivePlan();
    }, 3000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [fetchActivePlan, fetchAllPlans]);

  // ---- Step modification handlers ----

  const handleAddStep = useCallback(async (description: string) => {
    if (!plan) return;
    try {
      const resp = await fetch(`/api/v1/agents/${agentId}/plans/${plan.id}/steps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description, dependencies: [] }),
      });
      if (!resp.ok) throw new Error("Failed to add step");
      const updated: Plan = await resp.json();
      setPlan(updated);
      setShowAddForm(false);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [agentId, plan]);

  const handleDeleteStep = useCallback(async (stepId: string) => {
    if (!plan) return;
    try {
      const resp = await fetch(`/api/v1/agents/${agentId}/plans/${plan.id}/steps/${stepId}`, {
        method: "DELETE",
      });
      if (!resp.ok) throw new Error("Failed to remove step");
      await fetchActivePlan();
    } catch (err) {
      setError((err as Error).message);
    }
  }, [agentId, plan, fetchActivePlan]);

  const handleReorder = useCallback(async (stepId: string, direction: "up" | "down") => {
    if (!plan) return;
    const steps = [...plan.steps].sort((a, b) => a.step_number - b.step_number);
    const idx = steps.findIndex((s) => s.id === stepId);
    if (idx < 0) return;
    const swapIdx = direction === "up" ? idx - 1 : idx + 1;
    if (swapIdx < 0 || swapIdx >= steps.length) return;

    // Swap positions
    const reordered = [...steps];
    [reordered[idx], reordered[swapIdx]] = [reordered[swapIdx], reordered[idx]];
    const stepIds = reordered.map((s) => s.id);

    try {
      const resp = await fetch(`/api/v1/agents/${agentId}/plans/${plan.id}/reorder`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ step_ids: stepIds }),
      });
      if (!resp.ok) throw new Error("Failed to reorder");
      const updated: Plan = await resp.json();
      setPlan(updated);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [agentId, plan]);

  // ---- History selection ----

  const handleSelectHistoryPlan = useCallback(async (planId: string) => {
    setSelectedHistoryPlanId(planId);
    const p = await fetchPlanById(planId);
    if (p) setPlan(p);
  }, [fetchPlanById]);

  const handleBackToCurrent = useCallback(() => {
    setViewMode("current");
    setSelectedHistoryPlanId(null);
    fetchActivePlan();
  }, [fetchActivePlan]);

  // ---- Derived state ----

  const sortedSteps = useMemo(() => {
    if (!plan) return [];
    return [...plan.steps].sort((a, b) => a.step_number - b.step_number);
  }, [plan]);

  const displayPlan = plan;

  // ---- Render ----

  return (
    <div
      style={{
        fontFamily: "'Bricolage Grotesque', sans-serif",
        background: "var(--zen-paper, #f2f0e3)",
        borderRadius: 12,
        border: "1px solid var(--zen-subtle, #e0ddd0)",
        boxShadow: "0 2px 12px rgba(0,0,0,0.06)",
        overflow: "hidden",
        maxHeight: "80vh",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Spin animation for running icon */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* Header */}
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
          background: "var(--zen-paper, #f2f0e3)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--zen-coral, #F76F53)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
            <span style={{ fontSize: 15, fontWeight: 700, color: "var(--zen-dark, #2e2e2e)" }}>
              Execution Plan
            </span>
          </div>

          {/* View mode toggle */}
          <div style={{ display: "flex", gap: 2, background: "var(--zen-subtle, #e0ddd0)", borderRadius: 6, padding: 2 }}>
            <button
              onClick={() => { setViewMode("current"); handleBackToCurrent(); }}
              style={{
                padding: "3px 10px", borderRadius: 4, border: "none",
                background: viewMode === "current" ? "var(--zen-paper, #f2f0e3)" : "transparent",
                fontSize: 11, fontWeight: 600,
                color: viewMode === "current" ? "var(--zen-dark, #2e2e2e)" : "var(--zen-muted, #999)",
                cursor: "pointer", transition: "all 150ms",
              }}
            >
              Current
            </button>
            <button
              onClick={() => { setViewMode("history"); fetchAllPlans(); }}
              style={{
                padding: "3px 10px", borderRadius: 4, border: "none",
                background: viewMode === "history" ? "var(--zen-paper, #f2f0e3)" : "transparent",
                fontSize: 11, fontWeight: 600,
                color: viewMode === "history" ? "var(--zen-dark, #2e2e2e)" : "var(--zen-muted, #999)",
                cursor: "pointer", transition: "all 150ms",
              }}
            >
              History
            </button>
          </div>
        </div>

        {/* Plan goal */}
        {displayPlan && viewMode === "current" && (
          <div style={{ fontSize: 12, color: "var(--zen-muted, #999)", lineHeight: 1.4 }}>
            <span style={{ fontWeight: 600 }}>Goal:</span> {displayPlan.goal}
            <span style={{ marginLeft: 8, fontSize: 10, color: "var(--zen-muted, #999)" }}>
              (v{displayPlan.version} · {PLAN_STATUS_LABELS[displayPlan.status] || displayPlan.status})
            </span>
          </div>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div
          style={{
            padding: "8px 14px", background: "rgba(239, 68, 68, 0.08)",
            color: "#ef4444", fontSize: 12, display: "flex", justifyContent: "space-between", alignItems: "center",
          }}
        >
          <span>{error}</span>
          <button
            onClick={() => setError(null)}
            style={{ border: "none", background: "none", color: "#ef4444", cursor: "pointer", fontSize: 14, fontWeight: 700 }}
          >
            ×
          </button>
        </div>
      )}

      {/* Content area */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {loading ? (
          <div style={{ padding: "40px 14px", textAlign: "center", color: "var(--zen-muted, #999)", fontSize: 13 }}>
            Loading plan...
          </div>
        ) : viewMode === "current" ? (
          /* ---- Current plan view ---- */
          displayPlan ? (
            <>
              {/* Progress bar */}
              <ProgressBar steps={sortedSteps} />

              {/* Add step form */}
              {showAddForm && (
                <AddStepForm
                  onAdd={handleAddStep}
                  onCancel={() => setShowAddForm(false)}
                />
              )}

              {/* Steps list */}
              <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
                {sortedSteps.map((step, idx) => (
                  <StepRow
                    key={step.id}
                    step={step}
                    expanded={expandedStepId === step.id}
                    onToggle={() => setExpandedStepId(expandedStepId === step.id ? null : step.id)}
                    onDelete={() => handleDeleteStep(step.id)}
                    onMoveUp={() => handleReorder(step.id, "up")}
                    onMoveDown={() => handleReorder(step.id, "down")}
                    isFirst={idx === 0}
                    isLast={idx === sortedSteps.length - 1}
                    planStatus={displayPlan.status}
                  />
                ))}
              </ul>

              {/* Add step button */}
              {displayPlan.status === "active" && !showAddForm && (
                <div style={{ padding: "10px 14px" }}>
                  <button
                    onClick={() => setShowAddForm(true)}
                    style={{
                      display: "flex", alignItems: "center", gap: 6,
                      width: "100%", padding: "8px 12px", borderRadius: 8,
                      border: "1px dashed var(--zen-subtle, #e0ddd0)",
                      background: "transparent",
                      fontSize: 12, color: "var(--zen-muted, #999)",
                      cursor: "pointer", transition: "all 150ms",
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--zen-coral, #F76F53)";
                      (e.currentTarget as HTMLButtonElement).style.color = "var(--zen-coral, #F76F53)";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--zen-subtle, #e0ddd0)";
                      (e.currentTarget as HTMLButtonElement).style.color = "var(--zen-muted, #999)";
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                    Add step
                  </button>
                </div>
              )}
            </>
          ) : (
            <div style={{ padding: "40px 14px", textAlign: "center" }}>
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--zen-subtle, #e0ddd0)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: 12, display: "inline-block" }}>
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <p style={{ fontSize: 13, color: "var(--zen-muted, #999)", marginBottom: 4 }}>No active plan</p>
              <p style={{ fontSize: 11, color: "var(--zen-muted, #999)" }}>
                Send a goal to the agent in autonomous mode to generate a plan.
              </p>
            </div>
          )
        ) : (
          /* ---- History view ---- */
          allPlans.length > 0 ? (
            <>
              {/* History header */}
              {selectedHistoryPlanId && (
                <div style={{ padding: "8px 14px", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)" }}>
                  <button
                    onClick={() => { setSelectedHistoryPlanId(null); }}
                    style={{
                      display: "flex", alignItems: "center", gap: 4,
                      border: "none", background: "none", padding: 0,
                      fontSize: 12, color: "var(--zen-coral, #F76F53)", cursor: "pointer",
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="19" y1="12" x2="5" y2="12" />
                      <polyline points="12 19 5 12 12 5" />
                    </svg>
                    Back to versions
                  </button>
                </div>
              )}

              {selectedHistoryPlanId && displayPlan ? (
                /* Show selected plan detail */
                <>
                  <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--zen-subtle, #e0ddd0)" }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "var(--zen-dark, #2e2e2e)" }}>
                      Plan v{displayPlan.version}
                      <span
                        style={{
                          marginLeft: 8, fontSize: 10, fontWeight: 600,
                          padding: "1px 6px", borderRadius: 4,
                          color: displayPlan.status === "active" ? "var(--zen-green, #63f78b)" : "var(--zen-muted, #999)",
                          background: displayPlan.status === "active" ? "rgba(99, 247, 139, 0.12)" : "rgba(0,0,0,0.04)",
                        }}
                      >
                        {PLAN_STATUS_LABELS[displayPlan.status] || displayPlan.status}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: "var(--zen-muted, #999)", marginTop: 4 }}>
                      {displayPlan.goal}
                    </div>
                  </div>
                  <ProgressBar steps={displayPlan.steps} />
                  <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
                    {[...displayPlan.steps].sort((a, b) => a.step_number - b.step_number).map((step, idx, arr) => (
                      <StepRow
                        key={step.id}
                        step={step}
                        expanded={expandedStepId === step.id}
                        onToggle={() => setExpandedStepId(expandedStepId === step.id ? null : step.id)}
                        onDelete={() => {}}
                        onMoveUp={() => {}}
                        onMoveDown={() => {}}
                        isFirst={idx === 0}
                        isLast={idx === arr.length - 1}
                        planStatus={displayPlan.status}
                      />
                    ))}
                  </ul>
                </>
              ) : (
                /* Version list */
                allPlans.map((p) => (
                  <PlanHistoryItem
                    key={p.id}
                    plan={p}
                    isActive={p.id === selectedHistoryPlanId}
                    onSelect={() => handleSelectHistoryPlan(p.id)}
                  />
                ))
              )}
            </>
          ) : (
            <div style={{ padding: "40px 14px", textAlign: "center", color: "var(--zen-muted, #999)", fontSize: 13 }}>
              No plan history
            </div>
          )
        )}
      </div>
    </div>
  );
}
