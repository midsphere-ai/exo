import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { getHandlesForNodeType, HANDLE_COLORS, BRANCH_COLORS, type HandleSpec } from "./handleTypes";

/* ------------------------------------------------------------------ */
/* WorkflowNode — custom node with typed, color-coded handles          */
/* ------------------------------------------------------------------ */

interface WorkflowNodeData {
  label?: string;
  nodeType?: string;
  categoryColor?: string;
  tool_mode?: boolean;
  _relTint?: "root" | "upstream" | "downstream" | null;
  _missingConfig?: boolean;
  _unreachable?: boolean;
  _disconnectedInputs?: string[];
  _execStatus?: "running" | "completed" | "failed" | null;
  _debugPaused?: boolean;
  _hasBreakpoint?: boolean;
  [key: string]: unknown;
}

function WorkflowNode({ data, selected }: NodeProps) {
  const d = data as WorkflowNodeData;
  const nodeType = d.nodeType ?? "default";
  const categoryColor = d.categoryColor ?? "#999";
  const label = d.label ?? nodeType;
  const relTint = d._relTint ?? null;

  const toolMode = d.tool_mode ?? false;
  const hasMissingConfig = d._missingConfig ?? false;
  const isUnreachable = d._unreachable ?? false;
  const disconnectedInputs = new Set(d._disconnectedInputs ?? []);
  const execStatus = d._execStatus ?? null;
  const debugPaused = d._debugPaused ?? false;
  const hasBreakpoint = d._hasBreakpoint ?? false;

  const handles = getHandlesForNodeType(nodeType);
  const inputs = handles.filter((h) => h.type === "target");
  const outputs = handles.filter((h) => h.type === "source");

  /* Relationship mode tinting */
  const tintBorder =
    relTint === "root"
      ? "var(--zen-coral, #F76F53)"
      : relTint === "upstream"
        ? "var(--zen-blue, #6287f5)"
        : relTint === "downstream"
          ? "var(--zen-green, #63f78b)"
          : null;
  const tintShadow =
    relTint === "root"
      ? "0 0 0 3px rgba(247, 111, 83, 0.3)"
      : relTint === "upstream"
        ? "0 0 0 3px rgba(98, 135, 245, 0.25)"
        : relTint === "downstream"
          ? "0 0 0 3px rgba(99, 247, 139, 0.25)"
          : null;

  /* Execution status overrides relationship tinting */
  const execBorder =
    debugPaused
      ? "#a855f7"
      : execStatus === "running"
        ? "var(--zen-coral, #F76F53)"
        : execStatus === "completed"
          ? "var(--zen-green, #63f78b)"
          : execStatus === "failed"
            ? "#ef4444"
            : null;

  const borderColor =
    execBorder ?? tintBorder ?? (selected ? "var(--zen-coral, #F76F53)" : "var(--zen-subtle, #e0ddd0)");
  const debugShadow = debugPaused ? "0 0 0 4px rgba(168, 85, 247, 0.35)" : null;
  const shadow =
    debugShadow ?? tintShadow ?? (selected ? "0 0 0 2px rgba(247, 111, 83, 0.25)" : "0 1px 3px rgba(0,0,0,0.06)");

  return (
    <div
      style={{
        minWidth: 160,
        background: "var(--zen-paper, #f2f0e3)",
        border: `2px ${isUnreachable ? "dashed" : "solid"} ${borderColor}`,
        borderRadius: 10,
        fontFamily: "'Bricolage Grotesque', sans-serif",
        position: "relative",
        transition: "border-color 150ms, box-shadow 150ms, opacity 200ms",
        boxShadow: shadow,
      }}
    >
      {/* Execution status overlay badges */}
      {execStatus === "completed" && (
        <div
          title="Completed"
          style={{
            position: "absolute",
            top: -8,
            right: -8,
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: "var(--zen-green, #63f78b)",
            color: "#fff",
            fontSize: 13,
            fontWeight: 700,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10,
            boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
            lineHeight: 1,
          }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </div>
      )}
      {execStatus === "failed" && (
        <div
          title="Failed"
          style={{
            position: "absolute",
            top: -8,
            right: -8,
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: "#ef4444",
            color: "#fff",
            fontSize: 13,
            fontWeight: 700,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10,
            boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
            lineHeight: 1,
          }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </div>
      )}

      {/* Debug paused badge */}
      {debugPaused && (
        <div
          title="Paused"
          style={{
            position: "absolute",
            top: -8,
            right: -8,
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: "#a855f7",
            color: "#fff",
            fontSize: 13,
            fontWeight: 700,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10,
            boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
            lineHeight: 1,
          }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <rect x="6" y="4" width="4" height="16" rx="1" />
            <rect x="14" y="4" width="4" height="16" rx="1" />
          </svg>
        </div>
      )}

      {/* Breakpoint indicator (red dot on left) */}
      {hasBreakpoint && (
        <div
          title="Breakpoint"
          style={{
            position: "absolute",
            top: -6,
            left: -6,
            width: 14,
            height: 14,
            borderRadius: "50%",
            background: "#ef4444",
            border: "2px solid var(--zen-paper, #f2f0e3)",
            zIndex: 10,
            boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
          }}
        />
      )}

      {/* Tool Mode wrench badge (top-left, below breakpoint) */}
      {toolMode && (
        <div
          title="Tool Mode — available as agent tool"
          style={{
            position: "absolute",
            top: hasBreakpoint ? 12 : -6,
            left: -6,
            width: 16,
            height: 16,
            borderRadius: 4,
            background: "var(--zen-blue, #6287f5)",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10,
            boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
          }}
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
          </svg>
        </div>
      )}

      {/* Warning badge for missing config (top-right corner, hidden during execution) */}
      {hasMissingConfig && !execStatus && !debugPaused && (
        <div
          title="Missing required configuration"
          style={{
            position: "absolute",
            top: -8,
            right: -8,
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: "#f59e0b",
            color: "#fff",
            fontSize: 12,
            fontWeight: 700,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10,
            boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
            lineHeight: 1,
          }}
        >
          !
        </div>
      )}

      {/* Header bar with category color */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 12px",
          borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
        }}
      >
        <div
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: categoryColor,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: "var(--zen-dark, #2e2e2e)",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {label}
        </span>
      </div>

      {/* Handle labels area */}
      <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 12px" }}>
        {/* Input labels */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {inputs.map((h) => (
            <div
              key={h.id}
              style={{
                fontSize: 10,
                color: getHandleColor(h),
                fontWeight: 500,
              }}
            >
              {h.dataType}
            </div>
          ))}
        </div>
        {/* Output labels */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-end" }}>
          {outputs.map((h) => (
            <div
              key={h.id}
              style={{
                fontSize: 10,
                color: getHandleColor(h),
                fontWeight: h.label ? 700 : 500,
              }}
            >
              {h.label ?? h.dataType}
            </div>
          ))}
        </div>
      </div>

      {/* Input handles (left side) */}
      {inputs.map((h, i) => {
        const isDisconnected = disconnectedInputs.has(h.id);
        return (
          <Handle
            key={h.id}
            type="target"
            position={Position.Left}
            id={h.id}
            style={{
              top: computeHandleTop(i, inputs.length),
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: isDisconnected ? "#ef4444" : getHandleColor(h),
              border: `2px solid ${isDisconnected ? "#ef4444" : "var(--zen-paper, #f2f0e3)"}`,
              boxShadow: isDisconnected ? "0 0 0 2px rgba(239, 68, 68, 0.3)" : undefined,
            }}
          />
        );
      })}

      {/* Output handles (right side) */}
      {outputs.map((h, i) => (
        <Handle
          key={h.id}
          type="source"
          position={Position.Right}
          id={h.id}
          style={{
            top: computeHandleTop(i, outputs.length),
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: getHandleColor(h),
            border: "2px solid var(--zen-paper, #f2f0e3)",
          }}
        />
      ))}
    </div>
  );
}

/** Get display color for a handle — uses branch colors for labeled conditional outputs. */
function getHandleColor(h: HandleSpec): string {
  if (h.label === "True") return BRANCH_COLORS.true;
  if (h.label === "False") return BRANCH_COLORS.false;
  return HANDLE_COLORS[h.dataType];
}

/** Distribute handles evenly in the lower portion of the node (below the header). */
function computeHandleTop(index: number, total: number): string {
  // Header is ~37px, the handle labels area starts after.
  // Place handles in the zone 50%–90% of node height.
  if (total === 1) return "60%";
  const start = 45;
  const end = 90;
  const step = (end - start) / (total - 1);
  return `${start + step * index}%`;
}

export default memo(WorkflowNode);
