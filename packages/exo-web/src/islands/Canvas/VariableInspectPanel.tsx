import { useCallback, useMemo, useState } from "react";
import type { Node } from "@xyflow/react";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

type VarType = "string" | "number" | "boolean" | "array" | "object" | "null";

interface VariableRow {
  key: string;
  value: unknown;
  type: VarType;
  sourceNodeLabel: string;
}

interface VariableInspectPanelProps {
  variables: Record<string, unknown>;
  nodes: Node[];
  isDebugMode: boolean;
  onSetVariable?: (name: string, value: unknown) => void;
  onNavigateToNode: (nodeId: string) => void;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function getVarType(val: unknown): VarType {
  if (val === null || val === undefined) return "null";
  if (Array.isArray(val)) return "array";
  const t = typeof val;
  if (t === "string" || t === "number" || t === "boolean") return t;
  if (t === "object") return "object";
  return "string";
}

function formatValue(val: unknown, maxLen = 80): string {
  if (val === null || val === undefined) return "null";
  if (typeof val === "string") return val.length > maxLen ? val.slice(0, maxLen) + "\u2026" : val;
  const s = JSON.stringify(val);
  return s.length > maxLen ? s.slice(0, maxLen) + "\u2026" : s;
}

const TYPE_COLORS: Record<VarType, string> = {
  string: "#6287f5",
  number: "#F76F53",
  boolean: "#a855f7",
  array: "#63f78b",
  object: "#f59e0b",
  null: "#999",
};

/* ------------------------------------------------------------------ */
/* Copy icon SVG                                                       */
/* ------------------------------------------------------------------ */

const CopyIcon = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function VariableInspectPanel({
  variables,
  nodes,
  isDebugMode,
  onSetVariable,
  onNavigateToNode,
}: VariableInspectPanelProps) {
  const [filterNode, setFilterNode] = useState<string>("all");
  const [filterType, setFilterType] = useState<VarType | "all">("all");
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  /* Build node label lookup */
  const nodeLabels = useMemo(() => {
    const map: Record<string, string> = {};
    for (const n of nodes) {
      const d = n.data as { label?: string; nodeType?: string };
      map[n.id] = d.label ?? d.nodeType ?? n.id;
    }
    return map;
  }, [nodes]);

  /* Build rows with type info */
  const rows: VariableRow[] = useMemo(() => {
    return Object.entries(variables).map(([key, value]) => ({
      key,
      value,
      type: getVarType(value),
      sourceNodeLabel: nodeLabels[key] ?? key,
    }));
  }, [variables, nodeLabels]);

  /* Apply filters */
  const filteredRows = useMemo(() => {
    return rows.filter((r) => {
      if (filterNode !== "all" && r.key !== filterNode) return false;
      if (filterType !== "all" && r.type !== filterType) return false;
      return true;
    });
  }, [rows, filterNode, filterType]);

  /* Unique types present */
  const presentTypes = useMemo(() => {
    const types = new Set<VarType>();
    for (const r of rows) types.add(r.type);
    return Array.from(types).sort();
  }, [rows]);

  /* Unique source nodes present */
  const presentNodes = useMemo(() => {
    return rows.map((r) => ({ id: r.key, label: r.sourceNodeLabel }));
  }, [rows]);

  /* Edit handlers */
  const startEdit = useCallback((key: string, val: unknown) => {
    setEditingKey(key);
    setEditValue(typeof val === "string" ? val : JSON.stringify(val));
  }, []);

  const commitEdit = useCallback(() => {
    if (editingKey === null || !onSetVariable) return;
    let parsed: unknown = editValue;
    try {
      parsed = JSON.parse(editValue);
    } catch {
      // keep as string
    }
    onSetVariable(editingKey, parsed);
    setEditingKey(null);
    setEditValue("");
  }, [editingKey, editValue, onSetVariable]);

  const cancelEdit = useCallback(() => {
    setEditingKey(null);
    setEditValue("");
  }, []);

  /* Copy to clipboard */
  const copyValue = useCallback((key: string, val: unknown) => {
    const text = typeof val === "string" ? val : JSON.stringify(val, null, 2);
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1500);
  }, []);

  if (rows.length === 0) {
    return (
      <div
        className="nodrag nopan nowheel"
        style={{
          padding: "12px 16px",
          borderRadius: 10,
          background: "var(--zen-paper, #f2f0e3)",
          border: "1px solid var(--zen-subtle, #e0ddd0)",
          boxShadow: "0 -2px 12px rgba(0,0,0,0.08)",
          fontFamily: "'Bricolage Grotesque', sans-serif",
          fontSize: 12,
          color: "var(--zen-muted, #999)",
          textAlign: "center",
        }}
      >
        No variables yet. Run or debug a workflow to see variables.
      </div>
    );
  }

  return (
    <div
      className="nodrag nopan nowheel"
      style={{
        width: "min(720px, 80vw)",
        maxHeight: 280,
        display: "flex",
        flexDirection: "column",
        borderRadius: 10,
        background: "var(--zen-paper, #f2f0e3)",
        border: "1px solid var(--zen-subtle, #e0ddd0)",
        boxShadow: "0 -2px 12px rgba(0,0,0,0.08)",
        fontFamily: "'Bricolage Grotesque', sans-serif",
        fontSize: 12,
      }}
    >
      {/* Header + filters */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "8px 14px",
          borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            fontWeight: 600,
            fontSize: 11,
            color: "#a855f7",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            flexShrink: 0,
          }}
        >
          Variables ({filteredRows.length})
        </div>

        <div style={{ flex: 1 }} />

        {/* Node filter */}
        <select
          value={filterNode}
          onChange={(e) => setFilterNode(e.target.value)}
          style={{
            padding: "3px 6px",
            fontSize: 11,
            borderRadius: 5,
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            background: "var(--zen-paper, #f2f0e3)",
            color: "var(--zen-dark, #2e2e2e)",
            fontFamily: "'Bricolage Grotesque', sans-serif",
            cursor: "pointer",
          }}
        >
          <option value="all">All Nodes</option>
          {presentNodes.map((n) => (
            <option key={n.id} value={n.id}>{n.label}</option>
          ))}
        </select>

        {/* Type filter */}
        <select
          value={filterType}
          onChange={(e) => setFilterType(e.target.value as VarType | "all")}
          style={{
            padding: "3px 6px",
            fontSize: 11,
            borderRadius: 5,
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            background: "var(--zen-paper, #f2f0e3)",
            color: "var(--zen-dark, #2e2e2e)",
            fontFamily: "'Bricolage Grotesque', sans-serif",
            cursor: "pointer",
          }}
        >
          <option value="all">All Types</option>
          {presentTypes.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      {/* Table body */}
      <div style={{ overflowY: "auto", flex: 1, padding: "0 4px" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ position: "sticky", top: 0, background: "var(--zen-paper, #f2f0e3)", zIndex: 1 }}>
              <th style={{ ...thStyle, width: "22%" }}>Variable</th>
              <th style={{ ...thStyle, width: "45%" }}>Value</th>
              <th style={{ ...thStyle, width: "13%" }}>Type</th>
              <th style={{ ...thStyle, width: "15%" }}>Source</th>
              <th style={{ ...thStyle, width: "5%" }}></th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row) => (
              <tr
                key={row.key}
                style={{
                  borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
                }}
              >
                {/* Variable name (source node label) */}
                <td
                  style={{
                    padding: "5px 8px",
                    fontWeight: 600,
                    color: "var(--zen-dark, #2e2e2e)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    maxWidth: 0,
                  }}
                  title={row.key}
                >
                  {row.sourceNodeLabel}
                </td>

                {/* Value (editable in debug mode) */}
                <td style={{ padding: "5px 8px", maxWidth: 0 }}>
                  {editingKey === row.key ? (
                    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                      <input
                        autoFocus
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitEdit();
                          if (e.key === "Escape") cancelEdit();
                        }}
                        style={{
                          flex: 1,
                          padding: "2px 6px",
                          fontSize: 11,
                          border: "1px solid #a855f7",
                          borderRadius: 4,
                          background: "var(--zen-paper, #f2f0e3)",
                          color: "var(--zen-dark, #2e2e2e)",
                          fontFamily: "monospace",
                          outline: "none",
                        }}
                      />
                      <button
                        onClick={commitEdit}
                        style={{
                          padding: "1px 6px",
                          fontSize: 10,
                          borderRadius: 4,
                          border: "1px solid var(--zen-green, #63f78b)",
                          background: "var(--zen-green, #63f78b)",
                          color: "#fff",
                          cursor: "pointer",
                        }}
                      >
                        OK
                      </button>
                      <button
                        onClick={cancelEdit}
                        style={{
                          padding: "1px 6px",
                          fontSize: 10,
                          borderRadius: 4,
                          border: "1px solid var(--zen-subtle, #e0ddd0)",
                          background: "transparent",
                          color: "var(--zen-dark, #2e2e2e)",
                          cursor: "pointer",
                        }}
                      >
                        Esc
                      </button>
                    </div>
                  ) : (
                    <span
                      onClick={isDebugMode && onSetVariable ? () => startEdit(row.key, row.value) : undefined}
                      title={isDebugMode && onSetVariable ? "Click to edit" : String(row.value)}
                      style={{
                        fontFamily: "monospace",
                        color: "var(--zen-muted, #999)",
                        cursor: isDebugMode && onSetVariable ? "pointer" : "default",
                        wordBreak: "break-all",
                        display: "block",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                    >
                      {formatValue(row.value)}
                    </span>
                  )}
                </td>

                {/* Type badge */}
                <td style={{ padding: "5px 8px" }}>
                  <span
                    style={{
                      display: "inline-block",
                      padding: "1px 6px",
                      fontSize: 10,
                      fontWeight: 600,
                      borderRadius: 4,
                      background: TYPE_COLORS[row.type] + "18",
                      color: TYPE_COLORS[row.type],
                      border: `1px solid ${TYPE_COLORS[row.type]}40`,
                    }}
                  >
                    {row.type}
                  </span>
                </td>

                {/* Source node link */}
                <td style={{ padding: "5px 8px" }}>
                  <button
                    onClick={() => onNavigateToNode(row.key)}
                    title={`Navigate to ${row.sourceNodeLabel}`}
                    style={{
                      padding: "1px 6px",
                      fontSize: 10,
                      fontWeight: 500,
                      borderRadius: 4,
                      border: "1px solid var(--zen-subtle, #e0ddd0)",
                      background: "transparent",
                      color: "#6287f5",
                      cursor: "pointer",
                      fontFamily: "'Bricolage Grotesque', sans-serif",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      maxWidth: "100%",
                      display: "block",
                    }}
                  >
                    {row.sourceNodeLabel}
                  </button>
                </td>

                {/* Copy button */}
                <td style={{ padding: "5px 4px", width: 28 }}>
                  <button
                    title={copiedKey === row.key ? "Copied!" : "Copy value"}
                    onClick={() => copyValue(row.key, row.value)}
                    style={{
                      padding: 2,
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      color: copiedKey === row.key ? "var(--zen-green, #63f78b)" : "var(--zen-muted, #999)",
                      fontSize: 11,
                      transition: "color 150ms",
                    }}
                  >
                    {CopyIcon}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer with keyboard hint */}
      <div
        style={{
          padding: "4px 14px",
          borderTop: "1px solid var(--zen-subtle, #e0ddd0)",
          fontSize: 10,
          color: "var(--zen-muted, #999)",
          textAlign: "center",
          flexShrink: 0,
        }}
      >
        Press <kbd style={{ padding: "1px 4px", border: "1px solid var(--zen-subtle, #e0ddd0)", borderRadius: 3, fontSize: 10 }}>{navigator.userAgent?.includes("Mac") ? "\u2318" : "Ctrl"}+J</kbd> to toggle
        {isDebugMode && " \u00b7 Click value to edit in debug mode"}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Shared th style                                                     */
/* ------------------------------------------------------------------ */

const thStyle: React.CSSProperties = {
  padding: "6px 8px",
  fontWeight: 600,
  fontSize: 10,
  color: "var(--zen-muted, #999)",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  textAlign: "left",
  borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
};
