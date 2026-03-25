import { useState, useCallback } from "react";

/* ------------------------------------------------------------------ */
/* Node type catalog                                                    */
/* ------------------------------------------------------------------ */

export interface NodeTypeEntry {
  id: string;
  label: string;
  icon: React.ReactNode;
}

export interface NodeCategory {
  id: string;
  label: string;
  color: string;
  types: NodeTypeEntry[];
}

/* Compact 16×16 SVG icons per node type */
function Icon({ d }: { d: string }) {
  return (
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
      <path d={d} />
    </svg>
  );
}

/* Multi-path icon helper */
function MultiIcon({ children }: { children: React.ReactNode }) {
  return (
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
      {children}
    </svg>
  );
}

const NODE_CATEGORIES: NodeCategory[] = [
  {
    id: "triggers",
    label: "Triggers",
    color: "#F76F53", // coral
    types: [
      {
        id: "chat_input",
        label: "Chat Input",
        icon: (
          <MultiIcon>
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </MultiIcon>
        ),
      },
      {
        id: "webhook",
        label: "Webhook",
        icon: (
          <MultiIcon>
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
          </MultiIcon>
        ),
      },
      {
        id: "schedule",
        label: "Schedule",
        icon: (
          <MultiIcon>
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </MultiIcon>
        ),
      },
      {
        id: "manual",
        label: "Manual",
        icon: (
          <MultiIcon>
            <path d="M18 8h1a4 4 0 0 1 0 8h-1" />
            <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z" />
            <line x1="6" y1="1" x2="6" y2="4" />
            <line x1="10" y1="1" x2="10" y2="4" />
            <line x1="14" y1="1" x2="14" y2="4" />
          </MultiIcon>
        ),
      },
    ],
  },
  {
    id: "llm",
    label: "LLM",
    color: "#6287f5", // zen-blue
    types: [
      {
        id: "llm_call",
        label: "LLM Call",
        icon: (
          <MultiIcon>
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </MultiIcon>
        ),
      },
      {
        id: "prompt_template",
        label: "Prompt Template",
        icon: (
          <MultiIcon>
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
          </MultiIcon>
        ),
      },
      {
        id: "model_selector",
        label: "Model Selector",
        icon: (
          <MultiIcon>
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M9 9h6" />
            <path d="M12 9v6" />
          </MultiIcon>
        ),
      },
    ],
  },
  {
    id: "agent",
    label: "Agent",
    color: "#a78bfa", // purple
    types: [
      {
        id: "agent_node",
        label: "Agent Node",
        icon: (
          <MultiIcon>
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </MultiIcon>
        ),
      },
      {
        id: "sub_agent",
        label: "Sub-Agent",
        icon: (
          <MultiIcon>
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
            <circle cx="9" cy="7" r="4" />
            <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
            <path d="M16 3.13a4 4 0 0 1 0 7.75" />
          </MultiIcon>
        ),
      },
    ],
  },
  {
    id: "tools",
    label: "Tools",
    color: "#63f78b", // zen-green
    types: [
      {
        id: "function_tool",
        label: "Function Tool",
        icon: (
          <MultiIcon>
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
          </MultiIcon>
        ),
      },
      {
        id: "http_request",
        label: "HTTP Request",
        icon: (
          <MultiIcon>
            <circle cx="12" cy="12" r="10" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
          </MultiIcon>
        ),
      },
      {
        id: "code_python",
        label: "Code (Python)",
        icon: (
          <MultiIcon>
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
            <line x1="12" y1="2" x2="12" y2="22" />
          </MultiIcon>
        ),
      },
      {
        id: "code_javascript",
        label: "Code (JavaScript)",
        icon: (
          <MultiIcon>
            <path d="M20 3H4a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1z" />
            <path d="M9 17V10l-1.5 2" />
            <path d="M15 10c1 0 2 .5 2 2s-1 2-2 2 2 .5 2 2-1 2-2 2" />
          </MultiIcon>
        ),
      },
    ],
  },
  {
    id: "logic",
    label: "Logic",
    color: "#f59e0b", // amber
    types: [
      {
        id: "conditional",
        label: "Conditional (if/else)",
        icon: (
          <MultiIcon>
            <path d="M16 3h5v5" />
            <line x1="21" y1="3" x2="14" y2="10" />
            <path d="M8 21H3v-5" />
            <line x1="3" y1="21" x2="10" y2="14" />
            <line x1="3" y1="3" x2="21" y2="21" />
          </MultiIcon>
        ),
      },
      {
        id: "switch",
        label: "Switch",
        icon: (
          <MultiIcon>
            <circle cx="12" cy="12" r="2" />
            <path d="M12 2v4" />
            <path d="M12 18v4" />
            <path d="M4.93 4.93l2.83 2.83" />
            <path d="M16.24 16.24l2.83 2.83" />
            <path d="M2 12h4" />
            <path d="M18 12h4" />
          </MultiIcon>
        ),
      },
      {
        id: "loop_iterator",
        label: "Loop/Iterator",
        icon: (
          <MultiIcon>
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.13-9.36L23 10" />
          </MultiIcon>
        ),
      },
      {
        id: "aggregator",
        label: "Aggregator",
        icon: (
          <MultiIcon>
            <rect x="2" y="7" width="20" height="14" rx="2" />
            <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
          </MultiIcon>
        ),
      },
      {
        id: "approval_gate",
        label: "Approval Gate",
        icon: (
          <MultiIcon>
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            <polyline points="9 12 11 14 15 10" />
          </MultiIcon>
        ),
      },
    ],
  },
  {
    id: "data",
    label: "Data",
    color: "#14b8a6", // teal
    types: [
      {
        id: "variable_assigner",
        label: "Variable Assigner",
        icon: <Icon d="M20 6L9 17l-5-5" />,
      },
      {
        id: "template_jinja",
        label: "Template (Jinja)",
        icon: (
          <MultiIcon>
            <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
          </MultiIcon>
        ),
      },
      {
        id: "json_transform",
        label: "JSON Transform",
        icon: (
          <MultiIcon>
            <line x1="16" y1="3" x2="16" y2="21" />
            <line x1="8" y1="3" x2="8" y2="21" />
            <path d="M3 12h18" />
          </MultiIcon>
        ),
      },
      {
        id: "text_splitter",
        label: "Text Splitter",
        icon: (
          <MultiIcon>
            <line x1="18" y1="20" x2="18" y2="10" />
            <line x1="12" y1="20" x2="12" y2="4" />
            <line x1="6" y1="20" x2="6" y2="14" />
          </MultiIcon>
        ),
      },
    ],
  },
  {
    id: "knowledge",
    label: "Knowledge",
    color: "#8b5cf6", // violet
    types: [
      {
        id: "knowledge_retrieval",
        label: "Knowledge Retrieval",
        icon: (
          <MultiIcon>
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </MultiIcon>
        ),
      },
      {
        id: "document_loader",
        label: "Document Loader",
        icon: (
          <MultiIcon>
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
          </MultiIcon>
        ),
      },
      {
        id: "embedding_node",
        label: "Embedding Node",
        icon: (
          <MultiIcon>
            <rect x="1" y="4" width="22" height="16" rx="2" />
            <line x1="1" y1="10" x2="23" y2="10" />
          </MultiIcon>
        ),
      },
    ],
  },
  {
    id: "output",
    label: "Output",
    color: "#ec4899", // pink
    types: [
      {
        id: "chat_response",
        label: "Chat Response",
        icon: (
          <MultiIcon>
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
          </MultiIcon>
        ),
      },
      {
        id: "api_response",
        label: "API Response",
        icon: (
          <MultiIcon>
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </MultiIcon>
        ),
      },
      {
        id: "file_output",
        label: "File Output",
        icon: (
          <MultiIcon>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </MultiIcon>
        ),
      },
      {
        id: "notification",
        label: "Notification",
        icon: (
          <MultiIcon>
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 0 1-3.46 0" />
          </MultiIcon>
        ),
      },
    ],
  },
  {
    id: "integration",
    label: "Integration",
    color: "#06b6d4", // cyan
    types: [
      {
        id: "webhook_call",
        label: "Webhook Call",
        icon: (
          <MultiIcon>
            <path d="M22 2L11 13" />
            <path d="M22 2l-7 20-4-9-9-4 20-7z" />
          </MultiIcon>
        ),
      },
      {
        id: "mcp_client",
        label: "MCP Client",
        icon: (
          <MultiIcon>
            <rect x="2" y="2" width="20" height="8" rx="2" />
            <rect x="2" y="14" width="20" height="8" rx="2" />
            <line x1="6" y1="6" x2="6.01" y2="6" />
            <line x1="6" y1="18" x2="6.01" y2="18" />
          </MultiIcon>
        ),
      },
    ],
  },
];

export { NODE_CATEGORIES };

/* ------------------------------------------------------------------ */
/* Sidebar component                                                    */
/* ------------------------------------------------------------------ */

interface NodeSidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export default function NodeSidebar({ collapsed, onToggle }: NodeSidebarProps) {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    () => new Set(NODE_CATEGORIES.map((c) => c.id)),
  );
  const [search, setSearch] = useState("");

  const toggleCategory = useCallback((id: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const onDragStart = useCallback(
    (e: React.DragEvent, nodeType: string, label: string) => {
      e.dataTransfer.setData("application/reactflow-type", nodeType);
      e.dataTransfer.setData("application/reactflow-label", label);
      e.dataTransfer.effectAllowed = "move";
    },
    [],
  );

  const filteredCategories = search.trim()
    ? NODE_CATEGORIES.map((cat) => {
        const q = search.toLowerCase();
        const categoryMatch = cat.label.toLowerCase().includes(q);
        return {
          ...cat,
          types: categoryMatch
            ? cat.types
            : cat.types.filter((t) => t.label.toLowerCase().includes(q)),
        };
      }).filter((cat) => cat.types.length > 0)
    : NODE_CATEGORIES;

  if (collapsed) {
    return (
      <div
        style={{
          position: "absolute",
          top: 8,
          left: 8,
          zIndex: 10,
        }}
      >
        <button
          onClick={onToggle}
          title="Open node panel"
          className="nodrag nopan"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 36,
            height: 36,
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            borderRadius: 8,
            background: "var(--zen-paper, #f2f0e3)",
            color: "var(--zen-dark, #2e2e2e)",
            cursor: "pointer",
            boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
          }}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="3" y="3" width="7" height="7" />
            <rect x="14" y="3" width="7" height="7" />
            <rect x="14" y="14" width="7" height="7" />
            <rect x="3" y="14" width="7" height="7" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div
      className="nodrag nopan nowheel"
      style={{
        position: "absolute",
        top: 8,
        left: 8,
        bottom: 8,
        width: 240,
        zIndex: 10,
        display: "flex",
        flexDirection: "column",
        background: "var(--zen-paper, #f2f0e3)",
        border: "1px solid var(--zen-subtle, #e0ddd0)",
        borderRadius: 12,
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 12px 6px",
          borderBottom: "1px solid var(--zen-subtle, #e0ddd0)",
        }}
      >
        <span
          style={{
            fontWeight: 600,
            fontSize: 13,
            color: "var(--zen-dark, #2e2e2e)",
          }}
        >
          Nodes
        </span>
        <button
          onClick={onToggle}
          title="Close node panel"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 24,
            height: 24,
            border: "none",
            borderRadius: 4,
            background: "transparent",
            color: "var(--zen-muted, #999)",
            cursor: "pointer",
          }}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="11 17 6 12 11 7" />
            <polyline points="18 17 13 12 18 7" />
          </svg>
        </button>
      </div>

      {/* Search */}
      <div style={{ padding: "8px 10px" }}>
        <input
          type="text"
          placeholder="Search nodes..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            width: "100%",
            padding: "6px 10px",
            fontSize: 12,
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            borderRadius: 6,
            background: "var(--zen-paper, #f2f0e3)",
            color: "var(--zen-dark, #2e2e2e)",
            outline: "none",
            boxSizing: "border-box",
          }}
        />
      </div>

      {/* Categories */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "0 6px 8px",
        }}
      >
        {filteredCategories.map((cat) => (
          <div key={cat.id} style={{ marginBottom: 2 }}>
            {/* Category header */}
            <button
              onClick={() => toggleCategory(cat.id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                width: "100%",
                padding: "6px 6px",
                border: "none",
                borderRadius: 6,
                background: "transparent",
                cursor: "pointer",
                fontSize: 11,
                fontWeight: 600,
                color: "var(--zen-dark, #2e2e2e)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: cat.color,
                  flexShrink: 0,
                }}
              />
              <span style={{ flex: 1, textAlign: "left" }}>{cat.label}</span>
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  transform: expandedCategories.has(cat.id)
                    ? "rotate(90deg)"
                    : "rotate(0deg)",
                  transition: "transform 150ms",
                }}
              >
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </button>

            {/* Node type items */}
            {expandedCategories.has(cat.id) && (
              <div style={{ padding: "2px 0 4px" }}>
                {cat.types.map((nodeType) => (
                  <div
                    key={nodeType.id}
                    draggable
                    onDragStart={(e) =>
                      onDragStart(e, nodeType.id, nodeType.label)
                    }
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      padding: "5px 8px 5px 20px",
                      borderRadius: 6,
                      cursor: "grab",
                      fontSize: 12,
                      color: "var(--zen-dark, #2e2e2e)",
                      transition: "background 100ms",
                      userSelect: "none",
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.background =
                        `${cat.color}18`;
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.background =
                        "transparent";
                    }}
                  >
                    <span
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        width: 26,
                        height: 26,
                        borderRadius: 6,
                        background: `${cat.color}20`,
                        color: cat.color,
                        flexShrink: 0,
                      }}
                    >
                      {nodeType.icon}
                    </span>
                    <span style={{ lineHeight: 1.3 }}>{nodeType.label}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {filteredCategories.length === 0 && (
          <div
            style={{
              padding: "20px 12px",
              textAlign: "center",
              fontSize: 12,
              color: "var(--zen-muted, #999)",
            }}
          >
            No nodes match "{search}"
          </div>
        )}
      </div>
    </div>
  );
}
