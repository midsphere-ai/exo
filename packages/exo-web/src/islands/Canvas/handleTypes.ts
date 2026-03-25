/* ------------------------------------------------------------------ */
/* Handle type system for workflow canvas edges                         */
/* ------------------------------------------------------------------ */

/** Data types that can flow between node handles. */
export type HandleDataType =
  | "message"
  | "text"
  | "number"
  | "boolean"
  | "json"
  | "any";

/** Color assigned to each handle data type. */
export const HANDLE_COLORS: Record<HandleDataType, string> = {
  message: "#a78bfa", // purple
  text: "#6287f5", // blue
  number: "#63f78b", // green
  boolean: "#f59e0b", // orange
  json: "#eab308", // yellow
  any: "#999999", // gray
};

/** Distinct colors for conditional branch handles (not tied to data type). */
export const BRANCH_COLORS = {
  true: "#22c55e", // green
  false: "#ef4444", // red
} as const;

/** Specification of a single handle on a node. */
export interface HandleSpec {
  id: string;
  type: "source" | "target";
  dataType: HandleDataType;
  label?: string;
}

/** Returns the handles for a given nodeType. */
export function getHandlesForNodeType(nodeType: string): HandleSpec[] {
  const spec = NODE_HANDLE_MAP[nodeType];
  if (spec) return spec;
  // Default: one input, one output, both "any"
  return DEFAULT_HANDLES;
}

/**
 * Check if two handle data types are compatible for connection.
 * "any" type is compatible with everything.
 */
export function areTypesCompatible(
  sourceType: HandleDataType,
  targetType: HandleDataType,
): boolean {
  if (sourceType === "any" || targetType === "any") return true;
  return sourceType === targetType;
}

/* ------------------------------------------------------------------ */
/* Default handles (most nodes)                                        */
/* ------------------------------------------------------------------ */

const DEFAULT_HANDLES: HandleSpec[] = [
  { id: "input", type: "target", dataType: "any" },
  { id: "output", type: "source", dataType: "any" },
];

/* ------------------------------------------------------------------ */
/* Per-node-type handle definitions                                    */
/* ------------------------------------------------------------------ */

const NODE_HANDLE_MAP: Record<string, HandleSpec[]> = {
  // Triggers — output only (they start the flow)
  chat_input: [
    { id: "output", type: "source", dataType: "message" },
  ],
  webhook: [
    { id: "output", type: "source", dataType: "json" },
  ],
  schedule: [
    { id: "output", type: "source", dataType: "json" },
  ],
  manual: [
    { id: "output", type: "source", dataType: "message" },
  ],

  // LLM
  llm_call: [
    { id: "input", type: "target", dataType: "message" },
    { id: "output", type: "source", dataType: "text" },
  ],
  prompt_template: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "text" },
  ],
  model_selector: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "text" },
  ],

  // Agent
  agent_node: [
    { id: "input", type: "target", dataType: "message" },
    { id: "output", type: "source", dataType: "message" },
  ],
  sub_agent: [
    { id: "input", type: "target", dataType: "message" },
    { id: "output", type: "source", dataType: "message" },
  ],

  // Tools
  function_tool: [
    { id: "input", type: "target", dataType: "json" },
    { id: "output", type: "source", dataType: "json" },
  ],
  http_request: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "json" },
  ],
  code_python: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "any" },
  ],
  code_javascript: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "any" },
  ],

  // Logic
  conditional: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output-true", type: "source", dataType: "any", label: "True" },
    { id: "output-false", type: "source", dataType: "any", label: "False" },
  ],
  switch: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "any" },
  ],
  loop_iterator: [
    { id: "input", type: "target", dataType: "json" },
    { id: "output", type: "source", dataType: "any" },
  ],
  aggregator: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "json" },
  ],

  // Data
  variable_assigner: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "any" },
  ],
  template_jinja: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "text" },
  ],
  json_transform: [
    { id: "input", type: "target", dataType: "json" },
    { id: "output", type: "source", dataType: "json" },
  ],
  text_splitter: [
    { id: "input", type: "target", dataType: "text" },
    { id: "output", type: "source", dataType: "json" },
  ],

  // Knowledge
  knowledge_retrieval: [
    { id: "input", type: "target", dataType: "text" },
    { id: "output", type: "source", dataType: "json" },
  ],
  document_loader: [
    { id: "output", type: "source", dataType: "json" },
  ],
  embedding_node: [
    { id: "input", type: "target", dataType: "text" },
    { id: "output", type: "source", dataType: "json" },
  ],

  // Flow control
  approval_gate: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "any" },
  ],

  // Output — input only (they end the flow)
  chat_response: [
    { id: "input", type: "target", dataType: "message" },
  ],
  api_response: [
    { id: "input", type: "target", dataType: "json" },
  ],
  file_output: [
    { id: "input", type: "target", dataType: "any" },
  ],
  notification: [
    { id: "input", type: "target", dataType: "text" },
  ],

  // Integration
  webhook_call: [
    { id: "input", type: "target", dataType: "json" },
    { id: "output", type: "source", dataType: "json" },
  ],
  mcp_client: [
    { id: "input", type: "target", dataType: "any" },
    { id: "output", type: "source", dataType: "json" },
  ],
};
