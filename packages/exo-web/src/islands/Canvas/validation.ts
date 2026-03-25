/* ------------------------------------------------------------------ */
/* Workflow canvas real-time validation                                 */
/* ------------------------------------------------------------------ */

import type { Node, Edge } from "@xyflow/react";
import { getHandlesForNodeType } from "./handleTypes";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

export type IssueSeverity = "warning" | "error";

export interface ValidationIssue {
  nodeId: string;
  severity: IssueSeverity;
  message: string;
}

export interface ValidationResult {
  issues: ValidationIssue[];
  /** Node IDs with missing required config → show yellow warning badge */
  missingConfig: Set<string>;
  /** Map of nodeId → set of handle IDs that are disconnected required inputs */
  disconnectedInputs: Map<string, Set<string>>;
  /** Edge IDs that form a cycle → highlight in red */
  cycleEdges: Set<string>;
  /** Node IDs not reachable from any trigger → dashed border */
  unreachableNodes: Set<string>;
}

/* ------------------------------------------------------------------ */
/* Trigger node types (no input handles — they start the flow)         */
/* ------------------------------------------------------------------ */

const TRIGGER_TYPES = new Set(["chat_input", "webhook", "schedule", "manual"]);

/* ------------------------------------------------------------------ */
/* Node types that require specific configuration                      */
/* ------------------------------------------------------------------ */

/** Check if a node type has required configuration that may be missing. */
function isMissingConfig(data: Record<string, unknown>): boolean {
  const nodeType = (data.nodeType as string) ?? "default";

  switch (nodeType) {
    case "llm_call":
      // Requires a model provider + model
      return !data.provider || !data.model;
    case "agent_node":
    case "sub_agent":
      // Requires an agent to be selected (or inline config)
      return !data.agent_id && !data.inline;
    case "http_request":
      // Requires a URL
      return !data.url;
    case "code_python":
    case "code_javascript":
      // Requires code
      return !data.code;
    case "knowledge_retrieval":
      // Requires a knowledge base
      return !data.knowledge_base_id;
    case "conditional":
      // Requires a condition expression
      return !data.expression;
    case "prompt_template":
      // Requires prompt text
      return !data.prompt;
    default:
      return false;
  }
}

/* ------------------------------------------------------------------ */
/* Cycle detection via DFS                                             */
/* ------------------------------------------------------------------ */

function detectCycles(
  nodes: Node[],
  edges: Edge[],
): Set<string> {
  const cycleEdgeIds = new Set<string>();

  // Build adjacency list: source → [{target, edgeId}]
  const adj = new Map<string, { target: string; edgeId: string }[]>();
  for (const e of edges) {
    if (!adj.has(e.source)) adj.set(e.source, []);
    adj.get(e.source)!.push({ target: e.target, edgeId: e.id });
  }

  const WHITE = 0; // unvisited
  const GRAY = 1; // in current DFS path
  const BLACK = 2; // fully processed

  const color = new Map<string, number>();
  for (const n of nodes) color.set(n.id, WHITE);

  // Track parent edges in current DFS path for reconstruction
  const pathEdges: string[] = [];
  const pathNodes: string[] = [];

  function dfs(nodeId: string): void {
    color.set(nodeId, GRAY);
    pathNodes.push(nodeId);

    for (const { target, edgeId } of adj.get(nodeId) ?? []) {
      if (color.get(target) === GRAY) {
        // Found a back edge → cycle. Mark all edges in the cycle.
        cycleEdgeIds.add(edgeId);
        // Trace back through pathNodes to find cycle edges
        const cycleStart = pathNodes.indexOf(target);
        if (cycleStart >= 0) {
          for (let i = cycleStart; i < pathEdges.length; i++) {
            cycleEdgeIds.add(pathEdges[i]);
          }
        }
      } else if (color.get(target) === WHITE) {
        pathEdges.push(edgeId);
        dfs(target);
        pathEdges.pop();
      }
    }

    pathNodes.pop();
    color.set(nodeId, BLACK);
  }

  for (const n of nodes) {
    if (color.get(n.id) === WHITE) {
      dfs(n.id);
    }
  }

  return cycleEdgeIds;
}

/* ------------------------------------------------------------------ */
/* Reachability from triggers                                          */
/* ------------------------------------------------------------------ */

function findUnreachableNodes(
  nodes: Node[],
  edges: Edge[],
): Set<string> {
  // Build adjacency: source → targets (forward traversal from triggers)
  const childrenOf = new Map<string, string[]>();
  for (const e of edges) {
    if (!childrenOf.has(e.source)) childrenOf.set(e.source, []);
    childrenOf.get(e.source)!.push(e.target);
  }

  // Find all trigger nodes
  const triggers = nodes.filter((n) => {
    const nt = (n.data as { nodeType?: string }).nodeType ?? "default";
    return TRIGGER_TYPES.has(nt);
  });

  // BFS from all triggers
  const reachable = new Set<string>();
  const queue: string[] = [];
  for (const t of triggers) {
    reachable.add(t.id);
    queue.push(t.id);
  }

  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const child of childrenOf.get(current) ?? []) {
      if (!reachable.has(child)) {
        reachable.add(child);
        queue.push(child);
      }
    }
  }

  // Nodes not reached from any trigger
  const unreachable = new Set<string>();
  for (const n of nodes) {
    const nt = (n.data as { nodeType?: string }).nodeType ?? "default";
    // Don't flag triggers themselves as unreachable
    if (!TRIGGER_TYPES.has(nt) && !reachable.has(n.id)) {
      unreachable.add(n.id);
    }
  }

  return unreachable;
}

/* ------------------------------------------------------------------ */
/* Disconnected required inputs                                        */
/* ------------------------------------------------------------------ */

function findDisconnectedInputs(
  nodes: Node[],
  edges: Edge[],
): Map<string, Set<string>> {
  // Build set of connected (target, targetHandle) pairs
  const connectedHandles = new Set<string>();
  for (const e of edges) {
    connectedHandles.add(`${e.target}::${e.targetHandle ?? "input"}`);
  }

  const result = new Map<string, Set<string>>();

  for (const n of nodes) {
    const nt = (n.data as { nodeType?: string }).nodeType ?? "default";
    const handles = getHandlesForNodeType(nt);
    const inputs = handles.filter((h) => h.type === "target");

    for (const h of inputs) {
      const key = `${n.id}::${h.id}`;
      if (!connectedHandles.has(key)) {
        if (!result.has(n.id)) result.set(n.id, new Set());
        result.get(n.id)!.add(h.id);
      }
    }
  }

  return result;
}

/* ------------------------------------------------------------------ */
/* Main validation function                                            */
/* ------------------------------------------------------------------ */

export function validateWorkflow(
  nodes: Node[],
  edges: Edge[],
): ValidationResult {
  const issues: ValidationIssue[] = [];
  const missingConfig = new Set<string>();
  const disconnectedInputs = findDisconnectedInputs(nodes, edges);
  const cycleEdges = detectCycles(nodes, edges);
  const unreachableNodes = findUnreachableNodes(nodes, edges);

  // Check missing config
  for (const n of nodes) {
    const data = n.data as Record<string, unknown>;
    if (isMissingConfig(data)) {
      missingConfig.add(n.id);
      const nt = (data.nodeType as string) ?? "default";
      issues.push({
        nodeId: n.id,
        severity: "warning",
        message: `${data.label ?? nt}: missing required configuration`,
      });
    }
  }

  // Disconnected required inputs
  for (const [nodeId, handleIds] of disconnectedInputs) {
    const n = nodes.find((nd) => nd.id === nodeId);
    const label = (n?.data as { label?: string })?.label ?? nodeId;
    issues.push({
      nodeId,
      severity: "error",
      message: `${label}: ${handleIds.size} disconnected input${handleIds.size > 1 ? "s" : ""}`,
    });
  }

  // Cycles
  if (cycleEdges.size > 0) {
    // Find the nodes involved in cycle edges to report
    const cycleNodeIds = new Set<string>();
    for (const e of edges) {
      if (cycleEdges.has(e.id)) {
        cycleNodeIds.add(e.source);
        cycleNodeIds.add(e.target);
      }
    }
    for (const nodeId of cycleNodeIds) {
      const n = nodes.find((nd) => nd.id === nodeId);
      const label = (n?.data as { label?: string })?.label ?? nodeId;
      issues.push({
        nodeId,
        severity: "error",
        message: `${label}: part of a cycle`,
      });
    }
  }

  // Unreachable nodes
  for (const nodeId of unreachableNodes) {
    const n = nodes.find((nd) => nd.id === nodeId);
    const label = (n?.data as { label?: string })?.label ?? nodeId;
    issues.push({
      nodeId,
      severity: "warning",
      message: `${label}: not connected to any trigger`,
    });
  }

  return { issues, missingConfig, disconnectedInputs, cycleEdges, unreachableNodes };
}
