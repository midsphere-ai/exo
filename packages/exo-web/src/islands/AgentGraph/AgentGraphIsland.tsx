import { memo, useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  MiniMap,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  useReactFlow,
  getBezierPath,
  type Node,
  type Edge,
  type EdgeProps,
  type NodeProps,
  type ColorMode,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";

import "@xyflow/react/dist/style.css";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface Agent {
  id: string;
  name: string;
  description: string;
  model_provider: string;
  model_name: string;
  tools_json: string;
  handoffs_json: string;
  hooks_json: string;
}

interface HandoffTarget {
  id: string;
  name: string;
  description: string;
}

interface AgentNodeData {
  label: string;
  description: string;
  model: string;
  provider: string;
  toolCount: number;
  handoffCount: number;
  agentId: string;
  [key: string]: unknown;
}

/* ------------------------------------------------------------------ */
/* Dagre auto-layout                                                    */
/* ------------------------------------------------------------------ */

const NODE_WIDTH = 240;
const NODE_HEIGHT = 120;

function layoutNodes(
  nodes: Node[],
  edges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 60, ranksep: 80 });

  for (const node of nodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  const laid = nodes.map((node) => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 },
    };
  });
  return { nodes: laid, edges };
}

/* ------------------------------------------------------------------ */
/* AgentNode — custom node                                              */
/* ------------------------------------------------------------------ */

const AgentNode = memo(function AgentNode({ data, selected }: NodeProps) {
  const d = data as AgentNodeData;

  const providerIcon: Record<string, string> = {
    openai: "O",
    anthropic: "A",
    google: "G",
    ollama: "L",
  };
  const icon = providerIcon[d.provider?.toLowerCase()] ?? "?";
  const modelShort = d.model || "no model";

  return (
    <div
      style={{
        width: NODE_WIDTH,
        background: "var(--zen-paper, #f2f0e3)",
        border: `2px solid ${selected ? "var(--zen-coral, #F76F53)" : "var(--zen-subtle, #e0ddd0)"}`,
        borderRadius: 12,
        padding: "12px 14px",
        boxShadow: selected
          ? "0 0 0 2px rgba(247, 111, 83, 0.25)"
          : "0 1px 4px rgba(0,0,0,0.06)",
        cursor: "pointer",
        transition: "border-color 200ms, box-shadow 200ms",
        fontFamily: "var(--font-sans, system-ui)",
      }}
    >
      {/* Top handle for incoming edges */}
      <Handle
        type="target"
        position={Position.Top}
        style={{
          background: "var(--zen-blue, #6287f5)",
          width: 8,
          height: 8,
          border: "2px solid var(--zen-paper, #f2f0e3)",
        }}
        isConnectable={false}
      />

      {/* Agent name + status dot */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: d.model ? "var(--zen-green, #63f78b)" : "var(--zen-muted, #999)",
            flexShrink: 0,
          }}
        />
        <div
          style={{
            fontWeight: 600,
            fontSize: 14,
            color: "var(--zen-dark, #2e2e2e)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {d.label}
        </div>
      </div>

      {/* Model info */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 8,
          fontSize: 12,
          color: "var(--zen-muted, #999)",
        }}
      >
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 18,
            height: 18,
            borderRadius: 4,
            background: "var(--zen-subtle, #e8e5d8)",
            fontSize: 10,
            fontWeight: 700,
            color: "var(--zen-dark, #2e2e2e)",
          }}
        >
          {icon}
        </span>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {modelShort}
        </span>
      </div>

      {/* Tool + handoff counts */}
      <div
        style={{
          display: "flex",
          gap: 10,
          fontSize: 11,
          color: "var(--zen-muted, #999)",
        }}
      >
        <span>
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ display: "inline", verticalAlign: "-2px", marginRight: 3 }}
          >
            <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
          </svg>
          {d.toolCount} tool{d.toolCount !== 1 ? "s" : ""}
        </span>
        <span>
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ display: "inline", verticalAlign: "-2px", marginRight: 3 }}
          >
            <polyline points="15 14 20 9 15 4" />
            <path d="M4 20v-7a4 4 0 0 1 4-4h12" />
          </svg>
          {d.handoffCount} handoff{d.handoffCount !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Bottom handle for outgoing edges */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{
          background: "var(--zen-coral, #F76F53)",
          width: 8,
          height: 8,
          border: "2px solid var(--zen-paper, #f2f0e3)",
        }}
        isConnectable={false}
      />
    </div>
  );
});

/* ------------------------------------------------------------------ */
/* Custom edge with label                                               */
/* ------------------------------------------------------------------ */

function AgentEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  style = {},
}: EdgeProps) {
  const edgeType = (data?.edgeType as string) ?? "handoff";
  const isHandoff = edgeType === "handoff";

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={isHandoff ? "var(--zen-coral, #F76F53)" : "var(--zen-blue, #6287f5)"}
        strokeWidth={2}
        strokeDasharray={isHandoff ? undefined : "6 4"}
        markerEnd="url(#arrowhead)"
        style={style}
      />
      <foreignObject
        x={labelX - 40}
        y={labelY - 10}
        width={80}
        height={20}
        requiredExtensions="http://www.w3.org/1999/xhtml"
      >
        <div
          style={{
            fontSize: 10,
            fontWeight: 500,
            color: isHandoff ? "var(--zen-coral, #F76F53)" : "var(--zen-blue, #6287f5)",
            textAlign: "center",
            background: "var(--zen-paper, #f2f0e3)",
            borderRadius: 4,
            padding: "1px 6px",
            whiteSpace: "nowrap",
          }}
        >
          {isHandoff ? "handoff" : "delegates to"}
        </div>
      </foreignObject>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Node + edge types (stable refs)                                      */
/* ------------------------------------------------------------------ */

const nodeTypes = { agent: AgentNode };
const edgeTypes = { agent: AgentEdge };

/* ------------------------------------------------------------------ */
/* SVG arrow marker definition                                          */
/* ------------------------------------------------------------------ */

function ArrowMarkerDef() {
  return (
    <svg style={{ position: "absolute", width: 0, height: 0 }}>
      <defs>
        <marker
          id="arrowhead"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--zen-dark, #2e2e2e)" />
        </marker>
      </defs>
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Inner graph (needs useReactFlow)                                     */
/* ------------------------------------------------------------------ */

function AgentGraphInner({ projectId }: { projectId?: string }) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { fitView } = useReactFlow();

  /* Theme sync */
  const [colorMode, setColorMode] = useState<ColorMode>("light");
  useEffect(() => {
    const html = document.documentElement;
    const update = () => setColorMode(html.getAttribute("data-theme") === "dark" ? "dark" : "light");
    update();
    const obs = new MutationObserver(update);
    obs.observe(html, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);

  /* Fetch agents and build graph */
  useEffect(() => {
    const url = projectId
      ? `/api/v1/agents?project_id=${encodeURIComponent(projectId)}`
      : "/api/v1/agents";

    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to load agents");
        return r.json();
      })
      .then((body: any) => (body.data ?? body) as Agent[])
      .then((agents: Agent[]) => {
        const agentMap = new Map(agents.map((a) => [a.id, a]));
        const rawNodes: Node[] = [];
        const rawEdges: Edge[] = [];

        for (const agent of agents) {
          let toolCount = 0;
          try {
            const tools = JSON.parse(agent.tools_json || "[]");
            toolCount = Array.isArray(tools) ? tools.length : 0;
          } catch { /* empty */ }

          let handoffs: HandoffTarget[] = [];
          try {
            const parsed = JSON.parse(agent.handoffs_json || "[]");
            handoffs = Array.isArray(parsed) ? parsed : [];
          } catch { /* empty */ }

          rawNodes.push({
            id: agent.id,
            type: "agent",
            position: { x: 0, y: 0 },
            data: {
              label: agent.name,
              description: agent.description,
              model: agent.model_name,
              provider: agent.model_provider,
              toolCount,
              handoffCount: handoffs.length,
              agentId: agent.id,
            } satisfies AgentNodeData,
          });

          /* Handoff edges (solid, coral) */
          for (const h of handoffs) {
            if (agentMap.has(h.id)) {
              rawEdges.push({
                id: `handoff-${agent.id}-${h.id}`,
                source: agent.id,
                target: h.id,
                type: "agent",
                data: { edgeType: "handoff" },
              });
            }
          }

          /* Delegation edges: check hooks_json for _delegates_to */
          try {
            const hooks = JSON.parse(agent.hooks_json || "{}");
            const delegates: string[] = hooks._delegates_to ?? [];
            for (const targetId of delegates) {
              if (agentMap.has(targetId)) {
                rawEdges.push({
                  id: `delegate-${agent.id}-${targetId}`,
                  source: agent.id,
                  target: targetId,
                  type: "agent",
                  data: { edgeType: "delegate" },
                });
              }
            }
          } catch { /* empty */ }
        }

        const { nodes: laidNodes, edges: laidEdges } = layoutNodes(rawNodes, rawEdges);
        setNodes(laidNodes);
        setEdges(laidEdges);
        setLoading(false);

        setTimeout(() => fitView({ padding: 0.2, duration: 300 }), 50);
      })
      .catch((err) => {
        setError(err.message ?? "Failed to load agents");
        setLoading(false);
      });
  }, [projectId, setNodes, setEdges, fitView]);

  /* Click node -> navigate to agent edit */
  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    const agentId = (node.data as AgentNodeData).agentId;
    window.location.href = `/agents/${agentId}/edit`;
  }, []);

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--zen-muted, #999)" }}>
        Loading agents…
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--zen-coral, #F76F53)" }}>
        {error}
      </div>
    );
  }

  if (nodes.length === 0) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 12, color: "var(--zen-muted, #999)" }}>
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <path d="M8 12h.01M12 12h.01M16 12h.01" />
        </svg>
        <p style={{ fontSize: 14 }}>No agents found. Create some agents first.</p>
      </div>
    );
  }

  return (
    <>
      <ArrowMarkerDef />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        colorMode={colorMode}
        fitView
        minZoom={0.2}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        nodesDraggable
        nodesConnectable={false}
        edgesReconnectable={false}
        deleteKeyCode={null}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
        <MiniMap
          nodeColor={() => "var(--zen-coral, #F76F53)"}
          maskColor="rgba(0,0,0,0.08)"
          style={{ borderRadius: 8 }}
        />
      </ReactFlow>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Exported wrapper with ReactFlowProvider                              */
/* ------------------------------------------------------------------ */

export default function AgentGraphIsland({ projectId }: { projectId?: string }) {
  return (
    <ReactFlowProvider>
      <AgentGraphInner projectId={projectId} />
    </ReactFlowProvider>
  );
}
