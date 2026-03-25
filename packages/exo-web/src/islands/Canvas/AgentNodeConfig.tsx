import { useState, useEffect, useCallback, useRef } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface AgentNodeData {
  agent_id?: string;
  inline?: boolean;
  inline_name?: string;
  inline_model_provider?: string;
  inline_model_name?: string;
  inline_instructions?: string;
  inline_tools?: string[];
}

interface AgentFromAPI {
  id: string;
  name: string;
  description?: string;
  model_provider?: string;
  model_name?: string;
}

interface Provider {
  id: string;
  name: string;
  provider_type: string;
}

interface Model {
  id: string;
  model_name: string;
  provider_id: string;
}

interface ToolEntry {
  id: string;
  name: string;
  category: string;
}

interface AgentNodeConfigProps {
  data: AgentNodeData;
  onChange: (updates: Partial<AgentNodeData>) => void;
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

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  appearance: "none" as const,
  backgroundImage:
    'url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'12\' height=\'12\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'%23999\' stroke-width=\'2\'%3E%3Cpolyline points=\'6 9 12 15 18 9\'/%3E%3C/svg%3E")',
  backgroundRepeat: "no-repeat",
  backgroundPosition: "right 10px center",
  paddingRight: 30,
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
/* Component                                                            */
/* ------------------------------------------------------------------ */

export default function AgentNodeConfig({
  data,
  onChange,
}: AgentNodeConfigProps) {
  const [agents, setAgents] = useState<AgentFromAPI[]>([]);
  const [loadingAgents, setLoadingAgents] = useState(true);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [tools, setTools] = useState<ToolEntry[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const fetchedOnce = useRef(false);

  /* Fetch agents, providers, and tools on mount */
  useEffect(() => {
    if (fetchedOnce.current) return;
    fetchedOnce.current = true;

    Promise.all([
      fetch("/api/v1/agents").then((r) => r.json()).then((d) => d.data ?? d),
      fetch("/api/v1/providers").then((r) => r.json()),
      fetch("/api/v1/tools").then((r) => r.json()),
    ])
      .then(([agentList, providerList, toolList]) => {
        setAgents(agentList);
        setProviders(providerList);
        setTools(toolList);
      })
      .catch(() => {})
      .finally(() => setLoadingAgents(false));
  }, []);

  /* Fetch models when inline provider changes */
  const inlineProvider = providers.find(
    (p) => p.provider_type === data.inline_model_provider,
  );

  useEffect(() => {
    if (!data.inline || !inlineProvider) {
      setModels([]);
      return;
    }
    setLoadingModels(true);
    fetch(`/api/v1/models?provider_id=${inlineProvider.id}`)
      .then((r) => r.json())
      .then((list: Model[]) => setModels(list))
      .catch(() => setModels([]))
      .finally(() => setLoadingModels(false));
  }, [data.inline, inlineProvider]);

  /* Handler: select existing agent */
  const handleAgentSelect = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value;
      if (val === "__create_inline__") {
        onChange({ agent_id: undefined, inline: true });
      } else {
        onChange({
          agent_id: val || undefined,
          inline: false,
          inline_name: undefined,
          inline_model_provider: undefined,
          inline_model_name: undefined,
          inline_instructions: undefined,
          inline_tools: undefined,
        });
      }
    },
    [onChange],
  );

  /* Handler: switch back from inline to select */
  const switchToSelect = useCallback(() => {
    onChange({
      inline: false,
      inline_name: undefined,
      inline_model_provider: undefined,
      inline_model_name: undefined,
      inline_instructions: undefined,
      inline_tools: undefined,
    });
  }, [onChange]);

  /* Inline form handlers */
  const handleInlineName = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ inline_name: e.target.value });
    },
    [onChange],
  );

  const handleInlineProvider = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      onChange({
        inline_model_provider: e.target.value || undefined,
        inline_model_name: undefined,
      });
    },
    [onChange],
  );

  const handleInlineModel = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      onChange({ inline_model_name: e.target.value || undefined });
    },
    [onChange],
  );

  const handleInlineInstructions = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange({ inline_instructions: e.target.value });
    },
    [onChange],
  );

  const toggleTool = useCallback(
    (toolId: string) => {
      const current = data.inline_tools || [];
      const next = current.includes(toolId)
        ? current.filter((t) => t !== toolId)
        : [...current, toolId];
      onChange({ inline_tools: next });
    },
    [data.inline_tools, onChange],
  );

  /* Render selected agent info */
  const selectedAgent = agents.find((a) => a.id === data.agent_id);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Mode selection */}
      {!data.inline ? (
        <>
          {/* Agent selector */}
          <div>
            <label style={labelStyle}>Agent</label>
            <select
              value={data.agent_id || ""}
              onChange={handleAgentSelect}
              style={selectStyle}
            >
              <option value="">
                {loadingAgents ? "Loading..." : "Select agent"}
              </option>
              {agents.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                </option>
              ))}
              <option value="__create_inline__">
                + Create Inline Agent
              </option>
            </select>
          </div>

          {/* Selected agent info */}
          {selectedAgent && (
            <div
              style={{
                padding: "8px 10px",
                fontSize: 12,
                background: "var(--zen-subtle, #e0ddd0)",
                borderRadius: 8,
                lineHeight: 1.5,
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 2 }}>
                {selectedAgent.name}
              </div>
              {selectedAgent.description && (
                <div
                  style={{
                    color: "var(--zen-muted, #999)",
                    fontSize: 11,
                    marginBottom: 4,
                  }}
                >
                  {selectedAgent.description}
                </div>
              )}
              {selectedAgent.model_provider && (
                <div style={{ fontSize: 11, color: "var(--zen-muted, #999)" }}>
                  Model: {selectedAgent.model_provider}
                  {selectedAgent.model_name
                    ? `:${selectedAgent.model_name}`
                    : ""}
                </div>
              )}
            </div>
          )}
        </>
      ) : (
        <>
          {/* Inline agent form header */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <span
              style={{
                fontSize: 12,
                fontWeight: 600,
                color: "var(--zen-dark, #2e2e2e)",
              }}
            >
              Inline Agent
            </span>
            <button
              onClick={switchToSelect}
              style={{
                fontSize: 11,
                color: "var(--zen-coral, #F76F53)",
                background: "none",
                border: "none",
                cursor: "pointer",
                textDecoration: "underline",
                padding: 0,
              }}
            >
              Select existing
            </button>
          </div>

          {/* Inline: Name */}
          <div>
            <label style={labelStyle}>Name</label>
            <input
              type="text"
              value={data.inline_name || ""}
              onChange={handleInlineName}
              placeholder="Agent name"
              style={inputStyle}
              {...focusHandlers}
            />
          </div>

          {/* Inline: Provider */}
          <div>
            <label style={labelStyle}>Provider</label>
            <select
              value={data.inline_model_provider || ""}
              onChange={handleInlineProvider}
              style={selectStyle}
            >
              <option value="">Select provider</option>
              {providers.map((p) => (
                <option key={p.id} value={p.provider_type}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* Inline: Model */}
          <div>
            <label style={labelStyle}>Model</label>
            <select
              value={data.inline_model_name || ""}
              onChange={handleInlineModel}
              style={selectStyle}
              disabled={!inlineProvider}
            >
              <option value="">
                {loadingModels
                  ? "Loading..."
                  : !inlineProvider
                    ? "Select provider first"
                    : "Select model"}
              </option>
              {models.map((m) => (
                <option key={m.id} value={m.model_name}>
                  {m.model_name}
                </option>
              ))}
            </select>
          </div>

          {/* Inline: Instructions */}
          <div>
            <label style={labelStyle}>Instructions</label>
            <textarea
              value={data.inline_instructions || ""}
              onChange={handleInlineInstructions}
              placeholder="Agent instructions..."
              rows={4}
              style={{
                ...inputStyle,
                resize: "vertical",
                minHeight: 60,
                fontFamily: "inherit",
                lineHeight: 1.5,
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor =
                  "var(--zen-coral, #F76F53)";
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor =
                  "var(--zen-subtle, #e0ddd0)";
              }}
            />
          </div>

          {/* Inline: Tool assignments */}
          <div>
            <label style={labelStyle}>
              Tools{" "}
              <span style={{ fontWeight: 400, textTransform: "none" }}>
                ({(data.inline_tools || []).length} selected)
              </span>
            </label>
            <div
              style={{
                maxHeight: 140,
                overflowY: "auto",
                border: "1px solid var(--zen-subtle, #e0ddd0)",
                borderRadius: 8,
                padding: 4,
              }}
            >
              {tools.map((tool) => {
                const checked = (data.inline_tools || []).includes(tool.id);
                return (
                  <label
                    key={tool.id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      padding: "4px 6px",
                      borderRadius: 4,
                      cursor: "pointer",
                      fontSize: 12,
                      color: "var(--zen-dark, #2e2e2e)",
                      background: checked
                        ? "var(--zen-subtle, #e0ddd0)"
                        : "transparent",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleTool(tool.id)}
                      style={{ accentColor: "var(--zen-coral, #F76F53)" }}
                    />
                    <span>{tool.name}</span>
                  </label>
                );
              })}
              {tools.length === 0 && (
                <div
                  style={{
                    padding: "8px",
                    fontSize: 11,
                    color: "var(--zen-muted, #999)",
                    textAlign: "center",
                  }}
                >
                  No tools available
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
