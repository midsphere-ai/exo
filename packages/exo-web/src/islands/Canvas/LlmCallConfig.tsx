import { useState, useEffect, useCallback, useRef } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface LlmCallData {
  model_provider?: string;
  model_name?: string;
  prompt?: string;
  temperature?: number;
  max_tokens?: number;
  response_format?: "text" | "json";
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
  provider_name: string;
}

interface LlmCallConfigProps {
  data: LlmCallData;
  onChange: (updates: Partial<LlmCallData>) => void;
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

/* ------------------------------------------------------------------ */
/* Component                                                            */
/* ------------------------------------------------------------------ */

export default function LlmCallConfig({ data, onChange }: LlmCallConfigProps) {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [loadingProviders, setLoadingProviders] = useState(true);
  const [loadingModels, setLoadingModels] = useState(false);
  const fetchedProviders = useRef(false);

  /* Fetch providers on mount */
  useEffect(() => {
    if (fetchedProviders.current) return;
    fetchedProviders.current = true;
    fetch("/api/v1/providers")
      .then((r) => r.json())
      .then((list: Provider[]) => setProviders(list))
      .catch(() => {})
      .finally(() => setLoadingProviders(false));
  }, []);

  /* Fetch models when provider changes */
  const selectedProvider = providers.find(
    (p) => p.provider_type === data.model_provider,
  );

  useEffect(() => {
    if (!selectedProvider) {
      setModels([]);
      return;
    }
    setLoadingModels(true);
    fetch(`/api/v1/models?provider_id=${selectedProvider.id}`)
      .then((r) => r.json())
      .then((list: Model[]) => setModels(list))
      .catch(() => setModels([]))
      .finally(() => setLoadingModels(false));
  }, [selectedProvider]);

  const handleProviderChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value;
      onChange({ model_provider: val || undefined, model_name: undefined });
    },
    [onChange],
  );

  const handleModelChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      onChange({ model_name: e.target.value || undefined });
    },
    [onChange],
  );

  const handlePromptChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange({ prompt: e.target.value });
    },
    [onChange],
  );

  const handleTempChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ temperature: parseFloat(e.target.value) });
    },
    [onChange],
  );

  const handleMaxTokensChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = e.target.value;
      onChange({ max_tokens: val ? parseInt(val, 10) : undefined });
    },
    [onChange],
  );

  const handleFormatChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      onChange({
        response_format: e.target.value as "text" | "json",
      });
    },
    [onChange],
  );

  const temp = data.temperature ?? 0.7;

  /* Highlight {{variable}} patterns in prompt */
  const promptPreview = (data.prompt || "").replace(
    /\{\{(\w+)\}\}/g,
    '<span style="color: var(--zen-coral, #F76F53); font-weight: 600;">{{$1}}</span>',
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Provider selector */}
      <div>
        <label style={labelStyle}>Provider</label>
        <select
          value={data.model_provider || ""}
          onChange={handleProviderChange}
          style={selectStyle}
        >
          <option value="">
            {loadingProviders ? "Loading..." : "Select provider"}
          </option>
          {providers.map((p) => (
            <option key={p.id} value={p.provider_type}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      {/* Model selector */}
      <div>
        <label style={labelStyle}>Model</label>
        <select
          value={data.model_name || ""}
          onChange={handleModelChange}
          style={selectStyle}
          disabled={!selectedProvider}
        >
          <option value="">
            {loadingModels
              ? "Loading models..."
              : !selectedProvider
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

      {/* Prompt textarea with {{variable}} support */}
      <div>
        <label style={labelStyle}>
          Prompt{" "}
          <span style={{ fontWeight: 400, textTransform: "none" }}>
            (use {"{{var}}"} for variables)
          </span>
        </label>
        <div style={{ position: "relative" }}>
          <textarea
            value={data.prompt || ""}
            onChange={handlePromptChange}
            placeholder="Enter your prompt..."
            rows={5}
            style={{
              ...inputStyle,
              resize: "vertical",
              minHeight: 80,
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
          {/* Variable preview below textarea */}
          {data.prompt && /\{\{\w+\}\}/.test(data.prompt) && (
            <div
              style={{
                marginTop: 4,
                padding: "4px 8px",
                fontSize: 11,
                color: "var(--zen-muted, #999)",
                background: "var(--zen-subtle, #e0ddd0)",
                borderRadius: 4,
                lineHeight: 1.5,
              }}
              dangerouslySetInnerHTML={{ __html: `Preview: ${promptPreview}` }}
            />
          )}
        </div>
      </div>

      {/* Temperature slider */}
      <div>
        <label style={labelStyle}>
          Temperature{" "}
          <span style={{ float: "right", fontWeight: 400 }}>{temp.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          value={temp}
          onChange={handleTempChange}
          style={{
            width: "100%",
            accentColor: "var(--zen-coral, #F76F53)",
          }}
        />
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 10,
            color: "var(--zen-muted, #999)",
            marginTop: 2,
          }}
        >
          <span>Precise</span>
          <span>Creative</span>
        </div>
      </div>

      {/* Max tokens input */}
      <div>
        <label style={labelStyle}>Max Tokens</label>
        <input
          type="number"
          value={data.max_tokens ?? ""}
          onChange={handleMaxTokensChange}
          placeholder="Auto (model default)"
          min={1}
          max={128000}
          style={inputStyle}
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

      {/* Response format selector */}
      <div>
        <label style={labelStyle}>Response Format</label>
        <select
          value={data.response_format || "text"}
          onChange={handleFormatChange}
          style={selectStyle}
        >
          <option value="text">Text</option>
          <option value="json">JSON</option>
        </select>
      </div>
    </div>
  );
}
