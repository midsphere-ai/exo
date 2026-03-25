import { useState, useCallback } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface HeaderEntry {
  key: string;
  value: string;
}

interface HttpRequestData {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  url?: string;
  headers?: HeaderEntry[];
  body?: string;
  auth_type?: "none" | "bearer" | "basic" | "api_key";
  auth_token?: string;
  auth_username?: string;
  auth_password?: string;
  auth_header_name?: string;
  auth_header_value?: string;
  timeout_seconds?: number;
}

interface HttpRequestConfigProps {
  data: HttpRequestData;
  onChange: (updates: Partial<HttpRequestData>) => void;
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
  onFocus: (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    e.currentTarget.style.borderColor = "var(--zen-coral, #F76F53)";
  },
  onBlur: (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    e.currentTarget.style.borderColor = "var(--zen-subtle, #e0ddd0)";
  },
};

const METHOD_COLORS: Record<string, string> = {
  GET: "#22c55e",
  POST: "#3b82f6",
  PUT: "#f59e0b",
  DELETE: "#ef4444",
};

/* ------------------------------------------------------------------ */
/* Component                                                            */
/* ------------------------------------------------------------------ */

export default function HttpRequestConfig({ data, onChange }: HttpRequestConfigProps) {
  const [headersExpanded, setHeadersExpanded] = useState(true);

  const method = data.method || "GET";
  const headers = data.headers || [];
  const authType = data.auth_type || "none";

  /* Handlers */
  const handleMethodChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      onChange({ method: e.target.value as HttpRequestData["method"] });
    },
    [onChange],
  );

  const handleUrlChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ url: e.target.value });
    },
    [onChange],
  );

  const handleBodyChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange({ body: e.target.value });
    },
    [onChange],
  );

  const handleAuthTypeChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      onChange({
        auth_type: e.target.value as HttpRequestData["auth_type"],
        auth_token: undefined,
        auth_username: undefined,
        auth_password: undefined,
        auth_header_name: undefined,
        auth_header_value: undefined,
      });
    },
    [onChange],
  );

  const handleTimeoutChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = e.target.value;
      onChange({ timeout_seconds: val ? parseInt(val, 10) : undefined });
    },
    [onChange],
  );

  /* Header management */
  const addHeader = useCallback(() => {
    onChange({ headers: [...headers, { key: "", value: "" }] });
  }, [headers, onChange]);

  const updateHeader = useCallback(
    (index: number, field: "key" | "value", val: string) => {
      const updated = headers.map((h, i) =>
        i === index ? { ...h, [field]: val } : h,
      );
      onChange({ headers: updated });
    },
    [headers, onChange],
  );

  const removeHeader = useCallback(
    (index: number) => {
      onChange({ headers: headers.filter((_, i) => i !== index) });
    },
    [headers, onChange],
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Method + URL */}
      <div>
        <label style={labelStyle}>
          Request{" "}
          <span style={{ color: "var(--zen-coral, #F76F53)" }}>*</span>
        </label>
        <div style={{ display: "flex", gap: 6 }}>
          <select
            value={method}
            onChange={handleMethodChange}
            style={{
              ...selectStyle,
              width: 90,
              flex: "none",
              fontWeight: 700,
              fontSize: 12,
              color: METHOD_COLORS[method] || "var(--zen-dark)",
            }}
          >
            <option value="GET">GET</option>
            <option value="POST">POST</option>
            <option value="PUT">PUT</option>
            <option value="DELETE">DELETE</option>
          </select>
          <input
            type="text"
            value={data.url || ""}
            onChange={handleUrlChange}
            placeholder="https://api.example.com/endpoint"
            style={{ ...inputStyle, flex: 1 }}
            {...focusHandlers}
          />
        </div>
      </div>

      {/* Headers */}
      <div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 4,
          }}
        >
          <button
            onClick={() => setHeadersExpanded(!headersExpanded)}
            style={{
              ...labelStyle,
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 0,
              display: "flex",
              alignItems: "center",
              gap: 4,
              marginBottom: 0,
            }}
          >
            <svg
              width="10"
              height="10"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              style={{
                transform: headersExpanded ? "rotate(90deg)" : "rotate(0)",
                transition: "transform 150ms",
              }}
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
            Headers
            <span style={{ fontWeight: 400, textTransform: "none" }}>
              ({headers.length})
            </span>
          </button>
          <button
            onClick={addHeader}
            style={{
              fontSize: 11,
              color: "var(--zen-coral, #F76F53)",
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 0,
              fontWeight: 600,
            }}
          >
            + Add
          </button>
        </div>

        {headersExpanded && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {headers.map((header, index) => (
              <div key={index} style={{ display: "flex", gap: 4, alignItems: "center" }}>
                <input
                  type="text"
                  value={header.key}
                  onChange={(e) => updateHeader(index, "key", e.target.value)}
                  placeholder="Key"
                  style={{
                    ...inputStyle,
                    flex: 1,
                    fontSize: 11,
                    padding: "6px 8px",
                    fontFamily: "monospace",
                  }}
                  {...focusHandlers}
                />
                <input
                  type="text"
                  value={header.value}
                  onChange={(e) => updateHeader(index, "value", e.target.value)}
                  placeholder="Value"
                  style={{
                    ...inputStyle,
                    flex: 1,
                    fontSize: 11,
                    padding: "6px 8px",
                    fontFamily: "monospace",
                  }}
                  {...focusHandlers}
                />
                <button
                  onClick={() => removeHeader(index)}
                  title="Remove header"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 22,
                    height: 22,
                    border: "none",
                    borderRadius: 4,
                    background: "transparent",
                    color: "var(--zen-muted, #999)",
                    cursor: "pointer",
                    flexShrink: 0,
                    fontSize: 14,
                  }}
                >
                  ×
                </button>
              </div>
            ))}
            {headers.length === 0 && (
              <div
                style={{
                  fontSize: 11,
                  color: "var(--zen-muted, #999)",
                  padding: "8px",
                  textAlign: "center",
                  border: "1px dashed var(--zen-subtle, #e0ddd0)",
                  borderRadius: 6,
                }}
              >
                No headers. Click "+ Add" to add one.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Body (shown for POST/PUT) */}
      {(method === "POST" || method === "PUT") && (
        <div>
          <label style={labelStyle}>Body</label>
          <textarea
            value={data.body || ""}
            onChange={handleBodyChange}
            placeholder='{"key": "value"}'
            rows={5}
            style={{
              ...inputStyle,
              resize: "vertical",
              minHeight: 80,
              fontFamily: "monospace",
              fontSize: 12,
              lineHeight: 1.5,
            }}
            {...focusHandlers}
          />
        </div>
      )}

      {/* Auth selector */}
      <div>
        <label style={labelStyle}>Authentication</label>
        <select
          value={authType}
          onChange={handleAuthTypeChange}
          style={selectStyle}
        >
          <option value="none">None</option>
          <option value="bearer">Bearer Token</option>
          <option value="basic">Basic Auth</option>
          <option value="api_key">API Key</option>
        </select>
      </div>

      {/* Auth fields based on type */}
      {authType === "bearer" && (
        <div>
          <label style={labelStyle}>Bearer Token</label>
          <input
            type="password"
            value={data.auth_token || ""}
            onChange={(e) => onChange({ auth_token: e.target.value })}
            placeholder="Token value"
            style={inputStyle}
            {...focusHandlers}
          />
        </div>
      )}

      {authType === "basic" && (
        <>
          <div>
            <label style={labelStyle}>Username</label>
            <input
              type="text"
              value={data.auth_username || ""}
              onChange={(e) => onChange({ auth_username: e.target.value })}
              placeholder="Username"
              style={inputStyle}
              {...focusHandlers}
            />
          </div>
          <div>
            <label style={labelStyle}>Password</label>
            <input
              type="password"
              value={data.auth_password || ""}
              onChange={(e) => onChange({ auth_password: e.target.value })}
              placeholder="Password"
              style={inputStyle}
              {...focusHandlers}
            />
          </div>
        </>
      )}

      {authType === "api_key" && (
        <>
          <div>
            <label style={labelStyle}>Header Name</label>
            <input
              type="text"
              value={data.auth_header_name || ""}
              onChange={(e) => onChange({ auth_header_name: e.target.value })}
              placeholder="X-API-Key"
              style={{ ...inputStyle, fontFamily: "monospace", fontSize: 12 }}
              {...focusHandlers}
            />
          </div>
          <div>
            <label style={labelStyle}>Header Value</label>
            <input
              type="password"
              value={data.auth_header_value || ""}
              onChange={(e) => onChange({ auth_header_value: e.target.value })}
              placeholder="Your API key"
              style={inputStyle}
              {...focusHandlers}
            />
          </div>
        </>
      )}

      {/* Timeout */}
      <div>
        <label style={labelStyle}>Timeout (seconds)</label>
        <input
          type="number"
          value={data.timeout_seconds ?? ""}
          onChange={handleTimeoutChange}
          placeholder="30"
          min={1}
          max={300}
          style={inputStyle}
          {...focusHandlers}
        />
      </div>
    </div>
  );
}
