import { useCallback } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface ConditionalData {
  condition_expression?: string;
  true_label?: string;
  false_label?: string;
}

interface ConditionalConfigProps {
  data: ConditionalData;
  onChange: (updates: Partial<ConditionalData>) => void;
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

export default function ConditionalConfig({ data, onChange }: ConditionalConfigProps) {
  const handleExpressionChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange({ condition_expression: e.target.value });
    },
    [onChange],
  );

  const handleTrueLabelChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ true_label: e.target.value });
    },
    [onChange],
  );

  const handleFalseLabelChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ false_label: e.target.value });
    },
    [onChange],
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Condition expression */}
      <div>
        <label style={labelStyle}>
          Condition Expression{" "}
          <span style={{ color: "var(--zen-coral, #F76F53)" }}>*</span>
        </label>
        <textarea
          value={data.condition_expression || ""}
          onChange={handleExpressionChange}
          placeholder={"e.g. input.score > 0.8\n     input.status == 'approved'\n     len(input.items) > 0"}
          rows={4}
          style={{
            ...inputStyle,
            resize: "vertical",
            minHeight: 60,
            fontFamily: "monospace",
            fontSize: 12,
            lineHeight: 1.5,
          }}
          {...focusHandlers}
        />
        <div
          style={{
            marginTop: 4,
            fontSize: 10,
            color: "var(--zen-muted, #999)",
            lineHeight: 1.4,
          }}
        >
          Python-like expression evaluated at runtime. Access upstream data via{" "}
          <code style={{ background: "var(--zen-subtle, #e0ddd0)", padding: "1px 4px", borderRadius: 3 }}>
            input
          </code>
        </div>
      </div>

      {/* Output handles */}
      <div>
        <label style={labelStyle}>Output Handles</label>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {/* True branch */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: 20,
                height: 20,
                borderRadius: "50%",
                background: "#22c55e20",
                color: "#22c55e",
                fontSize: 10,
                fontWeight: 700,
                flexShrink: 0,
              }}
            >
              T
            </span>
            <input
              type="text"
              value={data.true_label || "True"}
              onChange={handleTrueLabelChange}
              placeholder="True"
              style={{ ...inputStyle, flex: 1 }}
              {...focusHandlers}
            />
          </div>

          {/* False branch */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: 20,
                height: 20,
                borderRadius: "50%",
                background: "#ef444420",
                color: "#ef4444",
                fontSize: 10,
                fontWeight: 700,
                flexShrink: 0,
              }}
            >
              F
            </span>
            <input
              type="text"
              value={data.false_label || "False"}
              onChange={handleFalseLabelChange}
              placeholder="False"
              style={{ ...inputStyle, flex: 1 }}
              {...focusHandlers}
            />
          </div>
        </div>
      </div>

      {/* Visual handle indicator */}
      <div
        style={{
          padding: "10px 12px",
          background: "var(--zen-subtle, #e0ddd0)",
          borderRadius: 8,
          fontSize: 11,
          color: "var(--zen-muted, #999)",
          lineHeight: 1.5,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 4, color: "var(--zen-dark, #2e2e2e)" }}>
          Handle Guide
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#22c55e",
              flexShrink: 0,
            }}
          />
          <span>
            Top output → <strong>{data.true_label || "True"}</strong> branch
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#ef4444",
              flexShrink: 0,
            }}
          />
          <span>
            Bottom output → <strong>{data.false_label || "False"}</strong> branch
          </span>
        </div>
      </div>
    </div>
  );
}
