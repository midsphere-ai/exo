import { useState, useEffect, useCallback } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface KnowledgeBase {
  id: string;
  name: string;
}

interface KnowledgeRetrievalData {
  knowledge_base_id?: string;
  top_k?: number;
  similarity_threshold?: number;
}

interface KnowledgeRetrievalConfigProps {
  data: KnowledgeRetrievalData;
  onChange: (updates: Partial<KnowledgeRetrievalData>) => void;
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

const selectStyle: React.CSSProperties = {
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
  appearance: "none" as const,
  backgroundImage:
    'url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'12\' height=\'12\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'%23999\' stroke-width=\'2\'%3E%3Cpolyline points=\'6 9 12 15 18 9\'/%3E%3C/svg%3E")',
  backgroundRepeat: "no-repeat",
  backgroundPosition: "right 10px center",
  paddingRight: 30,
};

const errorSelectStyle: React.CSSProperties = {
  ...selectStyle,
  borderColor: "var(--zen-coral, #F76F53)",
};

const errorMsgStyle: React.CSSProperties = {
  fontSize: 11,
  color: "var(--zen-coral, #F76F53)",
  marginTop: 4,
};

const focusHandlers = {
  onFocus: (e: React.FocusEvent<HTMLSelectElement>) => {
    e.currentTarget.style.borderColor = "var(--zen-coral, #F76F53)";
  },
  onBlur: (e: React.FocusEvent<HTMLSelectElement>) => {
    e.currentTarget.style.borderColor = "var(--zen-subtle, #e0ddd0)";
  },
};

/* ------------------------------------------------------------------ */
/* Component                                                            */
/* ------------------------------------------------------------------ */

export default function KnowledgeRetrievalConfig({ data, onChange }: KnowledgeRetrievalConfigProps) {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [touched, setTouched] = useState(false);

  const topK = data.top_k ?? 5;
  const threshold = data.similarity_threshold ?? 0.7;
  const selectedKbId = data.knowledge_base_id || "";

  /* Fetch knowledge bases on mount */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/v1/knowledge-bases");
        if (res.ok) {
          const json = await res.json();
          if (!cancelled) setKnowledgeBases(Array.isArray(json) ? json : []);
        }
      } catch {
        /* endpoint may not exist yet — graceful fallback */
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  /* Validation */
  const kbError = touched && !selectedKbId ? "Knowledge base is required" : "";

  /* Handlers */
  const handleKbChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setTouched(true);
      onChange({ knowledge_base_id: e.target.value || undefined });
    },
    [onChange],
  );

  const handleTopKChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ top_k: parseInt(e.target.value, 10) });
    },
    [onChange],
  );

  const handleThresholdChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ similarity_threshold: parseFloat(e.target.value) });
    },
    [onChange],
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Knowledge Base selector */}
      <div>
        <label style={labelStyle}>
          Knowledge Base{" "}
          <span style={{ color: "var(--zen-coral, #F76F53)" }}>*</span>
        </label>
        {loading ? (
          <div
            style={{
              fontSize: 12,
              color: "var(--zen-muted, #999)",
              padding: "8px 10px",
            }}
          >
            Loading…
          </div>
        ) : knowledgeBases.length === 0 ? (
          <>
            <select disabled style={{ ...selectStyle, opacity: 0.5 }}>
              <option>No knowledge bases available</option>
            </select>
            <div style={{ fontSize: 11, color: "var(--zen-muted, #999)", marginTop: 4 }}>
              Create a knowledge base in the RAG Pipeline section first.
            </div>
          </>
        ) : (
          <>
            <select
              value={selectedKbId}
              onChange={handleKbChange}
              style={kbError ? errorSelectStyle : selectStyle}
              {...focusHandlers}
            >
              <option value="">Select a knowledge base…</option>
              {knowledgeBases.map((kb) => (
                <option key={kb.id} value={kb.id}>
                  {kb.name}
                </option>
              ))}
            </select>
            {kbError && <div style={errorMsgStyle}>{kbError}</div>}
          </>
        )}
      </div>

      {/* Top-K slider */}
      <div>
        <label style={labelStyle}>
          Top-K Results{" "}
          <span style={{ color: "var(--zen-coral, #F76F53)" }}>*</span>
        </label>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={topK}
            onChange={handleTopKChange}
            style={{
              flex: 1,
              accentColor: "var(--zen-coral, #F76F53)",
              cursor: "pointer",
            }}
          />
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              fontFamily: "monospace",
              color: "var(--zen-dark, #2e2e2e)",
              minWidth: 24,
              textAlign: "right",
            }}
          >
            {topK}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 10,
            color: "var(--zen-muted, #999)",
            marginTop: 2,
          }}
        >
          <span>1</span>
          <span>20</span>
        </div>
      </div>

      {/* Similarity Threshold slider */}
      <div>
        <label style={labelStyle}>
          Similarity Threshold{" "}
          <span style={{ color: "var(--zen-coral, #F76F53)" }}>*</span>
        </label>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={threshold}
            onChange={handleThresholdChange}
            style={{
              flex: 1,
              accentColor: "var(--zen-coral, #F76F53)",
              cursor: "pointer",
            }}
          />
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              fontFamily: "monospace",
              color: "var(--zen-dark, #2e2e2e)",
              minWidth: 32,
              textAlign: "right",
            }}
          >
            {threshold.toFixed(2)}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 10,
            color: "var(--zen-muted, #999)",
            marginTop: 2,
          }}
        >
          <span>0.0</span>
          <span>1.0</span>
        </div>
      </div>

      {/* Info box */}
      <div
        style={{
          fontSize: 11,
          lineHeight: 1.5,
          color: "var(--zen-muted, #999)",
          padding: "10px 12px",
          background: "var(--zen-subtle, #e0ddd0)",
          borderRadius: 8,
        }}
      >
        <strong style={{ color: "var(--zen-dark, #2e2e2e)" }}>How it works:</strong>{" "}
        Retrieves the top-{topK} most similar documents from the selected knowledge base
        with a minimum similarity of {threshold.toFixed(2)}.
      </div>
    </div>
  );
}
