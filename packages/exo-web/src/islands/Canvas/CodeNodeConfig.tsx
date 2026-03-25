import { useState, useCallback, useRef, useEffect } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface CodeNodeData {
  language?: "python" | "javascript";
  code?: string;
  entry_function?: string;
  timeout_seconds?: number;
}

interface CodeNodeConfigProps {
  data: CodeNodeData;
  onChange: (updates: Partial<CodeNodeData>) => void;
  /** The node type from the sidebar (code_python or code_javascript) */
  initialLanguage: "python" | "javascript";
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
/* Syntax highlighting helpers                                          */
/* ------------------------------------------------------------------ */

const PYTHON_KEYWORDS = new Set([
  "def", "class", "return", "if", "elif", "else", "for", "while", "import",
  "from", "as", "try", "except", "finally", "with", "yield", "raise", "pass",
  "break", "continue", "and", "or", "not", "in", "is", "None", "True", "False",
  "lambda", "async", "await",
]);

const JS_KEYWORDS = new Set([
  "function", "return", "if", "else", "for", "while", "const", "let", "var",
  "class", "import", "export", "from", "try", "catch", "finally", "throw",
  "new", "this", "async", "await", "yield", "break", "continue", "switch",
  "case", "default", "true", "false", "null", "undefined", "typeof", "instanceof",
  "of", "in",
]);

function highlightCode(code: string, language: "python" | "javascript"): string {
  const keywords = language === "python" ? PYTHON_KEYWORDS : JS_KEYWORDS;
  const escaped = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  return escaped
    /* Comments */
    .replace(/(#.*$|\/\/.*$)/gm, '<span style="color:#6a9955;font-style:italic">$1</span>')
    /* Strings (single and double quoted) */
    .replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, '<span style="color:#ce9178">$&</span>')
    /* Numbers */
    .replace(/\b(\d+\.?\d*)\b/g, '<span style="color:#b5cea8">$1</span>')
    /* Keywords */
    .replace(/\b(\w+)\b/g, (match) =>
      keywords.has(match)
        ? `<span style="color:#569cd6;font-weight:600">${match}</span>`
        : match,
    );
}

/* ------------------------------------------------------------------ */
/* Default code templates                                               */
/* ------------------------------------------------------------------ */

const PYTHON_DEFAULT = `def process(input_data):
    """Process input and return result."""
    result = input_data
    return result`;

const JS_DEFAULT = `function process(inputData) {
  // Process input and return result
  const result = inputData;
  return result;
}`;

/* ------------------------------------------------------------------ */
/* Component                                                            */
/* ------------------------------------------------------------------ */

export default function CodeNodeConfig({ data, onChange, initialLanguage }: CodeNodeConfigProps) {
  const language = data.language || initialLanguage;
  const [lineCount, setLineCount] = useState(1);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);

  /* Sync line count */
  useEffect(() => {
    const lines = (data.code || "").split("\n").length;
    setLineCount(Math.max(lines, 1));
  }, [data.code]);

  /* Sync scroll between textarea and backdrop */
  const syncScroll = useCallback(() => {
    if (textareaRef.current && backdropRef.current) {
      backdropRef.current.scrollTop = textareaRef.current.scrollTop;
      backdropRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  }, []);

  const handleLanguageToggle = useCallback(
    (lang: "python" | "javascript") => {
      if (lang === language) return;
      const currentCode = data.code || "";
      const defaultCode = language === "python" ? PYTHON_DEFAULT : JS_DEFAULT;
      /* Only replace code with new default if current is empty or is the old default */
      const newCode =
        !currentCode || currentCode === defaultCode
          ? (lang === "python" ? PYTHON_DEFAULT : JS_DEFAULT)
          : currentCode;
      onChange({ language: lang, code: newCode });
    },
    [language, data.code, onChange],
  );

  const handleCodeChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange({ code: e.target.value });
    },
    [onChange],
  );

  const handleEntryFunctionChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ entry_function: e.target.value || undefined });
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

  /* Handle Tab key in textarea for indentation */
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Tab") {
        e.preventDefault();
        const ta = e.currentTarget;
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        const indent = language === "python" ? "    " : "  ";
        const newValue = ta.value.substring(0, start) + indent + ta.value.substring(end);
        onChange({ code: newValue });
        /* Restore cursor position after React re-renders */
        requestAnimationFrame(() => {
          ta.selectionStart = ta.selectionEnd = start + indent.length;
        });
      }
    },
    [language, onChange],
  );

  const codeValue = data.code || "";
  const highlighted = highlightCode(codeValue, language);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Language toggle */}
      <div>
        <label style={labelStyle}>Language</label>
        <div
          style={{
            display: "flex",
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            borderRadius: 8,
            overflow: "hidden",
          }}
        >
          <button
            onClick={() => handleLanguageToggle("python")}
            style={{
              flex: 1,
              padding: "7px 0",
              fontSize: 12,
              fontWeight: 600,
              border: "none",
              cursor: "pointer",
              background: language === "python" ? "var(--zen-coral, #F76F53)" : "var(--zen-paper, #f2f0e3)",
              color: language === "python" ? "#fff" : "var(--zen-dark, #2e2e2e)",
              transition: "all 150ms",
            }}
          >
            Python
          </button>
          <button
            onClick={() => handleLanguageToggle("javascript")}
            style={{
              flex: 1,
              padding: "7px 0",
              fontSize: 12,
              fontWeight: 600,
              border: "none",
              borderLeft: "1px solid var(--zen-subtle, #e0ddd0)",
              cursor: "pointer",
              background: language === "javascript" ? "var(--zen-coral, #F76F53)" : "var(--zen-paper, #f2f0e3)",
              color: language === "javascript" ? "#fff" : "var(--zen-dark, #2e2e2e)",
              transition: "all 150ms",
            }}
          >
            JavaScript
          </button>
        </div>
      </div>

      {/* Code editor with syntax highlighting overlay */}
      <div>
        <label style={labelStyle}>
          Code{" "}
          <span style={{ fontWeight: 400, textTransform: "none" }}>
            ({lineCount} {lineCount === 1 ? "line" : "lines"})
          </span>
        </label>
        <div
          style={{
            position: "relative",
            border: "1px solid var(--zen-subtle, #e0ddd0)",
            borderRadius: 8,
            overflow: "hidden",
            background: "#1e1e1e",
          }}
        >
          {/* Line numbers */}
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: 32,
              height: "100%",
              background: "#252526",
              borderRight: "1px solid #333",
              zIndex: 1,
              pointerEvents: "none",
              overflow: "hidden",
            }}
          >
            <div style={{ padding: "10px 0" }}>
              {Array.from({ length: lineCount }, (_, i) => (
                <div
                  key={i}
                  style={{
                    height: 18,
                    lineHeight: "18px",
                    fontSize: 11,
                    fontFamily: "monospace",
                    textAlign: "right",
                    paddingRight: 6,
                    color: "#858585",
                    userSelect: "none",
                  }}
                >
                  {i + 1}
                </div>
              ))}
            </div>
          </div>

          {/* Highlighted backdrop */}
          <div
            ref={backdropRef}
            style={{
              position: "absolute",
              top: 0,
              left: 32,
              right: 0,
              bottom: 0,
              padding: "10px 10px 10px 8px",
              fontFamily: "monospace",
              fontSize: 12,
              lineHeight: "18px",
              whiteSpace: "pre-wrap",
              wordWrap: "break-word",
              color: "#d4d4d4",
              overflow: "hidden",
              pointerEvents: "none",
            }}
            dangerouslySetInnerHTML={{ __html: highlighted + "\n" }}
          />

          {/* Transparent textarea */}
          <textarea
            ref={textareaRef}
            value={codeValue}
            onChange={handleCodeChange}
            onScroll={syncScroll}
            onKeyDown={handleKeyDown}
            placeholder={language === "python" ? PYTHON_DEFAULT : JS_DEFAULT}
            spellCheck={false}
            style={{
              position: "relative",
              width: "100%",
              minHeight: 160,
              maxHeight: 300,
              padding: "10px 10px 10px 40px",
              fontFamily: "monospace",
              fontSize: 12,
              lineHeight: "18px",
              border: "none",
              background: "transparent",
              color: "transparent",
              caretColor: "#d4d4d4",
              outline: "none",
              resize: "vertical",
              boxSizing: "border-box",
              zIndex: 2,
              whiteSpace: "pre-wrap",
              wordWrap: "break-word",
            }}
          />
        </div>
      </div>

      {/* Entry function */}
      <div>
        <label style={labelStyle}>Entry Function</label>
        <input
          type="text"
          value={data.entry_function || ""}
          onChange={handleEntryFunctionChange}
          placeholder={language === "python" ? "process" : "process"}
          style={{ ...inputStyle, fontFamily: "monospace", fontSize: 12 }}
          {...focusHandlers}
        />
        <div
          style={{
            marginTop: 3,
            fontSize: 10,
            color: "var(--zen-muted, #999)",
          }}
        >
          Function name to call as entry point (defaults to &ldquo;process&rdquo;)
        </div>
      </div>

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
