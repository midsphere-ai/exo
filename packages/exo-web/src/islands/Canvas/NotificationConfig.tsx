import { useState, useCallback, useEffect } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

interface NotificationNodeData {
  notification_type?: "slack" | "discord" | "email";
  template_id?: string;
  webhook_url?: string;
  channel?: string;
  username?: string;
  icon_emoji?: string;
  message_template?: string;
  avatar_url?: string;
  smtp_host?: string;
  smtp_port?: number;
  smtp_user?: string;
  smtp_password?: string;
  use_tls?: boolean;
  from_address?: string;
  to_addresses?: string;
  subject_template?: string;
  body_template?: string;
}

interface NotificationTemplate {
  id: string;
  name: string;
  type: string;
  config_json: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

interface NotificationConfigProps {
  data: NotificationNodeData;
  onChange: (updates: Partial<NotificationNodeData>) => void;
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
  cursor: "pointer",
  appearance: "auto" as const,
};

const focusHandlers = {
  onFocus: (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    e.currentTarget.style.borderColor = "var(--zen-coral, #F76F53)";
  },
  onBlur: (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    e.currentTarget.style.borderColor = "var(--zen-subtle, #e0ddd0)";
  },
};

const hintStyle: React.CSSProperties = {
  marginTop: 4,
  fontSize: 10,
  color: "var(--zen-muted, #999)",
  lineHeight: 1.4,
};

/* ------------------------------------------------------------------ */
/* Notification type icons & colors                                     */
/* ------------------------------------------------------------------ */

const TYPE_INFO: Record<string, { label: string; color: string; icon: string }> = {
  slack: { label: "Slack", color: "#4A154B", icon: "#" },
  discord: { label: "Discord", color: "#5865F2", icon: "D" },
  email: { label: "Email", color: "#EA4335", icon: "@" },
};

/* ------------------------------------------------------------------ */
/* Component                                                            */
/* ------------------------------------------------------------------ */

export default function NotificationConfig({ data, onChange }: NotificationConfigProps) {
  const [templates, setTemplates] = useState<NotificationTemplate[]>([]);
  const [templatesLoaded, setTemplatesLoaded] = useState(false);

  /* Fetch saved templates */
  useEffect(() => {
    if (templatesLoaded) return;
    fetch("/api/v1/notification-templates")
      .then((r) => (r.ok ? r.json() : []))
      .then((list: NotificationTemplate[]) => {
        setTemplates(list);
        setTemplatesLoaded(true);
      })
      .catch(() => setTemplatesLoaded(true));
  }, [templatesLoaded]);

  /* Load a saved template */
  const handleLoadTemplate = useCallback(
    (templateId: string) => {
      const tpl = templates.find((t) => t.id === templateId);
      if (!tpl) return;
      const cfg = tpl.config_json as Record<string, unknown>;
      onChange({
        template_id: tpl.id,
        notification_type: tpl.type as "slack" | "discord" | "email",
        webhook_url: (cfg.webhook_url as string) || "",
        channel: (cfg.channel as string) || "",
        username: (cfg.username as string) || "",
        icon_emoji: (cfg.icon_emoji as string) || "",
        message_template: (cfg.message_template as string) || "",
        avatar_url: (cfg.avatar_url as string) || "",
        smtp_host: (cfg.smtp_host as string) || "",
        smtp_port: (cfg.smtp_port as number) || 587,
        smtp_user: (cfg.smtp_user as string) || "",
        smtp_password: (cfg.smtp_password as string) || "",
        use_tls: cfg.use_tls !== false,
        from_address: (cfg.from_address as string) || "",
        to_addresses: Array.isArray(cfg.to_addresses)
          ? (cfg.to_addresses as string[]).join(", ")
          : (cfg.to_addresses as string) || "",
        subject_template: (cfg.subject_template as string) || "",
        body_template: (cfg.body_template as string) || "",
      });
    },
    [templates, onChange],
  );

  const notifType = data.notification_type || "slack";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Notification type selector */}
      <div>
        <label style={labelStyle}>Notification Type</label>
        <div style={{ display: "flex", gap: 6 }}>
          {(["slack", "discord", "email"] as const).map((t) => {
            const info = TYPE_INFO[t];
            const selected = notifType === t;
            return (
              <button
                key={t}
                onClick={() => onChange({ notification_type: t })}
                style={{
                  flex: 1,
                  padding: "8px 4px",
                  fontSize: 11,
                  fontWeight: 600,
                  border: selected ? `2px solid ${info.color}` : "1px solid var(--zen-subtle, #e0ddd0)",
                  borderRadius: 8,
                  background: selected ? `${info.color}15` : "var(--zen-paper, #f2f0e3)",
                  color: selected ? info.color : "var(--zen-muted, #999)",
                  cursor: "pointer",
                  transition: "all 150ms",
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: 16, marginBottom: 2 }}>{info.icon}</div>
                {info.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Load from saved template */}
      {templates.length > 0 && (
        <div>
          <label style={labelStyle}>Load From Template</label>
          <select
            value={data.template_id || ""}
            onChange={(e) => {
              if (e.target.value) handleLoadTemplate(e.target.value);
            }}
            style={selectStyle}
            {...focusHandlers}
          >
            <option value="">— Select a saved template —</option>
            {templates
              .filter((t) => t.type === notifType)
              .map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
          </select>
        </div>
      )}

      {/* Type-specific config forms */}
      {notifType === "slack" && (
        <SlackConfig data={data} onChange={onChange} />
      )}
      {notifType === "discord" && (
        <DiscordConfig data={data} onChange={onChange} />
      )}
      {notifType === "email" && (
        <EmailConfig data={data} onChange={onChange} />
      )}

      {/* Info box */}
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
          Template Variables
        </div>
        Use these in message templates:
        <div style={{ marginTop: 4, fontFamily: "monospace", fontSize: 10 }}>
          {"{{workflow_name}}"} {"{{status}}"} {"{{run_id}}"} {"{{timestamp}}"}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Slack config form                                                    */
/* ------------------------------------------------------------------ */

function SlackConfig({
  data,
  onChange,
}: {
  data: NotificationNodeData;
  onChange: (u: Partial<NotificationNodeData>) => void;
}) {
  return (
    <>
      <div>
        <label style={labelStyle}>Slack Webhook URL</label>
        <input
          type="url"
          value={data.webhook_url || ""}
          onChange={(e) => onChange({ webhook_url: e.target.value })}
          placeholder="https://hooks.slack.com/services/..."
          style={inputStyle}
          {...focusHandlers}
        />
      </div>
      <div>
        <label style={labelStyle}>Channel (optional)</label>
        <input
          type="text"
          value={data.channel || ""}
          onChange={(e) => onChange({ channel: e.target.value })}
          placeholder="#general"
          style={inputStyle}
          {...focusHandlers}
        />
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>Username</label>
          <input
            type="text"
            value={data.username || "Exo"}
            onChange={(e) => onChange({ username: e.target.value })}
            style={inputStyle}
            {...focusHandlers}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>Icon Emoji</label>
          <input
            type="text"
            value={data.icon_emoji || ":rocket:"}
            onChange={(e) => onChange({ icon_emoji: e.target.value })}
            style={inputStyle}
            {...focusHandlers}
          />
        </div>
      </div>
      <div>
        <label style={labelStyle}>Message Template</label>
        <textarea
          value={data.message_template || "Workflow *{{workflow_name}}* completed with status: {{status}}"}
          onChange={(e) => onChange({ message_template: e.target.value })}
          rows={3}
          style={{ ...inputStyle, resize: "vertical", minHeight: 50, lineHeight: 1.5 }}
          {...focusHandlers}
        />
        <div style={hintStyle}>Supports Slack markdown formatting</div>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Discord config form                                                  */
/* ------------------------------------------------------------------ */

function DiscordConfig({
  data,
  onChange,
}: {
  data: NotificationNodeData;
  onChange: (u: Partial<NotificationNodeData>) => void;
}) {
  return (
    <>
      <div>
        <label style={labelStyle}>Discord Webhook URL</label>
        <input
          type="url"
          value={data.webhook_url || ""}
          onChange={(e) => onChange({ webhook_url: e.target.value })}
          placeholder="https://discord.com/api/webhooks/..."
          style={inputStyle}
          {...focusHandlers}
        />
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>Username</label>
          <input
            type="text"
            value={data.username || "Exo"}
            onChange={(e) => onChange({ username: e.target.value })}
            style={inputStyle}
            {...focusHandlers}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>Avatar URL</label>
          <input
            type="url"
            value={data.avatar_url || ""}
            onChange={(e) => onChange({ avatar_url: e.target.value })}
            placeholder="https://..."
            style={inputStyle}
            {...focusHandlers}
          />
        </div>
      </div>
      <div>
        <label style={labelStyle}>Message Template</label>
        <textarea
          value={data.message_template || "Workflow **{{workflow_name}}** completed with status: {{status}}"}
          onChange={(e) => onChange({ message_template: e.target.value })}
          rows={3}
          style={{ ...inputStyle, resize: "vertical", minHeight: 50, lineHeight: 1.5 }}
          {...focusHandlers}
        />
        <div style={hintStyle}>Supports Discord markdown formatting</div>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Email config form                                                    */
/* ------------------------------------------------------------------ */

function EmailConfig({
  data,
  onChange,
}: {
  data: NotificationNodeData;
  onChange: (u: Partial<NotificationNodeData>) => void;
}) {
  return (
    <>
      <div style={{ display: "flex", gap: 8 }}>
        <div style={{ flex: 2 }}>
          <label style={labelStyle}>SMTP Host</label>
          <input
            type="text"
            value={data.smtp_host || ""}
            onChange={(e) => onChange({ smtp_host: e.target.value })}
            placeholder="smtp.gmail.com"
            style={inputStyle}
            {...focusHandlers}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>Port</label>
          <input
            type="number"
            value={data.smtp_port ?? 587}
            onChange={(e) => onChange({ smtp_port: parseInt(e.target.value, 10) || 587 })}
            style={inputStyle}
            {...focusHandlers}
          />
        </div>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>SMTP User</label>
          <input
            type="text"
            value={data.smtp_user || ""}
            onChange={(e) => onChange({ smtp_user: e.target.value })}
            style={inputStyle}
            {...focusHandlers}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label style={labelStyle}>SMTP Password</label>
          <input
            type="password"
            value={data.smtp_password || ""}
            onChange={(e) => onChange({ smtp_password: e.target.value })}
            style={inputStyle}
            {...focusHandlers}
          />
        </div>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <label style={{ ...labelStyle, marginBottom: 0 }}>Use TLS</label>
        <button
          onClick={() => onChange({ use_tls: !data.use_tls })}
          style={{
            position: "relative",
            width: 36,
            height: 20,
            borderRadius: 10,
            border: "none",
            cursor: "pointer",
            background: data.use_tls !== false
              ? "var(--zen-blue, #6287f5)"
              : "var(--zen-subtle, #e0ddd0)",
            transition: "background 200ms",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              position: "absolute",
              top: 2,
              left: data.use_tls !== false ? 18 : 2,
              width: 16,
              height: 16,
              borderRadius: "50%",
              background: "#fff",
              boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
              transition: "left 200ms",
            }}
          />
        </button>
      </div>
      <div>
        <label style={labelStyle}>From Address</label>
        <input
          type="email"
          value={data.from_address || ""}
          onChange={(e) => onChange({ from_address: e.target.value })}
          placeholder="noreply@example.com"
          style={inputStyle}
          {...focusHandlers}
        />
      </div>
      <div>
        <label style={labelStyle}>To Addresses</label>
        <input
          type="text"
          value={data.to_addresses || ""}
          onChange={(e) => onChange({ to_addresses: e.target.value })}
          placeholder="user@example.com, team@example.com"
          style={inputStyle}
          {...focusHandlers}
        />
        <div style={hintStyle}>Comma-separated list of email addresses</div>
      </div>
      <div>
        <label style={labelStyle}>Subject Template</label>
        <input
          type="text"
          value={data.subject_template || "Exo: {{workflow_name}} — {{status}}"}
          onChange={(e) => onChange({ subject_template: e.target.value })}
          style={inputStyle}
          {...focusHandlers}
        />
      </div>
      <div>
        <label style={labelStyle}>Body Template</label>
        <textarea
          value={data.body_template || "Workflow {{workflow_name}} completed with status: {{status}}"}
          onChange={(e) => onChange({ body_template: e.target.value })}
          rows={3}
          style={{ ...inputStyle, resize: "vertical", minHeight: 50, lineHeight: 1.5 }}
          {...focusHandlers}
        />
      </div>
    </>
  );
}
