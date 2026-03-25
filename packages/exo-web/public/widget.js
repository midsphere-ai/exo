/**
 * Exo Embeddable Chat Widget
 *
 * Usage:
 *   <script src="https://your-exo.example/widget.js"
 *     data-deployment-id="DEPLOYMENT_ID"
 *     data-base-url="https://your-exo.example"
 *     data-primary-color="#F76F53"
 *     data-position="bottom-right"
 *     data-welcome-message="Hi! How can I help you?"
 *     data-avatar-url=""
 *     defer></script>
 */
(function () {
  "use strict";

  /* ---- Read config from script tag ---- */
  var scripts = document.querySelectorAll("script[data-deployment-id]");
  var scriptTag = scripts[scripts.length - 1];
  if (!scriptTag) return;

  var deploymentId = scriptTag.getAttribute("data-deployment-id");
  var baseUrl = (scriptTag.getAttribute("data-base-url") || "").replace(/\/+$/, "");
  var primaryColor = scriptTag.getAttribute("data-primary-color") || "#F76F53";
  var position = scriptTag.getAttribute("data-position") || "bottom-right";
  var welcomeMessage = scriptTag.getAttribute("data-welcome-message") || "Hi! How can I help you?";
  var avatarUrl = scriptTag.getAttribute("data-avatar-url") || "";

  if (!deploymentId || !baseUrl) return;

  var apiKey = scriptTag.getAttribute("data-api-key") || "";
  var isRight = position !== "bottom-left";

  /* ---- Inject styles ---- */
  var styleId = "exo-widget-styles";
  if (!document.getElementById(styleId)) {
    var style = document.createElement("style");
    style.id = styleId;
    style.textContent = [
      ".exo-widget-bubble{position:fixed;bottom:20px;z-index:99999;width:56px;height:56px;border-radius:50%;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(0,0,0,.15);transition:transform .2s,box-shadow .2s}",
      ".exo-widget-bubble:hover{transform:scale(1.08);box-shadow:0 6px 20px rgba(0,0,0,.2)}",
      ".exo-widget-panel{position:fixed;bottom:88px;z-index:99999;width:380px;max-width:calc(100vw - 32px);height:520px;max-height:calc(100vh - 120px);border-radius:16px;background:#fff;box-shadow:0 12px 40px rgba(0,0,0,.12);display:none;flex-direction:column;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}",
      ".exo-widget-panel.open{display:flex}",
      ".exo-widget-header{padding:16px 20px;color:#fff;display:flex;align-items:center;gap:10px;flex-shrink:0}",
      ".exo-widget-header-avatar{width:32px;height:32px;border-radius:50%;object-fit:cover;background:rgba(255,255,255,.2)}",
      ".exo-widget-header-title{font-size:15px;font-weight:600}",
      ".exo-widget-close{margin-left:auto;background:none;border:none;color:rgba(255,255,255,.8);cursor:pointer;font-size:20px;line-height:1;padding:4px}",
      ".exo-widget-close:hover{color:#fff}",
      ".exo-widget-messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}",
      ".exo-widget-msg{max-width:85%;padding:10px 14px;border-radius:14px;font-size:14px;line-height:1.5;word-break:break-word}",
      ".exo-widget-msg.assistant{background:#f0f0f0;color:#333;align-self:flex-start;border-bottom-left-radius:4px}",
      ".exo-widget-msg.user{color:#fff;align-self:flex-end;border-bottom-right-radius:4px}",
      ".exo-widget-typing{align-self:flex-start;padding:10px 14px;background:#f0f0f0;border-radius:14px;border-bottom-left-radius:4px;font-size:14px;color:#999}",
      ".exo-widget-input-area{display:flex;padding:12px 16px;border-top:1px solid #eee;gap:8px;flex-shrink:0}",
      ".exo-widget-input{flex:1;border:1px solid #ddd;border-radius:24px;padding:10px 16px;font-size:14px;outline:none;font-family:inherit}",
      ".exo-widget-input:focus{border-color:" + primaryColor + "}",
      ".exo-widget-send{width:36px;height:36px;border-radius:50%;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:opacity .2s}",
      ".exo-widget-send:disabled{opacity:.4;cursor:not-allowed}",
      ".exo-widget-send svg{width:18px;height:18px;fill:#fff}",
    ].join("\n");
    document.head.appendChild(style);
  }

  /* ---- Create DOM ---- */
  var container = document.createElement("div");

  // Bubble
  var bubble = document.createElement("button");
  bubble.className = "exo-widget-bubble";
  bubble.style.backgroundColor = primaryColor;
  bubble.style[isRight ? "right" : "left"] = "20px";
  bubble.setAttribute("aria-label", "Open chat");
  bubble.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 21 1.9-5.7a8.5 8.5 0 1 1 3.8 3.8z"/></svg>';

  // Panel
  var panel = document.createElement("div");
  panel.className = "exo-widget-panel";
  panel.style[isRight ? "right" : "left"] = "20px";

  // Header
  var header = document.createElement("div");
  header.className = "exo-widget-header";
  header.style.backgroundColor = primaryColor;
  if (avatarUrl) {
    var avatarImg = document.createElement("img");
    avatarImg.className = "exo-widget-header-avatar";
    avatarImg.src = avatarUrl;
    avatarImg.alt = "Avatar";
    header.appendChild(avatarImg);
  }
  var titleSpan = document.createElement("span");
  titleSpan.className = "exo-widget-header-title";
  titleSpan.textContent = "Chat";
  header.appendChild(titleSpan);
  var closeBtn = document.createElement("button");
  closeBtn.className = "exo-widget-close";
  closeBtn.innerHTML = "&times;";
  closeBtn.setAttribute("aria-label", "Close chat");
  header.appendChild(closeBtn);
  panel.appendChild(header);

  // Messages area
  var messagesArea = document.createElement("div");
  messagesArea.className = "exo-widget-messages";
  panel.appendChild(messagesArea);

  // Input area
  var inputArea = document.createElement("div");
  inputArea.className = "exo-widget-input-area";
  var inputField = document.createElement("input");
  inputField.className = "exo-widget-input";
  inputField.type = "text";
  inputField.placeholder = "Type a message...";
  var sendBtn = document.createElement("button");
  sendBtn.className = "exo-widget-send";
  sendBtn.style.backgroundColor = primaryColor;
  sendBtn.disabled = true;
  sendBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M2.01 21 23 12 2.01 3 2 10l15 2-15 2z"/></svg>';
  inputArea.appendChild(inputField);
  inputArea.appendChild(sendBtn);
  panel.appendChild(inputArea);

  container.appendChild(bubble);
  container.appendChild(panel);
  document.body.appendChild(container);

  /* ---- Show welcome message ---- */
  if (welcomeMessage) {
    addMessage(welcomeMessage, "assistant");
  }

  /* ---- Toggle panel ---- */
  var isOpen = false;
  bubble.addEventListener("click", function () {
    isOpen = !isOpen;
    panel.classList.toggle("open", isOpen);
    bubble.innerHTML = isOpen
      ? '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
      : '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 21 1.9-5.7a8.5 8.5 0 1 1 3.8 3.8z"/></svg>';
    if (isOpen) inputField.focus();
  });
  closeBtn.addEventListener("click", function () {
    isOpen = false;
    panel.classList.remove("open");
    bubble.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 21 1.9-5.7a8.5 8.5 0 1 1 3.8 3.8z"/></svg>';
  });

  /* ---- Input handling ---- */
  inputField.addEventListener("input", function () {
    sendBtn.disabled = !inputField.value.trim();
  });
  inputField.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !sendBtn.disabled) sendMessage();
  });
  sendBtn.addEventListener("click", function () {
    if (!sendBtn.disabled) sendMessage();
  });

  /* ---- Message helpers ---- */
  function addMessage(text, role) {
    var msg = document.createElement("div");
    msg.className = "exo-widget-msg " + role;
    if (role === "user") {
      msg.style.backgroundColor = primaryColor;
    }
    msg.textContent = text;
    messagesArea.appendChild(msg);
    messagesArea.scrollTop = messagesArea.scrollHeight;
    return msg;
  }

  var sending = false;

  function sendMessage() {
    var text = inputField.value.trim();
    if (!text || sending) return;

    addMessage(text, "user");
    inputField.value = "";
    sendBtn.disabled = true;
    sending = true;

    // Typing indicator
    var typing = document.createElement("div");
    typing.className = "exo-widget-typing";
    typing.textContent = "Typing...";
    messagesArea.appendChild(typing);
    messagesArea.scrollTop = messagesArea.scrollHeight;

    var url = baseUrl + "/api/deployed/" + deploymentId + "/run";
    var headers = { "Content-Type": "application/json" };
    if (apiKey) headers["Authorization"] = "Bearer " + apiKey;

    fetch(url, {
      method: "POST",
      headers: headers,
      body: JSON.stringify({ input: text, stream: false }),
    })
      .then(function (resp) {
        if (!resp.ok) throw new Error("Request failed (" + resp.status + ")");
        return resp.json();
      })
      .then(function (data) {
        if (typing.parentNode) typing.remove();
        addMessage(data.output || "No response.", "assistant");
      })
      .catch(function (err) {
        if (typing.parentNode) typing.remove();
        addMessage("Sorry, something went wrong. Please try again.", "assistant");
        console.error("[Exo Widget]", err);
      })
      .finally(function () {
        sending = false;
      });
  }
})();
