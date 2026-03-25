/**
 * Client-side WebSocket singleton for Exo Web.
 *
 * Provides a single multiplexed connection that mirrors the server-side
 * {@link WebSocketManager} (US-156).  All frontend features share one
 * connection with channel-based routing and automatic reconnection.
 *
 * Message envelope format (matches server):
 * ```json
 * { "channel": "chat" | "execution" | "logs" | "sandbox" | "notifications" | "system",
 *   "type": "<message-type>",
 *   "payload": { ... } }
 * ```
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ConnectionState =
  | "connecting"
  | "connected"
  | "disconnected"
  | "reconnecting";

export type Channel =
  | "chat"
  | "execution"
  | "logs"
  | "sandbox"
  | "notifications"
  | "system";

export interface WsEnvelope {
  channel: Channel;
  type: string;
  payload: Record<string, unknown>;
}

export type ChannelCallback = (message: WsEnvelope) => void;
export type ConnectionChangeCallback = (state: ConnectionState) => void;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BACKOFF_INITIAL_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;
const BACKOFF_FACTOR = 2;

// ---------------------------------------------------------------------------
// ExoSocket
// ---------------------------------------------------------------------------

export class ExoSocket {
  // -- singleton -----------------------------------------------------------

  private static _instance: ExoSocket | null = null;

  static getInstance(): ExoSocket {
    if (!ExoSocket._instance) {
      ExoSocket._instance = new ExoSocket();
    }
    return ExoSocket._instance;
  }

  /** Reset singleton — useful for tests. */
  static resetInstance(): void {
    if (ExoSocket._instance) {
      ExoSocket._instance.close();
      ExoSocket._instance = null;
    }
  }

  // -- state ---------------------------------------------------------------

  private _ws: WebSocket | null = null;
  private _state: ConnectionState = "disconnected";

  /** channel → set of callbacks */
  private _subscriptions = new Map<Channel, Set<ChannelCallback>>();
  /** connection-state listeners */
  private _connectionListeners = new Set<ConnectionChangeCallback>();

  /** Messages queued while disconnected, flushed on reconnect. */
  private _queue: string[] = [];

  /** Current backoff delay (ms). */
  private _backoff = BACKOFF_INITIAL_MS;
  /** Scheduled reconnect timer. */
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  /** Whether close() was called intentionally. */
  private _intentionalClose = false;

  // -----------------------------------------------------------------------

  private constructor() {
    // private — use getInstance()
  }

  // -- public API ----------------------------------------------------------

  get state(): ConnectionState {
    return this._state;
  }

  /**
   * Subscribe to a channel. Automatically connects if not already connected.
   * Returns an unsubscribe function.
   */
  subscribe(channel: Channel, callback: ChannelCallback): () => void {
    let set = this._subscriptions.get(channel);
    if (!set) {
      set = new Set();
      this._subscriptions.set(channel, set);
    }
    set.add(callback);

    // Auto-connect on first subscription.
    if (this._state === "disconnected" && !this._intentionalClose) {
      this._connect();
    }

    // Tell the server about the subscription once connected.
    this._sendRaw(JSON.stringify({ type: "subscribe", channel }));

    return () => this.unsubscribe(channel, callback);
  }

  /** Remove a specific callback from a channel. */
  unsubscribe(channel: Channel, callback: ChannelCallback): void {
    const set = this._subscriptions.get(channel);
    if (!set) return;
    set.delete(callback);
    if (set.size === 0) {
      this._subscriptions.delete(channel);
      this._sendRaw(JSON.stringify({ type: "unsubscribe", channel }));
    }
  }

  /** Register a listener for connection-state changes. Returns unsubscribe fn. */
  onConnectionChange(callback: ConnectionChangeCallback): () => void {
    this._connectionListeners.add(callback);
    // Immediately notify with current state.
    callback(this._state);
    return () => {
      this._connectionListeners.delete(callback);
    };
  }

  /** Send a message through the WebSocket (queued if not connected). */
  send(envelope: WsEnvelope): void {
    this._sendRaw(JSON.stringify(envelope));
  }

  /** Intentionally close the connection (no reconnect). */
  close(): void {
    this._intentionalClose = true;
    this._clearReconnect();
    if (this._ws) {
      this._ws.close(1000, "client close");
      this._ws = null;
    }
    this._setState("disconnected");
  }

  // -- internal ------------------------------------------------------------

  private _setState(next: ConnectionState): void {
    if (this._state === next) return;
    this._state = next;
    for (const cb of this._connectionListeners) {
      try {
        cb(next);
      } catch {
        // Listener errors shouldn't crash the socket.
      }
    }
  }

  private _connect(): void {
    if (this._ws) return; // already connecting / connected
    this._intentionalClose = false;

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/api/v1/ws`;

    this._setState("connecting");
    const ws = new WebSocket(url);

    ws.addEventListener("open", () => {
      this._ws = ws;
      this._backoff = BACKOFF_INITIAL_MS;
      this._setState("connected");

      // Re-subscribe to all channels the client cares about.
      for (const channel of this._subscriptions.keys()) {
        ws.send(JSON.stringify({ type: "subscribe", channel }));
      }

      // Flush queued messages.
      for (const msg of this._queue) {
        ws.send(msg);
      }
      this._queue = [];
    });

    ws.addEventListener("message", (event) => {
      this._handleMessage(event.data as string);
    });

    ws.addEventListener("close", () => {
      this._ws = null;
      if (!this._intentionalClose) {
        this._scheduleReconnect();
      } else {
        this._setState("disconnected");
      }
    });

    ws.addEventListener("error", () => {
      // The close event always fires after error, so reconnection is
      // handled there. Just clean up the reference.
      ws.close();
    });
  }

  private _handleMessage(raw: string): void {
    let envelope: WsEnvelope;
    try {
      envelope = JSON.parse(raw) as WsEnvelope;
    } catch {
      return; // Ignore malformed messages.
    }

    // Respond to server heartbeat pings.
    if (envelope.channel === "system" && envelope.type === "ping") {
      this._sendRaw(JSON.stringify({ type: "pong" }));
      return;
    }

    // Dispatch to subscribers.
    const set = this._subscriptions.get(envelope.channel);
    if (set) {
      for (const cb of set) {
        try {
          cb(envelope);
        } catch {
          // Callback errors shouldn't crash the socket.
        }
      }
    }
  }

  /**
   * Send raw string — writes to socket if open, otherwise queues.
   */
  private _sendRaw(data: string): void {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(data);
    } else {
      this._queue.push(data);
    }
  }

  // -- reconnection --------------------------------------------------------

  private _scheduleReconnect(): void {
    this._clearReconnect();
    this._setState("reconnecting");
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._connect();
    }, this._backoff);
    // Exponential backoff with cap.
    this._backoff = Math.min(this._backoff * BACKOFF_FACTOR, BACKOFF_MAX_MS);
  }

  private _clearReconnect(): void {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }
}

// Convenience singleton accessor.
export const exoSocket = ExoSocket.getInstance();
