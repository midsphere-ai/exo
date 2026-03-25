/**
 * Global API fetch wrapper with session expiry handling.
 *
 * Intercepts 401 responses and redirects to /login?expired=true.
 * Provides an AbortController so all pending requests are cancelled
 * on session expiry (prevents cascading error toasts).
 *
 * Usage (from inline scripts in PageLayout):
 *   window.__exoApi.fetch(url, init)   — same as fetch() but with 401 handling
 *   window.__exoApi.expire()           — manually trigger session expiry flow
 *   window.__exoApi.abortAll()         — cancel all in-flight requests
 */

let _expired = false;
const _controllers = new Set<AbortController>();

/** Cancel all in-flight requests tracked by apiFetch. */
export function abortAll(): void {
  for (const ctrl of _controllers) {
    ctrl.abort();
  }
  _controllers.clear();
}

/** Redirect to login with ?expired=true. Cancels all pending requests first. */
export function expireSession(): void {
  if (_expired) return; // Prevent multiple redirects.
  _expired = true;
  abortAll();
  window.location.replace("/login?expired=true");
}

/**
 * Fetch wrapper that intercepts 401 responses.
 * Automatically cancels in-flight requests when session expires.
 */
export async function apiFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  if (_expired) {
    return Promise.reject(new DOMException("Session expired", "AbortError"));
  }

  const controller = new AbortController();
  _controllers.add(controller);

  // Chain caller's signal if provided.
  const callerSignal = init?.signal;
  if (callerSignal?.aborted) {
    _controllers.delete(controller);
    return Promise.reject(callerSignal.reason);
  }

  const combinedInit: RequestInit = {
    ...init,
    signal: controller.signal,
  };

  // If caller provided their own signal, abort our controller when theirs fires.
  if (callerSignal) {
    const onCallerAbort = () => controller.abort(callerSignal.reason);
    callerSignal.addEventListener("abort", onCallerAbort, { once: true });
  }

  try {
    const res = await fetch(input, combinedInit);

    if (res.status === 401) {
      expireSession();
      return Promise.reject(new DOMException("Session expired", "AbortError"));
    }

    return res;
  } finally {
    _controllers.delete(controller);
  }
}
