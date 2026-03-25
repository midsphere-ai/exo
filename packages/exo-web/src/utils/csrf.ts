/**
 * CSRF token management for Exo Web.
 *
 * Usage:
 *   import { getCsrfToken, clearCsrfToken } from "../utils/csrf";
 *
 *   const token = await getCsrfToken();
 *   fetch("/api/v1/...", {
 *     method: "POST",
 *     headers: { "Content-Type": "application/json", "X-CSRF-Token": token },
 *     body: JSON.stringify(data),
 *   });
 */

let _cachedToken: string | null = null;

/**
 * Fetch and cache the CSRF token for the current session.
 * Re-fetches if no cached value exists.
 */
export async function getCsrfToken(): Promise<string> {
  if (_cachedToken) return _cachedToken;

  const res = await fetch("/api/v1/auth/csrf", { credentials: "same-origin" });
  if (!res.ok) {
    throw new Error("Failed to fetch CSRF token");
  }
  const data = await res.json();
  _cachedToken = data.token as string;
  return _cachedToken;
}

/**
 * Clear the cached CSRF token. Call this on logout.
 */
export function clearCsrfToken(): void {
  _cachedToken = null;
}
