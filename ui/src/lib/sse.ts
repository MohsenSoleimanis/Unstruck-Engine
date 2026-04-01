/** SSE client — connects to /api/query/stream for real-time pipeline events. */

import type { StreamEvent } from "./types";

export type SSECallback = (event: StreamEvent) => void;

/**
 * Start an SSE stream for a query.
 *
 * Uses fetch + ReadableStream (not EventSource) because we need POST.
 * Returns an abort function to cancel the stream.
 */
export function startQueryStream(
  query: string,
  context: Record<string, unknown>,
  maxIterations: number,
  onEvent: SSECallback,
  onError: (err: Error) => void,
  onDone: () => void,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, context, max_iterations: maxIterations }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        throw new Error(`Stream failed: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ") && currentEvent) {
            try {
              const data = JSON.parse(line.slice(6));
              onEvent({ event: currentEvent as StreamEvent["event"], data });
            } catch {
              // Skip malformed events
            }
            currentEvent = "";
          }
        }
      }

      onDone();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      onError(err instanceof Error ? err : new Error(String(err)));
    }
  })();

  return () => controller.abort();
}
