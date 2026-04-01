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
  conversationId?: string,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          context,
          max_iterations: maxIterations,
          conversation_id: conversationId,
        }),
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

        // SSE events are separated by double newlines
        // Each event block: "event: <type>\ndata: <json>\n\n"
        const blocks = buffer.split("\n\n");
        // Last element may be incomplete — keep it in buffer
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          if (!block.trim()) continue;

          let eventType = "";
          let eventData = "";

          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              eventData = line.slice(6);
            }
          }

          if (eventType && eventData) {
            try {
              const data = JSON.parse(eventData);
              onEvent({ event: eventType as StreamEvent["event"], data });
            } catch {
              console.warn("SSE parse error:", eventType, eventData.slice(0, 100));
            }
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
