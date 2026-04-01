/** Stream state — manages SSE connection and streaming status. */

import { create } from "zustand";
import type { CostSummary } from "@/lib/types";

interface StreamState {
  isStreaming: boolean;
  abortFn: (() => void) | null;
  lastCost: CostSummary | null;
  error: string | null;

  setStreaming: (streaming: boolean, abortFn?: () => void) => void;
  setCost: (cost: CostSummary) => void;
  setError: (error: string | null) => void;
  stop: () => void;
}

export const useStreamStore = create<StreamState>((set, get) => ({
  isStreaming: false,
  abortFn: null,
  lastCost: null,
  error: null,

  setStreaming: (streaming, abortFn) => set({ isStreaming: streaming, abortFn: abortFn ?? null, error: null }),
  setCost: (cost) => set({ lastCost: cost }),
  setError: (error) => set({ error, isStreaming: false }),

  stop: () => {
    const { abortFn } = get();
    if (abortFn) abortFn();
    set({ isStreaming: false, abortFn: null });
  },
}));
