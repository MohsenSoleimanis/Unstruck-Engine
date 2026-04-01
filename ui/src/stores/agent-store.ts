/** Agent activity state — tracks real-time agent events during pipeline runs. */

import { create } from "zustand";
import type { AgentEvent } from "@/lib/types";

interface AgentState {
  events: AgentEvent[];
  currentPhase: string;
  isRunning: boolean;

  setPhase: (phase: string) => void;
  addEvent: (event: AgentEvent) => void;
  startRun: () => void;
  endRun: () => void;
  reset: () => void;
}

export const useAgentStore = create<AgentState>((set) => ({
  events: [],
  currentPhase: "idle",
  isRunning: false,

  setPhase: (phase) => set({ currentPhase: phase }),

  addEvent: (event) => set((s) => {
    // Update existing event if same task_id, otherwise append
    const existing = s.events.findIndex((e) => e.task_id === event.task_id);
    if (existing >= 0) {
      const updated = [...s.events];
      updated[existing] = event;
      return { events: updated };
    }
    return { events: [...s.events, event] };
  }),

  startRun: () => set({ events: [], currentPhase: "planning", isRunning: true }),
  endRun: () => set({ isRunning: false, currentPhase: "complete" }),
  reset: () => set({ events: [], currentPhase: "idle", isRunning: false }),
}));
