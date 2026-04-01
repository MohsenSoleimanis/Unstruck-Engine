/** Conversation state — CRUD + active conversation tracking. */

import { create } from "zustand";
import * as api from "@/lib/api";
import type { Conversation, Message } from "@/lib/types";

interface ConversationState {
  conversations: Conversation[];
  activeId: string | null;
  messages: Message[];
  loading: boolean;

  loadConversations: () => Promise<void>;
  createConversation: () => Promise<string>;
  selectConversation: (id: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
  addMessage: (role: string, content: string, metadata?: Record<string, unknown>) => Promise<void>;
  appendAssistantMessage: (content: string, metadata?: Record<string, unknown>) => void;
  updateLastAssistantMessage: (content: string) => void;
}

export const useConversationStore = create<ConversationState>((set, get) => ({
  conversations: [],
  activeId: null,
  messages: [],
  loading: false,

  loadConversations: async () => {
    set({ loading: true });
    try {
      const conversations = await api.listConversations();
      set({ conversations, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  createConversation: async () => {
    const conv = await api.createConversation();
    set((s) => ({
      conversations: [conv, ...s.conversations],
      activeId: conv.id,
      messages: [],
    }));
    return conv.id;
  },

  selectConversation: async (id) => {
    set({ loading: true, activeId: id });
    try {
      const data = await api.getConversation(id);
      set({ messages: data.messages ?? [], loading: false });
    } catch {
      set({ messages: [], loading: false });
    }
  },

  deleteConversation: async (id) => {
    await api.deleteConversation(id);
    const state = get();
    const filtered = state.conversations.filter((c) => c.id !== id);
    set({
      conversations: filtered,
      activeId: state.activeId === id ? null : state.activeId,
      messages: state.activeId === id ? [] : state.messages,
    });
  },

  addMessage: async (role, content, metadata) => {
    const { activeId } = get();
    if (!activeId) return;

    const msg = await api.addMessage(activeId, role, content, metadata);
    set((s) => ({ messages: [...s.messages, msg] }));
  },

  appendAssistantMessage: (content, metadata) => {
    const msg: Message = {
      id: crypto.randomUUID().slice(0, 8),
      role: "assistant",
      content,
      timestamp: new Date().toISOString(),
      ...metadata,
    };
    set((s) => ({ messages: [...s.messages, msg] }));
  },

  updateLastAssistantMessage: (content) => {
    set((s) => {
      const msgs = [...s.messages];
      const last = msgs[msgs.length - 1];
      if (last?.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, content };
      }
      return { messages: msgs };
    });
  },
}));
