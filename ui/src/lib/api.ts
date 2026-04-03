/** API client — typed fetch wrappers for v2 backend. */

import type { AgentInfo, Conversation, CostSummary, Message, ToolInfo, UploadedFile } from "./types";

const BASE = "/api";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

// --- Conversations ---

export async function listConversations(): Promise<Conversation[]> {
  return fetchJSON("/conversations");
}

export async function getConversation(id: string): Promise<Conversation> {
  return fetchJSON(`/conversations/${id}`);
}

export async function createConversation(title = "New Chat"): Promise<Conversation> {
  return fetchJSON("/conversations", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

export async function addMessage(conversationId: string, role: string, content: string): Promise<Message> {
  return fetchJSON(`/conversations/${conversationId}/messages`, {
    method: "POST",
    body: JSON.stringify({ role, content }),
  });
}

export async function deleteConversation(id: string): Promise<void> {
  await fetchJSON(`/conversations/${id}`, { method: "DELETE" });
}

// --- Files ---

export async function listFiles(): Promise<UploadedFile[]> {
  return fetchJSON("/files");
}

export async function uploadFile(file: File): Promise<{ name: string; size_bytes: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(BASE + "/files/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function deleteFile(name: string): Promise<void> {
  await fetchJSON(`/files/${name}`, { method: "DELETE" });
}

// --- System ---

export async function getAgents(): Promise<AgentInfo[]> {
  return fetchJSON("/agents");
}

export async function getTools(): Promise<ToolInfo[]> {
  return fetchJSON("/tools");
}

export async function getHealth() {
  return fetchJSON<Record<string, unknown>>("/health");
}

export async function getMetrics(): Promise<{ costs: CostSummary }> {
  return fetchJSON("/metrics");
}
