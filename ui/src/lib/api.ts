/** API client — typed fetch wrappers for all backend endpoints. */

import type { AgentInfo, Conversation, CostSummary, KGGraph, Message, UploadedFile } from "./types";

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

// --- Query ---

export async function querySync(query: string, context: Record<string, unknown> = {}, maxIter = 3) {
  return fetchJSON<{ status: string; results: Record<string, unknown>; cost: CostSummary; duration_ms: number }>(
    "/query",
    { method: "POST", body: JSON.stringify({ query, context, max_iterations: maxIter }) },
  );
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

export async function addMessage(conversationId: string, role: string, content: string, metadata?: Record<string, unknown>): Promise<Message> {
  return fetchJSON(`/conversations/${conversationId}/messages`, {
    method: "POST",
    body: JSON.stringify({ role, content, metadata }),
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

// --- Knowledge Graph ---

export async function getKGGraph(limit = 500): Promise<KGGraph> {
  return fetchJSON(`/kg/graph?limit=${limit}`);
}

export async function getKGSubgraph(entityId: string, depth = 2): Promise<KGGraph> {
  return fetchJSON(`/kg/subgraph/${entityId}?depth=${depth}`);
}

// --- System ---

export async function getAgents(): Promise<AgentInfo[]> {
  return fetchJSON("/agents");
}

export async function getHealth() {
  return fetchJSON<Record<string, unknown>>("/health");
}

export async function getMetrics(): Promise<{ costs: CostSummary }> {
  return fetchJSON("/metrics");
}
