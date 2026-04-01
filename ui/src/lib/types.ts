/** Shared TypeScript types — mirrors backend Pydantic models. */

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count?: number;
  messages?: Message[];
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  cost_usd?: number;
  duration_ms?: number;
  agent_activity?: AgentEvent[];
  sources?: Source[];
}

export interface AgentEvent {
  task_id: string;
  agent_type: string;
  status: "pending" | "running" | "success" | "partial" | "failed";
  instruction?: string;
  duration_ms?: number;
  tokens?: { input_tokens: number; output_tokens: number };
  cost_usd?: number;
}

export interface Source {
  id: string;
  text: string;
  metadata: Record<string, unknown>;
  distance?: number;
  rerank_score?: number;
}

export interface KGNode {
  id: string;
  entity_type?: string;
  name?: string;
  [key: string]: unknown;
}

export interface KGEdge {
  source: string;
  target: string;
  relation_type?: string;
  [key: string]: unknown;
}

export interface KGGraph {
  nodes: KGNode[];
  edges: KGEdge[];
  stats: { nodes: number; edges: number };
}

export interface CostSummary {
  session: {
    total_tokens: number;
    total_cost_usd: number;
    num_calls: number;
  };
  by_agent: Record<string, { tokens: number; cost: number }>;
}

export interface AgentInfo {
  agent_type: string;
  description: string;
  version: string;
}

export type StreamEventType = "phase" | "plan" | "task_complete" | "done" | "error";

export interface StreamEvent {
  event: StreamEventType;
  data: Record<string, unknown>;
}

export interface UploadedFile {
  name: string;
  size_bytes: number;
  modified?: number;
  extension: string;
}
