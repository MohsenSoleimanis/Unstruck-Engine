/** Shared TypeScript types — mirrors v2 backend models. */

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
}

export interface AgentEvent {
  task_id: string;
  agent_type: string;
  status: "pending" | "running" | "success" | "partial" | "failed";
  instruction?: string;
  duration_ms?: number;
  cost_usd?: number;
}

export interface CostSummary {
  total_cost_usd: number;
  total_tokens: number;
  total_calls: number;
  ceiling_usd: number;
  by_agent: Record<string, { tokens: number; cost: number }>;
}

export interface AgentInfo {
  agent_type: string;
  description: string;
  version: string;
  model_tier: string;
  allowed_tools: string[];
  trust_level: string;
}

export interface ToolInfo {
  name: string;
  description: string;
  permission_level: string;
}

export type StreamEventType =
  | "phase"
  | "plan"
  | "evaluation"
  | "decision"
  | "done"
  | "error";

export interface StreamEvent {
  event: StreamEventType;
  data: Record<string, unknown>;
}

export interface UploadedFile {
  name: string;
  size_bytes: number;
  extension: string;
}
