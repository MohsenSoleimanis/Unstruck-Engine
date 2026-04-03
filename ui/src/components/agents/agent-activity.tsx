import { CheckCircle2, Circle, Loader2, XCircle, AlertCircle } from "lucide-react";
import { useAgentStore } from "@/stores/agent-store";
import { cn, formatDuration, formatCost } from "@/lib/utils";
import type { AgentEvent } from "@/lib/types";

const STATUS_ICONS = {
  pending: Circle,
  running: Loader2,
  success: CheckCircle2,
  partial: AlertCircle,
  failed: XCircle,
} as const;

const STATUS_COLORS = {
  pending: "text-muted-foreground",
  running: "text-blue-400 animate-spin",
  success: "text-green-400",
  partial: "text-yellow-400",
  failed: "text-red-400",
} as const;

const PHASE_LABELS: Record<string, string> = {
  idle: "Idle",
  starting: "Starting...",
  understanding: "Understanding intent...",
  validating: "Validating input...",
  planning: "Creating task plan...",
  executing: "Executing agents...",
  evaluating: "Evaluating quality...",
  deciding: "Deciding next step...",
  replanning: "Replanning...",
  complete: "Complete",
};

export function AgentActivity() {
  const { events, currentPhase, isRunning } = useAgentStore();

  return (
    <div className="flex flex-col gap-3 p-4">
      {/* Phase indicator */}
      <div className="flex items-center gap-2 rounded-lg bg-secondary px-3 py-2">
        {isRunning && <Loader2 size={14} className="animate-spin text-blue-400" />}
        <span className="text-sm font-medium">{PHASE_LABELS[currentPhase] ?? currentPhase}</span>
      </div>

      {/* Agent events */}
      {events.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          Run a query to see agent activity
        </p>
      ) : (
        <div className="space-y-1">
          {events.map((event) => (
            <AgentEventRow key={event.task_id} event={event} />
          ))}
        </div>
      )}
    </div>
  );
}

function AgentEventRow({ event }: { event: AgentEvent }) {
  const Icon = STATUS_ICONS[event.status] ?? Circle;
  const color = STATUS_COLORS[event.status] ?? "text-muted-foreground";

  return (
    <div className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-accent/50 transition-colors">
      <Icon size={16} className={cn("shrink-0", color)} />
      <div className="min-w-0 flex-1">
        <p className="truncate font-medium">{event.agent_type}</p>
        {event.instruction && (
          <p className="truncate text-xs text-muted-foreground">{event.instruction}</p>
        )}
      </div>
      <div className="flex shrink-0 flex-col items-end text-xs text-muted-foreground">
        {event.duration_ms != null && <span>{formatDuration(event.duration_ms)}</span>}
        {event.cost_usd != null && event.cost_usd > 0 && <span>{formatCost(event.cost_usd)}</span>}
      </div>
    </div>
  );
}
