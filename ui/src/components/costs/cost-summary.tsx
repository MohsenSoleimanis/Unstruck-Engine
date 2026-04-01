import { DollarSign, Cpu, Zap } from "lucide-react";
import { useStreamStore } from "@/stores/stream-store";
import { formatCost, formatTokens } from "@/lib/utils";

export function CostSummary() {
  const { lastCost } = useStreamStore();

  if (!lastCost) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-sm text-muted-foreground">
        <DollarSign size={32} className="mb-2 opacity-50" />
        <p>No cost data yet</p>
        <p className="text-xs">Run a query to track costs</p>
      </div>
    );
  }

  const { session, by_agent } = lastCost;
  const agents = Object.entries(by_agent).sort((a, b) => b[1].cost - a[1].cost);

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Totals */}
      <div className="grid grid-cols-3 gap-2">
        <StatCard icon={DollarSign} label="Total Cost" value={formatCost(session.total_cost_usd)} />
        <StatCard icon={Cpu} label="Tokens" value={formatTokens(session.total_tokens)} />
        <StatCard icon={Zap} label="LLM Calls" value={String(session.num_calls)} />
      </div>

      {/* Per-agent breakdown */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold uppercase text-muted-foreground">By Agent</h3>
        {agents.map(([agentType, data]) => {
          const pct = session.total_cost_usd > 0 ? (data.cost / session.total_cost_usd) * 100 : 0;
          return (
            <div key={agentType} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">{agentType}</span>
                <span className="text-muted-foreground">{formatCost(data.cost)}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
                <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${pct}%` }} />
              </div>
              <p className="text-xs text-muted-foreground">{formatTokens(data.tokens)} tokens</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value }: { icon: typeof DollarSign; label: string; value: string }) {
  return (
    <div className="flex flex-col items-center gap-1 rounded-lg bg-secondary p-3 text-center">
      <Icon size={16} className="text-muted-foreground" />
      <span className="text-lg font-bold">{value}</span>
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}
