import { Activity, Network, BookOpen, DollarSign } from "lucide-react";
import { AgentActivity } from "@/components/agents/agent-activity";
import { CostSummary } from "@/components/costs/cost-summary";
import { KGViewer } from "@/components/kg/kg-viewer";
import { useUIStore, type RightPanelTab } from "@/stores/ui-store";
import { cn } from "@/lib/utils";

const TABS: { id: RightPanelTab; label: string; icon: typeof Activity }[] = [
  { id: "activity", label: "Agents", icon: Activity },
  { id: "kg", label: "Graph", icon: Network },
  { id: "sources", label: "Sources", icon: BookOpen },
  { id: "costs", label: "Costs", icon: DollarSign },
];

export function RightPanel() {
  const { rightPanelOpen, rightPanelTab, setRightPanelTab } = useUIStore();

  if (!rightPanelOpen) return null;

  return (
    <aside className="flex h-full w-[360px] flex-col border-l border-border bg-card">
      {/* Tab bar */}
      <div className="flex border-b border-border">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setRightPanelTab(id)}
            className={cn(
              "flex flex-1 items-center justify-center gap-1.5 py-2.5 text-xs font-medium transition-colors",
              rightPanelTab === id
                ? "border-b-2 border-primary text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {rightPanelTab === "activity" && <AgentActivity />}
        {rightPanelTab === "kg" && <KGViewer />}
        {rightPanelTab === "sources" && <SourcesPlaceholder />}
        {rightPanelTab === "costs" && <CostSummary />}
      </div>
    </aside>
  );
}

function SourcesPlaceholder() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-sm text-muted-foreground">
      <BookOpen size={32} className="mb-2 opacity-50" />
      <p>No sources yet</p>
      <p className="text-xs">Sources will appear after a query</p>
    </div>
  );
}
