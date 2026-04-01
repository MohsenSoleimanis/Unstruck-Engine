import { useEffect, useState } from "react";
import { Network, RefreshCw } from "lucide-react";
import { getKGGraph } from "@/lib/api";
import type { KGGraph } from "@/lib/types";

export function KGViewer() {
  const [graph, setGraph] = useState<KGGraph | null>(null);
  const [loading, setLoading] = useState(false);

  const loadGraph = async () => {
    setLoading(true);
    try {
      const data = await getKGGraph(200);
      setGraph(data);
    } catch {
      // Graph may be empty
    }
    setLoading(false);
  };

  useEffect(() => { loadGraph(); }, []);

  if (!graph || (graph.nodes.length === 0 && graph.edges.length === 0)) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-sm text-muted-foreground">
        <Network size={32} className="mb-2 opacity-50" />
        <p>No knowledge graph yet</p>
        <p className="text-xs">Process a document to build the graph</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 p-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="text-sm">
          <span className="font-medium">{graph.stats.nodes}</span>{" "}
          <span className="text-muted-foreground">nodes</span>
          <span className="mx-2 text-muted-foreground">/</span>
          <span className="font-medium">{graph.stats.edges}</span>{" "}
          <span className="text-muted-foreground">edges</span>
        </div>
        <button onClick={loadGraph} disabled={loading} className="rounded-lg p-1.5 text-muted-foreground hover:bg-accent">
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {/* Node list (fallback until react-force-graph is installed) */}
      <div className="max-h-[400px] overflow-y-auto space-y-1">
        {graph.nodes.map((node) => (
          <div key={node.id} className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-accent/50">
            <span className={`h-2.5 w-2.5 rounded-full ${typeColor(node.entity_type)}`} />
            <span className="flex-1 truncate font-medium">{node.name ?? node.id}</span>
            <span className="text-xs text-muted-foreground">{node.entity_type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function typeColor(type?: string): string {
  const colors: Record<string, string> = {
    Person: "bg-blue-400",
    Organization: "bg-green-400",
    Location: "bg-yellow-400",
    Concept: "bg-purple-400",
    Metric: "bg-orange-400",
    Drug: "bg-red-400",
  };
  return colors[type ?? ""] ?? "bg-muted-foreground";
}
