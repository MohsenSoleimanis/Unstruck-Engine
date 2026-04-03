import { Network } from "lucide-react";

export function KGViewer() {
  // KG visualization will connect to RAG-Anything's LightRAG graph
  // when the RAG service is fully integrated
  return (
    <div className="flex flex-col items-center justify-center py-12 text-sm text-muted-foreground">
      <Network size={32} className="mb-2 opacity-50" />
      <p>Knowledge Graph</p>
      <p className="text-xs mt-1">Process a document to build the graph</p>
      <p className="text-xs mt-1 opacity-60">Powered by RAG-Anything + LightRAG</p>
    </div>
  );
}
