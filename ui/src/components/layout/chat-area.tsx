import { useCallback } from "react";
import { PanelRightOpen, PanelLeftOpen } from "lucide-react";
import { MessageList } from "@/components/chat/message-list";
import { ChatInput } from "@/components/chat/chat-input";
import { useConversationStore } from "@/stores/conversation-store";
import { useAgentStore } from "@/stores/agent-store";
import { useStreamStore } from "@/stores/stream-store";
import { useUIStore } from "@/stores/ui-store";
import { uploadFile } from "@/lib/api";
import { startQueryStream } from "@/lib/sse";
import type { CostSummary, StreamEvent } from "@/lib/types";

export function ChatArea() {
  const { messages, activeId, createConversation, addMessage, appendAssistantMessage } = useConversationStore();
  const { startRun, endRun, setPhase, addEvent } = useAgentStore();
  const { setStreaming, setCost, setError } = useStreamStore();
  const { toggleSidebar, toggleRightPanel, sidebarOpen, rightPanelOpen } = useUIStore();

  const handleSend = useCallback(async (text: string, files?: File[]) => {
    let convId = activeId;
    if (!convId) {
      convId = await createConversation();
    }

    // Upload files first, collect paths
    const context: Record<string, unknown> = {};
    if (files && files.length > 0) {
      const uploaded = await Promise.all(files.map((f) => uploadFile(f)));
      // Single file → file_path, multiple → file_paths
      if (uploaded.length === 1) {
        context.file_path = `data/uploads/${uploaded[0].name}`;
      } else {
        context.file_paths = uploaded.map((u) => `data/uploads/${u.name}`);
      }
    }

    await addMessage("user", text);
    startRun();

    const abort = startQueryStream(
      text,
      context,
      3,
      (event: StreamEvent) => {
        switch (event.event) {
          case "phase":
            setPhase(event.data.phase as string);
            break;
          case "plan":
            for (const t of (event.data.tasks as Array<{ id: string; agent_type: string; instruction: string }>)) {
              addEvent({ task_id: t.id, agent_type: t.agent_type, status: "pending", instruction: t.instruction });
            }
            break;
          case "task_complete":
            addEvent({
              task_id: event.data.task_id as string,
              agent_type: event.data.agent_type as string,
              status: (event.data.status as string) === "success" ? "success" : "failed",
              duration_ms: event.data.duration_ms as number,
              cost_usd: event.data.cost_usd as number,
            });
            break;
          case "done": {
            const output = event.data.output as Record<string, unknown>;
            // Extract actual answer from analysis, rag_response, or synthesis
            const analysis = output.analysis as Record<string, unknown> | null;
            const ragResponse = output.rag_response as string;
            const synthesis = output.synthesis as Record<string, unknown>;

            let answer = "";
            if (analysis?.answer) {
              const raw = analysis.answer;
              answer = typeof raw === "string" ? raw : JSON.stringify(raw, null, 2);
            } else if (ragResponse) {
              answer = ragResponse;
            } else if (synthesis && Object.keys(synthesis).length > 0) {
              answer = typeof synthesis.answer === "string" ? synthesis.answer : JSON.stringify(synthesis, null, 2);
            } else {
              answer = JSON.stringify(output.results ?? output, null, 2);
            }

            appendAssistantMessage(answer, {
              cost_usd: (event.data.cost as CostSummary)?.session?.total_cost_usd,
              duration_ms: event.data.duration_ms as number,
            });
            setCost(event.data.cost as CostSummary);
            endRun();
            break;
          }
          case "error":
            setError(event.data.message as string);
            appendAssistantMessage(`Error: ${event.data.message as string}`);
            endRun();
            break;
        }
      },
      (err) => { setError(err.message); endRun(); },
      () => { setStreaming(false); },
      convId,  // pass conversation_id for session awareness
    );

    setStreaming(true, abort);
  }, [activeId, createConversation, addMessage, startRun, setPhase, addEvent, appendAssistantMessage, setCost, endRun, setError, setStreaming]);

  return (
    <main className="flex flex-1 flex-col overflow-hidden">
      <header className="flex h-12 items-center justify-between border-b border-border px-4">
        <button onClick={toggleSidebar} className="rounded-lg p-1.5 text-muted-foreground hover:bg-accent">
          <PanelLeftOpen size={18} className={sidebarOpen ? "opacity-50" : ""} />
        </button>
        <span className="text-sm font-medium text-muted-foreground">Multi-Agent Pipeline</span>
        <button onClick={toggleRightPanel} className="rounded-lg p-1.5 text-muted-foreground hover:bg-accent">
          <PanelRightOpen size={18} className={rightPanelOpen ? "opacity-50" : ""} />
        </button>
      </header>

      <MessageList messages={messages} />
      <ChatInput onSend={handleSend} />
    </main>
  );
}
