import { useEffect, useRef } from "react";
import { MessageBubble } from "./message-bubble";
import { Zap, FileText, BarChart3, Image } from "lucide-react";
import type { Message } from "@/lib/types";
import { useStreamStore } from "@/stores/stream-store";

interface Props {
  messages: Message[];
}

export function MessageList({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const { isStreaming } = useStreamStore();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, isStreaming]);

  if (messages.length === 0) {
    return <WelcomeScreen />;
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl py-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isStreaming && (
          <div className="flex gap-3 px-4 py-4">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-secondary">
              <div className="flex gap-1">
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:0ms]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:150ms]" />
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function WelcomeScreen() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-4">
      <div className="text-center">
        <h2 className="text-3xl font-bold">Multi-Agent System</h2>
        <p className="mt-2 text-muted-foreground">
          Upload documents, ask questions, get AI-powered analysis
        </p>
      </div>
      <div className="grid max-w-lg grid-cols-2 gap-3">
        {[
          { icon: FileText, label: "Analyze a document", desc: "PDF, DOCX, or text" },
          { icon: BarChart3, label: "Extract data", desc: "Tables, charts, metrics" },
          { icon: Zap, label: "Research a topic", desc: "Deep multi-agent analysis" },
          { icon: Image, label: "Process images", desc: "Diagrams, charts, photos" },
        ].map(({ icon: Icon, label, desc }) => (
          <button
            key={label}
            className="flex flex-col gap-1 rounded-xl border border-border p-4 text-left text-sm hover:bg-accent/50 transition-colors"
          >
            <Icon size={20} className="text-muted-foreground" />
            <span className="font-medium">{label}</span>
            <span className="text-xs text-muted-foreground">{desc}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
