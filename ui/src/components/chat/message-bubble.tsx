import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, User, Copy, Check } from "lucide-react";
import { useState } from "react";
import type { Message } from "@/lib/types";
import { cn, formatCost, formatDuration } from "@/lib/utils";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={cn("group flex gap-3 px-4 py-4", isUser ? "flex-row-reverse" : "")}>
      {/* Avatar */}
      <div className={cn(
        "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
        isUser ? "bg-primary text-primary-foreground" : "bg-secondary text-foreground",
      )}>
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Content */}
      <div className={cn("flex max-w-[75%] flex-col gap-1", isUser ? "items-end" : "items-start")}>
        <div className={cn(
          "rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground rounded-tr-md"
            : "bg-card border border-border rounded-tl-md",
        )}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Metadata + actions */}
        <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">
          {message.cost_usd != null && message.cost_usd > 0 && (
            <span>{formatCost(message.cost_usd)}</span>
          )}
          {message.duration_ms != null && message.duration_ms > 0 && (
            <span>{formatDuration(message.duration_ms)}</span>
          )}
          {!isUser && (
            <button onClick={handleCopy} className="ml-1 rounded p-0.5 hover:bg-accent">
              {copied ? <Check size={12} /> : <Copy size={12} />}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
