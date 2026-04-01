import { useEffect } from "react";
import { Plus, Trash2, MessageSquare, Search, Settings } from "lucide-react";
import { useConversationStore } from "@/stores/conversation-store";
import { useUIStore } from "@/stores/ui-store";
import { cn, timeAgo, truncate } from "@/lib/utils";

export function Sidebar() {
  const { conversations, activeId, loading, loadConversations, createConversation, selectConversation, deleteConversation } = useConversationStore();
  const { sidebarOpen } = useUIStore();

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  if (!sidebarOpen) return null;

  return (
    <aside className="flex h-full w-[280px] flex-col border-r border-border bg-card">
      {/* Header */}
      <div className="flex items-center justify-between p-4">
        <h1 className="text-lg font-semibold">MAS</h1>
        <button
          onClick={() => createConversation()}
          className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 transition-opacity"
        >
          <Plus size={16} />
          New Chat
        </button>
      </div>

      {/* Search */}
      <div className="px-3 pb-2">
        <div className="flex items-center gap-2 rounded-lg bg-secondary px-3 py-2 text-sm text-muted-foreground">
          <Search size={14} />
          <span>Search conversations...</span>
        </div>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto px-2">
        {loading && conversations.length === 0 ? (
          <div className="space-y-2 p-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 animate-pulse rounded-lg bg-secondary" />
            ))}
          </div>
        ) : conversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center text-sm text-muted-foreground">
            <MessageSquare size={32} className="mb-2 opacity-50" />
            <p>No conversations yet</p>
            <p className="text-xs">Start a new chat to begin</p>
          </div>
        ) : (
          <div className="space-y-0.5">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                onClick={() => selectConversation(conv.id)}
                className={cn(
                  "group flex items-center justify-between rounded-lg px-3 py-2.5 text-sm cursor-pointer transition-colors",
                  activeId === conv.id
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                )}
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium">{truncate(conv.title, 30)}</p>
                  <p className="text-xs opacity-60">{timeAgo(conv.updated_at)}</p>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id); }}
                  className="ml-2 hidden rounded p-1 hover:bg-destructive/20 group-hover:block"
                >
                  <Trash2 size={14} className="text-destructive" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border p-3">
        <button className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted-foreground hover:bg-accent/50 transition-colors">
          <Settings size={16} />
          Settings
        </button>
      </div>
    </aside>
  );
}
