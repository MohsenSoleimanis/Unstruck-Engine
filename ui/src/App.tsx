import { Sidebar } from "@/components/layout/sidebar";
import { ChatArea } from "@/components/layout/chat-area";
import { RightPanel } from "@/components/layout/right-panel";

export function App() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar />
      <ChatArea />
      <RightPanel />
    </div>
  );
}
