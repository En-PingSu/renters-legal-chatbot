"use client";

import { useState } from "react";
import { ChatConfig, DEFAULT_CONFIG } from "@/lib/types";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";

export default function Home() {
  const [config, setConfig] = useState<ChatConfig>(DEFAULT_CONFIG);
  const [chatKey, setChatKey] = useState(0);

  const handleNewChat = () => {
    setChatKey((k) => k + 1);
  };

  return (
    <div className="flex h-screen">
      <Sidebar
        config={config}
        onConfigChange={setConfig}
        onNewChat={handleNewChat}
      />
      <ChatPanel key={chatKey} config={config} />
    </div>
  );
}
