"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChatConfig, Message, Source, DEFAULT_CONFIG } from "@/lib/types";
import { sendMessage } from "@/lib/api";
import MessageBubble from "./MessageBubble";

interface ChatPanelProps {
  config: ChatConfig;
}

export default function ChatPanel({ config }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSubmit = async () => {
    const question = input.trim();
    if (!question || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
    };

    const assistantId = `assistant-${Date.now()}`;
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      sources: [],
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    await sendMessage(question, config, {
      onToken: (token) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: m.content + token } : m
          )
        );
      },
      onSources: (sources: Source[]) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, sources } : m))
        );
      },
      onDone: () => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m
          )
        );
        setIsStreaming(false);
      },
      onError: (err) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `Error: ${err}`, isStreaming: false }
              : m
          )
        );
        setIsStreaming(false);
      },
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-gray-50">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <p className="text-lg font-medium">MA Tenant Law Assistant</p>
              <p className="text-sm mt-1">
                Ask a question about Massachusetts tenant rights
              </p>
            </div>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 bg-white px-6 py-4">
        <div className="flex gap-3 items-end max-w-4xl mx-auto">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about Massachusetts tenant law..."
            disabled={isStreaming}
            rows={1}
            className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none disabled:opacity-50 disabled:bg-gray-50"
          />
          <button
            onClick={handleSubmit}
            disabled={isStreaming || !input.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white px-5 py-3 rounded-xl text-sm font-medium transition-colors shrink-0"
          >
            {isStreaming ? "..." : "Send"}
          </button>
        </div>
        <p className="text-xs text-gray-400 text-center mt-2">
          Legal information only — not legal advice. Consult an attorney for specific situations.
        </p>
      </div>
    </div>
  );
}
