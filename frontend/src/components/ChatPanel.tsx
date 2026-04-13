"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChatConfig, Message, Source, DEFAULT_CONFIG } from "@/lib/types";
import { sendMessage } from "@/lib/api";
import MessageBubble from "./MessageBubble";

const EXAMPLE_QUESTIONS = [
  "Can my landlord keep my security deposit for normal wear and tear?",
  "What are my rights if my apartment has no heat in winter?",
  "Can I withhold rent if my landlord won't fix a health code violation?",
  "How much notice does my landlord need to give before entering my apartment?",
];

interface ChatPanelProps {
  config: ChatConfig;
}

export default function ChatPanel({ config }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [showDisclaimer, setShowDisclaimer] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  }, [input]);

  const submitQuestion = async (question: string) => {
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

  const handleSubmit = () => submitQuestion(input.trim());

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-gray-50">
      {/* Legal disclaimer banner */}
      {showDisclaimer && (
        <div className="bg-amber-50 border-b border-amber-200 px-6 py-3 flex items-center justify-between">
          <p className="text-xs text-amber-800">
            <span className="font-semibold">Important:</span> This tool provides
            legal <em>information</em> about Massachusetts tenant law, not legal{" "}
            <em>advice</em>. For your specific situation, consult a licensed
            attorney or contact your local legal aid office.
          </p>
          <button
            onClick={() => setShowDisclaimer(false)}
            className="text-amber-600 hover:text-amber-800 ml-4 shrink-0 text-sm font-medium"
            aria-label="Dismiss disclaimer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-lg">
              <p className="text-lg font-medium text-gray-700">
                MA Tenant Law Assistant
              </p>
              <p className="text-sm text-gray-400 mt-1 mb-6">
                Ask a question about Massachusetts tenant rights
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {EXAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => submitQuestion(q)}
                    className="text-left text-sm text-gray-600 bg-white border border-gray-200 rounded-xl px-4 py-3 hover:border-blue-300 hover:bg-blue-50 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
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
          Legal information only — not legal advice. Press Shift+Enter for a new line.
        </p>
      </div>
    </div>
  );
}
