"use client";

import { ChatConfig } from "@/lib/types";

const MODELS = [
  { id: "mistralai/mistral-small-3.1-24b-instruct", label: "Mistral Small 24B" },
  { id: "openai/gpt-4o", label: "GPT-4o" },
  { id: "meta-llama/llama-3.3-70b-instruct", label: "Llama 3.3 70B" },
];

const RETRIEVERS = [
  "rerank",
  "vector",
  "bm25",
  "hybrid",
  "parent_child",
  "auto_merge",
  "hybrid_parent_child_rerank",
];

interface SidebarProps {
  config: ChatConfig;
  onConfigChange: (config: ChatConfig) => void;
  onNewChat: () => void;
}

export default function Sidebar({ config, onConfigChange, onNewChat }: SidebarProps) {
  const update = (partial: Partial<ChatConfig>) =>
    onConfigChange({ ...config, ...partial });

  return (
    <aside className="w-72 bg-gray-50 border-r border-gray-200 flex flex-col h-full">
      <div className="p-5 border-b border-gray-200">
        <h1 className="text-lg font-bold text-gray-900">MA Tenant Law</h1>
        <p className="text-xs text-gray-500 mt-0.5">RAG Legal Information Assistant</p>
      </div>

      <div className="flex-1 p-5 space-y-5 overflow-y-auto">
        <div>
          <label className="block text-xs font-medium text-gray-600 uppercase tracking-wide mb-1.5">
            Model
          </label>
          <select
            value={config.model}
            onChange={(e) => update({ model: e.target.value })}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          >
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-600 uppercase tracking-wide mb-1.5">
            Retriever
          </label>
          <select
            value={config.retriever}
            onChange={(e) => update({ retriever: e.target.value })}
            disabled={!config.use_rag}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none disabled:opacity-50 disabled:bg-gray-100"
          >
            {RETRIEVERS.map((r) => (
              <option key={r} value={r}>
                {r.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-600 uppercase tracking-wide mb-1.5">
            Top-K: {config.top_k}
          </label>
          <input
            type="range"
            min={1}
            max={15}
            value={config.top_k}
            onChange={(e) => update({ top_k: Number(e.target.value) })}
            disabled={!config.use_rag}
            className="w-full accent-blue-600 disabled:opacity-50"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-0.5">
            <span>1</span>
            <span>15</span>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <input
            type="checkbox"
            id="rag-toggle"
            checked={config.use_rag}
            onChange={(e) => update({ use_rag: e.target.checked })}
            className="w-4 h-4 accent-blue-600 rounded"
          />
          <label htmlFor="rag-toggle" className="text-sm text-gray-700 select-none">
            Use RAG retrieval
          </label>
        </div>
      </div>

      <div className="p-5 border-t border-gray-200">
        <button
          onClick={onNewChat}
          className="w-full bg-gray-200 hover:bg-gray-300 text-gray-700 font-medium py-2 rounded-lg text-sm transition-colors"
        >
          New Chat
        </button>
      </div>
    </aside>
  );
}
