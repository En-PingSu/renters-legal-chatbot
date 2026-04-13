export interface Source {
  chunk_id: string;
  content: string;
  metadata: {
    title: string;
    source_url: string;
    source_name: string;
    content_type: string;
  };
  distance: number | null;
  score_type?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
}

export interface ChatConfig {
  model: string;
  retriever: string;
  top_k: number;
  use_rag: boolean;
}

export const DEFAULT_CONFIG: ChatConfig = {
  model: "mistralai/mistral-small-3.1-24b-instruct",
  retriever: "rerank",
  top_k: 5,
  use_rag: true,
};
