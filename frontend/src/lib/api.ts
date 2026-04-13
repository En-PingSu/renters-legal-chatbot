import { ChatConfig, Source } from "./types";

const API_BASE = "http://localhost:8000";

export async function sendMessage(
  question: string,
  config: ChatConfig,
  callbacks: {
    onToken: (token: string) => void;
    onSources: (sources: Source[]) => void;
    onDone: () => void;
    onError: (err: string) => void;
  }
) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, ...config }),
  });

  if (!res.ok || !res.body) {
    callbacks.onError(`API error: ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse SSE events -- sse-starlette uses \r\n line endings
    // Normalize \r\n to \n, then split on double-newline
    buffer = buffer.replace(/\r\n/g, "\n");
    const parts = buffer.split("\n\n");
    buffer = parts.pop()!; // keep incomplete chunk

    for (const part of parts) {
      let eventType = "";
      let data = "";
      for (const line of part.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data: ")) data = line.slice(6);
        else if (line.startsWith("data:")) data = line.slice(5);
      }

      if (eventType === "token") callbacks.onToken(data || "\n");
      else if (eventType === "sources") {
        try {
          callbacks.onSources(JSON.parse(data));
        } catch {
          /* ignore parse errors */
        }
      } else if (eventType === "error") callbacks.onError(data);
      else if (eventType === "done") callbacks.onDone();
    }
  }

  // If stream ended without done event
  callbacks.onDone();
}

export async function fetchConfig(): Promise<{
  models: { id: string; label: string }[];
  retrievers: string[];
  defaults: { model: string; retriever: string; top_k: number };
}> {
  const res = await fetch(`${API_BASE}/api/config`);
  return res.json();
}
