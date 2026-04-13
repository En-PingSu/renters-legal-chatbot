"use client";

import ReactMarkdown from "react-markdown";
import SourceCard from "./SourceCard";
import { Message } from "@/lib/types";

const SECTIONS = [
  "Question Understanding",
  "Relevant Evidence",
  "Analysis",
  "Final Answer",
  "Confidence",
];

/**
 * Ensure each structured section header starts on its own paragraph.
 * Only applied to completed messages to avoid breaking partial markdown during streaming.
 */
function formatSections(text: string): string {
  let result = text;
  for (const section of SECTIONS) {
    // Normalize all variants to **Section:** with double newline before it:
    //   **Section:**  /  **Section**:  /  **Section:  (unclosed bold)
    const pattern = new RegExp(
      `\\*\\*${section}:?\\*\\*:?|\\*\\*${section}:?`,
      "g"
    );
    result = result.replace(pattern, `\n\n**${section}:**`);
  }
  // Clean up leading whitespace and collapse excessive newlines
  result = result.replace(/^\n+/, "");
  result = result.replace(/\n{3,}/g, "\n\n");
  return result;
}

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  // Only format sections for completed assistant messages
  const content = !isUser && !message.isStreaming
    ? formatSections(message.content)
    : message.content;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] ${
          isUser
            ? "bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-2.5"
            : "bg-white border border-gray-200 rounded-2xl rounded-bl-md px-5 py-4 shadow-sm"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <div className="overflow-wrap-anywhere break-words text-gray-700 text-sm leading-relaxed [&_p]:my-2.5 [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:my-2 [&_li]:my-1.5 [&_li]:leading-relaxed [&_a]:text-blue-600 [&_a]:underline [&_strong]:font-semibold [&_strong]:text-gray-900">
            <ReactMarkdown>{content}</ReactMarkdown>
            {message.isStreaming && (
              <span className="inline-block w-1.5 h-4 bg-blue-500 animate-pulse ml-0.5 align-text-bottom rounded-sm" />
            )}
          </div>
        )}
        {message.sources && message.sources.length > 0 && !message.isStreaming && (
          <div className="mt-4 pt-3 border-t border-gray-100 space-y-2">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
              Retrieved Sources ({message.sources.length})
            </p>
            {message.sources.map((source) => (
              <SourceCard key={source.chunk_id} source={source} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
