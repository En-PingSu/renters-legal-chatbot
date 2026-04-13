"use client";

import ReactMarkdown from "react-markdown";
import SourceCard from "./SourceCard";
import { Message } from "@/lib/types";

const SECTIONS = [
  "Question Understanding",
  "Relevant Evidence",
  "Analysis",
  "Final Answer",
];

/**
 * Parse the confidence score (1-10) from the response text.
 * Returns null if not found or not a valid number.
 */
function parseConfidence(text: string): number | null {
  const match = text.match(/\*\*Confidence:\*?\*?\s*(\d+)/);
  if (!match) return null;
  const score = parseInt(match[1], 10);
  return score >= 1 && score <= 10 ? score : null;
}

/**
 * Remove the **Confidence:** line from the markdown so we can render it separately.
 */
function stripConfidence(text: string): string {
  return text.replace(/\n*\*\*Confidence:\*?\*?\s*\d+.*/g, "").trimEnd();
}

function confidenceColor(score: number): string {
  if (score >= 8) return "bg-green-100 text-green-800 border-green-200";
  if (score >= 5) return "bg-yellow-100 text-yellow-800 border-yellow-200";
  return "bg-red-100 text-red-800 border-red-200";
}

function confidenceLabel(score: number): string {
  if (score >= 8) return "High";
  if (score >= 5) return "Medium";
  return "Low";
}

/**
 * Fix source citations: add missing closing brackets, then escape
 * both brackets so ReactMarkdown renders them literally.
 */
function fixSourceCitations(text: string): string {
  // Add missing closing ] for [Source... patterns that end at newline/end/next-citation
  let result = text.replace(
    /(\[Source\s*\d*\s*:[^\]\n]*?)(?=\n|$|\[Source)/gm,
    "$1]"
  );
  // Escape all [Source...] brackets for markdown
  result = result.replace(
    /\[Source\s*\d*\s*:[^\]]*?\]/g,
    (match) => "\\" + match.slice(0, -1) + "\\]"
  );
  return result;
}

/**
 * Ensure each structured section header starts on its own paragraph.
 * Only applied to completed messages to avoid breaking partial markdown during streaming.
 */
function formatSections(text: string): string {
  let result = text;
  for (const section of SECTIONS) {
    const pattern = new RegExp(
      `\\*\\*${section}:?\\*\\*:?|\\*\\*${section}:?`,
      "g"
    );
    result = result.replace(pattern, `\n\n**${section}:**`);
  }
  result = fixSourceCitations(result);
  result = result.replace(/^\n+/, "");
  result = result.replace(/\n{3,}/g, "\n\n");
  return result;
}

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const isComplete = !isUser && !message.isStreaming;

  const confidence = isComplete ? parseConfidence(message.content) : null;
  const formatted = isComplete
    ? formatSections(stripConfidence(message.content))
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
          <p className="whitespace-pre-wrap">{formatted}</p>
        ) : (
          <div className="overflow-wrap-anywhere break-words text-gray-700 text-sm leading-relaxed [&_p]:my-2.5 [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:my-2 [&_li]:my-1.5 [&_li]:leading-relaxed [&_a]:text-blue-600 [&_a]:underline [&_strong]:font-semibold [&_strong]:text-gray-900">
            <ReactMarkdown>{formatted}</ReactMarkdown>
            {message.isStreaming && (
              <span className="inline-block w-1.5 h-4 bg-blue-500 animate-pulse ml-0.5 align-text-bottom rounded-sm" />
            )}
          </div>
        )}
        {confidence !== null && (
          <div className="mt-3 pt-3 border-t border-gray-100 flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${confidenceColor(confidence)}`}
              title={`Confidence ${confidence}/10 — how well the retrieved sources cover this question. 8-10: sources directly address the question. 5-7: sources partially cover it. 1-4: limited source coverage, answer may be less reliable.`}
            >
              <span className="font-semibold">{confidence}/10</span>
              <span className="opacity-75">{confidenceLabel(confidence)}</span>
            </span>
            <span className="text-xs text-gray-400">source coverage</span>
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
