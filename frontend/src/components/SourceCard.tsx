"use client";

import { useState } from "react";
import { Source } from "@/lib/types";

function formatScore(distance: number, scoreType?: string): string {
  switch (scoreType) {
    case "cosine_distance":
      return `${Math.max(0, 1 - distance).toFixed(2)} sim`;
    case "bm25":
      return `${distance.toFixed(1)} BM25`;
    case "fusion":
      return `${(distance * 100).toFixed(0)}% relevance`;
    case "cross_encoder":
      return `${distance.toFixed(2)} relevance`;
    case "rrf":
      return `${distance.toFixed(3)} RRF`;
    default:
      return distance < 1
        ? `${(1 - distance).toFixed(2)} sim`
        : `${distance.toFixed(1)} score`;
  }
}

function scoreTooltip(scoreType?: string): string {
  switch (scoreType) {
    case "cosine_distance":
      return "Cosine similarity (0\u20131). Higher = more semantically similar to your query.";
    case "bm25":
      return "BM25 lexical score. Higher = more keyword overlap with your query.";
    case "fusion":
      return "Hybrid fusion score (0\u2013100%). Combines semantic similarity and keyword matching.";
    case "cross_encoder":
      return "Cross-encoder relevance score. A reranker model scored how relevant this chunk is to your query. Higher is better; values can be negative.";
    case "rrf":
      return "Reciprocal Rank Fusion score. Merges rankings from multiple query variants. Higher = appeared near the top in more variants.";
    default:
      return "Retrieval score";
  }
}

export default function SourceCard({ source }: { source: Source }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="border border-gray-200 rounded-lg text-sm cursor-pointer hover:border-gray-400 transition-colors"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center justify-between px-3 py-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-gray-400 text-xs shrink-0">
            {expanded ? "▼" : "▶"}
          </span>
          <span className="font-medium text-gray-700 truncate">
            {source.metadata.title}
          </span>
        </div>
        {source.distance !== null && (
          <span
            className="text-xs text-gray-400 shrink-0 ml-2 cursor-help border-b border-dotted border-gray-300"
            title={scoreTooltip(source.score_type)}
          >
            {formatScore(source.distance, source.score_type)}
          </span>
        )}
      </div>
      {expanded && (
        <div className="px-3 pb-3 border-t border-gray-100">
          <p className="text-gray-600 mt-2 whitespace-pre-wrap text-xs leading-relaxed">
            {source.content}
          </p>
          <a
            href={source.metadata.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline text-xs mt-2 inline-block"
            onClick={(e) => e.stopPropagation()}
          >
            {source.metadata.source_url}
          </a>
        </div>
      )}
    </div>
  );
}
