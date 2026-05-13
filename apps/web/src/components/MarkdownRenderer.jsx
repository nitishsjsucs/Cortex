/**
 * MarkdownRenderer — renders a subset of Markdown as React JSX.
 *
 * Supported syntax:
 *   # ## ###              — headings
 *   **bold** *italic*     — inline emphasis
 *   `code`                — inline code
 *   ```...```             — code block
 *   - item                — unordered list
 *   1. item               — ordered list
 *   > quote               — blockquote
 *   [text](url)           — hyperlink
 *   | a | b |             — table (followed by |---|)
 *   ---                   — horizontal rule
 *   \n\n                  — paragraph break
 *
 * Usage:
 *   <MarkdownRenderer text={content} className="prose" />
 */

import React from "react";
import { cn } from "./ui/index.jsx";

// ── Inline spans ──────────────────────────────────────────────────────────────

function InlineMarkdown({ text }) {
  if (!text) return null;

  // Split on all inline patterns in one pass using a combined regex
  const PATTERN = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[([^\]]+)\]\(([^)]+)\))/g;
  const parts = [];
  let last = 0;
  let match;
  let idx = 0;

  while ((match = PATTERN.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(<span key={idx++}>{text.slice(last, match.index)}</span>);
    }
    const m = match[0];
    if (m.startsWith("**")) {
      parts.push(<strong key={idx++} className="font-semibold text-white">{m.slice(2, -2)}</strong>);
    } else if (m.startsWith("*")) {
      parts.push(<em key={idx++} className="italic text-zinc-200">{m.slice(1, -1)}</em>);
    } else if (m.startsWith("`")) {
      parts.push(
        <code key={idx++} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[0.8em] font-mono text-violet-300">
          {m.slice(1, -1)}
        </code>
      );
    } else if (m.startsWith("[")) {
      const linkText = match[2];
      const href     = match[3];
      parts.push(
        <a key={idx++} href={href} target="_blank" rel="noreferrer"
          className="text-primary underline underline-offset-2 hover:text-primary/80 transition-colors">
          {linkText}
        </a>
      );
    }
    last = match.index + m.length;
  }

  if (last < text.length) {
    parts.push(<span key={idx++}>{text.slice(last)}</span>);
  }

  return <>{parts}</>;
}

// ── Block renderer ────────────────────────────────────────────────────────────

export default function MarkdownRenderer({ text = "", className = "", compact = false }) {
  if (!text) return null;

  const lines   = text.split("\n");
  const elements = [];
  let i = 0;
  let keyCounter = 0;
  const k = () => keyCounter++;

  while (i < lines.length) {
    const line = lines[i];

    // ── Code block ──────────────────────────────────────────────────────────
    if (line.startsWith("```")) {
      const lang  = line.slice(3).trim();
      const code  = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        code.push(lines[i]);
        i++;
      }
      elements.push(
        <pre key={k()} className="my-3 overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <code className="text-xs font-mono text-zinc-200 leading-relaxed whitespace-pre">
            {code.join("\n")}
          </code>
        </pre>
      );
      i++; continue;
    }

    // ── Table ────────────────────────────────────────────────────────────────
    if (line.startsWith("|") && i + 1 < lines.length && lines[i + 1].match(/^\|[-\s|:]+\|/)) {
      const headers = line.split("|").slice(1, -1).map(h => h.trim());
      i += 2; // skip header + separator
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        rows.push(lines[i].split("|").slice(1, -1).map(c => c.trim()));
        i++;
      }
      elements.push(
        <div key={k()} className="my-4 overflow-x-auto rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 bg-zinc-900/60">
                {headers.map((h, j) => (
                  <th key={j} className="px-4 py-2.5 text-left text-xs font-semibold text-zinc-300 uppercase tracking-wide">
                    <InlineMarkdown text={h} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr key={ri} className="border-b border-zinc-800/40 hover:bg-zinc-900/30">
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-4 py-2.5 text-zinc-300 text-xs">
                      <InlineMarkdown text={cell} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }

    // ── Horizontal rule ──────────────────────────────────────────────────────
    if (line.match(/^---+$/) || line.match(/^\*\*\*+$/)) {
      elements.push(<hr key={k()} className="my-4 border-zinc-800" />);
      i++; continue;
    }

    // ── Headings ─────────────────────────────────────────────────────────────
    if (line.startsWith("# ")) {
      elements.push(
        <h1 key={k()} className={cn("font-bold text-white mt-6 mb-2", compact ? "text-base" : "text-xl")}>
          <InlineMarkdown text={line.slice(2)} />
        </h1>
      );
      i++; continue;
    }
    if (line.startsWith("## ")) {
      elements.push(
        <h2 key={k()} className={cn("font-semibold text-white mt-5 mb-2 border-b border-zinc-800 pb-1", compact ? "text-sm" : "text-lg")}>
          <InlineMarkdown text={line.slice(3)} />
        </h2>
      );
      i++; continue;
    }
    if (line.startsWith("### ")) {
      elements.push(
        <h3 key={k()} className={cn("font-semibold text-white mt-4 mb-1.5", compact ? "text-xs" : "text-base")}>
          <InlineMarkdown text={line.slice(4)} />
        </h3>
      );
      i++; continue;
    }

    // ── Bold-only line (acts as h3) ───────────────────────────────────────────
    if (line.match(/^\*\*[^*]+\*\*$/) && !line.startsWith("- ")) {
      elements.push(
        <p key={k()} className={cn("font-semibold text-white mt-4 mb-1.5", compact ? "text-xs" : "text-sm")}>
          <InlineMarkdown text={line.replace(/\*\*/g, "")} />
        </p>
      );
      i++; continue;
    }

    // ── Blockquote ────────────────────────────────────────────────────────────
    if (line.startsWith("> ")) {
      const quotes = [];
      while (i < lines.length && lines[i].startsWith("> ")) {
        quotes.push(lines[i].slice(2));
        i++;
      }
      elements.push(
        <blockquote key={k()} className="my-3 border-l-2 border-primary pl-4 text-zinc-400 italic">
          {quotes.map((q, qi) => (
            <p key={qi} className="text-xs leading-relaxed"><InlineMarkdown text={q} /></p>
          ))}
        </blockquote>
      );
      continue;
    }

    // ── Unordered list ────────────────────────────────────────────────────────
    if (line.match(/^[-*] /)) {
      const items = [];
      while (i < lines.length && lines[i].match(/^[-*] /)) {
        items.push(lines[i].replace(/^[-*] /, ""));
        i++;
      }
      elements.push(
        <ul key={k()} className="my-2 space-y-1.5 pl-1">
          {items.map((item, j) => (
            <li key={j} className="flex items-start gap-2.5">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-400" />
              <span className={cn("leading-relaxed text-zinc-300", compact ? "text-xs" : "text-sm")}>
                <InlineMarkdown text={item} />
              </span>
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // ── Ordered list ──────────────────────────────────────────────────────────
    if (line.match(/^\d+\. /)) {
      const items = [];
      let num = 1;
      while (i < lines.length && lines[i].match(/^\d+\. /)) {
        items.push(lines[i].replace(/^\d+\. /, ""));
        i++;
      }
      elements.push(
        <ol key={k()} className="my-2 space-y-1.5 pl-1">
          {items.map((item, j) => (
            <li key={j} className="flex items-start gap-2.5">
              <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-indigo-500/20 text-[9px] font-bold text-indigo-400">
                {j + 1}
              </span>
              <span className={cn("leading-relaxed text-zinc-300", compact ? "text-xs" : "text-sm")}>
                <InlineMarkdown text={item} />
              </span>
            </li>
          ))}
        </ol>
      );
      continue;
    }

    // ── Empty line ────────────────────────────────────────────────────────────
    if (line.trim() === "") {
      if (!compact) elements.push(<div key={k()} className="h-1.5" />);
      i++; continue;
    }

    // ── Paragraph ─────────────────────────────────────────────────────────────
    elements.push(
      <p key={k()} className={cn("leading-relaxed text-zinc-300 my-1", compact ? "text-xs" : "text-sm")}>
        <InlineMarkdown text={line} />
      </p>
    );
    i++;
  }

  return <div className={cn("space-y-0", className)}>{elements}</div>;
}
