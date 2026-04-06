import React, { useState } from 'react';
import { User, Bot, ChevronDown, ChevronRight, Wrench, FileText, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message, RetrievalJudge } from '../types';
import './ChatMessage.css';

interface ChatMessageProps {
  message: Message;
  toolCalls?: any[];
  toolResults?: any[];
  retrievalJudge?: RetrievalJudge;
}

interface EvidenceItem {
  filePath: string;
  snippet: string;
}

interface ParsedAssistantAnswer {
  answerText: string;
  evidenceItems: EvidenceItem[];
}

const EVIDENCE_HEADER_REGEX = /(?:^|\n)#{2,3}\s*(证据|evidence)\s*\n/i;

const parseAssistantAnswerWithEvidence = (content: string): ParsedAssistantAnswer => {
  if (!content?.trim()) {
    return { answerText: '', evidenceItems: [] };
  }

  const headerMatch = EVIDENCE_HEADER_REGEX.exec(content);
  if (!headerMatch || headerMatch.index === undefined) {
    return { answerText: content, evidenceItems: [] };
  }

  const answerText = content.slice(0, headerMatch.index).trim();
  const evidenceText = content.slice(headerMatch.index + headerMatch[0].length).trim();
  const evidenceBlocks = evidenceText
    .split(/\n(?=\d+\.\s*(来源文件|source file)\s*:)/i)
    .map((item) => item.trim())
    .filter(Boolean);

  const evidenceItems: EvidenceItem[] = [];
  evidenceBlocks.forEach((block) => {
    const fileMatch = block.match(/\d+\.\s*(来源文件|source file)\s*:\s*`?([^\n`]+)`?/i);
    if (!fileMatch) {
      return;
    }

    const snippetMatch = block.match(/(?:证据片段|snippet)\s*:\s*["“]?([\s\S]*?)["”]?\s*$/i);
    const filePath = fileMatch[2].trim();
    let snippet = (snippetMatch?.[1] || '').trim();
    if (!snippet) {
      snippet = block
        .split('\n')
        .slice(1)
        .join(' ')
        .replace(/(?:证据片段|snippet)\s*:/i, '')
        .trim();
    }

    if (!filePath || !snippet) {
      return;
    }

    evidenceItems.push({
      filePath,
      snippet,
    });
  });

  return {
    answerText: answerText || content.trim(),
    evidenceItems,
  };
};

const ChatMessage: React.FC<ChatMessageProps> = ({ message, toolCalls, toolResults, retrievalJudge }) => {
  const isUser = message.role === 'user';
  const [isToolCallExpanded, setIsToolCallExpanded] = useState(true);
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set());
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  // Check if it's a tool call message
  const isToolCallMessage = message.content?.startsWith('[TOOL]');
  const parsedAssistantAnswer = !isUser
    ? parseAssistantAnswerWithEvidence(message.content || '')
    : null;
  const evidenceItems = parsedAssistantAnswer?.evidenceItems || [];
  const hasEvidenceItems = evidenceItems.length > 0;

  const handleCopy = async (content: string, index: number) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const toggleExpand = (index: number) => {
    setExpandedResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  return (
    <div className={`message ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className="message-icon">
        {isUser ? (
          <div className="icon-wrapper user-icon">
            <User size={18} />
          </div>
        ) : (
          <div className="icon-wrapper assistant-icon">
            <Bot size={18} />
          </div>
        )}
      </div>
      <div className="message-content">
        {isToolCallMessage && toolCalls ? (
          <div className="tool-call-container">
            <div
              className="tool-call-header"
              onClick={() => setIsToolCallExpanded(!isToolCallExpanded)}
            >
              <Wrench size={16} className="tool-icon" />
              <span className="tool-call-title">Tool Call</span>
              {isToolCallExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </div>
            {isToolCallExpanded && (
              <div className="tool-call-details">
                <div className="tool-section">
                  <div className="tool-section-title">
                    <Wrench size={14} />
                    <span>Input</span>
                  </div>
                  {toolCalls.map((toolCall, index) => {
                    try {
                      const args = JSON.parse(toolCall.function?.arguments || '{}');
                      const entries = Object.entries(args);
                      return (
                        <div key={index} className="tool-call-item">
                          <div className="tool-call-files">
                            {entries.length === 0 && (
                              <span className="file-badge">No arguments</span>
                            )}
                            {entries.map(([key, value]) => {
                              if (Array.isArray(value)) {
                                return value.map((item, i) => (
                                  <span key={`${key}-${i}`} className="file-badge">{`${key}: ${item}`}</span>
                                ));
                              }
                              return (
                                <span key={key} className="file-badge">
                                  {`${key}: ${String(value)}`}
                                </span>
                              );
                            })}
                          </div>
                        </div>
                      );
                    } catch (e) {
                      return null;
                    }
                  })}
                </div>

                {toolResults && toolResults.length > 0 && (
                  <div className="tool-section">
                    <div className="tool-section-title">
                      <FileText size={14} />
                      <span>Output</span>
                    </div>
                    {toolResults.map((result, index) => {
                      try {
                        const content = result.content || '';
                        const contentLength = content.length;
                        const MAX_DISPLAY_LENGTH = 500000;
                        const isTruncated = contentLength > MAX_DISPLAY_LENGTH;
                        const displayContent = isTruncated
                          ? content.substring(0, MAX_DISPLAY_LENGTH) + '\n\n... (content truncated for display)'
                          : content;
                        const isExpanded = expandedResults.has(index);

                        return (
                          <div key={index} className="tool-response">
                            <div className="tool-response-meta-row">
                              <span className="tool-response-meta">
                                {contentLength.toLocaleString()} characters retrieved
                                {isTruncated && ' (displaying first 500,000)'}
                              </span>
                              <div className="tool-response-actions">
                                <button
                                  className="icon-action-button"
                                  onClick={() => toggleExpand(index)}
                                  title={isExpanded ? 'Collapse' : 'Expand'}
                                >
                                  {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                                </button>
                                <button
                                  className="icon-action-button copy-btn"
                                  onClick={() => handleCopy(content, index)}
                                  title="Copy full content to clipboard"
                                >
                                  {copiedIndex === index ? <Check size={16} /> : <Copy size={16} />}
                                </button>
                              </div>
                            </div>
                            {isExpanded && (
                              <div className="tool-response-content">
                                {displayContent}
                              </div>
                            )}
                          </div>
                        );
                      } catch (e) {
                        console.error('Error rendering tool result:', e);
                        return (
                          <div key={index} className="tool-response">
                            <div className="tool-response-meta" style={{ color: 'red' }}>
                              Error rendering content
                            </div>
                          </div>
                        );
                      }
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="message-text">
            {isUser ? (
              <div className="user-text">{message.content}</div>
            ) : (
              <div className="assistant-answer-container">
                {retrievalJudge && (
                  <div className={`retrieval-judge ${retrievalJudge.stop ? 'stop' : 'continue'}`}>
                    检索停止判断：{retrievalJudge.stop ? '可停止' : '继续检索'}
                    {retrievalJudge.reason ? ` · ${retrievalJudge.reason}` : ''}
                  </div>
                )}
                <div className="markdown-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {hasEvidenceItems ? parsedAssistantAnswer?.answerText || '' : message.content || ''}
                  </ReactMarkdown>
                </div>

                {hasEvidenceItems ? (
                  <div className="evidence-panel">
                    <div className="evidence-panel-header">证据约束规范：来源文件 + 证据片段</div>
                    <div className="evidence-list">
                      {evidenceItems.map((item, index) => (
                        <div key={`${item.filePath}-${index}`} className="evidence-item">
                          <div className="evidence-source">
                            来源文件: <code>{item.filePath}</code>
                          </div>
                          <div className="evidence-snippet">证据片段: "{item.snippet}"</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="evidence-missing">
                    证据约束规范：未检测到“来源文件 + 证据片段”结构
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
