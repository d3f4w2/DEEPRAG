import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Settings, FileCode, MessageSquare, BarChart3, Mic, Image as ImageIcon } from 'lucide-react';
import { chatApi } from './api';
import type {
  BudgetSummary,
  ChatBudgetConfig,
  Message,
  ReasoningStage,
  ReasoningStageMetric,
  RetrievalCritic,
  StreamChunk,
} from './types';
import ChatMessage from './components/ChatMessage';
import SystemPromptPanel from './components/SystemPromptPanel';
import ConfigPanel from './components/ConfigPanel';
import EvaluationPanel from './components/EvaluationPanel';
import VoiceIngestionPanel from './components/VoiceIngestionPanel';
import VoiceErrorBoundary from './components/VoiceErrorBoundary';
import ImageIngestionPanel from './components/ImageIngestionPanel';
import './App.css';

const MAX_CONTEXT_MESSAGES = 12;
const TOOL_STATUS_PREFIX = '[TOOL]';
const BUDGET_CONFIG_STORAGE_KEY = 'deep_rag_budget_config_v1';
const DEFAULT_BUDGET_CONFIG: ChatBudgetConfig = {
  max_total_tokens: 60000,
  max_latency_ms: 90000,
  price_per_1m_tokens: 2,
  cost_multiplier: 1,
};
const STAGE_ORDER: Record<string, number> = {
  plan: 10,
  execute: 20,
  critic: 30,
};
const STAGE_LABELS: Record<string, string> = {
  plan: 'Plan',
  execute: 'Execute',
  critic: 'Judge',
};

const toNumber = (value: unknown, fallback: number): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const parseBudgetSummaryFromChunk = (chunk: StreamChunk): BudgetSummary => {
  const usage = chunk.usage || {};
  const limits = chunk.limits || {};
  const pricing = chunk.pricing || {};
  return {
    triggered: !!chunk.triggered,
    reason: chunk.reason || '',
    total_tokens: Math.max(0, Math.floor(toNumber(usage.total_tokens, 0))),
    elapsed_ms: Math.max(0, Math.floor(toNumber(usage.elapsed_ms, 0))),
    max_total_tokens: Math.max(0, Math.floor(toNumber(limits.max_total_tokens, 0))),
    max_latency_ms: Math.max(0, Math.floor(toNumber(limits.max_latency_ms, 0))),
    price_per_1m_tokens: Math.max(0, toNumber(pricing.price_per_1m_tokens, 0)),
    cost_multiplier: Math.max(0, toNumber(pricing.cost_multiplier, 1)),
    cost_estimate_usd: Math.max(0, toNumber(chunk.cost_estimate_usd, 0)),
  };
};

const normalizeReasoningMetrics = (metrics: StreamChunk['metrics']): ReasoningStageMetric[] => {
  if (!Array.isArray(metrics)) {
    return [];
  }
  return metrics.reduce<ReasoningStageMetric[]>((acc, item) => {
    const label = String(item?.label || '').trim();
    const value = String(item?.value || '').trim();
    if (!label || !value) {
      return acc;
    }
    const tone = String(item?.tone || '').trim().toLowerCase();
    acc.push({
      label,
      value,
      tone: tone || undefined,
    });
    return acc;
  }, []);
};

const parseReasoningStageFromChunk = (chunk: StreamChunk): ReasoningStage | null => {
  const stageKey = String(chunk.stage_key || '')
    .trim()
    .toLowerCase();
  if (!stageKey) {
    return null;
  }

  const defaultLabel = STAGE_LABELS[stageKey] || stageKey.toUpperCase();
  const mode = String(chunk.mode || 'llm').trim() || 'llm';
  const numericOrder = toNumber(chunk.order, Number.NaN);
  const order = Number.isFinite(numericOrder) ? numericOrder : (STAGE_ORDER[stageKey] ?? 99);

  return {
    stage_key: stageKey,
    stage_label: String(chunk.stage_label || defaultLabel).trim() || defaultLabel,
    title: String(chunk.title || `${defaultLabel} 更新`).trim() || `${defaultLabel} 更新`,
    summary: String(chunk.summary || '').trim(),
    status: String(chunk.status || 'running').trim().toLowerCase() || 'running',
    mode,
    badge: String(chunk.badge || mode.toUpperCase()).trim() || mode.toUpperCase(),
    order,
    updated_at: String(chunk.updated_at || new Date().toISOString()),
    metrics: normalizeReasoningMetrics(chunk.metrics),
  };
};

const mergeReasoningStages = (
  currentStages: ReasoningStage[],
  nextStage: ReasoningStage,
): ReasoningStage[] => {
  const stageMap = new Map<string, ReasoningStage>();
  currentStages.forEach((stage) => {
    stageMap.set(stage.stage_key, stage);
  });
  stageMap.set(nextStage.stage_key, nextStage);

  return Array.from(stageMap.values()).sort(
    (a, b) => a.order - b.order || a.stage_key.localeCompare(b.stage_key),
  );
};

function App() {
  const [activePage, setActivePage] = useState<'chat' | 'evaluation' | 'voice' | 'image'>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSystemPrompt, setShowSystemPrompt] = useState(false);
  const [showConfigPanel, setShowConfigPanel] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [messageToolCalls, setMessageToolCalls] = useState<Map<number, any[]>>(new Map());
  const [messageToolResults, setMessageToolResults] = useState<Map<number, any[]>>(new Map());
  const [messageReasoningStages, setMessageReasoningStages] = useState<Map<number, ReasoningStage[]>>(new Map());
  const [messageRetrievalCritic, setMessageRetrievalCritic] = useState<Map<number, RetrievalCritic>>(new Map());
  const [messageBudgetSummary, setMessageBudgetSummary] = useState<Map<number, BudgetSummary>>(new Map());
  const [showBudgetPanel, setShowBudgetPanel] = useState(false);
  const [budgetConfig, setBudgetConfig] = useState<ChatBudgetConfig>(DEFAULT_BUDGET_CONFIG);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadConfig();
    loadProviders();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(BUDGET_CONFIG_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw) as ChatBudgetConfig;
      setBudgetConfig({
        max_total_tokens: Math.max(0, Math.floor(toNumber(parsed.max_total_tokens, DEFAULT_BUDGET_CONFIG.max_total_tokens || 0))),
        max_latency_ms: Math.max(0, Math.floor(toNumber(parsed.max_latency_ms, DEFAULT_BUDGET_CONFIG.max_latency_ms || 0))),
        price_per_1m_tokens: Math.max(0, toNumber(parsed.price_per_1m_tokens, DEFAULT_BUDGET_CONFIG.price_per_1m_tokens || 0)),
        cost_multiplier: Math.max(0, toNumber(parsed.cost_multiplier, DEFAULT_BUDGET_CONFIG.cost_multiplier || 1)),
      });
    } catch (error) {
      console.error('Failed to load budget config:', error);
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(BUDGET_CONFIG_STORAGE_KEY, JSON.stringify(budgetConfig));
    } catch (error) {
      console.error('Failed to persist budget config:', error);
    }
  }, [budgetConfig]);

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  const loadConfig = async () => {
    try {
      const config = await chatApi.getConfig();
      setSelectedProvider(config.default_provider);
      setSelectedModel(config.default_model);
    } catch (error) {
      console.error('Failed to load config:', error);
    }
  };

  const loadProviders = async () => {
    try {
      await chatApi.getProviders();
    } catch (error) {
      console.error('Failed to load providers:', error);
    }
  };

  const findLatestAssistantAnswerIndex = (items: Message[]): number => {
    for (let i = items.length - 1; i >= 0; i -= 1) {
      const item = items[i];
      if (item.role === 'assistant' && !(item.content || '').startsWith(TOOL_STATUS_PREFIX)) {
        return i;
      }
    }
    return -1;
  };

  const updateBudgetConfig = (
    key: keyof ChatBudgetConfig,
    value: string,
    options: { float?: boolean } = {},
  ) => {
    setBudgetConfig((prev) => {
      const parsed = options.float ? Number.parseFloat(value) : Number.parseInt(value, 10);
      const fallback = options.float
        ? Number(prev[key] ?? (key === 'cost_multiplier' ? 1 : 0))
        : Number(prev[key] ?? 0);
      const nextValue = Number.isFinite(parsed) ? parsed : fallback;
      return {
        ...prev,
        [key]: Math.max(0, nextValue),
      };
    });
  };

  const handleSendMessage = async (messageText?: string) => {
    const textToSend = messageText || inputValue;
    if (!textToSend.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: textToSend,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    setTimeout(() => {
      if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
      }
    }, 50);

    try {
      const nextMessages = [...messages, userMessage];
      const compactMessages = nextMessages
        .filter((msg) => {
          const content = msg.content || '';
          return !(msg.role === 'assistant' && content.startsWith(TOOL_STATUS_PREFIX));
        })
        .slice(-MAX_CONTEXT_MESSAGES);

      const response = await chatApi.sendMessage({
        messages: compactMessages,
        provider: selectedProvider,
        model: selectedModel,
        budget: {
          max_total_tokens: Math.max(0, Math.floor(toNumber(budgetConfig.max_total_tokens, 0))),
          max_latency_ms: Math.max(0, Math.floor(toNumber(budgetConfig.max_latency_ms, 0))),
          price_per_1m_tokens: Math.max(0, toNumber(budgetConfig.price_per_1m_tokens, 0)),
          cost_multiplier: Math.max(0, toNumber(budgetConfig.cost_multiplier, 1)),
        },
      });

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = '';
      let toolCallsInfo = '';
      let currentToolCalls: any[] = [];
      let hasToolCall = false;
      let buffer = '';
      let latestCritic: RetrievalCritic | null = null;
      let latestBudgetSummary: BudgetSummary | null = null;
      let latestReasoningStages: ReasoningStage[] = [];

      const processChunk = async () => {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          buffer += chunk;
          const lines = buffer.split('\n');

          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              try {
                const parsed: StreamChunk = JSON.parse(data);

                if (parsed.type === 'content') {
                  assistantMessage += parsed.content || '';

                  const hasFinalAnswer = assistantMessage.includes('<|Final Answer|>');

                  if (hasFinalAnswer) {
                    const finalAnswerMatch = assistantMessage.match(/<\|Final Answer\|>\s*([\s\S]*)/);
                    const finalAnswerContent = finalAnswerMatch ? finalAnswerMatch[1].trim() : assistantMessage;

                    setMessages((prev) => {
                      const newMessages = [...prev];
                      const lastMessage = newMessages[newMessages.length - 1];
                      let targetMessageIndex = -1;

                      if (hasToolCall && (!lastMessage || lastMessage.content?.startsWith(TOOL_STATUS_PREFIX))) {
                        newMessages.push({
                          role: 'assistant',
                          content: finalAnswerContent,
                        });
                        targetMessageIndex = newMessages.length - 1;
                      } else if (lastMessage && lastMessage.role === 'assistant' && !lastMessage.content?.startsWith(TOOL_STATUS_PREFIX)) {
                        lastMessage.content = finalAnswerContent;
                        targetMessageIndex = newMessages.length - 1;
                      } else if (!hasToolCall) {
                        newMessages.push({
                          role: 'assistant',
                          content: finalAnswerContent,
                        });
                        targetMessageIndex = newMessages.length - 1;
                      }

                      if (latestCritic && targetMessageIndex >= 0) {
                        setMessageRetrievalCritic((prevCriticMap) => {
                          const newMap = new Map(prevCriticMap);
                          newMap.set(targetMessageIndex, latestCritic as RetrievalCritic);
                          return newMap;
                        });
                      }
                      if (latestReasoningStages.length > 0 && targetMessageIndex >= 0) {
                        setMessageReasoningStages((prevStageMap) => {
                          const newMap = new Map(prevStageMap);
                          newMap.set(targetMessageIndex, [...latestReasoningStages]);
                          return newMap;
                        });
                      }
                      if (latestBudgetSummary && targetMessageIndex >= 0) {
                        setMessageBudgetSummary((prevBudgetMap) => {
                          const newMap = new Map(prevBudgetMap);
                          newMap.set(targetMessageIndex, latestBudgetSummary as BudgetSummary);
                          return newMap;
                        });
                      }
                      return newMessages;
                    });
                  }
                } else if (parsed.type === 'tool_calls' && parsed.tool_calls) {
                  hasToolCall = true;
                  currentToolCalls = parsed.tool_calls;
                  toolCallsInfo = `${TOOL_STATUS_PREFIX} Retrieving evidence...`;

                  setMessages((prev) => {
                    const newMessages = [...prev];
                    newMessages.push({
                      role: 'assistant',
                      content: toolCallsInfo,
                    });

                    const messageIndex = newMessages.length - 1;
                    setMessageToolCalls((prevToolCalls) => {
                      const newMap = new Map(prevToolCalls);
                      newMap.set(messageIndex, currentToolCalls);
                      return newMap;
                    });

                    return newMessages;
                  });
                } else if (parsed.type === 'tool_results' && parsed.results) {
                  setMessages((prev) => {
                    const messageIndex = prev.length - 1;
                    setMessageToolResults((prevResults) => {
                      const newMap = new Map(prevResults);
                      newMap.set(messageIndex, parsed.results || []);
                      return newMap;
                    });
                    return [...prev];
                  });
                  assistantMessage = '';
                } else if (parsed.type === 'reasoning_stage') {
                  const stage = parseReasoningStageFromChunk(parsed);
                  if (!stage) {
                    continue;
                  }
                  latestReasoningStages = mergeReasoningStages(latestReasoningStages, stage);
                  setMessages((prev) => {
                    const messageIndex = findLatestAssistantAnswerIndex(prev);
                    if (messageIndex < 0) {
                      return prev;
                    }
                    setMessageReasoningStages((prevStageMap) => {
                      const newMap = new Map(prevStageMap);
                      const merged = mergeReasoningStages(newMap.get(messageIndex) || [], stage);
                      newMap.set(messageIndex, merged);
                      return newMap;
                    });
                    return [...prev];
                  });
                } else if (parsed.type === 'retrieval_critic') {
                  latestCritic = {
                    decision:
                      parsed.decision === 'accept' ||
                      parsed.decision === 'revise' ||
                      parsed.decision === 'refuse'
                        ? parsed.decision
                        : parsed.stop
                          ? 'accept'
                          : 'revise',
                    stop: !!parsed.stop,
                    mode: parsed.mode || 'rule',
                    reason: parsed.reason || '',
                    confidence: Math.max(0, Math.min(1, Number(parsed.confidence || 0))),
                    evidence_items: Math.max(0, Math.floor(Number(parsed.evidence_items || 0))),
                    distinct_files: Math.max(0, Math.floor(Number(parsed.distinct_files || 0))),
                    matched_evidence_items: Math.max(0, Math.floor(Number(parsed.matched_evidence_items || 0))),
                    total_snippet_chars: Math.max(0, Math.floor(Number(parsed.total_snippet_chars || 0))),
                    uncertain_answer: !!parsed.uncertain_answer,
                    evidence_sufficient: !!parsed.evidence_sufficient,
                    rule_reason: String(parsed.rule_reason || ''),
                    query_term_hits: Math.max(0, Math.floor(Number(parsed.query_term_hits || 0))),
                    query_terms: Math.max(0, Math.floor(Number(parsed.query_terms || 0))),
                    retrieval_rounds: Math.max(0, Math.floor(Number(parsed.retrieval_rounds || 0))),
                    max_retrieval_rounds: Math.max(0, Math.floor(Number(parsed.max_retrieval_rounds || 0))),
                    tool_call_rounds: Math.max(0, Math.floor(Number(parsed.tool_call_rounds || 0))),
                    max_tool_call_rounds: Math.max(0, Math.floor(Number(parsed.max_tool_call_rounds || 0))),
                    retry_count_total: Math.max(0, Math.floor(Number(parsed.retry_count_total || 0))),
                    retry_exhausted_count: Math.max(0, Math.floor(Number(parsed.retry_exhausted_count || 0))),
                    cache_hit_count: Math.max(0, Math.floor(Number(parsed.cache_hit_count || 0))),
                    auto_retrieval_rounds: Math.max(0, Math.floor(Number(parsed.auto_retrieval_rounds || 0))),
                    max_auto_retrieval_rounds: Math.max(0, Math.floor(Number(parsed.max_auto_retrieval_rounds || 0))),
                  };
                  setMessages((prev) => {
                    const messageIndex = findLatestAssistantAnswerIndex(prev);
                    if (messageIndex < 0) {
                      return prev;
                    }
                    setMessageRetrievalCritic((prevCriticMap) => {
                      const newMap = new Map(prevCriticMap);
                      newMap.set(messageIndex, {
                        decision: latestCritic?.decision || 'revise',
                        stop: latestCritic?.stop || false,
                        mode: latestCritic?.mode || 'rule',
                        reason: latestCritic?.reason || '',
                        confidence: latestCritic?.confidence || 0,
                        evidence_items: latestCritic?.evidence_items || 0,
                        distinct_files: latestCritic?.distinct_files || 0,
                        matched_evidence_items: latestCritic?.matched_evidence_items || 0,
                        total_snippet_chars: latestCritic?.total_snippet_chars || 0,
                        uncertain_answer: !!latestCritic?.uncertain_answer,
                        evidence_sufficient: !!latestCritic?.evidence_sufficient,
                        rule_reason: latestCritic?.rule_reason || '',
                        query_term_hits: latestCritic?.query_term_hits || 0,
                        query_terms: latestCritic?.query_terms || 0,
                        retrieval_rounds: latestCritic?.retrieval_rounds || 0,
                        max_retrieval_rounds: latestCritic?.max_retrieval_rounds || 0,
                        tool_call_rounds: latestCritic?.tool_call_rounds || 0,
                        max_tool_call_rounds: latestCritic?.max_tool_call_rounds || 0,
                        retry_count_total: latestCritic?.retry_count_total || 0,
                        retry_exhausted_count: latestCritic?.retry_exhausted_count || 0,
                        cache_hit_count: latestCritic?.cache_hit_count || 0,
                        auto_retrieval_rounds: latestCritic?.auto_retrieval_rounds || 0,
                        max_auto_retrieval_rounds: latestCritic?.max_auto_retrieval_rounds || 0,
                      });
                      return newMap;
                    });
                    return [...prev];
                  });
                } else if (
                  parsed.type === 'budget_update' ||
                  parsed.type === 'budget_guard' ||
                  parsed.type === 'budget_summary'
                ) {
                  latestBudgetSummary = parseBudgetSummaryFromChunk(parsed);
                  if (parsed.type !== 'budget_update') {
                    setMessages((prev) => {
                      const messageIndex = findLatestAssistantAnswerIndex(prev);
                      if (messageIndex < 0) {
                        return prev;
                      }
                      setMessageBudgetSummary((prevBudgetMap) => {
                        const newMap = new Map(prevBudgetMap);
                        newMap.set(messageIndex, latestBudgetSummary as BudgetSummary);
                        return newMap;
                      });
                      return [...prev];
                    });
                  }
                } else if (parsed.type === 'done') {
                  if (assistantMessage && !assistantMessage.includes('<|Final Answer|>')) {
                    setMessages((prev) => {
                      const newMessages = [...prev];
                      const lastMessage = newMessages[newMessages.length - 1];
                      let targetMessageIndex = -1;

                      if (lastMessage && lastMessage.role === 'assistant' && !lastMessage.content?.startsWith(TOOL_STATUS_PREFIX)) {
                        lastMessage.content = assistantMessage;
                        targetMessageIndex = newMessages.length - 1;
                      } else {
                        newMessages.push({
                          role: 'assistant',
                          content: assistantMessage,
                        });
                        targetMessageIndex = newMessages.length - 1;
                      }

                      if (latestCritic && targetMessageIndex >= 0) {
                        setMessageRetrievalCritic((prevCriticMap) => {
                          const newMap = new Map(prevCriticMap);
                          newMap.set(targetMessageIndex, latestCritic as RetrievalCritic);
                          return newMap;
                        });
                      }
                      if (latestReasoningStages.length > 0 && targetMessageIndex >= 0) {
                        setMessageReasoningStages((prevStageMap) => {
                          const newMap = new Map(prevStageMap);
                          newMap.set(targetMessageIndex, [...latestReasoningStages]);
                          return newMap;
                        });
                      }
                      if (latestBudgetSummary && targetMessageIndex >= 0) {
                        setMessageBudgetSummary((prevBudgetMap) => {
                          const newMap = new Map(prevBudgetMap);
                          newMap.set(targetMessageIndex, latestBudgetSummary as BudgetSummary);
                          return newMap;
                        });
                      }
                      return newMessages;
                    });
                  }
                  break;
                }
              } catch (e) {
                console.error('Failed to parse chunk:', e);
              }
            }
          }
        }
      };

      await processChunk();
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setMessageToolCalls(new Map());
    setMessageToolResults(new Map());
    setMessageReasoningStages(new Map());
    setMessageRetrievalCritic(new Map());
    setMessageBudgetSummary(new Map());
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="header-title">
            <h1>📳 Deep RAG</h1>
            <div className="header-nav">
              <button
                className={`nav-button ${activePage === 'chat' ? 'active' : ''}`}
                onClick={() => setActivePage('chat')}
                title="Chat"
              >
                <MessageSquare size={16} />
                <span>Chat</span>
              </button>
              <button
                className={`nav-button ${activePage === 'evaluation' ? 'active' : ''}`}
                onClick={() => setActivePage('evaluation')}
                title="Evaluation"
              >
                <BarChart3 size={16} />
                <span>测评</span>
              </button>
              <button
                className={`nav-button ${activePage === 'voice' ? 'active' : ''}`}
                onClick={() => setActivePage('voice')}
                title="Voice Ingestion"
              >
                <Mic size={16} />
                <span>语音入库</span>
              </button>
              <button
                className={`nav-button ${activePage === 'image' ? 'active' : ''}`}
                onClick={() => setActivePage('image')}
                title="Image Ingestion"
              >
                <ImageIcon size={16} />
                <span>图片入库</span>
              </button>
            </div>
          </div>
          {activePage === 'chat' && (
            <div className="header-actions">
              <button
                className="icon-button"
                onClick={() => setShowSystemPrompt(!showSystemPrompt)}
                title="System Prompt"
              >
                <FileCode size={20} />
              </button>
              <button
                className="icon-button"
                onClick={() => setShowConfigPanel(!showConfigPanel)}
                title="Configuration"
              >
                <Settings size={20} />
              </button>
            </div>
          )}
        </div>
      </header>

      <div className="main-content">
        {activePage === 'chat' ? (
          <>
            {showConfigPanel && (
              <ConfigPanel
                onClose={() => setShowConfigPanel(false)}
                onConfigUpdated={() => {
                  loadConfig();
                  loadProviders();
                }}
              />
            )}

            {showSystemPrompt && (
              <SystemPromptPanel onClose={() => setShowSystemPrompt(false)} />
            )}

            <div className="chat-container" ref={chatContainerRef}>
              {messages.length === 0 ? (
                <div className="welcome-screen">
                  <h2>Welcome to Deep RAG</h2>
                  <p>
                    Ask questions about your knowledge base and I'll help you find the answers.
                  </p>
                  <div className="example-questions">
                    <h3>Example Questions:</h3>
                    <ul>
                      <li onClick={() => handleSendMessage('What display types do we have besides AMOLED and OLED?')}>
                        What display types do we have besides AMOLED and OLED?
                      </li>
                      <li onClick={() => handleSendMessage('Which devices have waterproof ratings higher than IP67?')}>
                        Which devices have waterproof ratings higher than IP67?
                      </li>
                      <li onClick={() => handleSendMessage('Which Bluetooth audio device has the longest battery life?')}>
                        Which Bluetooth audio device has the longest battery life?
                      </li>
                      <li onClick={() => handleSendMessage('What was the total number of retail stores nationwide last year?')}>
                        What was the total number of retail stores nationwide last year?
                      </li>
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="messages">
                  {messages.map((message, index) => (
                    <ChatMessage
                      key={index}
                      message={message}
                      toolCalls={messageToolCalls.get(index)}
                      toolResults={messageToolResults.get(index)}
                      reasoningStages={messageReasoningStages.get(index)}
                      retrievalCritic={messageRetrievalCritic.get(index)}
                      budgetSummary={messageBudgetSummary.get(index)}
                    />
                  ))}
                  {isLoading && (
                    <div className="loading-indicator">
                      <Loader2 className="spinner" size={20} />
                      <span>Thinking...</span>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            <div className="input-container">
              <div className="budget-toolbar">
                <button
                  className={`budget-toggle ${showBudgetPanel ? 'active' : ''}`}
                  onClick={() => setShowBudgetPanel((prev) => !prev)}
                  type="button"
                >
                  Budget 设置
                </button>
                <span className="budget-toolbar-hint">
                  按问题控制 token/时延，并显示成本估算
                </span>
              </div>
              {showBudgetPanel && (
                <div className="budget-panel">
                  <label className="budget-field">
                    <span>Token 上限</span>
                    <input
                      type="number"
                      min={0}
                      step={1000}
                      value={Math.max(0, Math.floor(toNumber(budgetConfig.max_total_tokens, 0)))}
                      onChange={(e) => updateBudgetConfig('max_total_tokens', e.target.value)}
                    />
                  </label>
                  <label className="budget-field">
                    <span>时延上限(ms)</span>
                    <input
                      type="number"
                      min={0}
                      step={1000}
                      value={Math.max(0, Math.floor(toNumber(budgetConfig.max_latency_ms, 0)))}
                      onChange={(e) => updateBudgetConfig('max_latency_ms', e.target.value)}
                    />
                  </label>
                  <label className="budget-field">
                    <span>1M Token 单价($)</span>
                    <input
                      type="number"
                      min={0}
                      step={0.01}
                      value={Math.max(0, toNumber(budgetConfig.price_per_1m_tokens, 0))}
                      onChange={(e) => updateBudgetConfig('price_per_1m_tokens', e.target.value, { float: true })}
                    />
                  </label>
                  <label className="budget-field">
                    <span>倍率</span>
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      value={Math.max(0, toNumber(budgetConfig.cost_multiplier, 1))}
                      onChange={(e) => updateBudgetConfig('cost_multiplier', e.target.value, { float: true })}
                    />
                  </label>
                </div>
              )}
              <div className="input-wrapper">
                {messages.length > 0 && (
                  <button
                    className="clear-button"
                    onClick={handleClearChat}
                    title="Clear chat"
                  >
                    Clear
                  </button>
                )}
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  onFocus={() => setTimeout(scrollToBottom, 100)}
                  placeholder="Ask a question about your knowledge base..."
                  rows={1}
                  disabled={isLoading}
                />
                <button
                  onClick={() => handleSendMessage()}
                  disabled={!inputValue.trim() || isLoading}
                  className="send-button"
                >
                  {isLoading ? <Loader2 className="spinner" size={20} /> : <Send size={20} />}
                </button>
              </div>
            </div>
          </>
        ) : activePage === 'evaluation' ? (
          <div className="chat-container">
            <EvaluationPanel />
          </div>
        ) : activePage === 'voice' ? (
          <div className="chat-container">
            <VoiceErrorBoundary>
              <VoiceIngestionPanel provider={selectedProvider} />
            </VoiceErrorBoundary>
          </div>
        ) : (
          <div className="chat-container">
            <ImageIngestionPanel provider={selectedProvider} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
