import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Settings, FileCode, MessageSquare, BarChart3, Mic } from 'lucide-react';
import { chatApi } from './api';
import type { Message, RetrievalJudge, StreamChunk } from './types';
import ChatMessage from './components/ChatMessage';
import SystemPromptPanel from './components/SystemPromptPanel';
import ConfigPanel from './components/ConfigPanel';
import EvaluationPanel from './components/EvaluationPanel';
import VoiceIngestionPanel from './components/VoiceIngestionPanel';
import VoiceErrorBoundary from './components/VoiceErrorBoundary';
import './App.css';

const MAX_CONTEXT_MESSAGES = 12;
const TOOL_STATUS_PREFIX = '[TOOL]';

function App() {
  const [activePage, setActivePage] = useState<'chat' | 'evaluation' | 'voice'>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSystemPrompt, setShowSystemPrompt] = useState(false);
  const [showConfigPanel, setShowConfigPanel] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [messageToolCalls, setMessageToolCalls] = useState<Map<number, any[]>>(new Map());
  const [messageToolResults, setMessageToolResults] = useState<Map<number, any[]>>(new Map());
  const [messageRetrievalJudge, setMessageRetrievalJudge] = useState<Map<number, RetrievalJudge>>(new Map());

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadConfig();
    loadProviders();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

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
      let latestJudge: RetrievalJudge | null = null;

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

                      if (latestJudge && targetMessageIndex >= 0) {
                        setMessageRetrievalJudge((prevJudgeMap) => {
                          const newMap = new Map(prevJudgeMap);
                          newMap.set(targetMessageIndex, latestJudge as RetrievalJudge);
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
                } else if (parsed.type === 'retrieval_judge') {
                  latestJudge = {
                    stop: !!parsed.stop,
                    reason: parsed.reason || '',
                  };
                  setMessages((prev) => {
                    const messageIndex = prev.length - 1;
                    if (messageIndex < 0) {
                      return prev;
                    }
                    setMessageRetrievalJudge((prevJudgeMap) => {
                      const newMap = new Map(prevJudgeMap);
                      newMap.set(messageIndex, {
                        stop: latestJudge?.stop || false,
                        reason: latestJudge?.reason || '',
                      });
                      return newMap;
                    });
                    return [...prev];
                  });
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

                      if (latestJudge && targetMessageIndex >= 0) {
                        setMessageRetrievalJudge((prevJudgeMap) => {
                          const newMap = new Map(prevJudgeMap);
                          newMap.set(targetMessageIndex, latestJudge as RetrievalJudge);
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
                      retrievalJudge={messageRetrievalJudge.get(index)}
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
        ) : (
          <div className="chat-container">
            <VoiceErrorBoundary>
              <VoiceIngestionPanel provider={selectedProvider} />
            </VoiceErrorBoundary>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
