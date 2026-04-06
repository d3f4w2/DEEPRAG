import { useEffect, useMemo, useRef, useState } from 'react';
import { Mic, RefreshCw, Save } from 'lucide-react';
import { voiceApi } from '../api';

import './VoiceIngestionPanel.css';

type VoiceIngestionPanelProps = {
  provider: string;
};

type RecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

const getRecognitionCtor = (): (new () => RecognitionLike) | null => {
  const win = window as any;
  return win.SpeechRecognition || win.webkitSpeechRecognition || null;
};

const nowIsoLocal = () => new Date().toISOString();

function VoiceIngestionPanel({ provider }: VoiceIngestionPanelProps) {
  const [author, setAuthor] = useState('Unknown');
  const [source, setSource] = useState('Realtime microphone');
  const [speechSupported, setSpeechSupported] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [isDrafting, setIsDrafting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const [rawTranscript, setRawTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');

  const [aiDraftText, setAiDraftText] = useState('');
  const [aiSummary, setAiSummary] = useState('');
  const [draftWarning, setDraftWarning] = useState('');

  const [finalTranscript, setFinalTranscript] = useState('');
  const [finalSummary, setFinalSummary] = useState('');

  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const recognitionRef = useRef<RecognitionLike | null>(null);
  const draftReqIdRef = useRef(0);
  const textEditedRef = useRef(false);
  const summaryEditedRef = useRef(false);

  const liveTranscript = useMemo(() => {
    return [rawTranscript, interimTranscript].filter(Boolean).join(' ').trim();
  }, [rawTranscript, interimTranscript]);

  const requestDraft = async (transcript: string, forceApply = false, silent = false) => {
    const normalized = transcript.trim();
    if (!normalized) {
      return;
    }

    const reqId = ++draftReqIdRef.current;
    if (!silent) {
      setIsDrafting(true);
    }
    setErrorMessage('');

    try {
      const result = await voiceApi.draft({
        transcript: normalized,
        author: author.trim(),
        provider: provider || undefined,
      });

      if (reqId !== draftReqIdRef.current) {
        return;
      }

      setAiDraftText(result.polished_text || '');
      setAiSummary(result.summary || '');
      setDraftWarning(result.warning || '');

      if (forceApply || !textEditedRef.current) {
        setFinalTranscript(result.polished_text || '');
      }
      if (forceApply || !summaryEditedRef.current) {
        setFinalSummary(result.summary || '');
      }
    } catch (error) {
      if (reqId !== draftReqIdRef.current) {
        return;
      }
      const detail = error instanceof Error ? error.message : 'Failed to generate voice draft.';
      setErrorMessage(detail);
    } finally {
      if (!silent && reqId === draftReqIdRef.current) {
        setIsDrafting(false);
      }
    }
  };

  useEffect(() => {
    const Ctor = getRecognitionCtor();
    if (!Ctor) {
      setSpeechSupported(false);
      return;
    }

    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'zh-CN';
    recognition.onstart = () => {
      setIsRecording(true);
    };

    recognition.onresult = (event: any) => {
      try {
        const results = event?.results;
        if (!results || typeof results.length !== 'number') {
          return;
        }

        const safeStart = Number.isInteger(event?.resultIndex) ? Math.max(0, event.resultIndex) : 0;
        let finalPart = '';
        let interimPart = '';

        for (let i = safeStart; i < results.length; i += 1) {
          const current = results?.[i];
          const segment = String(current?.[0]?.transcript ?? '').trim();
          if (!segment) {
            continue;
          }
          if (Boolean(current?.isFinal)) {
            finalPart += ` ${segment}`;
          } else {
            interimPart += ` ${segment}`;
          }
        }

        if (finalPart.trim()) {
          setRawTranscript((prev) => `${prev} ${finalPart}`.trim());
        }
        setInterimTranscript(interimPart.trim());
      } catch (error) {
        console.error('Speech recognition result parse error:', error);
        setErrorMessage('Speech recognition parse error.');
      }
    };

    recognition.onerror = (event: any) => {
      setErrorMessage(`Speech recognition error: ${event?.error || 'unknown'}`);
      setIsRecording(false);
    };

    recognition.onend = () => {
      setIsRecording(false);
      setInterimTranscript('');
    };

    recognitionRef.current = recognition;
    return () => {
      try {
        recognition.stop();
      } catch (_e) {
        // ignore stop error during unmount
      }
      recognitionRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!liveTranscript) {
      return;
    }
    const timer = window.setTimeout(() => {
      void requestDraft(liveTranscript, false, true);
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [liveTranscript, author, provider]);

  const handleStartRecording = () => {
    if (!recognitionRef.current) {
      return;
    }
    setErrorMessage('');
    setStatusMessage('');
    try {
      recognitionRef.current.start();
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to start recording.';
      setErrorMessage(detail);
    }
  };

  const handleStopRecording = () => {
    if (!recognitionRef.current) {
      return;
    }
    try {
      recognitionRef.current.stop();
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to stop recording.';
      setErrorMessage(detail);
    } finally {
      setIsRecording(false);
    }
  };

  const handleUseLatestDraft = () => {
    textEditedRef.current = false;
    summaryEditedRef.current = false;
    setFinalTranscript(aiDraftText || liveTranscript);
    setFinalSummary(aiSummary || '');
  };

  const handleClear = () => {
    handleStopRecording();
    setRawTranscript('');
    setInterimTranscript('');
    setAiDraftText('');
    setAiSummary('');
    setFinalTranscript('');
    setFinalSummary('');
    setDraftWarning('');
    setErrorMessage('');
    setStatusMessage('');
    textEditedRef.current = false;
    summaryEditedRef.current = false;
  };

  const handleIngest = async () => {
    const finalText = finalTranscript.trim();
    const finalBrief = finalSummary.trim();
    const finalAuthor = author.trim();

    if (!finalText) {
      setErrorMessage('Please confirm transcript text before ingestion.');
      return;
    }
    if (!finalBrief) {
      setErrorMessage('Please confirm summary before ingestion.');
      return;
    }
    if (!finalAuthor) {
      setErrorMessage('Please set an author for this voice note.');
      return;
    }

    setIsSaving(true);
    setErrorMessage('');
    setStatusMessage('');
    try {
      const saved = await voiceApi.ingest({
        transcript: finalText,
        summary: finalBrief,
        author: finalAuthor,
        source: source.trim() || 'Realtime microphone',
        raw_transcript: liveTranscript,
        occurred_at: nowIsoLocal(),
      });
      setStatusMessage(`Saved to knowledge base: ${saved.file_path}`);
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to ingest voice note.';
      setErrorMessage(detail);
    } finally {
      setIsSaving(false);
    }
  };

  if (!speechSupported) {
    return (
      <div className="voice-page">
        <div className="voice-card">
          <h2>语音入库</h2>
          <p>当前浏览器不支持实时语音识别，请使用 Chrome/Edge 并允许麦克风权限。</p>
        </div>
      </div>
    );
  }

  return (
    <div className="voice-page" translate="no">
      <div className="voice-card voice-doc">
        <h2>语音资料入库（实时）</h2>
        <p>流程：实时语音转写 → LLM实时生成草稿与摘要 → 人工纠正 → 确认入库。</p>
      </div>

      <div className="voice-card voice-controls">
        <div className="voice-field">
          <label htmlFor="voice-author">作者（谁说的）</label>
          <input
            id="voice-author"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            placeholder="e.g. Alice"
          />
        </div>
        <div className="voice-field">
          <label htmlFor="voice-source">来源</label>
          <input
            id="voice-source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="e.g. Team Meeting / Phone Call"
          />
        </div>
        <div className="voice-buttons">
          <button
            className={`voice-action ${isRecording ? 'stop' : 'start'}`}
            onClick={isRecording ? handleStopRecording : handleStartRecording}
          >
            <Mic size={16} />
            <span className={`voice-rec-dot ${isRecording ? 'on' : 'off'}`} aria-hidden />
            <span>录音开/关</span>
          </button>
          <button
            className="voice-action secondary"
            onClick={() => void requestDraft(liveTranscript, true)}
            disabled={!liveTranscript || isDrafting}
          >
            <RefreshCw size={16} />
            <span>同步AI草稿</span>
          </button>
          <button className="voice-action secondary" onClick={handleClear}>
            清空
          </button>
        </div>
      </div>

      <div className="voice-grid">
        <div className="voice-card">
          <h3>实时转写（原始）</h3>
          <textarea value={liveTranscript} readOnly placeholder="录音后会实时显示转写文本..." />
        </div>

        <div className="voice-card">
          <h3>LLM实时草稿（建议）</h3>
          <textarea value={aiDraftText} readOnly placeholder="AI will draft polished text here..." />
          <label className="voice-sub-label">AI摘要（建议）</label>
          <textarea className="voice-summary" value={aiSummary} readOnly placeholder="AI summary..." />
          {draftWarning && <div className="voice-warning">{draftWarning}</div>}
        </div>

        <div className="voice-card voice-confirm">
          <h3>确认入库（可编辑）</h3>
          <textarea
            value={finalTranscript}
            onChange={(e) => {
              textEditedRef.current = true;
              setFinalTranscript(e.target.value);
            }}
            placeholder="请确认并纠正文本后再入库..."
          />
          <label className="voice-sub-label">确认摘要（可编辑）</label>
          <textarea
            className="voice-summary"
            value={finalSummary}
            onChange={(e) => {
              summaryEditedRef.current = true;
              setFinalSummary(e.target.value);
            }}
            placeholder="请确认摘要后再入库..."
          />
          <div className="voice-confirm-actions">
            <button className="voice-action secondary" onClick={handleUseLatestDraft} disabled={!aiDraftText}>
              使用最新AI草稿
            </button>
            <button className="voice-action primary" onClick={handleIngest} disabled={isSaving}>
              <Save size={16} />
              <span>确认入库</span>
            </button>
          </div>
        </div>
      </div>

      {statusMessage && <div className="voice-status ok">{statusMessage}</div>}
      {errorMessage && <div className="voice-status error">{errorMessage}</div>}
    </div>
  );
}

export default VoiceIngestionPanel;
