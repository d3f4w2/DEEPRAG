import axios from 'axios';
import type {
  ChatRequest,
  Provider,
  KnowledgeBaseInfo,
  EvaluationDataset,
  EvaluationSummaryItem,
  EvaluationStartRequest,
  EvaluationJob,
  VoiceDraftRequest,
  VoiceDraftResponse,
  VoiceIngestRequest,
  VoiceIngestResponse,
  ImageDraftResponse,
  ImageIngestResponse,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const chatApi = {
  sendMessage: async (request: ChatRequest) => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return response;
  },

  getConfig: async (): Promise<{ default_provider: string; default_model: string }> => {
    const response = await api.get('/config');
    return response.data;
  },

  getProviders: async (): Promise<{ providers: Provider[] }> => {
    const response = await api.get('/providers');
    return response.data;
  },

  getKnowledgeBaseInfo: async (): Promise<KnowledgeBaseInfo> => {
    const response = await api.get('/knowledge-base/info');
    return response.data;
  },

  getSystemPrompt: async (): Promise<{ system_prompt: string; mode: string }> => {
    const response = await api.get('/system-prompt');
    return response.data;
  },

  retrieveFiles: async (filePaths: string[]): Promise<{ content: string }> => {
    const response = await api.post('/knowledge-base/retrieve', {
      file_paths: filePaths,
    });
    return response.data;
  },
};

export const evaluationApi = {
  listDatasets: async (): Promise<{ datasets: EvaluationDataset[] }> => {
    const response = await api.get('/evaluation/datasets');
    return response.data;
  },

  listSummaries: async (): Promise<{ summaries: EvaluationSummaryItem[] }> => {
    const response = await api.get('/evaluation/summaries');
    return response.data;
  },

  listJobs: async (): Promise<{ jobs: EvaluationJob[] }> => {
    const response = await api.get('/evaluation/jobs');
    return response.data;
  },

  getJob: async (jobId: string): Promise<EvaluationJob> => {
    const response = await api.get(`/evaluation/jobs/${jobId}`);
    return response.data;
  },

  startJob: async (request: EvaluationStartRequest): Promise<EvaluationJob> => {
    const response = await api.post('/evaluation/jobs/start', request);
    return response.data;
  },
};

export const voiceApi = {
  draft: async (request: VoiceDraftRequest): Promise<VoiceDraftResponse> => {
    const response = await api.post('/voice/draft', request);
    return response.data;
  },

  ingest: async (request: VoiceIngestRequest): Promise<VoiceIngestResponse> => {
    const response = await api.post('/voice/ingest', request);
    return response.data;
  },
};

export const imageApi = {
  draft: async (formData: FormData): Promise<ImageDraftResponse> => {
    const response = await api.post('/image/draft', formData);
    return response.data;
  },

  ingest: async (formData: FormData): Promise<ImageIngestResponse> => {
    const response = await api.post('/image/ingest', formData);
    return response.data;
  },
};

export default api;
