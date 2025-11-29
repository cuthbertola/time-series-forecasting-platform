import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Dataset APIs
export const uploadDataset = async (file: File, name: string, description?: string, dateColumn?: string, targetColumn?: string) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', name);
  if (description) formData.append('description', description);
  if (dateColumn) formData.append('date_column', dateColumn);
  if (targetColumn) formData.append('target_column', targetColumn);

  const response = await api.post('/datasets/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const getDatasets = async () => {
  const response = await api.get('/datasets/');
  return response.data;
};

export const getDataset = async (id: number) => {
  const response = await api.get(`/datasets/${id}`);
  return response.data;
};

export const previewDataset = async (id: number, rows: number = 10) => {
  const response = await api.get(`/datasets/${id}/preview?rows=${rows}`);
  return response.data;
};

export const deleteDataset = async (id: number) => {
  const response = await api.delete(`/datasets/${id}`);
  return response.data;
};

// Training APIs
export const runAutoML = async (params: {
  dataset_id: number;
  target_column: string;
  date_column: string;
  feature_columns?: string[];
  forecast_horizon?: number;
  max_trials?: number;
  timeout_seconds?: number;
  algorithms?: string[];
}) => {
  const response = await api.post('/training/automl', params);
  return response.data;
};

export const getAutoMLRun = async (runId: number) => {
  const response = await api.get(`/training/automl/${runId}`);
  return response.data;
};

export const getModels = async (datasetId?: number) => {
  const url = datasetId ? `/training/models?dataset_id=${datasetId}` : '/training/models';
  const response = await api.get(url);
  return response.data;
};

export const getModel = async (modelId: number) => {
  const response = await api.get(`/training/models/${modelId}`);
  return response.data;
};

export const compareModels = async (datasetId: number) => {
  const response = await api.get(`/training/models/compare/${datasetId}`);
  return response.data;
};

// Forecast APIs
export const generateForecast = async (params: {
  model_id: number;
  forecast_horizon: number;
  confidence_level?: number;
}) => {
  const response = await api.post('/forecast/', params);
  return response.data;
};

export const getForecast = async (forecastId: number) => {
  const response = await api.get(`/forecast/${forecastId}`);
  return response.data;
};

export const getForecasts = async (datasetId: number) => {
  const response = await api.get(`/forecast/dataset/${datasetId}`);
  return response.data;
};

// Health check
export const healthCheck = async () => {
  const response = await axios.get('http://localhost:8001/health');
  return response.data;
};

export default api;
