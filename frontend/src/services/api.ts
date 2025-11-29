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

export const previewDataset = async (id: number) => {
  const response = await api.get(`/datasets/${id}/preview`);
  return response.data;
};

export const deleteDataset = async (id: number) => {
  const response = await api.delete(`/datasets/${id}`);
  return response.data;
};

// Training APIs
export const runAutoML = async (config: {
  dataset_id: number;
  target_column: string;
  date_column: string;
  feature_columns?: string[];
  forecast_horizon?: number;
  algorithms?: string[];
  max_trials?: number;
  timeout_seconds?: number;
}) => {
  const response = await api.post('/training/automl', config);
  return response.data;
};

export const getAutoMLRun = async (runId: number) => {
  const response = await api.get(`/training/automl/${runId}`);
  return response.data;
};

export const getModels = async (datasetId?: number) => {
  const params = datasetId ? { dataset_id: datasetId } : {};
  const response = await api.get('/training/models', { params });
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
export const generateForecast = async (config: {
  model_id: number;
  forecast_horizon: number;
  confidence_level: number;
}) => {
  const response = await api.post('/forecast/', config);
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

// Visualization APIs
export const getHistoricalData = async (datasetId: number) => {
  const response = await api.get(`/visualization/historical/${datasetId}`);
  return response.data;
};

export const getDatasetStatistics = async (datasetId: number) => {
  const response = await api.get(`/visualization/statistics/${datasetId}`);
  return response.data;
};

export const getSeasonalityAnalysis = async (datasetId: number) => {
  const response = await api.get(`/visualization/seasonality/${datasetId}`);
  return response.data;
};

// Export APIs
export const getModelExportUrl = (modelId: number) => {
  return `${API_BASE_URL}/export/model/${modelId}`;
};

export const getModelMetadata = async (modelId: number) => {
  const response = await api.get(`/export/model/${modelId}/metadata`);
  return response.data;
};

export const getModelPackageUrl = (modelId: number) => {
  return `${API_BASE_URL}/export/model/${modelId}/package`;
};

export const getForecastCsvUrl = (forecastId: number) => {
  return `${API_BASE_URL}/export/forecast/${forecastId}/csv`;
};

// Batch Prediction APIs
export const getBatchTemplateUrl = () => {
  return `${API_BASE_URL}/batch/template`;
};

export const uploadBatchPrediction = async (modelId: number, file: File, dateColumn: string, confidenceLevel: number = 0.95) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('date_column', dateColumn);
  formData.append('confidence_level', confidenceLevel.toString());
  
  const response = await api.post(`/batch/predict/${modelId}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob'
  });
  return response.data;
};

// Health check
export const healthCheck = async () => {
  const response = await axios.get('http://localhost:8001/health');
  return response.data;
};

export default api;
