export interface Dataset {
  id: number;
  name: string;
  description?: string;
  filename: string;
  row_count?: number;
  column_count?: number;
  date_column?: string;
  target_column?: string;
  feature_columns?: string[];
  frequency?: string;
  start_date?: string;
  end_date?: string;
  status: 'pending' | 'processing' | 'ready' | 'failed';
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export interface DatasetListResponse {
  datasets: Dataset[];
  total: number;
}

export interface TrainedModel {
  id: number;
  dataset_id: number;
  name: string;
  algorithm: string;
  hyperparameters?: Record<string, any>;
  feature_importance?: Record<string, number>;
  mape?: number;
  rmse?: number;
  mae?: number;
  r2_score?: number;
  training_time_seconds?: number;
  cv_scores?: number[];
  mlflow_run_id?: string;
  status: 'training' | 'completed' | 'failed' | 'deployed';
  is_best_model: boolean;
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export interface AutoMLRun {
  id: number;
  dataset_id: number;
  algorithms_tested: string[];
  best_algorithm?: string;
  best_model_id?: number;
  all_results?: Array<{
    algorithm: string;
    status: string;
    mape?: number;
    training_time?: number;
  }>;
  status: string;
  total_time_seconds?: number;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

export interface PredictionPoint {
  date: string;
  value: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface Forecast {
  id: number;
  dataset_id: number;
  model_id: number;
  forecast_horizon: number;
  confidence_level: number;
  predictions: PredictionPoint[];
  created_at: string;
}

export interface ModelComparison {
  models: TrainedModel[];
  best_model_id: number;
  comparison_metrics: Record<string, {
    mape?: number;
    rmse?: number;
    mae?: number;
    r2?: number;
    training_time?: number;
  }>;
}

export interface DatasetPreview {
  columns: string[];
  dtypes: Record<string, string>;
  data: Record<string, any>[];
  total_rows: number;
}
