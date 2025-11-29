import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp, Download, Upload, FileSpreadsheet } from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  Legend, ResponsiveContainer, ComposedChart, Area 
} from 'recharts';
import { 
  getModels, generateForecast, getForecastCsvUrl, 
  getBatchTemplateUrl, uploadBatchPrediction 
} from '../services/api';

interface Model {
  id: number;
  name: string;
  algorithm: string;
  mape: number | null;
  status: string;
}

interface PredictionPoint {
  date: string;
  value: number;
  lower_bound: number | null;
  upper_bound: number | null;
}

interface Forecast {
  id: number;
  forecast_horizon: number;
  confidence_level: number;
  predictions: PredictionPoint[];
}

const Forecasts: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<number | null>(null);
  const [forecastHorizon, setForecastHorizon] = useState(30);
  const [confidenceLevel, setConfidenceLevel] = useState(0.95);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'generate' | 'batch'>('generate');
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchDateColumn, setBatchDateColumn] = useState('date');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const data = await getModels();
      setModels(data.filter((m: Model) => m.status === 'completed'));
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const handleGenerateForecast = async () => {
    if (!selectedModel) return;
    
    setLoading(true);
    setError(null);
    try {
      const result = await generateForecast({
        model_id: selectedModel,
        forecast_horizon: forecastHorizon,
        confidence_level: confidenceLevel,
      });
      setForecast(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Forecast generation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleExportCsv = () => {
    if (forecast) {
      window.open(getForecastCsvUrl(forecast.id), '_blank');
    }
  };

  const handleDownloadTemplate = () => {
    window.open(getBatchTemplateUrl(), '_blank');
  };

  const handleBatchFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setBatchFile(e.target.files[0]);
    }
  };

  const handleBatchPredict = async () => {
    if (!selectedModel || !batchFile) return;
    
    setLoading(true);
    setError(null);
    try {
      const blob = await uploadBatchPrediction(
        selectedModel, 
        batchFile, 
        batchDateColumn, 
        confidenceLevel
      );
      
      // Download the result
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `batch_predictions_${selectedModel}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      setBatchFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Batch prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const chartData = forecast?.predictions.map(p => ({
    date: p.date,
    forecast: p.value,
    lower: p.lower_bound,
    upper: p.upper_bound,
  })) || [];

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Forecasts</h1>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-500 hover:text-red-700">
            &times;
          </button>
        </div>
      )}

      {models.length === 0 ? (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-lg">
          No trained models available. Please train a model first.
        </div>
      ) : (
        <>
          {/* Tab Navigation */}
          <div className="bg-white rounded-lg shadow">
            <div className="border-b border-gray-200">
              <nav className="flex -mb-px">
                <button
                  onClick={() => setActiveTab('generate')}
                  className={`px-6 py-3 border-b-2 font-medium text-sm ${
                    activeTab === 'generate'
                      ? 'border-purple-500 text-purple-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <TrendingUp className="w-4 h-4 inline mr-2" />
                  Generate Forecast
                </button>
                <button
                  onClick={() => setActiveTab('batch')}
                  className={`px-6 py-3 border-b-2 font-medium text-sm ${
                    activeTab === 'batch'
                      ? 'border-purple-500 text-purple-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <FileSpreadsheet className="w-4 h-4 inline mr-2" />
                  Batch Predictions
                </button>
              </nav>
            </div>

            <div className="p-6">
              {/* Generate Forecast Tab */}
              {activeTab === 'generate' && (
                <div>
                  <div className="flex items-center space-x-2 mb-6">
                    <TrendingUp className="w-6 h-6 text-purple-600" />
                    <h2 className="text-xl font-semibold">Generate New Forecast</h2>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Model *
                      </label>
                      <select
                        value={selectedModel || ''}
                        onChange={(e) => setSelectedModel(Number(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                      >
                        <option value="">Select a model</option>
                        {models.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.name} ({model.algorithm.toUpperCase()}) - MAPE: {model.mape ? `${(model.mape * 100).toFixed(2)}%` : 'N/A'}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Forecast Horizon (days)
                      </label>
                      <input
                        type="number"
                        value={forecastHorizon}
                        onChange={(e) => setForecastHorizon(Number(e.target.value))}
                        min={1}
                        max={365}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Confidence Level
                      </label>
                      <select
                        value={confidenceLevel}
                        onChange={(e) => setConfidenceLevel(Number(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                      >
                        <option value={0.90}>90%</option>
                        <option value={0.95}>95%</option>
                        <option value={0.99}>99%</option>
                      </select>
                    </div>
                  </div>

                  <button
                    onClick={handleGenerateForecast}
                    disabled={!selectedModel || loading}
                    className={`w-full py-3 rounded-lg font-medium flex items-center justify-center space-x-2 ${
                      !selectedModel || loading
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-purple-600 text-white hover:bg-purple-700'
                    }`}
                  >
                    {loading ? (
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    ) : (
                      <>
                        <TrendingUp className="w-5 h-5" />
                        <span>Generate Forecast</span>
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Batch Predictions Tab */}
              {activeTab === 'batch' && (
                <div>
                  <div className="flex items-center space-x-2 mb-6">
                    <FileSpreadsheet className="w-6 h-6 text-purple-600" />
                    <h2 className="text-xl font-semibold">Batch Predictions</h2>
                  </div>

                  <div className="mb-4">
                    <button
                      onClick={handleDownloadTemplate}
                      className="text-purple-600 hover:text-purple-700 text-sm flex items-center"
                    >
                      <Download className="w-4 h-4 mr-1" />
                      Download CSV Template
                    </button>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Model *
                      </label>
                      <select
                        value={selectedModel || ''}
                        onChange={(e) => setSelectedModel(Number(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                      >
                        <option value="">Select a model</option>
                        {models.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.name} ({model.algorithm.toUpperCase()})
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Date Column Name
                      </label>
                      <input
                        type="text"
                        value={batchDateColumn}
                        onChange={(e) => setBatchDateColumn(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                        placeholder="e.g., date"
                      />
                    </div>
                  </div>

                  <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Upload CSV with Future Dates
                    </label>
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv"
                        onChange={handleBatchFileChange}
                        className="hidden"
                        id="batch-file"
                      />
                      <label htmlFor="batch-file" className="cursor-pointer">
                        <Upload className="w-10 h-10 text-gray-400 mx-auto mb-2" />
                        <p className="text-gray-600">
                          {batchFile ? batchFile.name : 'Click to upload CSV file'}
                        </p>
                      </label>
                    </div>
                  </div>

                  <button
                    onClick={handleBatchPredict}
                    disabled={!selectedModel || !batchFile || loading}
                    className={`w-full py-3 rounded-lg font-medium flex items-center justify-center space-x-2 ${
                      !selectedModel || !batchFile || loading
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-purple-600 text-white hover:bg-purple-700'
                    }`}
                  >
                    {loading ? (
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    ) : (
                      <>
                        <FileSpreadsheet className="w-5 h-5" />
                        <span>Generate Batch Predictions</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Forecast Results */}
          {forecast && activeTab === 'generate' && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold">Forecast Results</h2>
                <button
                  onClick={handleExportCsv}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                >
                  <Download className="w-4 h-4" />
                  <span>Export CSV</span>
                </button>
              </div>

              {/* Metrics Cards */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-purple-50 rounded-lg p-4">
                  <p className="text-sm text-purple-600">Forecast Horizon</p>
                  <p className="text-2xl font-bold text-purple-700">{forecast.forecast_horizon} days</p>
                </div>
                <div className="bg-blue-50 rounded-lg p-4">
                  <p className="text-sm text-blue-600">Confidence Level</p>
                  <p className="text-2xl font-bold text-blue-700">{(forecast.confidence_level * 100).toFixed(0)}%</p>
                </div>
                <div className="bg-green-50 rounded-lg p-4">
                  <p className="text-sm text-green-600">Predictions</p>
                  <p className="text-2xl font-bold text-green-700">{forecast.predictions.length} points</p>
                </div>
              </div>

              {/* Chart */}
              <div className="h-80 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="upper"
                      fill="#E9D5FF"
                      stroke="none"
                      name="Upper Bound"
                    />
                    <Area
                      type="monotone"
                      dataKey="lower"
                      fill="#FFFFFF"
                      stroke="none"
                      name="Lower Bound"
                    />
                    <Line
                      type="monotone"
                      dataKey="forecast"
                      stroke="#7C3AED"
                      strokeWidth={2}
                      dot={false}
                      name="Forecast"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Predictions Table */}
              <div>
                <h3 className="font-semibold mb-3">Prediction Details</h3>
                <div className="overflow-x-auto max-h-64 overflow-y-auto">
                  <table className="w-full">
                    <thead className="sticky top-0 bg-white">
                      <tr className="text-left text-gray-500 text-sm border-b">
                        <th className="pb-2 font-medium">Date</th>
                        <th className="pb-2 font-medium">Forecast</th>
                        <th className="pb-2 font-medium">Lower Bound</th>
                        <th className="pb-2 font-medium">Upper Bound</th>
                      </tr>
                    </thead>
                    <tbody>
                      {forecast.predictions.map((pred, index) => (
                        <tr key={index} className="border-b last:border-0">
                          <td className="py-2">{pred.date}</td>
                          <td className="py-2">{pred.value.toFixed(2)}</td>
                          <td className="py-2">{pred.lower_bound?.toFixed(2) || 'N/A'}</td>
                          <td className="py-2">{pred.upper_bound?.toFixed(2) || 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Forecasts;
