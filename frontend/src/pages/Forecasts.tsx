import React, { useEffect, useState } from 'react';
import { TrendingUp, Loader, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, ComposedChart } from 'recharts';
import { getModels, generateForecast, getForecast } from '../services/api';
import { TrainedModel, Forecast, PredictionPoint } from '../types';

const Forecasts: React.FC = () => {
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [forecasts, setForecasts] = useState<Forecast[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedForecast, setSelectedForecast] = useState<Forecast | null>(null);

  const [formData, setFormData] = useState({
    modelId: '',
    forecastHorizon: 30,
    confidenceLevel: 0.95,
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const modelsRes = await getModels();
        setModels((modelsRes || []).filter((m: TrainedModel) => m.status === 'completed'));
      } catch (err) {
        setError('Failed to fetch models');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const handleGenerateForecast = async () => {
    if (!formData.modelId) {
      setError('Please select a model');
      return;
    }

    try {
      setGenerating(true);
      setError(null);
      const forecast = await generateForecast({
        model_id: parseInt(formData.modelId),
        forecast_horizon: formData.forecastHorizon,
        confidence_level: formData.confidenceLevel,
      });
      setSelectedForecast(forecast);
      setForecasts(prev => [forecast, ...prev]);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate forecast');
    } finally {
      setGenerating(false);
    }
  };

  const formatChartData = (predictions: PredictionPoint[]) => {
    return predictions.map(p => ({
      date: new Date(p.date).toLocaleDateString(),
      value: p.value,
      lower: p.lower_bound,
      upper: p.upper_bound,
    }));
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Forecasts</h1>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
          <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
          <span className="text-red-700">{error}</span>
          <button onClick={() => setError(null)} className="ml-auto">×</button>
        </div>
      )}

      {/* Generate Forecast Form */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="h-6 w-6 mr-2 text-purple-600" />
          Generate New Forecast
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model *</label>
            <select
              value={formData.modelId}
              onChange={(e) => setFormData(prev => ({ ...prev, modelId: e.target.value }))}
              className="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-purple-500"
            >
              <option value="">Select a model</option>
              {models.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.algorithm.toUpperCase()}) - MAPE: {m.mape?.toFixed(2)}%
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Forecast Horizon (days)</label>
            <input
              type="number"
              value={formData.forecastHorizon}
              onChange={(e) => setFormData(prev => ({ ...prev, forecastHorizon: parseInt(e.target.value) }))}
              min={1}
              max={365}
              className="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-purple-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Confidence Level</label>
            <select
              value={formData.confidenceLevel}
              onChange={(e) => setFormData(prev => ({ ...prev, confidenceLevel: parseFloat(e.target.value) }))}
              className="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-purple-500"
            >
              <option value={0.90}>90%</option>
              <option value={0.95}>95%</option>
              <option value={0.99}>99%</option>
            </select>
          </div>
        </div>

        <button
          onClick={handleGenerateForecast}
          disabled={generating || !formData.modelId}
          className="mt-4 w-full bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center justify-center"
        >
          {generating ? (
            <><Loader className="h-5 w-5 animate-spin mr-2" /> Generating...</>
          ) : (
            <><TrendingUp className="h-5 w-5 mr-2" /> Generate Forecast</>
          )}
        </button>
      </div>

      {/* Forecast Visualization */}
      {selectedForecast && selectedForecast.predictions && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Forecast Results</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-purple-50 rounded-lg p-4">
              <p className="text-sm text-purple-600">Forecast Horizon</p>
              <p className="text-2xl font-bold text-purple-900">{selectedForecast.forecast_horizon} days</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-4">
              <p className="text-sm text-blue-600">Confidence Level</p>
              <p className="text-2xl font-bold text-blue-900">{(selectedForecast.confidence_level * 100).toFixed(0)}%</p>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <p className="text-sm text-green-600">Predictions</p>
              <p className="text-2xl font-bold text-green-900">{selectedForecast.predictions.length} points</p>
            </div>
          </div>

          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={formatChartData(selectedForecast.predictions)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 12 }}
                  interval="preserveStartEnd"
                />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="upper"
                  stroke="none"
                  fill="#e0e7ff"
                  name="Upper Bound"
                />
                <Area
                  type="monotone"
                  dataKey="lower"
                  stroke="none"
                  fill="#ffffff"
                  name="Lower Bound"
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                  name="Forecast"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Predictions Table */}
          <div className="mt-6">
            <h3 className="font-medium text-gray-900 mb-2">Prediction Details</h3>
            <div className="max-h-64 overflow-y-auto">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="px-4 py-2 text-left font-medium text-gray-500">Date</th>
                    <th className="px-4 py-2 text-left font-medium text-gray-500">Forecast</th>
                    <th className="px-4 py-2 text-left font-medium text-gray-500">Lower Bound</th>
                    <th className="px-4 py-2 text-left font-medium text-gray-500">Upper Bound</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {selectedForecast.predictions.map((p, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="px-4 py-2">{new Date(p.date).toLocaleDateString()}</td>
                      <td className="px-4 py-2 font-medium">{p.value.toFixed(2)}</td>
                      <td className="px-4 py-2 text-gray-500">{p.lower_bound?.toFixed(2) || '-'}</td>
                      <td className="px-4 py-2 text-gray-500">{p.upper_bound?.toFixed(2) || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* No Models Warning */}
      {!loading && models.length === 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <TrendingUp className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
          <p className="text-yellow-800">No trained models available. Please train a model first!</p>
        </div>
      )}
    </div>
  );
};

export default Forecasts;
