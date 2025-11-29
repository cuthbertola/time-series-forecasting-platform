import React, { useEffect, useState } from 'react';
import { Brain, Play, Loader, CheckCircle, XCircle, Clock } from 'lucide-react';
import { getDatasets, runAutoML, getAutoMLRun, getModels } from '../services/api';
import { Dataset, AutoMLRun, TrainedModel } from '../types';

const Training: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [currentRun, setCurrentRun] = useState<AutoMLRun | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    datasetId: '',
    dateColumn: '',
    targetColumn: '',
    forecastHorizon: 30,
    maxTrials: 50,
    algorithms: ['prophet', 'arima', 'xgboost', 'lightgbm'],
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [datasetsRes, modelsRes] = await Promise.all([
          getDatasets(),
          getModels(),
        ]);
        setDatasets(datasetsRes.datasets || []);
        setModels(modelsRes || []);
      } catch (err) {
        setError('Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (currentRun && currentRun.status === 'running') {
      interval = setInterval(async () => {
        try {
          const updated = await getAutoMLRun(currentRun.id);
          setCurrentRun(updated);
          if (updated.status !== 'running') {
            const modelsRes = await getModels();
            setModels(modelsRes || []);
          }
        } catch (err) {
          console.error('Failed to fetch run status');
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [currentRun]);

  const handleDatasetChange = (datasetId: string) => {
    const dataset = datasets.find(d => d.id === parseInt(datasetId));
    setFormData(prev => ({
      ...prev,
      datasetId,
      dateColumn: dataset?.date_column || '',
      targetColumn: dataset?.target_column || '',
    }));
  };

  const handleStartTraining = async () => {
    if (!formData.datasetId || !formData.dateColumn || !formData.targetColumn) {
      setError('Please fill in all required fields');
      return;
    }

    try {
      setTraining(true);
      setError(null);
      const run = await runAutoML({
        dataset_id: parseInt(formData.datasetId),
        date_column: formData.dateColumn,
        target_column: formData.targetColumn,
        forecast_horizon: formData.forecastHorizon,
        max_trials: formData.maxTrials,
        algorithms: formData.algorithms,
      });
      setCurrentRun(run);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start training');
    } finally {
      setTraining(false);
    }
  };

  const toggleAlgorithm = (algo: string) => {
    setFormData(prev => ({
      ...prev,
      algorithms: prev.algorithms.includes(algo)
        ? prev.algorithms.filter(a => a !== algo)
        : [...prev.algorithms, algo],
    }));
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Model Training</h1>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
          <XCircle className="h-5 w-5 text-red-500 mr-2" />
          <span className="text-red-700">{error}</span>
          <button onClick={() => setError(null)} className="ml-auto">×</button>
        </div>
      )}

      {/* Training Form */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <Brain className="h-6 w-6 mr-2 text-green-600" />
          AutoML Configuration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Dataset *</label>
            <select
              value={formData.datasetId}
              onChange={(e) => handleDatasetChange(e.target.value)}
              className="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select a dataset</option>
              {datasets.filter(d => d.status === 'ready').map(d => (
                <option key={d.id} value={d.id}>{d.name} ({d.row_count} rows)</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Date Column *</label>
            <input
              type="text"
              value={formData.dateColumn}
              onChange={(e) => setFormData(prev => ({ ...prev, dateColumn: e.target.value }))}
              placeholder="e.g., date"
              className="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Target Column *</label>
            <input
              type="text"
              value={formData.targetColumn}
              onChange={(e) => setFormData(prev => ({ ...prev, targetColumn: e.target.value }))}
              placeholder="e.g., sales"
              className="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Forecast Horizon (days)</label>
            <input
              type="number"
              value={formData.forecastHorizon}
              onChange={(e) => setFormData(prev => ({ ...prev, forecastHorizon: parseInt(e.target.value) }))}
              min={1}
              max={365}
              className="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Algorithms</label>
          <div className="flex flex-wrap gap-2">
            {['prophet', 'arima', 'xgboost', 'lightgbm'].map(algo => (
              <button
                key={algo}
                onClick={() => toggleAlgorithm(algo)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  formData.algorithms.includes(algo)
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {algo.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handleStartTraining}
          disabled={training || (currentRun?.status === 'running')}
          className="mt-6 w-full bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center justify-center"
        >
          {training || currentRun?.status === 'running' ? (
            <><Loader className="h-5 w-5 animate-spin mr-2" /> Training in progress...</>
          ) : (
            <><Play className="h-5 w-5 mr-2" /> Start AutoML Training</>
          )}
        </button>
      </div>

      {/* Current Run Status */}
      {currentRun && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Training Status</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Status:</span>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                currentRun.status === 'completed' ? 'bg-green-100 text-green-800' :
                currentRun.status === 'running' ? 'bg-blue-100 text-blue-800' :
                'bg-red-100 text-red-800'
              }`}>
                {currentRun.status}
              </span>
            </div>
            {currentRun.best_algorithm && (
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Best Algorithm:</span>
                <span className="font-medium">{currentRun.best_algorithm.toUpperCase()}</span>
              </div>
            )}
            {currentRun.total_time_seconds && (
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Total Time:</span>
                <span className="font-medium">{currentRun.total_time_seconds.toFixed(1)}s</span>
              </div>
            )}
            {currentRun.all_results && (
              <div className="mt-4">
                <h3 className="font-medium mb-2">Results by Algorithm:</h3>
                <div className="space-y-2">
                  {currentRun.all_results.map((result, i) => (
                    <div key={i} className="flex items-center justify-between bg-gray-50 p-3 rounded-lg">
                      <span className="font-medium">{result.algorithm.toUpperCase()}</span>
                      <span className="text-sm">
                        MAPE: {result.mape?.toFixed(2) || 'N/A'}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Trained Models */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Trained Models</h2>
        {loading ? (
          <Loader className="h-8 w-8 animate-spin text-blue-500 mx-auto" />
        ) : models.length === 0 ? (
          <p className="text-center text-gray-500 py-8">No models trained yet. Start training above!</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Algorithm</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">MAPE</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Best</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {models.map((model) => (
                  <tr key={model.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 font-medium text-gray-900">{model.name}</td>
                    <td className="px-6 py-4 text-gray-500">{model.algorithm.toUpperCase()}</td>
                    <td className="px-6 py-4 text-gray-500">{model.mape?.toFixed(2) || '-'}%</td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        model.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {model.status}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      {model.is_best_model && <CheckCircle className="h-5 w-5 text-green-500" />}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Training;
