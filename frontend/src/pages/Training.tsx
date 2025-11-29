import React, { useState, useEffect } from 'react';
import { Brain, Play, CheckCircle, XCircle, Download, Package } from 'lucide-react';
import { getDatasets, runAutoML, getAutoMLRun, getModels, getModelExportUrl, getModelPackageUrl } from '../services/api';

interface Dataset {
  id: number;
  name: string;
  row_count: number;
  status: string;
}

interface Model {
  id: number;
  name: string;
  algorithm: string;
  mape: number | null;
  status: string;
  is_best_model: boolean;
}

interface AutoMLRun {
  id: number;
  status: string;
  best_algorithm: string | null;
  all_results: any[] | null;
  total_time_seconds: number | null;
}

const Training: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<number | null>(null);
  const [dateColumn, setDateColumn] = useState('date');
  const [targetColumn, setTargetColumn] = useState('sales');
  const [forecastHorizon, setForecastHorizon] = useState(30);
  const [algorithms, setAlgorithms] = useState(['prophet', 'arima', 'xgboost', 'lightgbm']);
  const [isTraining, setIsTraining] = useState(false);
  const [currentRun, setCurrentRun] = useState<AutoMLRun | null>(null);

  useEffect(() => {
    fetchDatasets();
    fetchModels();
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (currentRun && currentRun.status === 'running') {
      interval = setInterval(async () => {
        const run = await getAutoMLRun(currentRun.id);
        setCurrentRun(run);
        if (run.status !== 'running') {
          setIsTraining(false);
          fetchModels();
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [currentRun]);

  const fetchDatasets = async () => {
    try {
      const data = await getDatasets();
      setDatasets(data.filter((d: Dataset) => d.status === 'ready'));
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const data = await getModels();
      setModels(data);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const handleStartTraining = async () => {
    if (!selectedDataset) return;
    
    setIsTraining(true);
    try {
      const run = await runAutoML({
        dataset_id: selectedDataset,
        target_column: targetColumn,
        date_column: dateColumn,
        forecast_horizon: forecastHorizon,
        algorithms: algorithms,
      });
      setCurrentRun(run);
    } catch (error) {
      console.error('Error starting training:', error);
      setIsTraining(false);
    }
  };

  const toggleAlgorithm = (algo: string) => {
    setAlgorithms(prev => 
      prev.includes(algo) 
        ? prev.filter(a => a !== algo)
        : [...prev, algo]
    );
  };

  const handleExportModel = (modelId: number) => {
    window.open(getModelExportUrl(modelId), '_blank');
  };

  const handleExportPackage = (modelId: number) => {
    window.open(getModelPackageUrl(modelId), '_blank');
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Model Training</h1>

      {/* AutoML Configuration */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center space-x-2 mb-6">
          <Brain className="w-6 h-6 text-green-600" />
          <h2 className="text-xl font-semibold">AutoML Configuration</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Dataset *
            </label>
            <select
              value={selectedDataset || ''}
              onChange={(e) => setSelectedDataset(Number(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
            >
              <option value="">Select a dataset</option>
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.row_count} rows)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Date Column *
            </label>
            <input
              type="text"
              value={dateColumn}
              onChange={(e) => setDateColumn(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
              placeholder="e.g., date"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Target Column *
            </label>
            <input
              type="text"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
              placeholder="e.g., sales"
            />
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
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
            />
          </div>
        </div>

        <div className="mt-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Algorithms
          </label>
          <div className="flex flex-wrap gap-2">
            {['prophet', 'arima', 'xgboost', 'lightgbm'].map((algo) => (
              <button
                key={algo}
                onClick={() => toggleAlgorithm(algo)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  algorithms.includes(algo)
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {algo.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handleStartTraining}
          disabled={!selectedDataset || isTraining || algorithms.length === 0}
          className={`mt-6 w-full py-3 rounded-lg font-medium flex items-center justify-center space-x-2 ${
            !selectedDataset || isTraining || algorithms.length === 0
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-green-600 text-white hover:bg-green-700'
          }`}
        >
          {isTraining ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Training in progress...</span>
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              <span>Start AutoML Training</span>
            </>
          )}
        </button>
      </div>

      {/* Training Status */}
      {currentRun && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Training Status</h2>
          
          <div className="flex items-center justify-between mb-4">
            <span className="text-gray-600">Status:</span>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              currentRun.status === 'completed' ? 'bg-green-100 text-green-800' :
              currentRun.status === 'failed' ? 'bg-red-100 text-red-800' :
              'bg-blue-100 text-blue-800'
            }`}>
              {currentRun.status}
            </span>
          </div>

          {currentRun.best_algorithm && (
            <div className="flex items-center justify-between mb-4">
              <span className="text-gray-600">Best Algorithm:</span>
              <span className="font-semibold">{currentRun.best_algorithm.toUpperCase()}</span>
            </div>
          )}

          {currentRun.total_time_seconds && (
            <div className="flex items-center justify-between mb-4">
              <span className="text-gray-600">Total Time:</span>
              <span className="font-semibold">{currentRun.total_time_seconds.toFixed(2)}s</span>
            </div>
          )}

          {currentRun.all_results && currentRun.all_results.length > 0 && (
            <div className="mt-4">
              <h3 className="font-medium mb-2">Results by Algorithm:</h3>
              <div className="space-y-2">
                {currentRun.all_results.map((result, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="font-medium">{result.algorithm?.toUpperCase()}</span>
                    <span className="text-gray-600">
                      MAPE: {result.mape ? `${(result.mape * 100).toFixed(2)}%` : 'N/A'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Trained Models */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Trained Models</h2>
        
        {models.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-500 text-sm border-b">
                  <th className="pb-3 font-medium">MODEL</th>
                  <th className="pb-3 font-medium">ALGORITHM</th>
                  <th className="pb-3 font-medium">MAPE</th>
                  <th className="pb-3 font-medium">STATUS</th>
                  <th className="pb-3 font-medium">BEST</th>
                  <th className="pb-3 font-medium">ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.id} className="border-b last:border-0">
                    <td className="py-4">{model.name}</td>
                    <td className="py-4">{model.algorithm.toUpperCase()}</td>
                    <td className="py-4">
                      {model.mape ? `${(model.mape * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                    <td className="py-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        model.status === 'completed' ? 'bg-green-100 text-green-800' :
                        model.status === 'failed' ? 'bg-red-100 text-red-800' :
                        'bg-yellow-100 text-yellow-800'
                      }`}>
                        {model.status}
                      </span>
                    </td>
                    <td className="py-4">
                      {model.is_best_model && (
                        <CheckCircle className="w-5 h-5 text-green-600" />
                      )}
                    </td>
                    <td className="py-4">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleExportModel(model.id)}
                          className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg"
                          title="Download Model"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleExportPackage(model.id)}
                          className="p-2 text-purple-600 hover:bg-purple-50 rounded-lg"
                          title="Download Package (Model + Metadata)"
                        >
                          <Package className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            No models trained yet. Start training above!
          </p>
        )}
      </div>
    </div>
  );
};

export default Training;
