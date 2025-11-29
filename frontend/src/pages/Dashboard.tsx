import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Database, Brain, TrendingUp, Activity, CheckCircle, XCircle } from 'lucide-react';
import { getDatasets, getModels, healthCheck } from '../services/api';

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState({
    datasets: 0,
    models: 0,
    forecasts: 0,
  });
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [datasetsRes, modelsRes, health] = await Promise.all([
          getDatasets().catch(() => ({ datasets: [], total: 0 })),
          getModels().catch(() => []),
          healthCheck().catch(() => null),
        ]);

        setStats({
          datasets: datasetsRes.total || 0,
          models: Array.isArray(modelsRes) ? modelsRes.length : 0,
          forecasts: 0,
        });

        setApiStatus(health ? 'online' : 'offline');
      } catch (error) {
        setApiStatus('offline');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const statCards = [
    {
      title: 'Datasets',
      value: stats.datasets,
      icon: Database,
      color: 'bg-blue-500',
      link: '/datasets',
    },
    {
      title: 'Trained Models',
      value: stats.models,
      icon: Brain,
      color: 'bg-green-500',
      link: '/training',
    },
    {
      title: 'Forecasts',
      value: stats.forecasts,
      icon: TrendingUp,
      color: 'bg-purple-500',
      link: '/forecasts',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <div className="flex items-center space-x-2">
          <Activity className="h-5 w-5 text-gray-500" />
          <span className="text-sm text-gray-600">API Status:</span>
          {apiStatus === 'checking' ? (
            <span className="text-yellow-500">Checking...</span>
          ) : apiStatus === 'online' ? (
            <span className="flex items-center text-green-500">
              <CheckCircle className="h-4 w-4 mr-1" /> Online
            </span>
          ) : (
            <span className="flex items-center text-red-500">
              <XCircle className="h-4 w-4 mr-1" /> Offline
            </span>
          )}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {statCards.map((card) => {
          const Icon = card.icon;
          return (
            <Link
              key={card.title}
              to={card.link}
              className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
            >
              <div className="flex items-center">
                <div className={`${card.color} rounded-lg p-3`}>
                  <Icon className="h-6 w-6 text-white" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">{card.title}</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {loading ? '...' : card.value}
                  </p>
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            to="/datasets"
            className="flex items-center p-4 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
          >
            <Database className="h-8 w-8 text-blue-600" />
            <div className="ml-4">
              <p className="font-medium text-gray-900">Upload Dataset</p>
              <p className="text-sm text-gray-500">Import your time series data</p>
            </div>
          </Link>

          <Link
            to="/training"
            className="flex items-center p-4 bg-green-50 rounded-lg hover:bg-green-100 transition-colors"
          >
            <Brain className="h-8 w-8 text-green-600" />
            <div className="ml-4">
              <p className="font-medium text-gray-900">Train Models</p>
              <p className="text-sm text-gray-500">Run AutoML on your data</p>
            </div>
          </Link>

          <Link
            to="/forecasts"
            className="flex items-center p-4 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors"
          >
            <TrendingUp className="h-8 w-8 text-purple-600" />
            <div className="ml-4">
              <p className="font-medium text-gray-900">Generate Forecasts</p>
              <p className="text-sm text-gray-500">Predict future values</p>
            </div>
          </Link>
        </div>
      </div>

      {/* Features */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Platform Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-start">
            <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
            <div className="ml-3">
              <p className="font-medium text-gray-900">AutoML</p>
              <p className="text-sm text-gray-500">Automatic model selection and hyperparameter tuning</p>
            </div>
          </div>
          <div className="flex items-start">
            <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
            <div className="ml-3">
              <p className="font-medium text-gray-900">Multiple Algorithms</p>
              <p className="text-sm text-gray-500">Prophet, ARIMA, XGBoost, LightGBM</p>
            </div>
          </div>
          <div className="flex items-start">
            <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
            <div className="ml-3">
              <p className="font-medium text-gray-900">Feature Engineering</p>
              <p className="text-sm text-gray-500">Automated lag, rolling, and calendar features</p>
            </div>
          </div>
          <div className="flex items-start">
            <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
            <div className="ml-3">
              <p className="font-medium text-gray-900">Confidence Intervals</p>
              <p className="text-sm text-gray-500">Uncertainty quantification in predictions</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
