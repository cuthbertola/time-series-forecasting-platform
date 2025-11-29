import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, ComposedChart, Area 
} from 'recharts';
import { BarChart3, TrendingUp, Calendar, Activity } from 'lucide-react';
import { getDatasets, getHistoricalData, getDatasetStatistics, getSeasonalityAnalysis } from '../services/api';

interface Dataset {
  id: number;
  name: string;
  status: string;
}

const Visualization: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<number | null>(null);
  const [historicalData, setHistoricalData] = useState<any>(null);
  const [statistics, setStatistics] = useState<any>(null);
  const [seasonality, setSeasonality] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'historical' | 'seasonality' | 'statistics'>('historical');

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      fetchVisualizationData(selectedDataset);
    }
  }, [selectedDataset]);

  const fetchDatasets = async () => {
    try {
      const data = await getDatasets();
      setDatasets(data.filter((d: Dataset) => d.status === 'ready'));
    } catch (error) {
      console.error('Error fetching datasets:', error);
    }
  };

  const fetchVisualizationData = async (datasetId: number) => {
    setLoading(true);
    try {
      const [hist, stats, season] = await Promise.all([
        getHistoricalData(datasetId),
        getDatasetStatistics(datasetId),
        getSeasonalityAnalysis(datasetId)
      ]);
      setHistoricalData(hist);
      setStatistics(stats);
      setSeasonality(season);
    } catch (error) {
      console.error('Error fetching visualization data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Data Visualization</h1>

      {/* Dataset Selection */}
      <div className="bg-white rounded-lg shadow p-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Dataset
        </label>
        <select
          value={selectedDataset || ''}
          onChange={(e) => setSelectedDataset(Number(e.target.value))}
          className="w-full md:w-1/3 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Choose a dataset...</option>
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.name}
            </option>
          ))}
        </select>
      </div>

      {selectedDataset && !loading && (
        <>
          {/* Tab Navigation */}
          <div className="bg-white rounded-lg shadow">
            <div className="border-b border-gray-200">
              <nav className="flex -mb-px">
                <button
                  onClick={() => setActiveTab('historical')}
                  className={`px-6 py-3 border-b-2 font-medium text-sm ${
                    activeTab === 'historical'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <TrendingUp className="w-4 h-4 inline mr-2" />
                  Historical Data
                </button>
                <button
                  onClick={() => setActiveTab('seasonality')}
                  className={`px-6 py-3 border-b-2 font-medium text-sm ${
                    activeTab === 'seasonality'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <Calendar className="w-4 h-4 inline mr-2" />
                  Seasonality
                </button>
                <button
                  onClick={() => setActiveTab('statistics')}
                  className={`px-6 py-3 border-b-2 font-medium text-sm ${
                    activeTab === 'statistics'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <BarChart3 className="w-4 h-4 inline mr-2" />
                  Statistics
                </button>
              </nav>
            </div>

            <div className="p-6">
              {/* Historical Data Tab */}
              {activeTab === 'historical' && historicalData && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">
                    {historicalData.dataset_name} - Time Series
                  </h3>
                  <p className="text-sm text-gray-500 mb-4">
                    {historicalData.total_points} data points
                  </p>
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={historicalData.data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          dataKey="date" 
                          tick={{ fontSize: 10 }}
                          interval="preserveStartEnd"
                        />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="value" 
                          stroke="#3B82F6" 
                          dot={false}
                          name={historicalData.target_column}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Seasonality Tab */}
              {activeTab === 'seasonality' && seasonality && (
                <div className="space-y-8">
                  {/* Weekly Pattern */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4">Weekly Pattern</h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={seasonality.weekly_pattern}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="day" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="value" fill="#3B82F6" name="Average Value" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Monthly Pattern */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4">Monthly Pattern</h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={seasonality.monthly_pattern}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="month" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="value" fill="#10B981" name="Average Value" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Trend */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4">Trend Analysis (7-day Rolling Average)</h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={seasonality.trend_data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="value" stroke="#94A3B8" dot={false} name="Actual" />
                          <Line type="monotone" dataKey="trend" stroke="#EF4444" dot={false} strokeWidth={2} name="Trend" />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              )}

              {/* Statistics Tab */}
              {activeTab === 'statistics' && statistics && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">Statistical Summary</h3>
                  
                  {/* Date Range */}
                  <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                    <h4 className="font-medium text-gray-700 mb-2">Date Range</h4>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className="text-sm text-gray-500">Start Date</p>
                        <p className="font-semibold">{statistics.date_range.start_date}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">End Date</p>
                        <p className="font-semibold">{statistics.date_range.end_date}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Total Days</p>
                        <p className="font-semibold">{statistics.date_range.total_days}</p>
                      </div>
                    </div>
                  </div>

                  {/* Statistics Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <p className="text-sm text-blue-600">Count</p>
                      <p className="text-2xl font-bold text-blue-700">{statistics.statistics.count}</p>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <p className="text-sm text-green-600">Mean</p>
                      <p className="text-2xl font-bold text-green-700">{statistics.statistics.mean.toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg">
                      <p className="text-sm text-purple-600">Median</p>
                      <p className="text-2xl font-bold text-purple-700">{statistics.statistics.median.toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-orange-50 rounded-lg">
                      <p className="text-sm text-orange-600">Std Dev</p>
                      <p className="text-2xl font-bold text-orange-700">{statistics.statistics.std.toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-red-50 rounded-lg">
                      <p className="text-sm text-red-600">Min</p>
                      <p className="text-2xl font-bold text-red-700">{statistics.statistics.min.toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-teal-50 rounded-lg">
                      <p className="text-sm text-teal-600">Max</p>
                      <p className="text-2xl font-bold text-teal-700">{statistics.statistics.max.toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-indigo-50 rounded-lg">
                      <p className="text-sm text-indigo-600">25th Percentile</p>
                      <p className="text-2xl font-bold text-indigo-700">{statistics.statistics.q25.toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-pink-50 rounded-lg">
                      <p className="text-sm text-pink-600">75th Percentile</p>
                      <p className="text-2xl font-bold text-pink-700">{statistics.statistics.q75.toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {loading && (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      )}

      {!selectedDataset && (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500">Select a dataset to view visualizations</p>
        </div>
      )}
    </div>
  );
};

export default Visualization;
