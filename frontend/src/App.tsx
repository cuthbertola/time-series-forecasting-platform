import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Home, Upload, Brain, TrendingUp, BarChart3, Download, FileSpreadsheet } from 'lucide-react';
import { Dashboard, Datasets, Training, Forecasts } from './pages';
import Visualization from './pages/Visualization';

const Navigation: React.FC = () => {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/datasets', label: 'Datasets', icon: Upload },
    { path: '/visualization', label: 'Visualization', icon: BarChart3 },
    { path: '/training', label: 'Training', icon: Brain },
    { path: '/forecasts', label: 'Forecasts', icon: TrendingUp },
  ];

  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <BarChart3 className="w-8 h-8 text-blue-600" />
              <span className="text-xl font-bold text-gray-900">Time Series Forecasting</span>
            </Link>
          </div>
          <div className="flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="w-4 h-4 mr-1.5" />
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main className="max-w-7xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/visualization" element={<Visualization />} />
            <Route path="/training" element={<Training />} />
            <Route path="/forecasts" element={<Forecasts />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
