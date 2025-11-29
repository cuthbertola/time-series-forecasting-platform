import React, { useEffect, useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Trash2, Eye, Database, AlertCircle, CheckCircle, Loader } from 'lucide-react';
import { getDatasets, uploadDataset, deleteDataset, previewDataset } from '../services/api';
import { Dataset, DatasetPreview } from '../types';

const Datasets: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewDatasetId, setPreviewDatasetId] = useState<number | null>(null);

  const [uploadForm, setUploadForm] = useState({
    name: '',
    description: '',
    dateColumn: '',
    targetColumn: '',
  });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const response = await getDatasets();
      setDatasets(response.datasets || []);
    } catch (err) {
      setError('Failed to fetch datasets');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
      setUploadForm(prev => ({
        ...prev,
        name: acceptedFiles[0].name.replace(/\.[^/.]+$/, ''),
      }));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
  });

  const handleUpload = async () => {
    if (!selectedFile || !uploadForm.name) {
      setError('Please select a file and provide a name');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      await uploadDataset(
        selectedFile,
        uploadForm.name,
        uploadForm.description,
        uploadForm.dateColumn,
        uploadForm.targetColumn
      );
      setSelectedFile(null);
      setUploadForm({ name: '', description: '', dateColumn: '', targetColumn: '' });
      fetchDatasets();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload dataset');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;
    try {
      await deleteDataset(id);
      fetchDatasets();
    } catch (err) {
      setError('Failed to delete dataset');
    }
  };

  const handlePreview = async (id: number) => {
    try {
      const data = await previewDataset(id);
      setPreview(data);
      setPreviewDatasetId(id);
    } catch (err) {
      setError('Failed to load preview');
    }
  };

  const getStatusBadge = (status: string) => {
    const styles = {
      ready: 'bg-green-100 text-green-800',
      pending: 'bg-yellow-100 text-yellow-800',
      processing: 'bg-blue-100 text-blue-800',
      failed: 'bg-red-100 text-red-800',
    };
    return styles[status as keyof typeof styles] || styles.pending;
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Datasets</h1>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
          <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
          <span className="text-red-700">{error}</span>
          <button onClick={() => setError(null)} className="ml-auto text-red-500 hover:text-red-700">×</button>
        </div>
      )}

      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload New Dataset</h2>
        
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          {selectedFile ? (
            <p className="text-green-600 font-medium">{selectedFile.name}</p>
          ) : isDragActive ? (
            <p className="text-blue-600">Drop the file here...</p>
          ) : (
            <p className="text-gray-600">Drag & drop a CSV, Excel, or JSON file here, or click to select</p>
          )}
        </div>

        {selectedFile && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <input
              type="text"
              placeholder="Dataset Name *"
              value={uploadForm.name}
              onChange={(e) => setUploadForm(prev => ({ ...prev, name: e.target.value }))}
              className="border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <input
              type="text"
              placeholder="Description (optional)"
              value={uploadForm.description}
              onChange={(e) => setUploadForm(prev => ({ ...prev, description: e.target.value }))}
              className="border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <input
              type="text"
              placeholder="Date Column (e.g., date)"
              value={uploadForm.dateColumn}
              onChange={(e) => setUploadForm(prev => ({ ...prev, dateColumn: e.target.value }))}
              className="border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <input
              type="text"
              placeholder="Target Column (e.g., sales)"
              value={uploadForm.targetColumn}
              onChange={(e) => setUploadForm(prev => ({ ...prev, targetColumn: e.target.value }))}
              className="border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="md:col-span-2 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center"
            >
              {uploading ? <Loader className="h-5 w-5 animate-spin mr-2" /> : <Upload className="h-5 w-5 mr-2" />}
              {uploading ? 'Uploading...' : 'Upload Dataset'}
            </button>
          </div>
        )}
      </div>

      {/* Datasets List */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Your Datasets</h2>
        
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader className="h-8 w-8 animate-spin text-blue-500" />
          </div>
        ) : datasets.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Database className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>No datasets uploaded yet. Upload your first dataset above!</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rows</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Columns</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {datasets.map((dataset) => (
                  <tr key={dataset.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="font-medium text-gray-900">{dataset.name}</div>
                      <div className="text-sm text-gray-500">{dataset.filename}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-500">{dataset.row_count || '-'}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-500">{dataset.column_count || '-'}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs rounded-full ${getStatusBadge(dataset.status)}`}>
                        {dataset.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-500">
                      {new Date(dataset.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={() => handlePreview(dataset.id)}
                        className="text-blue-600 hover:text-blue-800 mr-3"
                        title="Preview"
                      >
                        <Eye className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => handleDelete(dataset.id)}
                        className="text-red-600 hover:text-red-800"
                        title="Delete"
                      >
                        <Trash2 className="h-5 w-5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Preview Modal */}
      {preview && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="text-lg font-semibold">Dataset Preview</h3>
              <button onClick={() => { setPreview(null); setPreviewDatasetId(null); }} className="text-gray-500 hover:text-gray-700">×</button>
            </div>
            <div className="p-4 overflow-auto max-h-[60vh]">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    {preview.columns.map((col) => (
                      <th key={col} className="px-4 py-2 text-left font-medium text-gray-500">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {preview.data.map((row, i) => (
                    <tr key={i}>
                      {preview.columns.map((col) => (
                        <td key={col} className="px-4 py-2 text-gray-700">{String(row[col] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Datasets;
