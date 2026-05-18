import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileText, CheckCircle, Clock } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState({ processed: 0, total: 1, status: '' });
  
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/logs/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setJobId(data.job_id);
      
      // Start WebSocket connection
      const ws = new WebSocket(`ws://localhost:8000/api/logs/ws/progress/${data.job_id}`);
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        setProgress(msg);
        if (msg.status === 'completed') {
          ws.close();
        }
      };
    } catch (err) {
      console.error(err);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] }
  });

  return (
    <div className="min-h-screen bg-background p-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-8">
        
        <header className="flex justify-between items-center pb-6 border-b border-gray-800">
          <div>
            <h1 className="text-3xl font-bold text-white tracking-tight">Log Analytics</h1>
            <p className="text-gray-400 mt-1">Hybrid Classification Pipeline</p>
          </div>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 shadow-sm flex items-center space-x-4">
             <div className="p-3 bg-blue-500/10 text-blue-500 rounded-lg">
                <FileText size={24} />
             </div>
             <div>
               <p className="text-sm text-gray-400">Total Logs</p>
               <h3 className="text-2xl font-bold text-white">4,281</h3>
             </div>
          </div>
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 shadow-sm flex items-center space-x-4">
             <div className="p-3 bg-green-500/10 text-green-500 rounded-lg">
                <CheckCircle size={24} />
             </div>
             <div>
               <p className="text-sm text-gray-400">Avg. Confidence</p>
               <h3 className="text-2xl font-bold text-white">94.2%</h3>
             </div>
          </div>
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 shadow-sm flex items-center space-x-4">
             <div className="p-3 bg-purple-500/10 text-purple-500 rounded-lg">
                <Clock size={24} />
             </div>
             <div>
               <p className="text-sm text-gray-400">Processing Time</p>
               <h3 className="text-2xl font-bold text-white">1.2s</h3>
             </div>
          </div>
        </div>

        <div className="bg-gray-900 rounded-xl border border-gray-800 p-8 shadow-sm">
          <h2 className="text-xl font-semibold text-white mb-6">Upload Dataset</h2>
          <div 
            {...getRootProps()} 
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-primary bg-primary/5' : 'border-gray-700 hover:border-gray-500 hover:bg-gray-800/50'
            }`}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center justify-center space-y-4">
              <div className="p-4 bg-gray-800 rounded-full">
                <UploadCloud size={32} className="text-gray-400" />
              </div>
              <div>
                <p className="text-lg font-medium text-gray-300">Drag & drop your CSV file here</p>
                <p className="text-sm text-gray-500 mt-1">or click to select a file from your computer</p>
              </div>
            </div>
          </div>

          {jobId && (
            <div className="mt-8 space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Processing Job: {jobId}</span>
                <span className="text-white font-medium">
                  {Math.round((progress.processed / progress.total) * 100)}%
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                <div 
                  className="bg-primary h-2 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${(progress.processed / progress.total) * 100}%` }}
                />
              </div>
              {progress.status === 'completed' && (
                <p className="text-green-500 text-sm mt-2 flex items-center">
                  <CheckCircle size={16} className="mr-1" /> Analysis completed successfully
                </p>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
