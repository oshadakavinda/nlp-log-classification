import { useState, useCallback, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  UploadCloud, 
  FileText, 
  CheckCircle, 
  Search, 
  Filter, 
  Cpu, 
  Zap, 
  Database, 
  RefreshCw, 
  AlertTriangle,
  Trash2
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';

interface LogEntry {
  id: number;
  source: string;
  log_message: string;
  target_label: string | null;
  classification_method: string | null;
  created_at: string;
}

interface JobProgress {
  processed: number;
  total: number;
  status: string;
}

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<JobProgress>({ processed: 0, total: 1, status: '' });
  const [searchTerm, setSearchTerm] = useState('');
  const [methodFilter, setMethodFilter] = useState('all');
  const [labelFilter, setLabelFilter] = useState('all');
  const [isClearing, setIsClearing] = useState(false);

  const handleClearLogs = async () => {
    if (!window.confirm("Are you sure you want to clear all stored logs? This action cannot be undone.")) {
      return;
    }
    setIsClearing(true);
    try {
      const response = await fetch('http://localhost:8000/api/logs', {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to clear logs');
      refetchLogs();
    } catch (err) {
      console.error(err);
      alert("Failed to clear logs. Please try again.");
    } finally {
      setIsClearing(false);
    }
  };

  const { data: logs = [], refetch: refetchLogs, isFetching } = useQuery<LogEntry[]>({
    queryKey: ['logs'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/logs?limit=500');
      if (!response.ok) {
        throw new Error('Failed to fetch logs');
      }
      return response.json();
    }
  });

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
      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();
      setJobId(data.job_id);
      setProgress({ processed: 0, total: 1, status: 'processing' });
      
      // Start WebSocket connection for real-time progress
      const ws = new WebSocket(`ws://localhost:8000/api/logs/ws/progress/${data.job_id}`);
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        setProgress(msg);
        if (msg.status === 'completed') {
          ws.close();
          refetchLogs();
        }
      };
      ws.onerror = () => {
        console.error("WebSocket connection error");
      };
    } catch (err) {
      console.error(err);
    }
  }, [refetchLogs]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] }
  });

  // Calculate dynamic stats from logs
  const stats = useMemo(() => {
    const total = logs.length;
    const regexCount = logs.filter(l => l.classification_method === 'Regex').length;
    const mlCount = logs.filter(l => l.classification_method === 'ML').length;
    const llmCount = logs.filter(l => l.classification_method === 'LLM').length;

    const regexPercent = total > 0 ? Math.round((regexCount / total) * 100) : 0;
    const mlPercent = total > 0 ? Math.round((mlCount / total) * 100) : 0;
    const llmPercent = total > 0 ? Math.round((llmCount / total) * 100) : 0;

    return {
      total,
      regex: { count: regexCount, percent: regexPercent },
      ml: { count: mlCount, percent: mlPercent },
      llm: { count: llmCount, percent: llmPercent },
      complex: { count: mlCount + llmCount, percent: mlPercent + llmPercent }
    };
  }, [logs]);

  // List of unique labels for the filter dropdown
  const uniqueLabels = useMemo(() => {
    const labels = new Set<string>();
    logs.forEach(l => {
      if (l.target_label) labels.add(l.target_label);
    });
    return Array.from(labels);
  }, [logs]);

  // Filter logs based on search term and dropdown filters
  const filteredLogs = useMemo(() => {
    return logs.filter(log => {
      const matchesSearch = 
        log.log_message.toLowerCase().includes(searchTerm.toLowerCase()) ||
        log.source.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesMethod = 
        methodFilter === 'all' || 
        log.classification_method === methodFilter;

      const matchesLabel = 
        labelFilter === 'all' || 
        log.target_label === labelFilter;

      return matchesSearch && matchesMethod && matchesLabel;
    });
  }, [logs, searchTerm, methodFilter, labelFilter]);

  // Styling helper for classification method badge
  const getMethodBadge = (method: string | null) => {
    if (!method) return null;
    switch(method) {
      case 'Regex':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Zap size={12} className="mr-1" /> Regex
          </span>
        );
      case 'ML':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <Cpu size={12} className="mr-1" /> ML Model
          </span>
        );
      case 'LLM':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Database size={12} className="mr-1" /> LLM
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-500/10 text-gray-400 border border-gray-500/20">
            {method}
          </span>
        );
    }
  };

  // Styling helper for source badges
  const getSourceBadge = (source: string) => {
    let colorClasses = "bg-gray-500/10 text-gray-400 border-gray-500/20";
    switch(source) {
      case 'LegacyCRM':
        colorClasses = "bg-amber-500/10 text-amber-400 border-amber-500/20";
        break;
      case 'ModernCRM':
        colorClasses = "bg-sky-500/10 text-sky-400 border-sky-500/20";
        break;
      case 'BillingSystem':
        colorClasses = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
        break;
      case 'AnalyticsEngine':
        colorClasses = "bg-rose-500/10 text-rose-400 border-rose-500/20";
        break;
      case 'ModernHR':
        colorClasses = "bg-fuchsia-500/10 text-fuchsia-400 border-fuchsia-500/20";
        break;
    }
    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-semibold border ${colorClasses}`}>
        {source}
      </span>
    );
  };

  // Styling helper for classification target labels
  const getLabelBadge = (label: string | null) => {
    if (!label) return <span className="text-gray-600">—</span>;
    let colorClasses = "bg-gray-500/10 text-gray-400 border-gray-500/20";
    
    const lower = label.toLowerCase();
    if (lower.includes('error') || lower.includes('fail')) {
      colorClasses = "bg-red-500/10 text-red-400 border-red-500/20";
    } else if (lower.includes('warn') || lower.includes('retire') || lower.includes('deprecation')) {
      colorClasses = "bg-yellow-500/10 text-yellow-400 border-yellow-500/20";
    } else if (lower.includes('action') || lower.includes('login') || lower.includes('create')) {
      colorClasses = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
    } else if (lower.includes('notification') || lower.includes('success') || lower.includes('backup')) {
      colorClasses = "bg-cyan-500/10 text-cyan-400 border-cyan-500/20";
    }

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-bold border ${colorClasses}`}>
        {label}
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-gray-100 p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header Section */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-gray-800 gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-emerald-500 animate-pulse" />
              <h1 className="text-3xl font-extrabold text-white tracking-tight bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
                Log Analytics Platform
              </h1>
            </div>
            <p className="text-gray-400 mt-1">Hybrid Classification Pipeline (Regex + Sentence Transformer + LLM)</p>
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={handleClearLogs}
              disabled={isClearing || logs.length === 0}
              className="flex items-center gap-2 px-4 py-2 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl hover:bg-red-500/20 disabled:opacity-40 disabled:cursor-not-allowed text-sm font-medium transition-all"
            >
              <Trash2 size={16} className={isClearing ? "animate-pulse" : ""} />
              {isClearing ? "Clearing..." : "Clear Logs"}
            </button>
            <button 
              onClick={() => refetchLogs()}
              className="flex items-center gap-2 px-4 py-2 bg-[#131926] border border-[#222E45] rounded-xl hover:bg-[#1E273A] text-sm font-medium transition-colors"
            >
              <RefreshCw size={16} className={isFetching ? "animate-spin text-blue-400" : "text-gray-400"} />
              Refresh Logs
            </button>
          </div>
        </header>

        {/* Metric Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-[#131926] rounded-2xl p-6 border border-[#222E45] shadow-lg flex items-center space-x-4 hover:border-[#304161] transition-all">
             <div className="p-3 bg-blue-500/10 text-blue-400 rounded-xl">
                <FileText size={28} />
             </div>
             <div>
               <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">Total Stored Logs</p>
               <h3 className="text-3xl font-bold text-white mt-1">{stats.total.toLocaleString()}</h3>
             </div>
          </div>
          
          <div className="bg-[#131926] rounded-2xl p-6 border border-[#222E45] shadow-lg flex items-center space-x-4 hover:border-[#304161] transition-all">
             <div className="p-3 bg-indigo-500/10 text-indigo-400 rounded-xl">
                <Cpu size={28} />
             </div>
             <div>
               <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">ML & LLM Processed</p>
               <h3 className="text-3xl font-bold text-white mt-1">
                 {stats.complex.count.toLocaleString()}
                 <span className="text-sm font-normal text-indigo-400 ml-2">({stats.complex.percent}%)</span>
               </h3>
             </div>
          </div>

          <div className="bg-[#131926] rounded-2xl p-6 border border-[#222E45] shadow-lg flex items-center space-x-4 hover:border-[#304161] transition-all">
             <div className="p-3 bg-emerald-500/10 text-emerald-400 rounded-xl">
                <Zap size={28} />
             </div>
             <div>
               <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">Regex Pattern Matches</p>
               <h3 className="text-3xl font-bold text-white mt-1">
                 {stats.regex.count.toLocaleString()}
                 <span className="text-sm font-normal text-emerald-400 ml-2">({stats.regex.percent}%)</span>
               </h3>
             </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Upload panel (Left) */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-[#131926] rounded-2xl border border-[#222E45] p-6 shadow-lg">
              <h2 className="text-lg font-semibold text-white mb-4">Upload Log Dataset</h2>
              <p className="text-xs text-gray-400 mb-4">
                Upload a CSV file containing logs. The CSV must have column headers <code className="bg-gray-800 text-gray-300 px-1 rounded font-mono">source</code> and <code className="bg-gray-800 text-gray-300 px-1 rounded font-mono">log_message</code>.
              </p>
              
              <div 
                {...getRootProps()} 
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                  isDragActive 
                    ? 'border-blue-500 bg-blue-500/5' 
                    : 'border-[#2D3E5D] hover:border-gray-500 hover:bg-[#182030]'
                }`}
              >
                <input {...getInputProps()} />
                <div className="flex flex-col items-center justify-center space-y-4">
                  <div className="p-3 bg-gray-800/80 rounded-full text-gray-400">
                    <UploadCloud size={28} />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-gray-300">Drag & drop CSV here</p>
                    <p className="text-xs text-gray-500 mt-1">or click to choose file</p>
                  </div>
                </div>
              </div>

              {/* Upload & processing progress */}
              {jobId && (
                <div className="mt-6 p-4 bg-[#182030] rounded-xl border border-[#2D3E5D] space-y-3">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-gray-400 font-mono text-ellipsis overflow-hidden max-w-[150px]">Job: {jobId}</span>
                    <span className="text-white font-semibold">
                      {Math.round((progress.processed / progress.total) * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                    <div 
                      className="bg-blue-500 h-1.5 rounded-full transition-all duration-300 ease-out"
                      style={{ width: `${(progress.processed / progress.total) * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-400">
                    <span>Processed: {progress.processed}</span>
                    <span>Total: {progress.total}</span>
                  </div>
                  {progress.status === 'completed' && (
                    <p className="text-emerald-400 text-xs mt-2 flex items-center gap-1 font-medium bg-emerald-500/5 p-2 rounded border border-emerald-500/10">
                      <CheckCircle size={14} /> Analysis completed! Table refreshed.
                    </p>
                  )}
                </div>
              )}
            </div>
            
            {/* Quick tips panel */}
            <div className="bg-[#131926] rounded-2xl border border-[#222E45] p-6 shadow-lg space-y-3 text-xs text-gray-400">
              <h3 className="font-semibold text-white flex items-center gap-1">
                <AlertTriangle size={14} className="text-amber-500" /> Pipeline Orchestration
              </h3>
              <p>The backend classifies log messages using a hybrid cascade workflow:</p>
              <ul className="list-disc pl-4 space-y-1">
                <li><strong>Regex Rules</strong> are applied first to handle high-frequency, structured patterns.</li>
                <li><strong>BERT Embeddings + Logistic Regression</strong> is used next for complex matching.</li>
                <li><strong>Groq DeepSeek-R1</strong> is invoked as a fallback when sufficient training data is absent (e.g. for LegacyCRM logs).</li>
              </ul>
            </div>
          </div>

          {/* Logs View (Right - taking up 2 cols) */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-[#131926] rounded-2xl border border-[#222E45] shadow-lg overflow-hidden">
              
              {/* Filter and Search Bar */}
              <div className="p-4 bg-[#171E2E] border-b border-[#222E45] flex flex-col md:flex-row gap-3 items-center">
                <div className="relative w-full md:flex-1">
                  <span className="absolute inset-y-0 left-3 flex items-center text-gray-500">
                    <Search size={16} />
                  </span>
                  <input
                    type="text"
                    placeholder="Search logs by message or source..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full bg-[#1C2538] text-white pl-10 pr-4 py-2 rounded-xl text-sm border border-[#2D3E5D] focus:outline-none focus:border-blue-500 transition-colors"
                  />
                </div>
                
                <div className="flex gap-2 w-full md:w-auto">
                  <div className="flex items-center gap-1 bg-[#1C2538] border border-[#2D3E5D] rounded-xl px-2.5 py-1.5 text-xs text-gray-400">
                    <Filter size={14} />
                    <select
                      value={methodFilter}
                      onChange={(e) => setMethodFilter(e.target.value)}
                      className="bg-transparent text-gray-200 focus:outline-none cursor-pointer text-xs"
                    >
                      <option value="all" className="bg-[#1C2538]">All Methods</option>
                      <option value="Regex" className="bg-[#1C2538]">Regex</option>
                      <option value="ML" className="bg-[#1C2538]">ML Model</option>
                      <option value="LLM" className="bg-[#1C2538]">LLM</option>
                    </select>
                  </div>

                  <div className="flex items-center gap-1 bg-[#1C2538] border border-[#2D3E5D] rounded-xl px-2.5 py-1.5 text-xs text-gray-400">
                    <Filter size={14} />
                    <select
                      value={labelFilter}
                      onChange={(e) => setLabelFilter(e.target.value)}
                      className="bg-transparent text-gray-200 focus:outline-none cursor-pointer text-xs"
                    >
                      <option value="all" className="bg-[#1C2538]">All Labels</option>
                      {uniqueLabels.map(label => (
                        <option key={label} value={label} className="bg-[#1C2538]">{label}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Table list */}
              <div className="overflow-x-auto max-h-[580px]">
                {filteredLogs.length === 0 ? (
                  <div className="py-20 text-center flex flex-col items-center justify-center space-y-4">
                    <div className="p-4 bg-gray-800/40 rounded-full text-gray-500">
                      <FileText size={32} />
                    </div>
                    <div>
                      <p className="text-base font-semibold text-gray-300">No logs found</p>
                      <p className="text-xs text-gray-500 mt-1">
                        {logs.length === 0 
                          ? "Upload a log CSV file on the left to start classification."
                          : "Try adjusting your search query or filters."
                        }
                      </p>
                    </div>
                  </div>
                ) : (
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-[#222E45] bg-[#171E2E]/50 text-xs font-semibold text-gray-400 uppercase tracking-wider">
                        <th className="py-3 px-4">Source</th>
                        <th className="py-3 px-4">Log Message</th>
                        <th className="py-3 px-4">Classification Label</th>
                        <th className="py-3 px-4 text-right">Method</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1E293B] text-xs">
                      {filteredLogs.map((log) => (
                        <tr 
                          key={log.id} 
                          className="hover:bg-[#1C2538]/50 transition-colors"
                        >
                          <td className="py-3.5 px-4 whitespace-nowrap">
                            {getSourceBadge(log.source)}
                          </td>
                          <td className="py-3.5 px-4 font-mono text-gray-200 break-all max-w-sm">
                            {log.log_message}
                          </td>
                          <td className="py-3.5 px-4 whitespace-nowrap">
                            {getLabelBadge(log.target_label)}
                          </td>
                          <td className="py-3.5 px-4 text-right whitespace-nowrap">
                            {getMethodBadge(log.classification_method)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>

              {/* Table Footer */}
              <div className="p-3 bg-[#171E2E]/60 border-t border-[#222E45] text-xs text-gray-500 flex justify-between items-center">
                <span>Showing {filteredLogs.length} of {logs.length} logs</span>
                <span>Max showing limit: 500 logs</span>
              </div>

            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
