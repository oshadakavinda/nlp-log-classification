import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
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
  Trash2,
  Server,
  Activity,
  Terminal,
  Layers,
  History,
  Download,
  AlertCircle,
  Settings,
  Plus
} from 'lucide-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell, 
  Tooltip, 
  Legend, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid 
} from 'recharts';

interface LogEntry {
  id: number;
  source: string;
  log_message: string;
  target_label: string | null;
  classification_method: string | null;
  confidence: number | null;
  created_at: string;
}

interface JobProgress {
  processed: number;
  total: number;
  status: string;
  message?: string;
  error?: string;
}

interface ModelVersion {
  id: number;
  version_tag: string;
  dataset_name: string;
  num_records: number;
  accuracy: number;
  metrics_json: string;
  is_active: boolean;
  created_at: string;
}

interface RegexRule {
  id: number;
  pattern: string;
  target_label: string;
  created_at: string;
}

interface SystemStatus {
  database: string;
  redis: string;
  celery_worker: string;
  groq_api: string;
}

interface ActiveModel {
  version_tag?: string;
  dataset_name?: string;
  num_records?: number;
  accuracy?: number;
  created_at?: string;
  report?: any;
  classes?: string[];
  status?: string;
  message?: string;
}

export default function App() {
  const queryClient = useQueryClient();
  
  // Navigation tabs: 'analytics' | 'training' | 'monitoring'
  const [activeTab, setActiveTab] = useState<'analytics' | 'training' | 'monitoring'>('analytics');
  
  const uploadWsRef = useRef<WebSocket | null>(null);
  const trainWsRef = useRef<WebSocket | null>(null);
  const terminalEndRef = useRef<HTMLDivElement | null>(null);
  
  // Upload Job State
  const [uploadJobId, setUploadJobId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<JobProgress>({ processed: 0, total: 1, status: '' });
  
  // Model Training Job State
  const [trainJobId, setTrainJobId] = useState<string | null>(null);
  const [trainProgress, setTrainProgress] = useState<JobProgress>({ processed: 0, total: 100, status: '' });
  const [trainingLogs, setTrainingLogs] = useState<Array<{ timestamp: string; message: string }>>([]);
  const [isTraining, setIsTraining] = useState(false);
  
  // Custom Regex Rule Form State
  const [newPattern, setNewPattern] = useState('');
  const [newTargetLabel, setNewTargetLabel] = useState('');
  const [isAddingRule, setIsAddingRule] = useState(false);
  
  // Search & Filter State
  const [searchTerm, setSearchTerm] = useState('');
  const [methodFilter, setMethodFilter] = useState('all');
  const [labelFilter, setLabelFilter] = useState('all');
  
  // Live Activity Monitoring Feed State
  const [liveLogs, setLiveLogs] = useState<LogEntry[]>([]);

  // API Queries
  const { data: logs = [], refetch: refetchLogs } = useQuery<LogEntry[]>({
    queryKey: ['logs'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/logs?limit=500');
      if (!response.ok) throw new Error('Failed to fetch logs');
      return response.json();
    },
    staleTime: 5000,
  });

  const { data: activeModel, refetch: refetchActiveModel } = useQuery<ActiveModel>({
    queryKey: ['activeModel'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/logs/active-model');
      if (!response.ok) throw new Error('Failed to fetch active model');
      return response.json();
    }
  });

  const { data: modelVersions = [], refetch: refetchModelVersions } = useQuery<ModelVersion[]>({
    queryKey: ['modelVersions'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/logs/model-versions');
      if (!response.ok) throw new Error('Failed to fetch model versions');
      return response.json();
    }
  });

  const { data: regexRules = [], refetch: refetchRegexRules } = useQuery<RegexRule[]>({
    queryKey: ['regexRules'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/logs/regex-rules');
      if (!response.ok) throw new Error('Failed to fetch regex rules');
      return response.json();
    }
  });

  const { data: systemStatus } = useQuery<SystemStatus>({
    queryKey: ['systemStatus'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/logs/system-status');
      if (!response.ok) throw new Error('Failed to fetch system status');
      return response.json();
    },
    refetchInterval: 10000, // Poll system health every 10s
  });

  // Keep live feed updated with recent log additions
  useEffect(() => {
    if (logs.length > 0) {
      setLiveLogs(logs.slice(0, 5));
    }
  }, [logs]);

  // Autoscroll terminal console
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [trainingLogs]);

  // Clean logs
  const handleClearLogs = async () => {
    if (!window.confirm("Are you sure you want to clear all stored logs? This action cannot be undone.")) {
      return;
    }
    try {
      const response = await fetch('http://localhost:8000/api/logs', {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to clear logs');
      queryClient.setQueryData(['logs'], []);
      refetchLogs();
    } catch (err) {
      console.error(err);
      alert("Failed to clear logs.");
    }
  };

  // Upload Log Dataset logic
  const handleLogUpload = useCallback(async (acceptedFiles: File[]) => {
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
      setUploadJobId(data.job_id);
      setUploadProgress({ processed: 0, total: 1, status: 'processing' });
      
      if (uploadWsRef.current) {
        uploadWsRef.current.close();
      }

      const ws = new WebSocket(`ws://localhost:8000/api/logs/ws/progress/${data.job_id}`);
      uploadWsRef.current = ws;
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        setUploadProgress(msg);
        if (msg.status === 'completed') {
          ws.close();
          uploadWsRef.current = null;
          setTimeout(() => {
            refetchLogs();
          }, 1000);
        }
      };
    } catch (err) {
      console.error(err);
    }
  }, [queryClient, refetchLogs]);

  const { getRootProps: getLogUploadProps, getInputProps: getLogUploadInputProps, isDragActive: isLogDragActive } = useDropzone({
    onDrop: handleLogUpload,
    accept: { 'text/csv': ['.csv'] }
  });

  // Model Training Dataset Upload logic
  const handleTrainingUpload = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setIsTraining(true);
    setTrainingLogs([]);

    try {
      const response = await fetch('http://localhost:8000/api/logs/train', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Training upload failed');
      const data = await response.json();
      setTrainJobId(data.job_id);
      setTrainProgress({ processed: 0, total: 100, status: 'processing', message: 'Reading dataset...' });
      
      if (trainWsRef.current) {
        trainWsRef.current.close();
      }

      const ws = new WebSocket(`ws://localhost:8000/api/logs/ws/progress/${data.job_id}`);
      trainWsRef.current = ws;
      
      const pollLogs = async () => {
        try {
          const res = await fetch(`http://localhost:8000/api/logs/train/logs/${data.job_id}`);
          if (res.ok) {
            const logsData = await res.json();
            setTrainingLogs(logsData);
          }
        } catch (e) {
          console.error(e);
        }
      };

      const logInterval = setInterval(pollLogs, 1500);

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        setTrainProgress(msg);
        
        if (msg.status === 'completed') {
          clearInterval(logInterval);
          ws.close();
          trainWsRef.current = null;
          setIsTraining(false);
          setTimeout(() => {
            refetchActiveModel();
            refetchModelVersions();
          }, 1500);
          pollLogs();
        }
      };
    } catch (err) {
      console.error(err);
      setIsTraining(false);
    }
  }, [refetchActiveModel, refetchModelVersions]);

  const { getRootProps: getTrainUploadProps, getInputProps: getTrainUploadInputProps, isDragActive: isTrainDragActive } = useDropzone({
    onDrop: handleTrainingUpload,
    accept: { 'text/csv': ['.csv'] },
    disabled: isTraining
  });

  // Activate model version
  const handleActivateModel = async (versionId: number) => {
    try {
      const res = await fetch(`http://localhost:8000/api/logs/model-versions/${versionId}/activate`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error("Failed to activate model");
      refetchActiveModel();
      refetchModelVersions();
      refetchLogs();
    } catch (err) {
      console.error(err);
      alert("Failed to activate model version.");
    }
  };

  // Add regex rule
  const handleAddRegexRule = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPattern.trim() || !newTargetLabel.trim()) return;
    setIsAddingRule(true);
    try {
      const res = await fetch('http://localhost:8000/api/logs/regex-rules', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pattern: newPattern, target_label: newTargetLabel })
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Failed to add rule");
      }
      setNewPattern('');
      setNewTargetLabel('');
      refetchRegexRules();
    } catch (err: any) {
      alert(err.message || "Failed to add regex rule");
    } finally {
      setIsAddingRule(false);
    }
  };

  // Delete regex rule
  const handleDeleteRegexRule = async (ruleId: number) => {
    if (!window.confirm("Are you sure you want to delete this regex rule?")) return;
    try {
      const res = await fetch(`http://localhost:8000/api/logs/regex-rules/${ruleId}`, {
        method: 'DELETE'
      });
      if (!res.ok) throw new Error("Failed to delete rule");
      refetchRegexRules();
    } catch (err) {
      console.error(err);
      alert("Failed to delete regex rule.");
    }
  };

  // Download training logs dataset
  const handleDownloadDataset = () => {
    window.open('http://localhost:8000/api/logs/download-dataset', '_blank');
  };

  // Dynamic stats computations
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

  // Chart data formatting
  const methodChartData = useMemo(() => {
    return [
      { name: 'Regex', value: stats.regex.count, color: '#3B82F6' },
      { name: 'ML Model', value: stats.ml.count, color: '#6366F1' },
      { name: 'LLM Fallback', value: stats.llm.count, color: '#8B5CF6' }
    ].filter(d => d.value > 0);
  }, [stats]);

  const labelChartData = useMemo(() => {
    const counts: Record<string, number> = {};
    logs.forEach(l => {
      const label = l.target_label || 'Unclassified';
      counts[label] = (counts[label] || 0) + 1;
    });
    return Object.keys(counts).map(name => ({
      name,
      count: counts[name]
    })).sort((a, b) => b.count - a.count).slice(0, 8); // Top 8 classes
  }, [logs]);

  // Unique labels list
  const uniqueLabels = useMemo(() => {
    const labels = new Set<string>();
    logs.forEach(l => {
      if (l.target_label) labels.add(l.target_label);
    });
    return Array.from(labels);
  }, [logs]);

  // Filter logs logic
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
    if (lower.includes('error') || lower.includes('fail') || lower.includes('crash')) {
      colorClasses = "bg-red-500/10 text-red-400 border-red-500/20";
    } else if (lower.includes('warn') || lower.includes('retire') || lower.includes('deprecation')) {
      colorClasses = "bg-yellow-500/10 text-yellow-400 border-yellow-500/20";
    } else if (lower.includes('action') || lower.includes('login') || lower.includes('create')) {
      colorClasses = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
    } else if (lower.includes('notification') || lower.includes('success') || lower.includes('backup') || lower.includes('complete')) {
      colorClasses = "bg-cyan-500/10 text-cyan-400 border-cyan-500/20";
    }

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-bold border ${colorClasses}`}>
        {label}
      </span>
    );
  };

  // Evaluation classification report format converter
  const classReport = useMemo(() => {
    if (!activeModel || !activeModel.report) return [];
    
    return Object.keys(activeModel.report)
      .filter(key => key !== 'accuracy' && key !== 'macro avg' && key !== 'weighted avg')
      .map(className => ({
        name: className,
        precision: activeModel.report[className].precision,
        recall: activeModel.report[className].recall,
        f1Score: activeModel.report[className]['f1-score'],
        support: activeModel.report[className].support,
      }));
  }, [activeModel]);

  return (
    <div className="min-h-screen bg-[#0B0F19] text-gray-100 p-8 font-sans transition-all selection:bg-indigo-500/30 selection:text-indigo-200">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header Section */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-gray-800 gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-emerald-500 animate-pulse" />
              <h1 className="text-3xl font-extrabold text-white tracking-tight bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-500 bg-clip-text text-transparent">
                Log Analytics Platform
              </h1>
            </div>
            <p className="text-gray-400 mt-1">Hybrid Log Classification & Real-Time Training Pipeline</p>
          </div>
          
          {/* Navigation Controls */}
          <div className="flex bg-[#131926]/90 border border-[#222E45] p-1 rounded-xl shadow-inner select-none animate-fade-in">
            <button 
              onClick={() => setActiveTab('analytics')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeTab === 'analytics' 
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-md' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/40'
              }`}
            >
              <Layers size={15} /> Logs Analytics
            </button>
            <button 
              onClick={() => setActiveTab('training')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeTab === 'training' 
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-md' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/40'
              }`}
            >
              <Terminal size={15} /> Model Training
            </button>
            <button 
              onClick={() => setActiveTab('monitoring')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeTab === 'monitoring' 
                  ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-md' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/40'
              }`}
            >
              <Activity size={15} /> System Monitoring
            </button>
          </div>
        </header>

        {/* ==================== TAB 1: LOGS ANALYTICS ==================== */}
        {activeTab === 'analytics' && (
          <>
            {/* Metric Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-[#131926] rounded-2xl p-6 border border-[#222E45] shadow-lg flex items-center space-x-4 hover:border-[#304161] transition-all group duration-300">
                 <div className="p-3 bg-blue-500/10 text-blue-400 rounded-xl group-hover:scale-105 transition-transform">
                    <FileText size={28} />
                 </div>
                 <div>
                   <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">Total Stored Logs</p>
                   <h3 className="text-3xl font-bold text-white mt-1">{stats.total.toLocaleString()}</h3>
                 </div>
              </div>
              
              <div className="bg-[#131926] rounded-2xl p-6 border border-[#222E45] shadow-lg flex items-center space-x-4 hover:border-[#304161] transition-all group duration-300">
                 <div className="p-3 bg-indigo-500/10 text-indigo-400 rounded-xl group-hover:scale-105 transition-transform">
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

              <div className="bg-[#131926] rounded-2xl p-6 border border-[#222E45] shadow-lg flex items-center space-x-4 hover:border-[#304161] transition-all group duration-300">
                 <div className="p-3 bg-emerald-500/10 text-emerald-400 rounded-xl group-hover:scale-105 transition-transform">
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
                  <h2 className="text-lg font-semibold text-white mb-2">Upload Logs to Classify</h2>
                  <p className="text-xs text-gray-400 mb-4">
                    Choose a CSV file with logs. The headers must map to <code className="bg-gray-800 text-gray-300 px-1 rounded font-mono">source</code> and <code className="bg-gray-800 text-gray-300 px-1 rounded font-mono">log_message</code>.
                  </p>
                  
                  <div 
                    {...getLogUploadProps()} 
                    className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                      isLogDragActive 
                        ? 'border-blue-500 bg-blue-500/5' 
                        : 'border-[#2D3E5D] hover:border-gray-500 hover:bg-[#182030]'
                    }`}
                  >
                    <input {...getLogUploadInputProps()} />
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

                  {/* Upload Progress */}
                  {uploadJobId && (
                    <div className="mt-6 p-4 bg-[#182030] rounded-xl border border-[#2D3E5D] space-y-3">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-gray-400 font-mono text-ellipsis overflow-hidden max-w-[150px]">Job: {uploadJobId}</span>
                        <span className="text-white font-semibold">
                          {Math.round((uploadProgress.processed / uploadProgress.total) * 100)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                        <div 
                          className="bg-blue-500 h-1.5 rounded-full transition-all duration-300 ease-out"
                          style={{ width: `${(uploadProgress.processed / uploadProgress.total) * 100}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-xs text-gray-400">
                        <span>Processed: {uploadProgress.processed}</span>
                        <span>Total: {uploadProgress.total}</span>
                      </div>
                      {uploadProgress.status === 'completed' && (
                        <p className="text-emerald-400 text-xs mt-2 flex items-center gap-1 font-medium bg-emerald-500/5 p-2 rounded border border-emerald-500/10">
                          <CheckCircle size={14} /> Processing complete! Table updated.
                        </p>
                      )}
                    </div>
                  )}
                </div>
                
                {/* Active Model Indicator */}
                <div className="bg-[#131926] rounded-2xl border border-[#222E45] p-6 shadow-lg space-y-4">
                  <h3 className="font-semibold text-white flex items-center gap-2 text-sm border-b border-gray-800 pb-2">
                    <Cpu size={16} className="text-indigo-400" /> Active Machine Learning Model
                  </h3>
                  {activeModel && activeModel.version_tag ? (
                    <div className="space-y-3 text-xs text-gray-400">
                      <div className="flex justify-between">
                        <span>Version Tag:</span>
                        <span className="font-mono font-semibold text-indigo-300">{activeModel.version_tag}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Dataset Used:</span>
                        <span className="text-gray-200 text-ellipsis overflow-hidden max-w-[150px]" title={activeModel.dataset_name}>{activeModel.dataset_name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Accuracy score:</span>
                        <span className="font-bold text-emerald-400">{((activeModel.accuracy ?? 0) * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Last Trained:</span>
                        <span>{new Date(activeModel.created_at || '').toLocaleDateString()}</span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-gray-500">No active custom model is loaded. Standard rules apply.</p>
                  )}
                </div>
              </div>

              {/* Search & Logs list (Right) */}
              <div className="lg:col-span-2 space-y-4">
                <div className="bg-[#131926] rounded-2xl border border-[#222E45] shadow-lg overflow-hidden">
                  
                  {/* Search and Filters */}
                  <div className="p-4 bg-[#171E2E] border-b border-[#222E45] flex flex-col sm:flex-row gap-3 items-center">
                    <div className="relative w-full sm:flex-1">
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
                    
                    <div className="flex gap-2 w-full sm:w-auto">
                      <div className="flex items-center gap-1 bg-[#1C2538] border border-[#2D3E5D] rounded-xl px-2 py-1.5 text-xs text-gray-400">
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

                      <div className="flex items-center gap-1 bg-[#1C2538] border border-[#2D3E5D] rounded-xl px-2 py-1.5 text-xs text-gray-400">
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

                  {/* Logs Table */}
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
                              ? "Upload a log dataset on the left to analyze."
                              : "Try adjusting filters or search term."
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
                            <th className="py-3 px-4">Label</th>
                            <th className="py-3 px-4">Confidence</th>
                            <th className="py-3 px-4 text-right">Method</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#1E293B] text-xs">
                          {filteredLogs.map((log) => (
                            <tr key={log.id} className="hover:bg-[#1C2538]/50 transition-colors">
                              <td className="py-3 px-4 whitespace-nowrap">
                                {getSourceBadge(log.source)}
                              </td>
                              <td className="py-3 px-4 font-mono text-gray-200 max-w-xs truncate" title={log.log_message}>
                                {log.log_message}
                              </td>
                              <td className="py-3 px-4 whitespace-nowrap">
                                {getLabelBadge(log.target_label)}
                              </td>
                              <td className="py-3 px-4 whitespace-nowrap font-mono font-medium">
                                {log.confidence !== null ? (
                                  <span className={log.confidence >= 0.8 ? "text-emerald-400" : log.confidence >= 0.65 ? "text-amber-400" : "text-red-400"}>
                                    {(log.confidence * 100).toFixed(1)}%
                                  </span>
                                ) : (
                                  <span className="text-gray-500">—</span>
                                )}
                              </td>
                              <td className="py-3 px-4 text-right whitespace-nowrap">
                                {getMethodBadge(log.classification_method)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>

                  {/* Footer */}
                  <div className="p-3 bg-[#171E2E]/60 border-t border-[#222E45] text-xs text-gray-500 flex justify-between items-center">
                    <span>Showing {filteredLogs.length} of {logs.length} logs</span>
                    <button 
                      onClick={handleClearLogs}
                      disabled={logs.length === 0}
                      className="text-red-400/80 hover:text-red-400 font-medium disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1 transition-all"
                    >
                      <Trash2 size={13} /> Clear Stored Logs
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* ==================== TAB 2: MODEL TRAINING ==================== */}
        {activeTab === 'training' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Left: Trigger panel & Active stats */}
            <div className="lg:col-span-1 space-y-6">
              
              {/* Training upload */}
              <div className="bg-[#131926] rounded-2xl border border-[#222E45] p-6 shadow-lg">
                <h2 className="text-lg font-semibold text-white mb-2">Train Log Classifier</h2>
                <p className="text-xs text-gray-400 mb-4">
                  Upload a labeled training CSV dataset containing <code className="bg-gray-800 text-gray-300 px-1 rounded font-mono">log_message</code> and <code className="bg-gray-800 text-gray-300 px-1 rounded font-mono">target_label</code> columns to train a new Logistic Regression classifier.
                </p>
                
                <div 
                  {...getTrainUploadProps()} 
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                    isTraining ? 'opacity-40 cursor-not-allowed border-gray-700 bg-gray-900/10' :
                    isTrainDragActive ? 'border-indigo-500 bg-indigo-500/5' : 'border-[#2D3E5D] hover:border-gray-500 hover:bg-[#182030]'
                  }`}
                >
                  <input {...getTrainUploadInputProps()} />
                  <div className="flex flex-col items-center justify-center space-y-4">
                    <div className="p-3 bg-gray-800/80 rounded-full text-indigo-400">
                      <Cpu size={28} className={isTraining ? "animate-spin" : ""} />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-gray-300">
                        {isTraining ? "Training running..." : "Upload Training CSV"}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {isTraining ? "Please wait for completion" : "Drag CSV here or click to upload"}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-4">
                  <button 
                    onClick={handleDownloadDataset}
                    className="w-full flex items-center justify-center gap-2 py-2 bg-[#1C2538] hover:bg-[#26324D] border border-[#2D3E5D] text-xs font-semibold rounded-xl text-indigo-300 transition-colors"
                  >
                    <Download size={14} /> Download Sample Training Dataset
                  </button>
                </div>

                {/* Training status card */}
                {trainJobId && (
                  <div className="mt-6 p-4 bg-[#182030] rounded-xl border border-[#2D3E5D] space-y-3">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-gray-400 font-mono">Job: {trainJobId.slice(0, 8)}...</span>
                      <span className="text-indigo-400 font-semibold">{trainProgress.processed}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                      <div 
                        className="bg-indigo-500 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${trainProgress.processed}%` }}
                      />
                    </div>
                    <div className="text-xs text-gray-300 font-medium">
                      Status: <span className="text-white">{trainProgress.message || "Initializing..."}</span>
                    </div>
                    {trainProgress.error && (
                      <div className="text-red-400 text-xs mt-1 p-2 bg-red-500/5 rounded border border-red-500/10 flex gap-1 items-start">
                        <AlertCircle size={14} className="shrink-0 mt-0.5" />
                        <span>Error: {trainProgress.error}</span>
                      </div>
                    )}
                    {trainProgress.status === 'completed' && !trainProgress.error && (
                      <p className="text-emerald-400 text-xs mt-2 flex items-center gap-1 font-medium bg-emerald-500/5 p-2 rounded border border-emerald-500/10 animate-fade-in">
                        <CheckCircle size={14} /> Training completed! Model updated.
                      </p>
                    )}
                  </div>
                )}
              </div>

              {/* Active model summary card */}
              <div className="bg-[#131926] rounded-2xl border border-[#222E45] p-6 shadow-lg space-y-4">
                <h3 className="font-semibold text-white flex items-center gap-2 text-sm border-b border-gray-800 pb-2">
                  <CheckCircle size={16} className="text-emerald-400" /> Active Model Overview
                </h3>
                {activeModel && activeModel.version_tag ? (
                  <div className="space-y-3 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Active Tag:</span>
                      <span className="font-mono font-bold text-gray-200">{activeModel.version_tag}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total Dataset:</span>
                      <span className="font-semibold text-gray-200">{activeModel.num_records?.toLocaleString()} records</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Overall Accuracy:</span>
                      <span className="font-bold text-emerald-400">{((activeModel.accuracy ?? 0) * 100).toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Trained Date:</span>
                      <span className="text-gray-200">{new Date(activeModel.created_at || '').toLocaleString()}</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-gray-500">No custom trained model is active.</p>
                )}
              </div>
            </div>

            {/* Right: Rules manager, Live Console, and Version Registry */}
            <div className="lg:col-span-2 space-y-6">
              
              {/* Custom Regex Rules Manager */}
              <div className="bg-[#131926] rounded-2xl border border-[#222E45] shadow-lg p-6 space-y-4">
                <h3 className="font-semibold text-white text-sm flex items-center gap-2 border-b border-gray-800 pb-2">
                  <Settings size={16} className="text-blue-400 animate-spin-slow" /> Custom Regex Rules Manager
                </h3>
                <p className="text-xs text-gray-400">
                  Define static rules to override the ML classifier. If a log message matches the regex pattern, it will instantly classify with 100% confidence.
                </p>
                
                {/* Add rule form */}
                <form onSubmit={handleAddRegexRule} className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="sm:col-span-2">
                    <input 
                      type="text" 
                      placeholder="Regex pattern (e.g. Inference completed.*)"
                      value={newPattern}
                      onChange={(e) => setNewPattern(e.target.value)}
                      className="w-full bg-[#1C2538] text-xs text-white px-3 py-2 rounded-xl border border-[#2D3E5D] focus:outline-none focus:border-blue-500"
                      required
                    />
                  </div>
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      placeholder="Target label (e.g. Success)"
                      value={newTargetLabel}
                      onChange={(e) => setNewTargetLabel(e.target.value)}
                      className="w-full bg-[#1C2538] text-xs text-white px-3 py-2 rounded-xl border border-[#2D3E5D] focus:outline-none focus:border-blue-500"
                      required
                    />
                    <button 
                      type="submit"
                      disabled={isAddingRule}
                      className="px-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl flex items-center justify-center shrink-0 transition-colors disabled:opacity-50"
                    >
                      <Plus size={16} />
                    </button>
                  </div>
                </form>

                {/* Rules List */}
                <div className="max-h-[160px] overflow-y-auto border border-[#1E293B] rounded-xl">
                  {regexRules.length === 0 ? (
                    <div className="py-6 text-center text-xs text-gray-500">No custom regex rules registered. Default system rules apply.</div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-[#222E45] bg-[#171E2E]/30 text-[10px] font-bold text-gray-400 uppercase">
                          <th className="py-2 px-3">Pattern</th>
                          <th className="py-2 px-3">Target Label</th>
                          <th className="py-2 px-3 text-right">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#1E293B] text-[11px]">
                        {regexRules.map((rule) => (
                          <tr key={rule.id} className="hover:bg-[#1C2538]/30 transition-colors">
                            <td className="py-2 px-3 font-mono text-gray-300 break-all">{rule.pattern}</td>
                            <td className="py-2 px-3">{getLabelBadge(rule.target_label)}</td>
                            <td className="py-2 px-3 text-right">
                              <button 
                                onClick={() => handleDeleteRegexRule(rule.id)}
                                className="text-red-400 hover:text-red-300 p-1 rounded hover:bg-red-500/10 transition-all"
                              >
                                <Trash2 size={13} />
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>

              {/* Scrollable Logs Console */}
              {isTraining && (
                <div className="bg-[#070A13] border border-[#242F4D] rounded-2xl shadow-xl overflow-hidden flex flex-col">
                  <div className="p-3 bg-[#111728] border-b border-[#242F4D] flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1.5">
                        <span className="h-2.5 w-2.5 rounded-full bg-red-500/80" />
                        <span className="h-2.5 w-2.5 rounded-full bg-yellow-500/80" />
                        <span className="h-2.5 w-2.5 rounded-full bg-green-500/80" />
                      </div>
                      <span className="text-xs font-mono font-semibold text-gray-400 ml-2">Training Console Logs</span>
                    </div>
                    <span className="text-[10px] bg-indigo-500/10 text-indigo-400 px-2 py-0.5 rounded font-mono uppercase tracking-wide animate-pulse">Live Stream</span>
                  </div>
                  <div className="p-4 font-mono text-xs text-indigo-300/95 max-h-[200px] overflow-y-auto space-y-2 bg-[#090C16] h-[200px]">
                    {trainingLogs.length === 0 ? (
                      <div className="text-gray-500 italic">Console starting. Waiting for log output...</div>
                    ) : (
                      trainingLogs.map((log, index) => (
                        <div key={index} className="flex gap-2">
                          <span className="text-gray-600 select-none">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
                          <span className="text-gray-200 break-all">{log.message}</span>
                        </div>
                      ))
                    )}
                    <div ref={terminalEndRef} />
                  </div>
                </div>
              )}

              {/* Classification Report Card */}
              {activeModel && activeModel.report && (
                <div className="bg-[#131926] rounded-2xl border border-[#222E45] shadow-lg overflow-hidden">
                  <div className="p-4 bg-[#171E2E] border-b border-[#222E45]">
                    <h3 className="font-semibold text-white text-sm flex items-center gap-2">
                      <Activity size={16} className="text-purple-400" /> Active Model Classification Report
                    </h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-[#222E45] bg-[#171E2E]/30 text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                          <th className="py-2.5 px-4">Class Label</th>
                          <th className="py-2.5 px-4 text-center">Precision</th>
                          <th className="py-2.5 px-4 text-center">Recall</th>
                          <th className="py-2.5 px-4 text-center">F1-Score</th>
                          <th className="py-2.5 px-4 text-right">Support</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#1E293B] text-xs">
                        {classReport.map((row) => (
                          <tr key={row.name} className="hover:bg-[#1C2538]/30 transition-colors">
                            <td className="py-2.5 px-4 font-semibold text-gray-300">{row.name}</td>
                            <td className="py-2.5 px-4 text-center font-mono font-medium text-gray-300">{(row.precision * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-4 text-center font-mono font-medium text-gray-300">{(row.recall * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-4 text-center font-mono font-bold text-indigo-400">{(row.f1Score * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-4 text-right font-mono text-gray-400">{row.support}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Trained Versions History Table */}
              <div className="bg-[#131926] rounded-2xl border border-[#222E45] shadow-lg overflow-hidden">
                <div className="p-4 bg-[#171E2E] border-b border-[#222E45]">
                  <h3 className="font-semibold text-white text-sm flex items-center gap-2">
                    <History size={16} className="text-indigo-400" /> Trained Model History & Version Registry
                  </h3>
                </div>
                
                {modelVersions.length === 0 ? (
                  <div className="py-10 text-center text-gray-500 text-xs">No trained models registered yet.</div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-[#222E45] bg-[#171E2E]/30 text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                          <th className="py-2.5 px-4">Version Tag</th>
                          <th className="py-2.5 px-4">Dataset Name</th>
                          <th className="py-2.5 px-4 text-center">Records</th>
                          <th className="py-2.5 px-4 text-center">Accuracy</th>
                          <th className="py-2.5 px-4 text-center">Active</th>
                          <th className="py-2.5 px-4 text-right">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#1E293B] text-xs">
                        {modelVersions.map((version) => (
                          <tr key={version.id} className="hover:bg-[#1C2538]/30 transition-colors">
                            <td className="py-2.5 px-4 font-mono font-semibold text-gray-300">{version.version_tag}</td>
                            <td className="py-2.5 px-4 text-gray-400 max-w-[140px] truncate" title={version.dataset_name}>{version.dataset_name}</td>
                            <td className="py-2.5 px-4 text-center font-mono">{version.num_records.toLocaleString()}</td>
                            <td className="py-2.5 px-4 text-center font-mono font-semibold text-emerald-400">{(version.accuracy * 100).toFixed(2)}%</td>
                            <td className="py-2.5 px-4 text-center">
                              {version.is_active ? (
                                <span className="inline-flex px-1.5 py-0.5 text-[10px] bg-emerald-500/10 text-emerald-400 rounded border border-emerald-500/20 font-bold uppercase">Active</span>
                              ) : (
                                <span className="text-gray-600">—</span>
                              )}
                            </td>
                            <td className="py-2.5 px-4 text-right">
                              {!version.is_active && (
                                <button 
                                  onClick={() => handleActivateModel(version.id)}
                                  className="px-2.5 py-1 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 border border-indigo-500/20 text-xs font-semibold rounded-lg transition-colors"
                                >
                                  Activate
                                </button>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

            </div>

          </div>
        )}

        {/* ==================== TAB 3: SYSTEM MONITORING ==================== */}
        {activeTab === 'monitoring' && (
          <div className="space-y-8 animate-fade-in">
            
            {/* System Status Indicators Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              
              {/* postgres */}
              <div className="bg-[#131926] rounded-2xl p-5 border border-[#222E45] shadow-lg flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2.5 rounded-xl ${systemStatus?.database === 'healthy' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                    <Server size={22} />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">PostgreSQL DB</h4>
                    <p className="text-sm font-bold text-white mt-0.5 capitalize">
                      {systemStatus?.database || 'Checking...'}
                    </p>
                  </div>
                </div>
                <span className={`h-2.5 w-2.5 rounded-full ${systemStatus?.database === 'healthy' ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
              </div>

              {/* redis */}
              <div className="bg-[#131926] rounded-2xl p-5 border border-[#222E45] shadow-lg flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2.5 rounded-xl ${systemStatus?.redis === 'healthy' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                    <Database size={22} />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Redis Cache</h4>
                    <p className="text-sm font-bold text-white mt-0.5 capitalize">
                      {systemStatus?.redis || 'Checking...'}
                    </p>
                  </div>
                </div>
                <span className={`h-2.5 w-2.5 rounded-full ${systemStatus?.redis === 'healthy' ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
              </div>

              {/* worker */}
              <div className="bg-[#131926] rounded-2xl p-5 border border-[#222E45] shadow-lg flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2.5 rounded-xl ${systemStatus?.celery_worker === 'active' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                    <Cpu size={22} />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Celery Worker</h4>
                    <p className="text-sm font-bold text-white mt-0.5 capitalize">
                      {systemStatus?.celery_worker || 'Checking...'}
                    </p>
                  </div>
                </div>
                <span className={`h-2.5 w-2.5 rounded-full ${systemStatus?.celery_worker === 'active' ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
              </div>

              {/* groq */}
              <div className="bg-[#131926] rounded-2xl p-5 border border-[#222E45] shadow-lg flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2.5 rounded-xl ${systemStatus?.groq_api === 'configured' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'}`}>
                    <Activity size={22} />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Groq API Key</h4>
                    <p className="text-sm font-bold text-white mt-0.5 capitalize">
                      {systemStatus?.groq_api === 'configured' ? 'Configured' : 'Not Configured'}
                    </p>
                  </div>
                </div>
                <span className={`h-2.5 w-2.5 rounded-full ${systemStatus?.groq_api === 'configured' ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'}`} />
              </div>

            </div>

            {/* Visual Recharts Charts Section */}
            {logs.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
                {/* Method Breakdown Chart */}
                <div className="bg-[#131926] rounded-2xl border border-[#222E45] p-5 shadow-lg flex flex-col h-[320px]">
                  <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Classification Methods Ratio</h4>
                  <div className="flex-1 min-h-0 relative">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={methodChartData}
                          innerRadius={60}
                          outerRadius={80}
                          paddingAngle={3}
                          dataKey="value"
                        >
                          {methodChartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#131926', border: '1px solid #2D3E5D', borderRadius: '10px' }}
                          itemStyle={{ color: '#fff', fontSize: '12px' }}
                        />
                        <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: '#9CA3AF' }} />
                      </PieChart>
                    </ResponsiveContainer>
                    {/* Inner Center text */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none mt-[-36px]">
                      <span className="text-[10px] uppercase text-gray-400 tracking-wider">Total Logs</span>
                      <span className="text-xl font-bold text-white">{stats.total.toLocaleString()}</span>
                    </div>
                  </div>
                </div>

                {/* Log Categories Distribution Bar Chart */}
                <div className="bg-[#131926] rounded-2xl border border-[#222E45] p-5 shadow-lg flex flex-col h-[320px]">
                  <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Log Categories Distribution</h4>
                  <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={labelChartData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222E45" vertical={false} />
                        <XAxis dataKey="name" stroke="#9CA3AF" fontSize={10} tickLine={false} />
                        <YAxis stroke="#9CA3AF" fontSize={10} tickLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#131926', border: '1px solid #2D3E5D', borderRadius: '10px' }}
                          itemStyle={{ color: '#fff', fontSize: '12px' }}
                          labelStyle={{ color: '#8B5CF6', fontSize: '11px', fontWeight: 'bold' }}
                        />
                        <Bar dataKey="count" fill="url(#barGradient)" radius={[4, 4, 0, 0]}>
                          {labelChartData.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#6366F1' : '#3B82F6'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {/* Visual Analytics grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              
              {/* Method Usage progress bars */}
              <div className="bg-[#131926] rounded-2xl border border-[#222E45] p-6 shadow-lg space-y-6">
                <h3 className="text-base font-semibold text-white border-b border-gray-800 pb-3 flex items-center gap-2">
                  <Zap size={16} className="text-blue-400" /> Pipeline Method Performance
                </h3>
                
                {logs.length === 0 ? (
                  <div className="py-20 text-center text-xs text-gray-500">No logs classified. Upload data to view classification pipeline analytics.</div>
                ) : (
                  <div className="space-y-5">
                    
                    {/* Regex usage */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs font-medium">
                        <span className="text-blue-400 flex items-center gap-1"><Zap size={12} /> Regex Pattern Matcher</span>
                        <span className="text-white font-bold">{stats.regex.count} logs ({stats.regex.percent}%)</span>
                      </div>
                      <div className="w-full bg-gray-800 rounded-full h-2">
                        <div className="bg-blue-500 h-2 rounded-full transition-all duration-500" style={{ width: `${stats.regex.percent}%` }} />
                      </div>
                    </div>

                    {/* ML usage */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs font-medium">
                        <span className="text-indigo-400 flex items-center gap-1"><Cpu size={12} /> BERT ML Classifier</span>
                        <span className="text-white font-bold">{stats.ml.count} logs ({stats.ml.percent}%)</span>
                      </div>
                      <div className="w-full bg-gray-800 rounded-full h-2">
                        <div className="bg-indigo-500 h-2 rounded-full transition-all duration-500" style={{ width: `${stats.ml.percent}%` }} />
                      </div>
                    </div>

                    {/* LLM usage */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs font-medium">
                        <span className="text-purple-400 flex items-center gap-1"><Database size={12} /> Groq LLM (Fallback / Legacy)</span>
                        <span className="text-white font-bold">{stats.llm.count} logs ({stats.llm.percent}%)</span>
                      </div>
                      <div className="w-full bg-gray-800 rounded-full h-2">
                        <div className="bg-purple-500 h-2 rounded-full transition-all duration-500" style={{ width: `${stats.llm.percent}%` }} />
                      </div>
                    </div>

                  </div>
                )}
              </div>

              {/* Real-time Activity Feed */}
              <div className="bg-[#131926] rounded-2xl border border-[#222E45] p-6 shadow-lg flex flex-col">
                <h3 className="text-base font-semibold text-white border-b border-gray-800 pb-3 flex justify-between items-center">
                  <span className="flex items-center gap-2"><Activity size={16} className="text-pink-400 animate-pulse" /> Live Log Classification Stream</span>
                  <span className="text-[10px] text-gray-500">Last 5 activities</span>
                </h3>
                
                {liveLogs.length === 0 ? (
                  <div className="py-20 text-center text-xs text-gray-500 flex-1 flex items-center justify-center">No live log activity recorded.</div>
                ) : (
                  <div className="divide-y divide-[#1E293B] flex-1 overflow-y-auto">
                    {liveLogs.map((log) => (
                      <div key={log.id} className="py-3 flex justify-between items-start text-xs border-b border-[#1E293B] last:border-0 hover:bg-[#1C2538]/20 px-2 rounded-lg transition-colors animate-fade-in">
                        <div className="space-y-1 max-w-[70%]">
                          <p className="font-mono text-gray-300 truncate" title={log.log_message}>{log.log_message}</p>
                          <div className="flex items-center gap-2">
                            {getSourceBadge(log.source)}
                            <span className="text-[10px] text-gray-500 font-mono">{new Date(log.created_at).toLocaleTimeString()}</span>
                          </div>
                        </div>
                        <div className="text-right space-y-1">
                          <div>{getLabelBadge(log.target_label)}</div>
                          <div>{getMethodBadge(log.classification_method)}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

            </div>

          </div>
        )}

      </div>
    </div>
  );
}
