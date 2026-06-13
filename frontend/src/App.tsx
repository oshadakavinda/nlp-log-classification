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
  Plus,
  Play,
  Pause,
  Radio
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
  user_corrected?: boolean;
}

interface LogsResponse {
  total: number;
  logs: LogEntry[];
}

interface LogStats {
  total: number;
  regex: number;
  ml: number;
  llm: number;
  label_distribution: Array<{ name: string; count: number }>;
  unique_labels: string[];
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

  // Correction and Inspection states
  const [editingLogId, setEditingLogId] = useState<number | null>(null);
  const [inspectingVersion, setInspectingVersion] = useState<ModelVersion | null>(null);
  const [devTab, setDevTab] = useState<'curl' | 'python' | 'node'>('curl');

  // Search & Filter State
  const [searchTerm, setSearchTerm] = useState('');
  const [methodFilter, setMethodFilter] = useState('all');
  const [labelFilter, setLabelFilter] = useState('all');

  // Live Activity Monitoring Feed State
  const [liveLogs, setLiveLogs] = useState<LogEntry[]>([]);

  // Live Log Stream Simulator State
  const [simulatorActive, setSimulatorActive] = useState(false);
  const [simulatorInterval, setSimulatorInterval] = useState(1500); // ms between log sends
  const [simulatorConsole, setSimulatorConsole] = useState<Array<{ timestamp: string; message: string; label: string; method: string; source: string; confidence: number }>>([]);
  const [simulatorLogsGenerated, setSimulatorLogsGenerated] = useState(0);
  const simulatorConsoleEndRef = useRef<HTMLDivElement | null>(null);
  const simulatorIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // API Queries
  const { data: logsData = { total: 0, logs: [] }, refetch: refetchLogs } = useQuery<LogsResponse>({
    queryKey: ['logs', searchTerm, methodFilter, labelFilter],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (searchTerm) params.append('search', searchTerm);
      if (methodFilter !== 'all') params.append('method', methodFilter);
      if (labelFilter !== 'all') params.append('label', labelFilter);

      const response = await fetch(`http://localhost:8000/api/logs?${params.toString()}`);
      if (!response.ok) throw new Error('Failed to fetch logs');
      return response.json();
    },
    staleTime: 5000,
  });

  const logs = logsData.logs;
  const totalMatchingLogsCount = logsData.total;

  const { data: logStats = { total: 0, regex: 0, ml: 0, llm: 0, label_distribution: [], unique_labels: [] }, refetch: refetchStats } = useQuery<LogStats>({
    queryKey: ['logStats'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/logs/stats');
      if (!response.ok) throw new Error('Failed to fetch log stats');
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

  // Autoscroll simulator console
  useEffect(() => {
    if (simulatorConsoleEndRef.current) {
      simulatorConsoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [simulatorConsole]);

  // --- Live Log Stream Simulator Templates ---
  const LOG_TEMPLATES = useMemo(() => [
    "GET /api/v2/users/{id} status=200 response_time={rt}ms client={ip}",
    "POST /api/v1/orders status=201 response_time={rt}ms client={ip}",
    "Database connection pool synchronized. Active connections: {conns}.",
    "User admin successfully logged in from IP {ip}.",
    "Payment gateway timeout after {rt}ms for transaction txn_{txnid}.",
    "File upload completed: path=/uploads/report_{fid}.pdf size={size}KB.",
    "Authentication token expired for user_id={uid}. Session terminated.",
    "Cache miss for key=user_profile:{uid}. Falling back to database.",
    "SSL certificate for *.example.com expires in 14 days. Renewal required.",
    "Rate limiting triggered for client {ip}: {conns} requests in 60s.",
    "Disk usage alert: /var/log at 92% capacity. Cleanup recommended.",
    "Scheduled backup completed successfully. Archive size: {size}MB.",
    "Memory usage critical: {conns}% of available RAM consumed by worker process.",
    "WARN: Deprecated API endpoint /api/v1/legacy accessed by client {ip}.",
    "ERROR: Unhandled exception in /api/v2/reports: NullReferenceException.",
    "Load balancer health check passed for node-{uid}. Latency: {rt}ms.",
    "New deployment detected: version v2.{fid}.0 rolling out to production.",
    "Firewall blocked suspicious request from {ip}: SQL injection pattern detected.",
    "Websocket connection established for session_{txnid}. Protocol: wss.",
    "Container orchestrator scaled service 'api-gateway' from 3 to 5 replicas.",
  ], []);

  const generateRandomLog = useCallback(() => {
    const template = LOG_TEMPLATES[Math.floor(Math.random() * LOG_TEMPLATES.length)];
    return template
      .replace('{id}', String(Math.floor(Math.random() * 9999) + 1))
      .replace('{rt}', String(Math.floor(Math.random() * 450) + 10))
      .replace('{ip}', `${Math.floor(Math.random() * 223) + 1}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`)
      .replace('{conns}', String(Math.floor(Math.random() * 90) + 5))
      .replace('{txnid}', Math.random().toString(36).substring(2, 10))
      .replace('{fid}', String(Math.floor(Math.random() * 9999)))
      .replace('{size}', String(Math.floor(Math.random() * 2048) + 64))
      .replace('{uid}', String(Math.floor(Math.random() * 99999) + 1000));
  }, [LOG_TEMPLATES]);

  // Simulator interval effect
  useEffect(() => {
    if (simulatorIntervalRef.current) {
      clearInterval(simulatorIntervalRef.current);
      simulatorIntervalRef.current = null;
    }

    if (!simulatorActive) return;

    let localCounter = 0;

    const sendLog = async () => {
      const logMsg = generateRandomLog();
      try {
        const res = await fetch('http://localhost:8000/api/logs/classify?save_to_db=true', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ log_message: logMsg }),
        });
        if (res.ok) {
          const data = await res.json();
          setSimulatorConsole(prev => [
            ...prev.slice(-49), // keep last 50 entries
            {
              timestamp: new Date().toISOString(),
              message: logMsg,
              label: data.target_label || 'Unknown',
              method: data.classification_method || 'N/A',
              source: data.source || 'Unknown',
              confidence: data.confidence || 0,
            },
          ]);
          setSimulatorLogsGenerated(prev => prev + 1);
          localCounter++;

          // Invalidate queries every 5 logs to update charts
          if (localCounter % 5 === 0) {
            queryClient.invalidateQueries({ queryKey: ['logs'] });
            queryClient.invalidateQueries({ queryKey: ['logStats'] });
          }
        }
      } catch (err) {
        console.error('Simulator error:', err);
      }
    };

    // Send first log immediately
    sendLog();
    simulatorIntervalRef.current = setInterval(sendLog, simulatorInterval);

    return () => {
      if (simulatorIntervalRef.current) {
        clearInterval(simulatorIntervalRef.current);
        simulatorIntervalRef.current = null;
      }
    };
  }, [simulatorActive, simulatorInterval, generateRandomLog, queryClient]);

  // Cleanup on unmount and refresh logs when simulator stops
  useEffect(() => {
    return () => {
      if (simulatorIntervalRef.current) {
        clearInterval(simulatorIntervalRef.current);
      }
    };
  }, []);

  const handleToggleSimulator = () => {
    if (simulatorActive) {
      // Stopping — refresh all data
      setSimulatorActive(false);
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['logs'] });
        queryClient.invalidateQueries({ queryKey: ['logStats'] });
      }, 500);
    } else {
      setSimulatorConsole([]);
      setSimulatorLogsGenerated(0);
      setSimulatorActive(true);
    }
  };

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
      queryClient.setQueryData(['logs'], { total: 0, logs: [] });
      queryClient.setQueryData(['logStats'], { total: 0, regex: 0, ml: 0, llm: 0, label_distribution: [], unique_labels: [] });
      refetchLogs();
      refetchStats();
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
            refetchStats();
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
      refetchStats();
    } catch (err) {
      console.error(err);
      alert("Failed to activate model version.");
    }
  };

  // Delete model version
  const handleDeleteModelVersion = async (versionId: number) => {
    if (!window.confirm("Are you sure you want to delete this model version? This will remove it from the database and delete its files from disk.")) return;
    try {
      const res = await fetch(`http://localhost:8000/api/logs/model-versions/${versionId}`, {
        method: 'DELETE'
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to delete model version");
      }
      refetchModelVersions();
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to delete model version.");
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

  const handleSaveLabelCorrection = async (logId: number, label: string) => {
    let finalLabel = label;
    if (label === '__custom__') {
      const customVal = window.prompt("Enter custom label name:");
      if (!customVal || !customVal.trim()) {
        setEditingLogId(null);
        return;
      }
      finalLabel = customVal.trim();
    }
    
    try {
      const res = await fetch(`http://localhost:8000/api/logs/${logId}/correct`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ corrected_label: finalLabel })
      });
      if (!res.ok) throw new Error("Failed to correct label");
      refetchLogs();
      refetchStats();
    } catch (err) {
      console.error(err);
      alert("Failed to correct label.");
    } finally {
      setEditingLogId(null);
    }
  };

  const handleTrainFromDB = async () => {
    setIsTraining(true);
    setTrainingLogs([]);
    try {
      const response = await fetch('http://localhost:8000/api/logs/train/db', {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Database training trigger failed');
      const data = await response.json();
      setTrainJobId(data.job_id);
      setTrainProgress({ processed: 0, total: 100, status: 'processing', message: 'Fetching DB logs...' });

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
            refetchLogs();
            refetchStats();
          }, 1500);
          pollLogs();
        }
      };
    } catch (err) {
      console.error(err);
      setIsTraining(false);
    }
  };

  const dbTrainingEligibleCount = useMemo(() => {
    return logs.filter(l => l.user_corrected || (l.confidence !== null && l.confidence >= 0.8)).length;
  }, [logs]);

  const inspectReportData = useMemo(() => {
    if (!inspectingVersion || !inspectingVersion.metrics_json) return [];
    try {
      const parsed = JSON.parse(inspectingVersion.metrics_json);
      return Object.keys(parsed)
        .filter(key => key !== 'accuracy' && key !== 'macro avg' && key !== 'weighted avg')
        .map(className => ({
          name: className,
          precision: parsed[className].precision,
          recall: parsed[className].recall,
          f1Score: parsed[className]['f1-score'],
          support: parsed[className].support,
        }));
    } catch (e) {
      return [];
    }
  }, [inspectingVersion]);

  // Dynamic stats computations
  const stats = useMemo(() => {
    const total = logStats.total;
    const regexCount = logStats.regex;
    const mlCount = logStats.ml;
    const llmCount = logStats.llm;

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
  }, [logStats]);

  // Chart data formatting
  const methodChartData = useMemo(() => {
    return [
      { name: 'Regex', value: logStats.regex, color: '#3B82F6' },
      { name: 'ML Model', value: logStats.ml, color: '#6366F1' },
      { name: 'LLM Fallback', value: logStats.llm, color: '#8B5CF6' }
    ].filter(d => d.value > 0);
  }, [logStats]);

  const labelChartData = logStats.label_distribution;

  // Unique labels list
  const uniqueLabels = logStats.unique_labels;

  // Filter logs logic (now managed by backend)
  const filteredLogs = logs;

  // Styling helper for classification method badge
  const getMethodBadge = (method: string | null) => {
    if (!method) return null;
    switch (method) {
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

  const getSourceBadge = (source: string) => {
    let colorClasses = "bg-gray-500/10 text-gray-400 border-gray-500/20";
    const cleanSource = source.trim();
    switch (cleanSource) {
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
      case 'DatabasePool':
        colorClasses = "bg-indigo-500/10 text-indigo-400 border-indigo-500/20";
        break;
      case 'AuthService':
        colorClasses = "bg-emerald-500/10 text-emerald-450 border-emerald-500/20";
        break;
      case 'PaymentGateway':
        colorClasses = "bg-yellow-500/10 text-yellow-400 border-yellow-500/20";
        break;
      case 'APIGateway':
        colorClasses = "bg-cyan-500/10 text-cyan-400 border-cyan-500/20";
        break;
      case 'SystemAgent':
        colorClasses = "bg-orange-500/10 text-orange-400 border-orange-500/20";
        break;
      case 'SecurityMonitor':
        colorClasses = "bg-red-500/10 text-red-400 border-red-500/20";
        break;
      case 'FileService':
        colorClasses = "bg-teal-500/10 text-teal-400 border-teal-500/20";
        break;
      case 'Orchestrator':
        colorClasses = "bg-purple-500/10 text-purple-400 border-purple-500/20";
        break;
      case 'LoadBalancer':
        colorClasses = "bg-pink-500/10 text-pink-400 border-pink-500/20";
        break;
      default:
        if (cleanSource.endsWith('Service')) {
          colorClasses = "bg-violet-500/10 text-violet-400 border-violet-500/20";
        }
        break;
    }
    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-semibold border ${colorClasses}`}>
        {cleanSource}
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
    <div className="min-h-screen text-gray-100 p-8 font-sans transition-all selection:bg-indigo-500/30 selection:text-indigo-200 relative overflow-hidden">
      {/* Background glowing shapes */}
      <div className="absolute top-[-10%] left-1/4 w-[500px] h-[500px] glow-purple rounded-full filter blur-[120px] pointer-events-none -z-10 animate-pulse-subtle" />
      <div className="absolute top-[30%] right-1/4 w-[400px] h-[400px] glow-blue rounded-full filter blur-[100px] pointer-events-none -z-10" />
      <div className="absolute bottom-[10%] left-1/3 w-[450px] h-[450px] glow-pink rounded-full filter blur-[110px] pointer-events-none -z-10 animate-pulse-subtle" />

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
          <div className="flex glass-card p-1 rounded-xl shadow-inner select-none animate-fade-in">
            <button
              onClick={() => setActiveTab('analytics')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${activeTab === 'analytics'
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-md scale-[1.02]'
                  : 'text-gray-400 hover:text-white hover:bg-white/5 hover:scale-[1.02]'
                }`}
            >
              <Layers size={15} /> Logs Analytics
            </button>
            <button
              onClick={() => setActiveTab('training')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${activeTab === 'training'
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-md scale-[1.02]'
                  : 'text-gray-400 hover:text-white hover:bg-white/5 hover:scale-[1.02]'
                }`}
            >
              <Terminal size={15} /> Model Training
            </button>
            <button
              onClick={() => setActiveTab('monitoring')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${activeTab === 'monitoring'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-md scale-[1.02]'
                  : 'text-gray-400 hover:text-white hover:bg-white/5 hover:scale-[1.02]'
                }`}
            >
              <Activity size={15} /> System Monitoring
            </button>
          </div>
        </header>

        {/* ==================== TAB 1: LOGS ANALYTICS ==================== */}
        {activeTab === 'analytics' && (
          <div className="animate-fade-in animate-slide-up space-y-8">
            {/* Metric Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="glass-card glass-card-hover rounded-2xl p-6 shadow-lg flex items-center space-x-4 group">
                <div className="p-3 bg-blue-500/10 text-blue-400 rounded-xl group-hover:scale-110 transition-transform duration-300">
                  <FileText size={28} />
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">Total Stored Logs</p>
                  <h3 className="text-3xl font-bold text-white mt-1 tracking-tight">{stats.total.toLocaleString()}</h3>
                </div>
              </div>

              <div className="glass-card glass-card-hover rounded-2xl p-6 shadow-lg flex items-center space-x-4 group">
                <div className="p-3 bg-indigo-500/10 text-indigo-400 rounded-xl group-hover:scale-110 transition-transform duration-300">
                  <Cpu size={28} />
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">ML & LLM Processed</p>
                  <h3 className="text-3xl font-bold text-white mt-1 tracking-tight">
                    {stats.complex.count.toLocaleString()}
                    <span className="text-sm font-normal text-indigo-400 ml-2">({stats.complex.percent}%)</span>
                  </h3>
                </div>
              </div>

              <div className="glass-card glass-card-hover rounded-2xl p-6 shadow-lg flex items-center space-x-4 group">
                <div className="p-3 bg-emerald-500/10 text-emerald-400 rounded-xl group-hover:scale-110 transition-transform duration-300">
                  <Zap size={28} />
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">Regex Pattern Matches</p>
                  <h3 className="text-3xl font-bold text-white mt-1 tracking-tight">
                    {stats.regex.count.toLocaleString()}
                    <span className="text-sm font-normal text-emerald-400 ml-2">({stats.regex.percent}%)</span>
                  </h3>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Upload panel (Left) */}
              <div className="lg:col-span-1 space-y-6">
                <div className="glass-card rounded-2xl p-6 shadow-lg animate-fade-in">
                  <h2 className="text-lg font-semibold text-white mb-2">Upload Logs to Classify</h2>
                  <p className="text-xs text-gray-400 mb-4">
                    Choose a CSV file with logs. The headers must map to <code className="bg-gray-800/50 text-gray-300 px-1.5 py-0.5 rounded font-mono">source</code> and <code className="bg-gray-800/50 text-gray-300 px-1.5 py-0.5 rounded font-mono">log_message</code>.
                  </p>

                  <div
                    {...getLogUploadProps()}
                    className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 ${isLogDragActive
                        ? 'border-blue-500 bg-blue-500/10 scale-[0.99]'
                        : 'border-[#2D3E5D] hover:border-blue-500/50 hover:bg-[#182030]/30'
                      }`}
                  >
                    <input {...getLogUploadInputProps()} />
                    <div className="flex flex-col items-center justify-center space-y-4">
                      <div className="p-3 bg-gray-800/80 rounded-full text-gray-400 transition-all duration-300 hover:scale-110">
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
                    <div className="mt-6 p-4 bg-[#182030]/65 rounded-xl border border-[#2D3E5D]/80 space-y-3 animate-fade-in">
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
                        <p className="text-emerald-450 text-xs mt-2 flex items-center gap-1 font-medium bg-emerald-500/10 p-2 rounded border border-emerald-500/20">
                          <CheckCircle size={14} /> Processing complete! Table updated.
                        </p>
                      )}
                    </div>
                  )}
                </div>

                {/* Active Model Indicator */}
                <div className="glass-card rounded-2xl p-6 shadow-lg space-y-4 animate-fade-in">
                  <h3 className="font-semibold text-white flex items-center gap-2 text-sm border-b border-gray-800/80 pb-2">
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
                <div className="glass-card rounded-2xl shadow-lg overflow-hidden animate-fade-in">

                  {/* Search and Filters */}
                  <div className="p-4 bg-[#171E2E]/40 border-b border-[#222E45]/60 flex flex-col sm:flex-row gap-3 items-center">
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
                                {editingLogId === log.id ? (
                                  <select
                                    defaultValue={log.target_label || ''}
                                    onChange={(e) => handleSaveLabelCorrection(log.id, e.target.value)}
                                    onBlur={() => setEditingLogId(null)}
                                    className="bg-[#1C2538] text-xs text-white border border-[#2D3E5D] rounded px-2 py-1 focus:outline-none focus:border-blue-500"
                                    autoFocus
                                  >
                                    <option value="" disabled>Select label...</option>
                                    {['INFO_ACCESS', 'SERVER_ERROR', 'SUCCESS', 'WARN', 'USER_ACTION', 'DATABASE_ERROR', 'NETWORK_TRAFFIC'].map(lbl => (
                                      <option key={lbl} value={lbl}>{lbl}</option>
                                    ))}
                                    {uniqueLabels.filter(lbl => !['INFO_ACCESS', 'SERVER_ERROR', 'SUCCESS', 'WARN', 'USER_ACTION', 'DATABASE_ERROR', 'NETWORK_TRAFFIC'].includes(lbl)).map(lbl => (
                                      <option key={lbl} value={lbl}>{lbl}</option>
                                    ))}
                                    <option value="__custom__">Custom label...</option>
                                  </select>
                                ) : (
                                  <div 
                                    className="flex items-center gap-1.5 group cursor-pointer" 
                                    onClick={() => setEditingLogId(log.id)}
                                    title="Click to correct label"
                                  >
                                    {getLabelBadge(log.target_label)}
                                    <span className="text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity text-[10px] ml-1">✏️</span>
                                  </div>
                                )}
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
                  <div className="p-3 bg-[#171E2E]/60 border-t border-[#222E45] text-xs text-gray-400 flex justify-between items-center">
                    <span>
                      Showing {filteredLogs.length} of {totalMatchingLogsCount.toLocaleString()} matching logs (Total stored: {logStats.total.toLocaleString()})
                    </span>
                    <button
                      onClick={handleClearLogs}
                      disabled={logStats.total === 0}
                      className="text-red-400/80 hover:text-red-400 font-medium disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1 transition-all"
                    >
                      <Trash2 size={13} /> Clear Stored Logs
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================== TAB 2: MODEL TRAINING ==================== */}
        {activeTab === 'training' && (
          <div className="animate-fade-in animate-slide-up grid grid-cols-1 lg:grid-cols-3 gap-8">

            {/* Left: Trigger panel & Active stats */}
            <div className="lg:col-span-1 space-y-6">

              {/* Training upload */}
              <div className="glass-card rounded-2xl p-6 shadow-lg">
                <h2 className="text-lg font-semibold text-white mb-2">Train Log Classifier</h2>
                <p className="text-xs text-gray-400 mb-4">
                  Upload a labeled training CSV dataset containing <code className="bg-gray-800 text-gray-300 px-1 rounded font-mono">log_message</code> and <code className="bg-gray-800 text-gray-300 px-1 rounded font-mono">target_label</code> columns to train a new Logistic Regression classifier.
                </p>

                <div
                  {...getTrainUploadProps()}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${isTraining ? 'opacity-40 cursor-not-allowed border-gray-700 bg-gray-900/10' :
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

                <div className="mt-4 flex flex-col gap-2">
                  <button
                    onClick={handleDownloadDataset}
                    className="w-full flex items-center justify-center gap-2 py-2 bg-[#1C2538] hover:bg-[#26324D] border border-[#2D3E5D] text-xs font-semibold rounded-xl text-indigo-300 transition-colors"
                  >
                    <Download size={14} /> Download Sample Training Dataset
                  </button>
                  <button
                    onClick={handleTrainFromDB}
                    disabled={isTraining || dbTrainingEligibleCount < 10}
                    className="w-full flex items-center justify-center gap-2 py-2 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-650 hover:to-purple-750 text-xs font-semibold rounded-xl text-white transition-all shadow-md disabled:opacity-40 disabled:cursor-not-allowed border border-indigo-550/20"
                    title={dbTrainingEligibleCount < 10 ? "Requires at least 10 high-confidence or manually corrected logs" : "Train ML model using SQLite collected logs"}
                  >
                    <Cpu size={14} /> Retrain on DB Logs ({dbTrainingEligibleCount} ready)
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
              <div className="glass-card rounded-2xl p-6 shadow-lg space-y-4 animate-fade-in">
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
              <div className="glass-card rounded-2xl shadow-lg p-6 space-y-4 animate-fade-in">
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
                <div className="glass-card rounded-2xl shadow-lg overflow-hidden animate-fade-in">
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
              <div className="glass-card rounded-2xl shadow-lg overflow-hidden animate-fade-in">
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
                            <td className="py-2.5 px-4 text-right flex justify-end gap-1.5 whitespace-nowrap">
                              <button
                                onClick={() => setInspectingVersion(version)}
                                className="px-2.5 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 text-xs font-semibold rounded-lg transition-colors"
                              >
                                Inspect
                              </button>
                              {!version.is_active && (
                                <>
                                  <button
                                    onClick={() => handleActivateModel(version.id)}
                                    className="px-2.5 py-1 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 border border-indigo-500/20 text-xs font-semibold rounded-lg transition-colors"
                                  >
                                    Activate
                                  </button>
                                  <button
                                    onClick={() => handleDeleteModelVersion(version.id)}
                                    className="px-2.5 py-1 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 text-xs font-semibold rounded-lg transition-colors"
                                  >
                                    Delete
                                  </button>
                                </>
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
          <div className="animate-fade-in animate-slide-up space-y-8">

            {/* System Status Indicators Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">

              {/* postgres */}
              <div className="glass-card glass-card-hover rounded-2xl p-5 shadow-lg flex items-center justify-between group">
                <div className="flex items-center space-x-3">
                  <div className={`p-2.5 rounded-xl transition-all duration-300 group-hover:scale-105 ${systemStatus?.database === 'healthy' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                    <Server size={22} />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">PostgreSQL DB</h4>
                    <p className="text-sm font-bold text-white mt-0.5 capitalize">
                      {systemStatus?.database || 'Checking...'}
                    </p>
                  </div>
                </div>
                <span className={`h-2.5 w-2.5 rounded-full ${systemStatus?.database === 'healthy' ? 'bg-emerald-500 animate-pulse-subtle' : 'bg-red-500'}`} />
              </div>

              {/* redis */}
              <div className="glass-card glass-card-hover rounded-2xl p-5 shadow-lg flex items-center justify-between group">
                <div className="flex items-center space-x-3">
                  <div className={`p-2.5 rounded-xl transition-all duration-300 group-hover:scale-105 ${systemStatus?.redis === 'healthy' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                    <Database size={22} />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Redis Cache</h4>
                    <p className="text-sm font-bold text-white mt-0.5 capitalize">
                      {systemStatus?.redis || 'Checking...'}
                    </p>
                  </div>
                </div>
                <span className={`h-2.5 w-2.5 rounded-full ${systemStatus?.redis === 'healthy' ? 'bg-emerald-500 animate-pulse-subtle' : 'bg-red-500'}`} />
              </div>

              {/* worker */}
              <div className="glass-card glass-card-hover rounded-2xl p-5 shadow-lg flex items-center justify-between group">
                <div className="flex items-center space-x-3">
                  <div className={`p-2.5 rounded-xl transition-all duration-300 group-hover:scale-105 ${systemStatus?.celery_worker === 'active' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                    <Cpu size={22} />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Celery Worker</h4>
                    <p className="text-sm font-bold text-white mt-0.5 capitalize">
                      {systemStatus?.celery_worker || 'Checking...'}
                    </p>
                  </div>
                </div>
                <span className={`h-2.5 w-2.5 rounded-full ${systemStatus?.celery_worker === 'active' ? 'bg-emerald-500 animate-pulse-subtle' : 'bg-red-500'}`} />
              </div>

              {/* groq */}
              <div className="glass-card glass-card-hover rounded-2xl p-5 shadow-lg flex items-center justify-between group">
                <div className="flex items-center space-x-3">
                  <div className={`p-2.5 rounded-xl transition-all duration-300 group-hover:scale-105 ${systemStatus?.groq_api === 'configured' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'}`}>
                    <Activity size={22} />
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Groq API Key</h4>
                    <p className="text-sm font-bold text-white mt-0.5 capitalize">
                      {systemStatus?.groq_api === 'configured' ? 'Configured' : 'Not Configured'}
                    </p>
                  </div>
                </div>
                <span className={`h-2.5 w-2.5 rounded-full ${systemStatus?.groq_api === 'configured' ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse-subtle'}`} />
              </div>

            </div>

            {/* Visual Recharts Charts Section */}
            {logs.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
                {/* Method Breakdown Chart */}
                <div className="glass-card rounded-2xl p-5 shadow-lg flex flex-col h-[320px]">
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
                <div className="glass-card rounded-2xl p-5 shadow-lg flex flex-col h-[320px]">
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
              <div className="glass-card rounded-2xl p-6 shadow-lg space-y-6">
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
              <div className="glass-card rounded-2xl p-6 shadow-lg flex flex-col">
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

            {/* ====== LIVE LOG STREAM SIMULATOR ====== */}
            <div className="glass-card rounded-2xl shadow-xl overflow-hidden animate-fade-in">
              <div className="p-5 bg-gradient-to-r from-[#131926] to-[#171E2E] border-b border-[#222E45]">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                  <div className="flex items-center gap-3">
                    <div className={`p-2.5 rounded-xl transition-all duration-300 ${simulatorActive
                        ? 'bg-emerald-500/15 text-emerald-400 shadow-lg shadow-emerald-500/10'
                        : 'bg-gray-700/20 text-gray-400'
                      }`}>
                      <Radio size={22} className={simulatorActive ? 'animate-pulse' : ''} />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        Live Log Stream Simulator
                        {simulatorActive && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] bg-emerald-500/10 text-emerald-400 rounded-full border border-emerald-500/20 font-mono uppercase tracking-wider animate-pulse">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Streaming
                          </span>
                        )}
                      </h3>
                      <p className="text-xs text-gray-500 mt-0.5">Generate synthetic production logs and stream them through the classification pipeline in real-time.</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    {/* Interval Speed Control */}
                    <div className="flex flex-col items-end gap-1">
                      <label className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Interval: {(simulatorInterval / 1000).toFixed(1)}s</label>
                      <input
                        type="range"
                        min={500}
                        max={3000}
                        step={100}
                        value={simulatorInterval}
                        onChange={(e) => setSimulatorInterval(Number(e.target.value))}
                        disabled={simulatorActive}
                        className="w-28 h-1 accent-indigo-500 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                      />
                    </div>

                    {/* Toggle Button */}
                    <button
                      onClick={handleToggleSimulator}
                      className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 shadow-lg ${simulatorActive
                          ? 'bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25 hover:shadow-red-500/10'
                          : 'bg-gradient-to-r from-emerald-500 to-cyan-500 text-white hover:shadow-emerald-500/25 hover:scale-[1.02]'
                        }`}
                    >
                      {simulatorActive ? <><Pause size={16} /> Stop Stream</> : <><Play size={16} /> Start Stream</>}
                    </button>
                  </div>
                </div>

                {/* Stats bar */}
                {simulatorLogsGenerated > 0 && (
                  <div className="mt-4 flex items-center gap-6 text-xs animate-fade-in">
                    <div className="flex items-center gap-1.5">
                      <span className="text-gray-500">Logs Generated:</span>
                      <span className="font-bold font-mono text-white">{simulatorLogsGenerated}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-gray-500">Speed:</span>
                      <span className="font-mono text-indigo-400">{(simulatorInterval / 1000).toFixed(1)}s/log</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-gray-500">Console Buffer:</span>
                      <span className="font-mono text-gray-300">{simulatorConsole.length}/50</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Scrolling Console */}
              <div className="bg-[#070A13] border-t border-[#242F4D]">
                <div className="p-3 bg-[#0C1120] border-b border-[#1E293B] flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1.5">
                      <span className="h-2.5 w-2.5 rounded-full bg-red-500/80" />
                      <span className="h-2.5 w-2.5 rounded-full bg-yellow-500/80" />
                      <span className="h-2.5 w-2.5 rounded-full bg-green-500/80" />
                    </div>
                    <span className="text-xs font-mono font-semibold text-gray-400 ml-1">simulator_output.log</span>
                  </div>
                  {simulatorActive && (
                    <span className="text-[10px] bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded font-mono uppercase tracking-wide animate-pulse">● Live</span>
                  )}
                </div>
                <div className="p-4 font-mono text-xs max-h-[240px] overflow-y-auto space-y-1.5 bg-[#090C16] min-h-[120px]">
                  {simulatorConsole.length === 0 ? (
                    <div className="text-gray-600 italic py-8 text-center">
                      {simulatorActive ? 'Initializing stream...' : 'Click "Start Stream" to begin generating synthetic production logs.'}
                    </div>
                  ) : (
                    simulatorConsole.map((entry, idx) => (
                      <div key={idx} className="flex gap-2 items-start group hover:bg-[#0E1425] px-1 py-0.5 rounded transition-colors animate-fade-in">
                        <span className="text-gray-600 select-none shrink-0">[{new Date(entry.timestamp).toLocaleTimeString()}]</span>
                        <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-bold border ${entry.method === 'ML' ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20' :
                            entry.method === 'Regex' ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' :
                              entry.method === 'LLM' ? 'bg-purple-500/10 text-purple-400 border-purple-500/20' :
                                'bg-gray-500/10 text-gray-400 border-gray-500/20'
                          }`}>{entry.method}</span>
                        <span className="text-cyan-400/70 shrink-0">{entry.source}</span>
                        <span className="text-gray-300 break-all">{entry.message}</span>
                        <span className="ml-auto text-emerald-400/80 shrink-0 font-semibold">→ {entry.label}</span>
                        <span className="text-gray-500 shrink-0">({(entry.confidence * 100).toFixed(0)}%)</span>
                      </div>
                    ))
                  )}
                  <div ref={simulatorConsoleEndRef} />
                </div>
              </div>
            </div>

            {/* ====== DEVELOPER INTEGRATION explorer ====== */}
            <div className="glass-card rounded-2xl shadow-xl overflow-hidden animate-fade-in p-6 space-y-4">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Settings size={18} className="text-indigo-400" /> Developer Integration API
                </h3>
                <p className="text-xs text-gray-500 mt-0.5">Integrate the real-time classification pipeline into your external microservices and scripts.</p>
              </div>

              <div className="flex gap-2 border-b border-[#222E45]/80 pb-2">
                {['curl', 'python', 'node'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setDevTab(tab as 'curl' | 'python' | 'node')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all ${
                      devTab === tab 
                        ? 'bg-[#1E293B] text-indigo-400 border border-indigo-500/20' 
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    {tab === 'node' ? 'Node.js' : tab}
                  </button>
                ))}
              </div>

              <div className="bg-[#070A13] border border-[#242F4D]/50 rounded-xl p-4 font-mono text-xs text-gray-300 relative group max-h-[200px] overflow-y-auto">
                {devTab === 'curl' && (
                  <pre className="whitespace-pre-wrap">
                    {`curl -X POST "http://localhost:8000/api/logs/classify?save_to_db=true" \\
  -H "Content-Type: application/json" \\
  -d '{"log_message": "GET /api/v1/checkout status=500 response_time=150ms client=10.0.0.5."}'`}
                  </pre>
                )}
                {devTab === 'python' && (
                  <pre className="whitespace-pre-wrap">
                    {`import requests

url = "http://localhost:8000/api/logs/classify?save_to_db=true"
payload = {
    "log_message": "GET /api/v1/checkout status=500 response_time=150ms client=10.0.0.5."
}
response = requests.post(url, json=payload)
print(response.json())`}
                  </pre>
                )}
                {devTab === 'node' && (
                  <pre className="whitespace-pre-wrap">
                    {`fetch("http://localhost:8000/api/logs/classify?save_to_db=true", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    log_message: "GET /api/v1/checkout status=500 response_time=150ms client=10.0.0.5."
  })
})
.then(res => res.json())
.then(data => console.log(data));`}
                  </pre>
                )}
              </div>
            </div>

          </div>
        )}

      </div>

      {/* inspectingVersion details modal */}
      {inspectingVersion && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
          <div className="glass-card w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden border border-gray-700/80 m-4 animate-scale-in">
            <div className="p-5 bg-[#171E2E] border-b border-gray-800 flex justify-between items-center">
              <div>
                <h3 className="font-bold text-white text-base">Model Version Details</h3>
                <p className="text-xs text-indigo-400 font-mono mt-0.5">{inspectingVersion.version_tag}</p>
              </div>
              <button 
                onClick={() => setInspectingVersion(null)}
                className="text-gray-450 hover:text-white hover:bg-white/5 p-1 rounded-lg transition-colors text-lg"
              >
                ✕
              </button>
            </div>
            <div className="p-6 space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                <div className="bg-[#182030]/50 p-3 rounded-xl border border-[#2D3E5D]/30">
                  <span className="text-gray-400 block mb-0.5">Records Used</span>
                  <span className="text-white font-mono font-bold">{inspectingVersion.num_records.toLocaleString()}</span>
                </div>
                <div className="bg-[#182030]/50 p-3 rounded-xl border border-[#2D3E5D]/30">
                  <span className="text-gray-400 block mb-0.5">Overall Accuracy</span>
                  <span className="text-emerald-400 font-mono font-bold">{(inspectingVersion.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div className="bg-[#182030]/50 p-3 rounded-xl border border-[#2D3E5D]/30">
                  <span className="text-gray-400 block mb-0.5">Dataset Source</span>
                  <span className="text-white font-semibold truncate block" title={inspectingVersion.dataset_name}>{inspectingVersion.dataset_name}</span>
                </div>
                <div className="bg-[#182030]/50 p-3 rounded-xl border border-[#2D3E5D]/30">
                  <span className="text-gray-400 block mb-0.5">Trained On</span>
                  <span className="text-white font-semibold block">{new Date(inspectingVersion.created_at).toLocaleDateString()}</span>
                </div>
              </div>

              <div className="border border-gray-800/80 rounded-xl overflow-hidden">
                <div className="p-3 bg-[#171E2E]/40 border-b border-gray-800/80">
                  <h4 className="text-xs font-semibold text-gray-300">Classification Report Metrics</h4>
                </div>
                <div className="overflow-x-auto max-h-[240px]">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-[#222E45]/80 bg-[#171E2E]/20 text-[10px] font-bold text-gray-400 uppercase">
                        <th className="py-2 px-4">Class Label</th>
                        <th className="py-2 px-4 text-center">Precision</th>
                        <th className="py-2 px-4 text-center">Recall</th>
                        <th className="py-2 px-4 text-center">F1-Score</th>
                        <th className="py-2 px-4 text-right">Support</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1E293B] text-[11px]">
                      {inspectReportData.length === 0 ? (
                        <tr>
                          <td colSpan={5} className="py-6 text-center text-gray-500 italic">No per-class metrics available.</td>
                        </tr>
                      ) : (
                        inspectReportData.map((row) => (
                          <tr key={row.name} className="hover:bg-[#1C2538]/30 transition-colors">
                            <td className="py-2 px-4 font-semibold text-gray-300">{row.name}</td>
                            <td className="py-2 px-4 text-center font-mono font-medium text-gray-300">{(row.precision * 100).toFixed(1)}%</td>
                            <td className="py-2 px-4 text-center font-mono font-medium text-gray-300">{(row.recall * 100).toFixed(1)}%</td>
                            <td className="py-2 px-4 text-center font-mono font-bold text-indigo-400">{(row.f1Score * 100).toFixed(1)}%</td>
                            <td className="py-2 px-4 text-right font-mono text-gray-400">{row.support}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="p-4 bg-[#111728] border-t border-gray-800 flex justify-end gap-2">
              <button
                onClick={() => setInspectingVersion(null)}
                className="px-4 py-2 bg-gray-850 hover:bg-gray-800 border border-gray-700/80 text-xs font-semibold rounded-lg text-white transition-colors"
              >
                Close
              </button>
              {!inspectingVersion.is_active && (
                <button
                  onClick={() => {
                    handleActivateModel(inspectingVersion.id);
                    setInspectingVersion(null);
                  }}
                  className="px-4 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-xs font-semibold rounded-lg text-white transition-colors"
                >
                  Activate Version
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
