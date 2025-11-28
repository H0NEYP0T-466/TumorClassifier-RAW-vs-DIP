import React, { useState, useRef, type ChangeEvent, type DragEvent } from 'react';
import { 
  BrainCircuit, 
  Scan, 
  Upload, 
  Activity, 
  FileCheck, 
  ChevronRight, 
  Zap,
  Binary,
  X,
  AlertCircle,
  CheckCircle,
  type LucideIcon
} from 'lucide-react';

// API base URL - adjust if your backend runs on a different port
const API_BASE_URL = 'http://localhost:8888';

interface PredictionResult {
  success: boolean;
  prediction: number;
  class_name: string;
  message: string;
}

const LandingPage: React.FC = () => {
  // State for image preview URL
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  // State for the actual file (for upload)
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  // State for drag-and-drop visual feedback
  const [isDragging, setIsDragging] = useState<boolean>(false);
  // State for loading during prediction
  const [isLoading, setIsLoading] = useState<boolean>(false);
  // State for prediction result
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  // State for error message
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // Reference to the hidden file input
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- Handlers ---

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(URL.createObjectURL(file));
      setSelectedFile(file);
      // Clear previous results
      setPredictionResult(null);
      setErrorMessage(null);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedImage(URL.createObjectURL(file));
      setSelectedFile(file);
      // Clear previous results
      setPredictionResult(null);
      setErrorMessage(null);
    }
  };

  const triggerUpload = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    
    setIsLoading(true);
    setPredictionResult(null);
    setErrorMessage(null);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }
      
      const result: PredictionResult = await response.json();
      setPredictionResult(result);
      console.log('Prediction result:', result);
      
    } catch (error) {
      console.error('Prediction error:', error);
      setErrorMessage(error instanceof Error ? error.message : 'An error occurred during prediction');
    } finally {
      setIsLoading(false);
    }
  };

  const resetImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedImage(null);
    setSelectedFile(null);
    setPredictionResult(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const dismissResult = () => {
    setPredictionResult(null);
    setErrorMessage(null);
  };

  return (
    <div className="h-screen w-full bg-[#0a0f1c] text-slate-200 overflow-hidden relative selection:bg-cyan-500/30 font-sans">
      
      {/* Background Gradients (Ambient Lighting) */}
      <div className="absolute top-[-20%] right-[-10%] w-[800px] h-[800px] bg-cyan-900/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] left-[-10%] w-[600px] h-[600px] bg-blue-900/10 rounded-full blur-[100px] pointer-events-none" />

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 h-full max-w-screen-2xl mx-auto">
        
        {/* --- LEFT PANEL: Information & Context --- */}
        <div className="lg:col-span-5 flex flex-col justify-center p-12 lg:p-16 relative z-10">
          
          {/* Logo */}
          <div className="flex items-center gap-3 mb-12">
            <div className="w-10 h-10 bg-cyan-500/10 border border-cyan-500/20 rounded-xl flex items-center justify-center text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
              <BrainCircuit size={24} />
            </div>
            <span className="text-2xl font-bold tracking-tight text-white">
              Tumor<span className="text-cyan-400">Classifier</span>
            </span>
          </div>

          {/* Headline */}
          <h1 className="text-5xl lg:text-6xl font-extrabold text-white leading-[1.1] mb-6 tracking-tight">
            Brain Tumor <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
              Detection AI
            </span>
          </h1>

          {/* Description */}
          <p className="text-lg text-slate-400 mb-8 max-w-md leading-relaxed">
            A comparative study of Raw Image Processing vs Digital Image Processing (DIP) pipelines utilizing Linear SVM classifiers for high-precision diagnosis.
          </p>

          {/* Feature Tags */}
          <div className="flex flex-wrap gap-4 mb-10">
            <FeatureTag icon={Binary} text="DIP Enhanced Pipeline" color="text-cyan-400" />
            <FeatureTag icon={Activity} text="Linear SVM Model" color="text-emerald-400" />
          </div>

          {/* Footer Status */}
          <div className="mt-auto pt-8 border-t border-slate-800/50 flex gap-6 text-xs text-slate-500 font-mono">
            <span>V 1.0.0 RELEASE</span>
            <span>SYSTEM IDLE</span>
            <span className="text-cyan-500/80 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
              ONLINE
            </span>
          </div>
        </div>

        {/* --- RIGHT PANEL: Interactive Upload Area --- */}
        <div className="lg:col-span-7 h-full bg-slate-900/30 backdrop-blur-sm border-l border-white/5 relative flex items-center justify-center p-8">
          
          {/* Decorative Background Grid */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />

          {/* Glass Card Container */}
          <div className="w-full max-w-lg bg-[#0f1623]/80 backdrop-blur-xl border border-slate-700/50 rounded-3xl shadow-2xl p-8 relative z-20 overflow-hidden">
            
            {/* Card Header */}
            <div className="flex justify-between items-center mb-8">
              <div>
                <h3 className="text-xl font-semibold text-white">MRI Analysis</h3>
                <p className="text-sm text-slate-400">Upload scan for classification</p>
              </div>
              <div className="p-2 bg-slate-800 rounded-lg border border-slate-700">
                <Scan size={20} className="text-cyan-400" />
              </div>
            </div>

            {/* Upload Zone */}
            <div 
              className={`
                relative group cursor-pointer transition-all duration-300 ease-in-out
                border-2 border-dashed rounded-2xl h-64 flex flex-col items-center justify-center
                ${isDragging 
                  ? 'border-cyan-500 bg-cyan-500/10' 
                  : 'border-slate-700 hover:border-slate-500 hover:bg-slate-800/30 bg-slate-900/50'
                }
                ${selectedImage ? 'border-none p-0 overflow-hidden' : ''}
              `}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={triggerUpload}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept="image/*"
                onChange={handleFileChange}
              />

              {selectedImage ? (
                <>
                  <img 
                    src={selectedImage} 
                    alt="Preview" 
                    className="w-full h-full object-cover rounded-2xl opacity-90 transition-transform duration-700 hover:scale-105" 
                  />
                  {/* Success Overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-transparent to-transparent flex items-end p-6 pointer-events-none">
                    <div className="flex items-center gap-2 text-white">
                      <FileCheck size={18} className="text-emerald-400" />
                      <span className="text-sm font-medium">Scan Uploaded</span>
                    </div>
                  </div>
                  {/* Reset Button */}
                  <button 
                    onClick={resetImage}
                    className="absolute top-4 right-4 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full backdrop-blur-md transition-colors border border-white/10 z-30"
                  >
                    <Upload size={16} />
                  </button>
                </>
              ) : (
                <div className="text-center space-y-4 pointer-events-none">
                  <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-2 group-hover:scale-110 transition-transform duration-300 border border-slate-700">
                    <Upload size={28} className="text-cyan-400" />
                  </div>
                  <div>
                    <p className="text-slate-200 font-medium">Click or drag MRI scan here</p>
                    <p className="text-slate-500 text-sm mt-1">Supports JPG, PNG, DICOM</p>
                  </div>
                </div>
              )}
            </div>

            {/* Action Buttons Area */}
            <div className="mt-8">
              {selectedImage ? (
                <button 
                  onClick={handleAnalyze}
                  disabled={isLoading}
                  className={`w-full group relative overflow-hidden bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-300 shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:shadow-[0_0_30px_rgba(6,182,212,0.5)] transform active:scale-[0.98] ${isLoading ? 'opacity-70 cursor-not-allowed' : ''}`}
                >
                  <div className="flex items-center justify-center gap-2 relative z-10">
                    {isLoading ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <Zap size={20} className="fill-white" />
                        <span>Run Diagnostic Analysis</span>
                      </>
                    )}
                  </div>
                  {/* Shine Animation Effect */}
                  {!isLoading && (
                    <div className="absolute top-0 -left-full w-1/2 h-full bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-[25deg] group-hover:animate-shine" />
                  )}
                </button>
              ) : (
                <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-slate-600" />
                    <span className="text-slate-400 text-sm">Waiting for input...</span>
                  </div>
                  <ChevronRight size={16} className="text-slate-600" />
                </div>
              )}
            </div>

          </div>
        </div>
      </div>

      {/* Toggle Message for Prediction Result */}
      {(predictionResult || errorMessage) && (
        <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50 animate-slide-up">
          <div 
            className={`
              flex items-center gap-4 px-6 py-4 rounded-2xl shadow-2xl backdrop-blur-xl border
              ${predictionResult?.prediction === 1 
                ? 'bg-red-900/80 border-red-500/50 text-red-100' 
                : predictionResult?.prediction === 0
                  ? 'bg-emerald-900/80 border-emerald-500/50 text-emerald-100'
                  : 'bg-red-900/80 border-red-500/50 text-red-100'
              }
            `}
          >
            {/* Icon */}
            <div className={`
              w-12 h-12 rounded-full flex items-center justify-center
              ${predictionResult?.prediction === 1 
                ? 'bg-red-500/30' 
                : predictionResult?.prediction === 0
                  ? 'bg-emerald-500/30'
                  : 'bg-red-500/30'
              }
            `}>
              {errorMessage ? (
                <AlertCircle size={24} />
              ) : predictionResult?.prediction === 1 ? (
                <AlertCircle size={24} />
              ) : (
                <CheckCircle size={24} />
              )}
            </div>

            {/* Message Content */}
            <div className="flex flex-col">
              <span className="font-bold text-lg">
                {errorMessage ? 'Error' : predictionResult?.class_name}
              </span>
              <span className="text-sm opacity-80">
                {errorMessage || predictionResult?.message}
              </span>
            </div>

            {/* Dismiss Button */}
            <button 
              onClick={dismissResult}
              className="ml-4 p-2 rounded-full hover:bg-white/10 transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>
      )}
      
      {/* Inline Styles for Custom Shine Animation */}
      <style>{`
        @keyframes shine {
          100% { left: 125%; }
        }
        .animate-shine {
          animation: shine 0.75s;
        }
        @keyframes slide-up {
          from {
            opacity: 0;
            transform: translateX(-50%) translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
          }
        }
        .animate-slide-up {
          animation: slide-up 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

// --- Sub-Components ---

interface FeatureTagProps {
  icon: LucideIcon;
  text: string;
  color: string;
}

const FeatureTag: React.FC<FeatureTagProps> = ({ icon: Icon, text, color }) => (
  <div className="px-4 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg backdrop-blur-sm flex items-center gap-2 transition-colors hover:bg-slate-800 hover:border-cyan-500/30">
    <Icon size={16} className={color} />
    <span className="text-sm font-medium text-slate-300">{text}</span>
  </div>
);

export default LandingPage;