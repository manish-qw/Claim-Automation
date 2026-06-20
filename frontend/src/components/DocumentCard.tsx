import React from 'react';
import { Upload, CheckCircle2, X, AlertCircle } from 'lucide-react';

export interface FileState {
  file: File | null;
  status: 'idle' | 'uploading' | 'scanning' | 'extracting' | 'success' | 'failed' | 'validation_error';
  timeMs?: number;
  result?: any;
  errorMsg?: string;
}

interface DocumentCardProps {
  docKey: string;
  title: string;
  desc: string;
  fileState: FileState;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>, docKey: string) => void;
  onRemoveFile: (docKey: string) => void;
}

export function DocumentCard({ docKey, title, desc, fileState, onFileChange, onRemoveFile }: DocumentCardProps) {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between gap-6 hover:border-slate-700 transition-all group relative overflow-hidden">
      <div className="flex flex-col gap-2">
        <div className="flex justify-between items-start">
          <h3 className="font-semibold text-slate-100 group-hover:text-white transition-colors text-sm">
            {title}
          </h3>
        </div>
        <p className="text-xs text-slate-400">
          {desc}
        </p>
      </div>

      {!fileState.file ? (
        <label className="border border-dashed border-slate-800 hover:border-indigo-500/50 bg-slate-950/40 hover:bg-slate-900/20 rounded-xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-3">
          <Upload className="w-8 h-8 text-indigo-500 group-hover:-translate-y-1 transition-transform" />
          <span className="text-xs font-semibold text-slate-300">Choose File</span>
          <span className="text-[10px] text-slate-500">PDF, PNG, JPG</span>
          <input 
            type="file" 
            className="hidden" 
            accept="image/*,application/pdf"
            onChange={(e) => onFileChange(e, docKey)} 
          />
        </label>
      ) : (
        <div className="flex items-center justify-between bg-slate-950/60 border border-slate-800/80 rounded-xl px-4 py-3 text-xs">
          <div className="flex items-center gap-2 overflow-hidden max-w-[80%]">
            <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0" />
            <span className="truncate text-slate-200 font-medium">{fileState.file.name}</span>
          </div>
          <button onClick={() => onRemoveFile(docKey)} className="text-slate-500 hover:text-red-400 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
      
      {fileState.status === 'failed' && (
        <div className="absolute inset-0 bg-red-950/90 rounded-2xl p-5 flex flex-col justify-center items-center text-center animate-fade-in border border-red-900/50 z-10 backdrop-blur-sm">
          <X className="text-red-500 mb-2" size={32} />
          <p className="text-white font-bold mb-1">Upload Failed</p>
          <p className="text-red-400 text-xs mb-4">{fileState.errorMsg}</p>
          <button 
            onClick={(e) => { e.stopPropagation(); onRemoveFile(docKey); }}
            className="px-4 py-1.5 bg-red-900/50 hover:bg-red-800 rounded text-xs font-medium text-white transition-colors border border-red-800"
          >
            Try Again
          </button>
        </div>
      )}

      {fileState.status === 'validation_error' && fileState.result && (
        <div className="absolute inset-0 bg-amber-950/95 rounded-2xl p-4 flex flex-col justify-center items-center text-center animate-fade-in border border-amber-900/80 z-20 overflow-y-auto">
          <AlertCircle className="text-amber-500 mb-2 shrink-0" size={28} />
          <p className="text-white font-bold text-sm mb-1 leading-tight">{fileState.result.title}</p>
          <p className="text-amber-200/80 text-[10px] mb-2 leading-snug">{fileState.result.message}</p>
          {fileState.result.missing_fields && fileState.result.missing_fields.length > 0 && (
            <div className="bg-amber-900/30 border border-amber-900/50 rounded p-2 mb-3 w-full text-left">
              <span className="text-[9px] text-amber-500 uppercase font-bold tracking-wider mb-1 block">Missing / Unclear</span>
              <p className="text-[10px] text-amber-100">{fileState.result.missing_fields.join(', ')}</p>
            </div>
          )}
          <button 
            onClick={(e) => { e.stopPropagation(); onRemoveFile(docKey); }}
            className="px-4 py-1.5 bg-amber-600 hover:bg-amber-500 rounded text-[10px] font-bold text-white transition-colors shadow-lg mt-auto w-full"
          >
            {fileState.result.action || "Re-upload Document"}
          </button>
        </div>
      )}
    </div>
  );
}
