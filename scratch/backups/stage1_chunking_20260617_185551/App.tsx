import React, { useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import { DocumentCard, FileState } from './components/DocumentCard';

type DocType = 'CLAIMANT_STATEMENT_FORM' | 'DEATH_CERTIFICATE' | 'IDENTITY_PROOF' | 'ADDRESS_PROOF' | 'PAN_CARD' | 'BANK_PROOF';

const STEP1_DOC_CONFIG: Record<DocType, { title: string; desc: string }> = {
  CLAIMANT_STATEMENT_FORM: { title: 'Claimant Statement', desc: 'Signed claimant statement details form.' },
  DEATH_CERTIFICATE: { title: 'Death Certificate', desc: 'Official death registry certificate confirming details.' },
  IDENTITY_PROOF: { title: 'Photo Identity Proof', desc: 'Upload Aadhaar, Passport, DL, or Voter ID. Auto-detected.' },
  ADDRESS_PROOF: { title: 'Address Proof', desc: 'Upload Aadhaar, Passport, DL, or Voter ID. Auto-detected.' },
  PAN_CARD: { title: 'PAN Card / Form 60', desc: 'Permanent Account Number card.' },
  BANK_PROOF: { title: 'Bank Account Proof', desc: 'Cancelled cheque / bank statement.' },
};

const NATURAL_DOC_CONFIG: Record<string, { title: string; desc: string }> = {
  MEDICO_LEGAL_CERT: { title: 'Medico-legal Certificate', desc: 'Medical cause of death certificate' },
  HOSPITALIZATION_RECORDS: { title: 'Hospitalization Records', desc: 'Admission, ICPs, Discharge, Labs, etc. (Combine into one PDF/Image)' },
  TREATING_DOCTOR_CERT: { title: 'Treating Doctor Certificate', desc: 'Duly filled by the treating doctor.' },
  HOSPITAL_ATTENDANT_CERT: { title: 'Hospital Attendant Certificate', desc: 'Duly filled medical attendant certificate.' },
  EMPLOYER_CERT: { title: 'Employer Certificate', desc: 'Only if Life Assured was a salaried individual.' },
};

const UNNATURAL_DOC_CONFIG: Record<string, { title: string; desc: string }> = {
  MEDICO_LEGAL_CERT: { title: 'Medico-legal Certificate', desc: 'Medical cause of death certificate' },
  FIR: { title: 'FIR', desc: 'First Information Report from police.' },
  INQUEST_REPORT: { title: 'Inquest / Panchnama', desc: 'Inquest or Panchnama Report.' },
  FINAL_POLICE_REPORT: { title: 'Final Police Report', desc: 'Final police investigation report.' },
  POSTMORTEM_REPORT: { title: 'Postmortem Report (PMR)', desc: 'Issued by the hospital.' },
  VISCERA_REPORT: { title: 'Viscera / Chemical Report', desc: 'Viscera or chemical examination report.' },
  NEWSPAPER_CUTTING: { title: 'Newspaper Cutting', desc: 'News cutting regarding the incident, if any.' },
  DRIVING_LICENCE_STEP2: { title: 'Driving License', desc: 'Of the Life Assured if death due to road accident.' },
  HOSPITALIZATION_RECORDS: { title: 'Hospitalization Records', desc: 'If any treatment was provided.' },
  HOSPITAL_ATTENDANT_CERT: { title: 'Hospital Attendant Certificate', desc: 'Duly filled medical attendant certificate.' },
  EMPLOYER_CERT: { title: 'Employer Certificate', desc: 'Only if Life Assured was a salaried individual.' },
};

const createInitialState = (config: Record<string, any>) => {
  const state: Record<string, FileState> = {};
  Object.keys(config).forEach(k => { state[k] = { file: null, status: 'idle' }; });
  return state;
};

export default function App() {
  const [currentStep, setCurrentStep] = useState<1 | 2>(1);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [deathCategory, setDeathCategory] = useState<'NATURAL_OR_MEDICAL' | 'UNNATURAL' | null>(null);
  const [showCategoryModal, setShowCategoryModal] = useState(false);
  const [modalDetectionType, setModalDetectionType] = useState<'AUTO' | 'MANUAL'>('MANUAL');

  // Step 1 Files
  const [files, setFiles] = useState<Record<string, FileState>>(createInitialState(STEP1_DOC_CONFIG));
  
  // Step 2 Files
  const [step2Files, setStep2Files] = useState<Record<string, FileState>>({});

  const activeFiles = currentStep === 1 ? files : step2Files;
  const setActiveFiles = currentStep === 1 ? setFiles : setStep2Files;
  const activeConfig = currentStep === 1 ? STEP1_DOC_CONFIG : (deathCategory === 'NATURAL_OR_MEDICAL' ? NATURAL_DOC_CONFIG : UNNATURAL_DOC_CONFIG);

  const isProceedReady = Object.values(activeFiles).some(f => f.file !== null);
  const hasResults = Object.values(activeFiles).some(f => f.result !== null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, docType: string) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setActiveFiles(prev => ({
        ...prev,
        [docType]: { file, status: 'idle' }
      }));
    }
  };

  const removeFile = (docType: string) => {
    setActiveFiles(prev => ({
      ...prev,
      [docType]: { file: null, status: 'idle' }
    }));
  };

  const triggerParallelOCR = async () => {
    if (!isProceedReady) return;
    setIsProcessing(true);

    setActiveFiles(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(k => {
        if (updated[k].file) {
          updated[k].status = 'scanning';
          updated[k].errorMsg = undefined;
          updated[k].result = null;
        }
      });
      return updated;
    });

    const currentFilesState = currentStep === 1 ? files : step2Files;
    const selectedEntries = Object.entries(currentFilesState).filter(([, state]) => state.file);
    const batchStartedAt = Date.now();
    let results: any[] = [];

    try {
      const formData = new FormData();
      selectedEntries.forEach(([docType, fileState]) => {
        if (!fileState.file) return;
        formData.append('files', fileState.file);
        formData.append('doc_types', docType);
      });

      const startResponse = await fetch('/v1/claims/pipeline/start', {
        method: 'POST',
        body: formData,
      });
      if (!startResponse.ok) {
        const errData = await startResponse.json().catch(() => ({ detail: 'Pipeline start failed' }));
        throw new Error(errData.detail || `Server returned ${startResponse.status}`);
      }

      let batch = await startResponse.json();
      const applyBatchDocs = (docs: any[]) => {
        setActiveFiles(prev => {
          const updated = { ...prev };
          docs.forEach(doc => {
            const existing = updated[doc.doc_type];
            if (!existing) return;
            let status: FileState['status'] = 'scanning';
            if (doc.status === 'queued_llm' || doc.status === 'llm_running' || doc.status === 'queued_validation' || doc.status === 'validation_running') {
              status = 'extracting';
            } else if (doc.status === 'success') {
              status = 'success';
            } else if (doc.status === 'validation_error') {
              status = 'validation_error';
            } else if (doc.status === 'failed') {
              status = 'failed';
            }
            updated[doc.doc_type] = {
              ...existing,
              status,
              timeMs: Date.now() - batchStartedAt,
              result: doc.result || existing.result,
              errorMsg: doc.errorMsg || existing.errorMsg,
            };
          });
          return updated;
        });
      };

      applyBatchDocs(batch.docs || []);
      while (batch.status !== 'completed') {
        await new Promise(resolve => setTimeout(resolve, 700));
        const statusResponse = await fetch(`/v1/claims/pipeline/${batch.batch_id}`);
        if (!statusResponse.ok) {
          const errData = await statusResponse.json().catch(() => ({ detail: 'Pipeline status failed' }));
          throw new Error(errData.detail || `Server returned ${statusResponse.status}`);
        }
        batch = await statusResponse.json();
        applyBatchDocs(batch.docs || []);
      }

      results = (batch.docs || []).map((doc: any) => ({
        docType: doc.doc_type,
        data: doc.result,
        error: doc.status === 'failed' ? doc.errorMsg : undefined,
      }));
    } catch (err: any) {
      setActiveFiles(prev => {
        const updated = { ...prev };
        selectedEntries.forEach(([docType]) => {
          updated[docType] = {
            ...updated[docType],
            status: 'failed',
            timeMs: Date.now() - batchStartedAt,
            errorMsg: err.message || 'Pipeline failed',
          };
        });
        return updated;
      });
      setIsProcessing(false);
      return;
    }
    setIsProcessing(false);

    if (currentStep === 1) {
      // Evaluate DeathCategory
      let detectedCategory = null;
      results.forEach(res => {
        if (res?.data?.extracted_data?.DeathCategory?.value) {
          detectedCategory = res.data.extracted_data.DeathCategory.value;
        }
      });

      if (detectedCategory && (detectedCategory === 'NATURAL_OR_MEDICAL' || detectedCategory === 'UNNATURAL')) {
        setDeathCategory(detectedCategory);
        setModalDetectionType('AUTO');
      } else {
        setDeathCategory('NATURAL_OR_MEDICAL'); // default fallback selection
        setModalDetectionType('MANUAL');
      }
      setShowCategoryModal(true);
    }
  };

  const proceedToStep2 = () => {
    setShowCategoryModal(false);
    setCurrentStep(2);
    if (deathCategory === 'NATURAL_OR_MEDICAL') {
      setStep2Files(createInitialState(NATURAL_DOC_CONFIG));
    } else {
      setStep2Files(createInitialState(UNNATURAL_DOC_CONFIG));
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-xl px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-tr from-indigo-500 to-blue-600 w-10 h-10 rounded-xl flex items-center justify-center font-bold text-lg text-white shadow-lg shadow-indigo-500/30">
            CO
          </div>
          <div>
            <h1 className="font-bold text-xl tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              CLAIMOS AI
            </h1>
            <span className="text-[10px] text-indigo-400 font-semibold tracking-wider uppercase bg-indigo-500/10 border border-indigo-500/20 px-1.5 py-0.5 rounded-md">
              Intake Portal - Step {currentStep}
            </span>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 flex flex-col gap-8">
        <div className="flex flex-col gap-8 animate-fade-in">
          <div className="max-w-2xl">
            <h2 className="text-3xl font-extrabold tracking-tight text-white mb-3">
              {currentStep === 1 ? 'Mandatory Document Intake' : 'Supporting Contextual Documents'}
            </h2>
            <p className="text-slate-400 text-sm leading-relaxed">
              {currentStep === 1 
                ? 'Upload any of the required documents below. The files will be processed in parallel using the chosen OCR engine. At least one document is required to proceed.'
                : 'Based on the identified cause of death, please upload the following specific supporting documents. You can combine multiple pages into a single PDF if necessary.'}
            </p>
          </div>

          {/* Document Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {Object.entries(activeConfig).map(([docKey, config]) => (
              <DocumentCard 
                key={docKey}
                docKey={docKey}
                title={config.title}
                desc={config.desc}
                fileState={activeFiles[docKey]}
                onFileChange={handleFileChange}
                onRemoveFile={removeFile}
              />
            ))}
          </div>

          {/* Controls Bar */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 flex flex-col sm:flex-row items-center justify-end gap-6">
            {currentStep === 1 && hasResults && !isProcessing && (
              <button
                onClick={() => setShowCategoryModal(true)}
                className="w-full sm:w-auto px-6 py-3 rounded-xl text-sm font-bold text-amber-500 hover:text-amber-400 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/20 transition-all"
              >
                Review Cause of Death & Proceed
              </button>
            )}
            {currentStep === 2 && (
              <button
                onClick={() => setCurrentStep(1)}
                className="w-full sm:w-auto px-6 py-3 rounded-xl text-sm font-bold text-slate-300 hover:text-white transition-colors"
              >
                Back to Step 1
              </button>
            )}
            <button
              onClick={triggerParallelOCR}
              disabled={!isProceedReady || isProcessing}
              className={`w-full sm:w-auto px-8 py-3 rounded-xl text-sm font-bold text-white shadow-lg transition-all ${
                isProceedReady && !isProcessing
                  ? 'bg-gradient-to-r from-indigo-500 to-blue-600 shadow-indigo-500/20 hover:scale-[1.02] cursor-pointer'
                  : 'bg-slate-800 text-slate-500 cursor-not-allowed opacity-50'
              }`}
            >
              {isProcessing ? 'Processing...' : 'Upload & Extract'}
            </button>
          </div>
        </div>

        {/* Live Parallel Scan Progress Status & Failure Indicators */}
        {(isProcessing || Object.values(activeFiles).some(f => f.status === 'scanning' || f.status === 'extracting' || f.status === 'failed')) && (
          <div className="flex flex-col gap-4 bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping"></span>
              Parallel Document Scan Status
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {Object.keys(activeFiles).map(type => {
                const status = activeFiles[type].status;
                const err = activeFiles[type].errorMsg;
                const time = activeFiles[type].timeMs;

                let statusColor = 'text-indigo-400';
                let statusLabel = 'Extracting Verbatim Text...';

                if (status === 'success') {
                  statusColor = 'text-emerald-400';
                  statusLabel = `Scanned in ${time}ms`;
                } else if (status === 'extracting') {
                  statusColor = 'text-blue-400';
                  statusLabel = `OCR done, generating JSON...`;
                } else if (status === 'failed') {
                  statusColor = 'text-red-400';
                  statusLabel = `Failed: ${err || 'Error'}`;
                } else if (status === 'validation_error') {
                  statusColor = 'text-amber-400';
                  statusLabel = `Validation Failed`;
                }

                return (
                  <div key={type} className={`bg-slate-950/60 border rounded-xl p-4 flex flex-col gap-2 ${
                    status === 'failed' ? 'border-red-900/50' : 'border-slate-800/80'
                  }`}>
                    <span className="text-xs font-semibold text-slate-300">
                      {type.split('_').slice(0, 2).join(' ')}
                    </span>
                    <span className={`text-[10px] font-semibold ${statusColor}`}>
                      {statusLabel}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Live Verbatim Extracted Text Display */}
        {hasResults && (
          <div className="flex flex-col gap-6 animate-fade-in border-t border-slate-800 pt-8 mt-4">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Verbatim Extracted OCR Text</h2>
              <p className="text-slate-400 text-xs">Verify the text extracted from each uploaded document below.</p>
            </div>

            <div className="flex flex-col gap-6">
              {Object.entries(activeFiles).map(([docKey, fileData]) => {
                if (!fileData.result?.pages) return null;
                return (
                  <div key={docKey} className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 flex flex-col gap-4">
                    <div className="flex justify-between items-center">
                      <div className="flex flex-col gap-1">
                        <h3 className="text-sm font-bold text-white uppercase">{docKey.replace(/_/g, ' ')}</h3>
                      </div>
                    </div>
                    <div className="flex flex-col lg:flex-row gap-4">
                      {/* Verbatim OCR Text */}
                      <div className="flex-1 bg-slate-950 border border-slate-850 rounded-xl p-4 text-xs max-h-[400px] overflow-y-auto leading-relaxed text-white font-mono">
                        <h4 className="text-slate-400 font-bold mb-3 pb-2 border-b border-slate-800 uppercase text-[10px] tracking-wider">Raw OCR Verbatim</h4>
                        {fileData.result.pages.map((page: any, pIdx: number) => (
                          <div key={pIdx} className="mb-4">
                            {page.words && page.words.length > 0 ? (
                              page.words.map((word: any, wIdx: number) => (
                                <span key={wIdx} className="mr-2 inline-block">
                                  {word.text} <span className="text-slate-500 opacity-50">({word.confidence}%)</span>
                                </span>
                              ))
                            ) : (
                              <span className="text-slate-500 italic">No text extracted.</span>
                            )}
                          </div>
                        ))}
                      </div>

                      {/* Structured JSON Output */}
                      <div className="flex-1 bg-indigo-950/20 border border-indigo-900/50 rounded-xl p-4 text-[11px] max-h-[400px] overflow-y-auto leading-relaxed text-indigo-100 font-mono">
                        <h4 className="text-indigo-400 font-bold mb-3 pb-2 border-b border-indigo-900/50 uppercase tracking-wider">Structured JSON</h4>
                        {fileData.result.extracted_data ? (
                          <pre className="whitespace-pre-wrap break-words">
                            {JSON.stringify(fileData.result.extracted_data, null, 2)}
                          </pre>
                        ) : (
                          <div className="flex h-full items-center justify-center text-slate-500 italic">No structured data extracted.</div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </main>

      {/* Cause of Death Modal */}
      {showCategoryModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg shadow-2xl flex flex-col overflow-hidden animate-fade-in">
            <div className="flex justify-between items-center p-5 border-b border-slate-800">
              <h3 className="font-bold text-lg text-white">Cause of Death Verification</h3>
              <button onClick={() => setShowCategoryModal(false)} className="text-slate-400 hover:text-white transition-colors">
                <X size={20} />
              </button>
            </div>
            
            <div className="p-6 flex flex-col gap-5">
              {modalDetectionType === 'AUTO' ? (
                <div className="bg-indigo-950/40 border border-indigo-900/50 rounded-xl p-4">
                  <span className="text-[10px] text-indigo-400 font-bold tracking-widest uppercase block mb-2">AI Extraction Success</span>
                  <p className="text-slate-200 text-sm">
                    We automatically detected the cause of death from your documents as <strong className="text-white">{deathCategory?.replace(/_/g, ' ')}</strong>.
                  </p>
                </div>
              ) : (
                <div className="bg-amber-950/20 border border-amber-900/30 rounded-xl p-4">
                  <span className="text-[10px] text-amber-500 font-bold tracking-widest uppercase block mb-2 flex items-center gap-1">
                    <AlertTriangle size={12} /> AI Extraction Incomplete
                  </span>
                  <p className="text-slate-300 text-sm mb-4">
                    The cause of death was not clearly identified in the uploaded documents. Please select it manually to proceed.
                  </p>
                  <label className="flex flex-col gap-2">
                    <span className="text-xs font-semibold text-slate-400">Select Death Category</span>
                    <select 
                      value={deathCategory || 'NATURAL_OR_MEDICAL'} 
                      onChange={(e) => setDeathCategory(e.target.value as any)}
                      className="bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm text-white focus:border-indigo-500 outline-none"
                    >
                      <option value="NATURAL_OR_MEDICAL">Death at Home or hospital (Natural / Medical reasons)</option>
                      <option value="UNNATURAL">Unnatural Causes (Accidents, Murder, Suicide, etc.)</option>
                    </select>
                  </label>
                </div>
              )}
            </div>

            <div className="p-5 border-t border-slate-800 bg-slate-950/50 flex justify-end gap-3">
              <button 
                onClick={() => setShowCategoryModal(false)}
                className="px-5 py-2.5 rounded-lg text-sm font-semibold text-slate-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button 
                onClick={proceedToStep2}
                className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-bold text-white shadow-lg shadow-indigo-500/20 transition-all"
              >
                Proceed to Next Step
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
