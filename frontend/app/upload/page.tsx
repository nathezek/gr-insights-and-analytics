'use client';

import { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import { uploadSession } from '@/lib/api';
import { useRouter } from 'next/navigation';
import ProgressBar from '@/components/ProgressBar';

export default function UploadPage() {
    const [files, setFiles] = useState<{
        lap_start: File | null;
        lap_end: File | null;
        lap_time: File | null;
        telemetry: File | null;
    }>({
        lap_start: null,
        lap_end: null,
        lap_time: null,
        telemetry: null
    });

    const [uploadProgress, setUploadProgress] = useState(0);
    const [statusMessage, setStatusMessage] = useState('');
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const router = useRouter();

    const handleFileChange = (fileType: keyof typeof files, file: File | null) => {
        setFiles(prev => ({ ...prev, [fileType]: file }));
    };

    const allFilesSelected = files.lap_start && files.lap_end && files.lap_time && files.telemetry;

    const handleUpload = async () => {
        if (!allFilesSelected) return;

        setUploading(true);
        setError(null);
        setUploadProgress(0);
        setStatusMessage('Uploading files...');

        try {
            const response = await uploadSession(
                {
                    lap_start: files.lap_start!,
                    lap_end: files.lap_end!,
                    lap_time: files.lap_time!,
                    telemetry: files.telemetry!
                },
                (progress) => {
                    setUploadProgress(progress);
                    if (progress < 100) {
                        setStatusMessage(`Uploading... ${progress}%`);
                    } else {
                        setStatusMessage('Processing data...');
                    }
                }
            );

            const data = response.data;

            setStatusMessage('Upload complete!');

            // Store session ID in localStorage
            localStorage.setItem('sessionId', data.session_id);

            // Redirect to dashboard
            setTimeout(() => {
                router.push('/dashboard');
            }, 500);

        } catch (err: any) {
            console.error('Upload error:', err);
            setError(err.response?.data?.detail || err.message || 'Upload failed');
            setStatusMessage('');
            setUploadProgress(0);
        } finally {
            setUploading(false);
        }
    };

    const FileInput = ({
        label,
        fileType,
        description
    }: {
        label: string;
        fileType: keyof typeof files;
        description: string;
    }) => (
        <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-4">
            <label className="block mb-2">
                <span className="text-sm font-medium text-gray-300">{label}</span>
                <span className="text-xs text-gray-500 ml-2">({description})</span>
            </label>
            <div className="flex items-center gap-3">
                <label className="flex-1 cursor-pointer">
                    <div className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${files[fileType]
                            ? 'border-green-500 bg-green-500/10'
                            : 'border-[#2C2C2B] hover:border-red-500/50'
                        }`}>
                        {files[fileType] ? (
                            <div className="flex items-center justify-center gap-2 text-green-500">
                                <CheckCircle size={20} />
                                <span className="text-sm">{files[fileType]!.name}</span>
                            </div>
                        ) : (
                            <div className="flex items-center justify-center gap-2 text-gray-400">
                                <FileText size={20} />
                                <span className="text-sm">Click to select file</span>
                            </div>
                        )}
                    </div>
                    <input
                        type="file"
                        accept=".csv"
                        className="hidden"
                        onChange={(e) => {
                            const file = e.target.files?.[0] || null;
                            handleFileChange(fileType, file);
                        }}
                    />
                </label>
                {files[fileType] && (
                    <button
                        onClick={() => handleFileChange(fileType, null)}
                        className="px-3 py-2 text-sm text-red-400 hover:text-red-300 border border-red-500/30 rounded-md"
                    >
                        Clear
                    </button>
                )}
            </div>
        </div>
    );

    return (
        <div className="max-w-4xl mx-auto p-8">
            <div className="mb-8">
                <h1 className="text-3xl font-bold mb-2">Upload Session Data</h1>
                <p className="text-gray-400">
                    Upload all 4 CSV files for your racing session
                </p>
            </div>

            {error && (
                <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-3">
                    <AlertCircle className="text-red-500 mt-0.5" size={20} />
                    <div>
                        <p className="text-red-400 font-medium">Upload Error</p>
                        <p className="text-sm text-red-300">{error}</p>
                    </div>
                </div>
            )}

            <div className="space-y-4 mb-6">
                <FileInput
                    label="Lap Start Data"
                    fileType="lap_start"
                    description="lap_start.csv"
                />
                <FileInput
                    label="Lap End Data"
                    fileType="lap_end"
                    description="lap_end.csv"
                />
                <FileInput
                    label="Lap Time Data"
                    fileType="lap_time"
                    description="lap_time.csv"
                />
                <FileInput
                    label="Telemetry Data"
                    fileType="telemetry"
                    description="telemetry.csv"
                />
            </div>

            {uploading && (
                <div className="mb-6">
                    <ProgressBar progress={uploadProgress} />
                    <p className="text-sm text-gray-400 mt-2 text-center">{statusMessage}</p>
                </div>
            )}

            <div className="flex gap-4">
                <button
                    onClick={handleUpload}
                    disabled={!allFilesSelected || uploading}
                    className={`flex-1 py-3 px-6 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors ${allFilesSelected && !uploading
                            ? 'bg-red-600 hover:bg-red-700 text-white'
                            : 'bg-gray-700 text-gray-400 cursor-not-allowed'
                        }`}
                >
                    <Upload size={20} />
                    {uploading ? 'Uploading...' : 'Upload Session'}
                </button>

                <button
                    onClick={() => router.push('/dashboard')}
                    className="px-6 py-3 border border-[#2C2C2B] rounded-lg hover:bg-[#242324] transition-colors"
                >
                    Cancel
                </button>
            </div>

            <div className="mt-8 p-4 bg-[#242324] border border-[#2C2C2B] rounded-lg">
                <h3 className="text-sm font-medium mb-2">Required Files:</h3>
                <ul className="text-sm text-gray-400 space-y-1">
                    <li>• <strong>lap_start.csv</strong> - Lap start timestamps/distances</li>
                    <li>• <strong>lap_end.csv</strong> - Lap end timestamps/distances</li>
                    <li>• <strong>lap_time.csv</strong> - Lap timing data</li>
                    <li>• <strong>telemetry.csv</strong> - Main telemetry data (speed, throttle, brake, etc.)</li>
                </ul>
            </div>
        </div>
    );
}
