'use client';

import { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import { uploadFile } from '@/lib/api';
import { useRouter } from 'next/navigation';
import ProgressBar from '@/components/ProgressBar';

export default function UploadPage() {
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [statusMessage, setStatusMessage] = useState('');

    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const router = useRouter();

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        setError(null);
        setUploadProgress(0);
        setStatusMessage('Uploading file...');

        try {
            const data = await uploadFile(file, (progress) => {
                setUploadProgress(progress);
                if (progress === 100) {
                    setStatusMessage('Processing data (this may take a moment)...');
                }
            });

            // Store data in local storage or state management for now
            localStorage.setItem('telemetryData', JSON.stringify(data));
            router.push('/dashboard');
        } catch (err: any) {
            setError('Upload failed. Please try again.');
            console.error(err);
        } finally {
            setUploading(false);
        }
    };

    // ... (in return statement)

    <div className="flex gap-4 w-full justify-center">
        {!uploading ? (
            <>
                <button
                    onClick={() => setFile(null)}
                    className="px-6 py-2 border border-[#2C2C2B] rounded-md hover:bg-[#242324] transition-colors"
                >
                    Cancel
                </button>
                <button
                    onClick={handleUpload}
                    className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-md flex items-center gap-2"
                >
                    Start Analysis
                </button>
            </>
        ) : (
            <div className="w-full max-w-md">
                <ProgressBar progress={uploadProgress} label={statusMessage} color="green" />
            </div>
        )}
    </div>

    return (
        <div className="max-w-4xl mx-auto mt-10">
            <h1 className="text-3xl font-bold mb-2">Upload Telemetry Data</h1>
            <p className="text-gray-400 mb-8">Upload your CSV or VBO file to start analysis.</p>

            <div
                className={`border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center transition-colors ${dragActive ? 'border-red-600 bg-[#242324]' : 'border-[#2C2C2B] bg-[#1e1d1d]'
                    }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    className="hidden"
                    id="file-upload"
                    onChange={handleChange}
                    accept=".csv,.vbo"
                />

                {!file ? (
                    <>
                        <div className="bg-[#242324] p-4 rounded-full mb-4">
                            <Upload size={40} className="text-red-600" />
                        </div>
                        <p className="text-xl font-medium mb-2">Drag & drop your file here</p>
                        <p className="text-gray-500 mb-6">or</p>
                        <label
                            htmlFor="file-upload"
                            className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-md cursor-pointer transition-colors"
                        >
                            Browse Files
                        </label>
                    </>
                ) : (
                    <div className="flex flex-col items-center">
                        <FileText size={48} className="text-gray-300 mb-4" />
                        <p className="text-xl font-medium mb-2">{file.name}</p>
                        <p className="text-gray-500 mb-6">{(file.size / 1024 / 1024).toFixed(2)} MB</p>

                        <div className="flex gap-4">
                            <button
                                onClick={() => setFile(null)}
                                className="px-6 py-2 border border-[#2C2C2B] rounded-md hover:bg-[#242324] transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleUpload}
                                disabled={uploading}
                                className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-md disabled:opacity-50 flex items-center gap-2"
                            >
                                {uploading ? 'Processing...' : 'Start Analysis'}
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {error && (
                <div className="mt-6 p-4 bg-red-900/20 border border-red-900/50 rounded-md flex items-center gap-3 text-red-200">
                    <AlertCircle size={20} />
                    <span>{error}</span>
                </div>
            )}
        </div>
    );
}
