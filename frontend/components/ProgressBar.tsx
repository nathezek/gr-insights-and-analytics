import React from 'react';

interface ProgressBarProps {
    progress: number; // 0 to 100
    label?: string;
    color?: 'green' | 'blue' | 'yellow';
}

const ProgressBar: React.FC<ProgressBarProps> = ({ progress, label, color = 'green' }) => {
    const getGradient = () => {
        switch (color) {
            case 'blue':
                return 'from-blue-500 to-blue-400';
            case 'yellow':
                return 'from-yellow-500 to-yellow-400';
            case 'green':
            default:
                return 'from-green-500 to-green-400';
        }
    };

    return (
        <div className="w-full">
            {label && (
                <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-gray-300">{label}</span>
                    <span className="text-sm font-medium text-gray-400">{Math.round(progress)}%</span>
                </div>
            )}
            <div className="w-full bg-[#2C2C2B] rounded-full h-4 border border-[#3C3C3B] overflow-hidden">
                <div
                    className={`bg-gradient-to-r ${getGradient()} h-4 rounded-full transition-all duration-300 ease-out shadow-[0_0_10px_rgba(34,197,94,0.3)]`}
                    style={{ width: `${progress}%` }}
                >
                    {/* Striped animation overlay */}
                    <div className="w-full h-full opacity-20 bg-[linear-gradient(45deg,rgba(255,255,255,0.15)_25%,transparent_25%,transparent_50%,rgba(255,255,255,0.15)_50%,rgba(255,255,255,0.15)_75%,transparent_75%,transparent)] bg-[length:1rem_1rem] animate-[progress-stripes_1s_linear_infinite]" />
                </div>
            </div>
        </div>
    );
};

export default ProgressBar;
