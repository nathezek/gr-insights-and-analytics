import Link from 'next/link';
import { ArrowRight, Activity, Zap, Database } from 'lucide-react';

export default function Home() {
    return (
        <div className="max-w-6xl mx-auto mt-20">
            <div className="text-center mb-16">
                <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-red-600 to-white bg-clip-text text-transparent">
                    Gazoo Insights & Analytics
                </h1>
                <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                    Advanced telemetry analysis for race engineers. Identify mistakes, optimize lap times, and visualize performance with AI-powered insights.
                </p>

                <div className="mt-10">
                    <Link
                        href="/upload"
                        className="bg-red-600 hover:bg-red-700 text-white px-8 py-4 rounded-full text-lg font-medium inline-flex items-center gap-2 transition-transform hover:scale-105"
                    >
                        Start Analysis <ArrowRight size={20} />
                    </Link>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <FeatureCard
                    icon={<Activity className="text-red-500" size={32} />}
                    title="Telemetry Visualization"
                    description="Interactive, high-performance charts for speed, throttle, brake, and steering data."
                />
                <FeatureCard
                    icon={<Zap className="text-yellow-500" size={32} />}
                    title="AI Mistake Detection"
                    description="Automatically identify corners where time was lost due to braking or throttle errors."
                />
                <FeatureCard
                    icon={<Database className="text-blue-500" size={32} />}
                    title="Data Management"
                    description="Upload and process VBO/CSV files with automatic cleaning and normalization."
                />
            </div>
        </div>
    );
}

const FeatureCard = ({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) => {
    return (
        <div className="bg-[#242324] border border-[#2C2C2B] p-8 rounded-xl hover:border-red-900/50 transition-colors">
            <div className="mb-4">{icon}</div>
            <h3 className="text-xl font-bold mb-2">{title}</h3>
            <p className="text-gray-400">{description}</p>
        </div>
    );
};
