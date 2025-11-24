import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, Activity, Zap, Database } from 'lucide-react';

export default function Home() {
    return (
        <div className="max-w-6xl mx-auto mt-20">
            <div className="text-center mb-16">
                <div className="flex justify-center mb-8">
                    <Image
                        src="/logo_best.png"
                        alt="Gazoo Insights"
                        width={600}
                        height={60}
                        priority
                    />
                </div>

                <p className="text-sm text-gray-400 max-w-2xl mx-auto">
                    Advanced telemetry analysis for race engineers. Identify mistakes, optimize lap times, and visualize performance with AI-powered insights.
                </p>

                <div className="mt-10">
                    <Link
                        href="/upload"
                        className="rounded-md h-10 w-fit px-6 text-base bg-red-600 hover:bg-red-700 text-white inline-flex items-center gap-2 transition-transform hover:scale-105"
                    >
                        Start Analysis <ArrowRight size={16} />
                    </Link>
                </div>
            </div>

            <div className="bg-[#242324] border border-[#2C2C2B] rounded-xl p-8 max-w-4xl mx-auto">
                <h2 className="text-2xl font-bold mb-6 text-center">Race Engineer Capabilities</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <CapabilityItem text="Analyze telemetry data for speed, throttle, and brake inputs" />
                    <CapabilityItem text="Detect driving mistakes and time loss automatically" />
                    <CapabilityItem text="Compare lap times and sector performance" />
                    <CapabilityItem text="Visualize track heatmaps with speed overlays" />
                    <CapabilityItem text="Receive AI-powered coaching suggestions" />
                    <CapabilityItem text="Export detailed analysis reports" />
                </div>
            </div>
        </div>
    );
}

const CapabilityItem = ({ text }: { text: string }) => {
    return (
        <div className="flex items-center gap-3 p-3 bg-[#191818] rounded-lg border border-[#2C2C2B]">
            <div className="h-2 w-2 rounded-full bg-red-600 shrink-0" />
            <span className="text-gray-300">{text}</span>
        </div>
    );
};
