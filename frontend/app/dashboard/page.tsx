'use client';

import { useEffect, useState } from 'react';
import Graph from '@/components/Graph';
import { useRouter } from 'next/navigation';

export default function DashboardPage() {
    const [data, setData] = useState<any[]>([]);
    const [insights, setInsights] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        const storedData = localStorage.getItem('telemetryData');
        if (storedData) {
            try {
                const parsed = JSON.parse(storedData);
                setData(parsed.data || []);
                setInsights(parsed.insights || []);
            } catch (e) {
                console.error("Failed to parse data", e);
            }
        } else {
            // Redirect to upload if no data
            // router.push('/upload');
        }
        setLoading(false);
    }, [router]);

    if (loading) return <div className="p-8">Loading...</div>;

    if (data.length === 0) {
        return (
            <div className="p-8 text-center">
                <h2 className="text-2xl font-bold mb-4">No Data Available</h2>
                <p className="text-gray-400 mb-6">Please upload a telemetry file first.</p>
                <button
                    onClick={() => router.push('/upload')}
                    className="bg-red-600 text-white px-6 py-2 rounded-md"
                >
                    Go to Upload
                </button>
            </div>
        );
    }

    // Prepare data for plots
    const x_axis = data.map((d: any) => d.Laptrigger_lapdist_dls || d.index);

    const speedTrace = {
        x: x_axis,
        y: data.map((d: any) => d.speed),
        type: 'scatter',
        mode: 'lines',
        name: 'Speed',
        line: { color: '#1f77b4', width: 2 }
    };

    const gearTrace = {
        x: x_axis,
        y: data.map((d: any) => d.gear),
        type: 'scatter',
        mode: 'lines',
        name: 'Gear',
        line: { color: '#ff7f0e', width: 2 }
    };

    const throttleTrace = {
        x: x_axis,
        y: data.map((d: any) => d.aps),
        type: 'scatter',
        mode: 'lines',
        name: 'Throttle',
        line: { color: '#2ca02c', width: 2 }
    };

    const brakeTrace = {
        x: x_axis,
        y: data.map((d: any) => d.pbrake_f),
        type: 'scatter',
        mode: 'lines',
        name: 'Brake',
        line: { color: '#d62728', width: 2 }
    };

    const steeringTrace = {
        x: x_axis,
        y: data.map((d: any) => d.Steering_Angle),
        type: 'scatter',
        mode: 'lines',
        name: 'Steering',
        line: { color: '#9467bd', width: 2 }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-2xl font-bold">Telemetry Dashboard</h1>
                <div className="text-sm text-gray-400">
                    {data.length.toLocaleString()} data points
                </div>
            </div>

            {/* Insights Section */}
            {insights.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {insights.map((insight, i) => (
                        <div key={i} className="bg-[#242324] border border-[#2C2C2B] p-4 rounded-lg text-sm">
                            {insight}
                        </div>
                    ))}
                </div>
            )}

            {/* Main Graphs Grid */}
            <div className="grid grid-cols-1 gap-6">
                <Graph
                    title="Speed (km/h)"
                    data={[speedTrace]}
                    height={350}
                />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Graph
                        title="Gear"
                        data={[gearTrace]}
                        height={250}
                    />
                    <Graph
                        title="Throttle (%)"
                        data={[throttleTrace]}
                        height={250}
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Graph
                        title="Brake Pressure (bar)"
                        data={[brakeTrace]}
                        height={250}
                    />
                    <Graph
                        title="Steering Angle (°)"
                        data={[steeringTrace]}
                        height={250}
                    />
                </div>
            </div>
        </div>
    );
}
