'use client';

import { useEffect, useState } from 'react';
import Graph from '@/components/Graph';
import { useRouter } from 'next/navigation';

export default function DashboardPage() {
    const [data, setData] = useState<any[]>([]);
    const [insights, setInsights] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    const [selectedLap, setSelectedLap] = useState<number | null>(null);

    useEffect(() => {
        const storedData = localStorage.getItem('telemetryData');
        if (storedData) {
            try {
                const parsed = JSON.parse(storedData);
                const rawData = parsed.data || [];
                setData(rawData);
                setInsights(parsed.insights || []);

                // Set initial selected lap (first available)
                if (rawData.length > 0) {
                    const laps = Array.from(new Set(rawData.map((d: any) => d.lap))).sort((a: any, b: any) => a - b);
                    if (laps.length > 0) {
                        setSelectedLap(laps[0] as number);
                    }
                }
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

    // Get available laps
    const availableLaps = Array.from(new Set(data.map((d: any) => d.lap))).sort((a: any, b: any) => a - b);

    // Filter data by selected lap
    const filteredData = selectedLap !== null
        ? data.filter((d: any) => d.lap === selectedLap)
        : data;

    // Prepare data for plots
    // Sort data by x-axis (distance or index) to ensure clean line plotting
    const sortedData = [...filteredData].sort((a, b) => {
        const valA = a.Laptrigger_lapdist_dls ?? a.index ?? 0;
        const valB = b.Laptrigger_lapdist_dls ?? b.index ?? 0;
        return valA - valB;
    });

    const x_axis = sortedData.map((d: any) => d.Laptrigger_lapdist_dls ?? d.index);

    const speedTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.speed),
        type: 'scatter',
        mode: 'lines',
        name: 'Speed',
        line: { color: '#1f77b4', width: 2 }
    };

    const gearTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.gear),
        type: 'scatter',
        mode: 'lines',
        name: 'Gear',
        line: { color: '#ff7f0e', width: 2 }
    };

    const throttleTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.aps),
        type: 'scatter',
        mode: 'lines',
        name: 'Throttle',
        line: { color: '#2ca02c', width: 2 }
    };

    const brakeTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.pbrake_f),
        type: 'scatter',
        mode: 'lines',
        name: 'Brake',
        line: { color: '#d62728', width: 2 }
    };

    const steeringTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.Steering_Angle),
        type: 'scatter',
        mode: 'lines',
        name: 'Steering',
        line: { color: '#9467bd', width: 2 }
    };

    // Placeholder turn data for Barber Motorsports Park (approximate distances in meters)
    // User can adjust these values based on their specific telemetry
    const barberTurns = [
        { name: 'T1', start: 200, end: 350 },
        { name: 'T2', start: 600, end: 750 },
        { name: 'T3', start: 1100, end: 1200 },
        { name: 'T4', start: 1900, end: 2100 },
        { name: 'T5', start: 2300, end: 2500 },
        { name: 'T6', start: 2900, end: 3100 },
        { name: 'T7', start: 3300, end: 3500 },
        { name: 'T8', start: 3800, end: 4000 },
    ];

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-2xl font-bold">Telemetry Dashboard</h1>

                <div className="flex items-center gap-4">
                    <div className="text-sm text-gray-400">
                        {sortedData.length.toLocaleString()} points
                    </div>

                    {/* Lap Selector */}
                    <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-gray-300">Lap:</label>
                        <select
                            value={selectedLap ?? ''}
                            onChange={(e) => setSelectedLap(Number(e.target.value))}
                            className="bg-[#242324] border border-[#2C2C2B] text-white text-sm rounded-md focus:ring-red-500 focus:border-red-500 block p-2"
                        >
                            {availableLaps.map((lap: any) => (
                                <option key={lap} value={lap}>
                                    Lap {lap}
                                </option>
                            ))}
                        </select>
                    </div>
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

            {/* Main Graphs Stack */}
            <div className="flex flex-col gap-6">
                <Graph
                    title="Speed (km/h)"
                    data={[speedTrace]}
                    height={400}
                    turns={barberTurns}
                />

                <Graph
                    title="Gear"
                    data={[gearTrace]}
                    height={200}
                    turns={barberTurns}
                />

                <Graph
                    title="Throttle (%)"
                    data={[throttleTrace]}
                    height={300}
                    turns={barberTurns}
                />

                <Graph
                    title="Brake Pressure (bar)"
                    data={[brakeTrace]}
                    height={300}
                    turns={barberTurns}
                />

                <Graph
                    title="Steering Angle (°)"
                    data={[steeringTrace]}
                    height={300}
                    turns={barberTurns}
                />
            </div>
        </div>
    );
}
