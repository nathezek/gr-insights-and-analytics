'use client';

import { useEffect, useState, useCallback } from 'react';
import Image from 'next/image';
import Graph from '@/components/Graph';
import { useRouter } from 'next/navigation';
import { getSessionLaps, getLapData } from '@/lib/api';

export default function DashboardPage() {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [availableLaps, setAvailableLaps] = useState<any[]>([]);
    const [selectedLap, setSelectedLap] = useState<number | null>(null);
    const [lapData, setLapData] = useState<any[]>([]);
    const [insights, setInsights] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadingLap, setLoadingLap] = useState(false);
    const router = useRouter();

    // Load session metadata on mount
    useEffect(() => {
        const storedSessionId = localStorage.getItem('sessionId');
        if (!storedSessionId) {
            router.push('/upload');
            return;
        }

        setSessionId(storedSessionId);

        // Fetch available laps
        getSessionLaps(storedSessionId)
            .then(response => {
                const data = response.data;
                setAvailableLaps(data.laps || []);
                setInsights(data.insights || []);

                // Select first lap by default
                if (data.laps && data.laps.length > 0) {
                    setSelectedLap(data.laps[0].lap);
                }
                setLoading(false);
            })
            .catch(error => {
                console.error('Failed to load session:', error);
                setLoading(false);
                router.push('/upload');
            });
    }, [router]);

    // Load lap data when selected lap changes
    useEffect(() => {
        if (!sessionId || selectedLap === null) return;

        setLoadingLap(true);
        getLapData(sessionId, selectedLap)
            .then(response => {
                setLapData(response.data.data || []);
                setLoadingLap(false);
            })
            .catch(error => {
                console.error('Failed to load lap data:', error);
                setLoadingLap(false);
            });
    }, [sessionId, selectedLap]);

    if (loading) {
        return (
            <div className="p-8 text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600 mx-auto mb-4"></div>
                <p className="text-gray-400">Loading session...</p>
            </div>
        );
    }

    if (availableLaps.length === 0) {
        return (
            <div className="p-8 text-center">
                <h2 className="text-2xl font-bold mb-4">No Data Available</h2>
                <p className="text-gray-400 mb-6">Please upload session files first.</p>
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
    const sortedData = [...lapData].sort((a, b) => {
        const valA = a.Laptrigger_lapdist_dls ?? a.index ?? 0;
        const valB = b.Laptrigger_lapdist_dls ?? b.index ?? 0;
        return valA - valB;
    });

    const x_axis = sortedData.map((d: any) => d.Laptrigger_lapdist_dls ?? d.index);

    // Extract mistakes (where mistake column is 1 or true)
    const mistakes = sortedData
        .map((d: any, idx: number) => ({
            x: x_axis[idx],
            y: d.speed,
            isMistake: d.mistake === 1 || d.mistake === true
        }))
        .filter(m => m.isMistake);

    const speedMistakes = mistakes.map(m => ({ x: m.x, y: m.y, text: `Mistake at ${m.x.toFixed(0)}m` }));

    const speedTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.speed),
        type: 'scatter',
        mode: 'lines',
        name: 'Speed',
        line: { color: '#1f77b4', width: 2 },
        hovertemplate: '<b>Speed:</b> %{y:.2f} km/h<br><b>Distance:</b> %{x:.0f}m<extra></extra>'
    };

    const gearTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.gear),
        type: 'scatter',
        mode: 'lines',
        name: 'Gear',
        line: { color: '#ff7f0e', width: 2 },
        hovertemplate: '<b>Gear:</b> %{y}<br><b>Distance:</b> %{x:.0f}m<extra></extra>'
    };

    const throttleTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.aps),
        type: 'scatter',
        mode: 'lines',
        name: 'Throttle',
        line: { color: '#2ca02c', width: 2 },
        hovertemplate: '<b>Throttle:</b> %{y:.1f}%<br><b>Distance:</b> %{x:.0f}m<extra></extra>'
    };

    const brakeTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.pbrake_f),
        type: 'scatter',
        mode: 'lines',
        name: 'Brake',
        line: { color: '#d62728', width: 2 },
        hovertemplate: '<b>Brake Pressure:</b> %{y:.1f} bar<br><b>Distance:</b> %{x:.0f}m<extra></extra>'
    };

    const steeringTrace = {
        x: x_axis,
        y: sortedData.map((d: any) => d.Steering_Angle),
        type: 'scatter',
        mode: 'lines',
        name: 'Steering',
        line: { color: '#9467bd', width: 2 },
        hovertemplate: '<b>Steering Angle:</b> %{y:.1f}°<br><b>Distance:</b> %{x:.0f}m<extra></extra>'
    };

    // Turn data for Barber Motorsports Park
    const barberTurns = [
        { name: 'T1', start: 0, end: 150 },
        { name: 'T2', start: 400, end: 550 },
        { name: 'T3', start: 700, end: 850 },
        { name: 'T4', start: 1000, end: 1150 },
        { name: 'T5', start: 1300, end: 1500 },
        { name: 'T6', start: 1600, end: 1750 },
        { name: 'T7', start: 1900, end: 2050 },
        { name: 'T8', start: 2200, end: 2350 },
        { name: 'T9', start: 2500, end: 2650 },
        { name: 'T10', start: 2800, end: 2950 },
        { name: 'T11', start: 3050, end: 3200 },
        { name: 'T12', start: 3300, end: 3400 },
        { name: 'T13', start: 3450, end: 3600 },
    ];

    // Get current lap metadata
    const currentLapMeta = availableLaps.find(l => l.lap === selectedLap);

    return (
        <div className="space-y-6">
            {/* Header with Controls */}
            <div className="flex justify-between items-center">
                <h1 className="text-2xl font-bold">Telemetry Analysis</h1>

                <div className="flex items-center gap-4">
                    <div className="text-sm text-gray-400">
                        {sortedData.length.toLocaleString()} points
                    </div>

                    {/* Lap Info */}
                    {currentLapMeta && (
                        <div className="text-sm text-gray-400">
                            {currentLapMeta.duration_seconds && (
                                <span>Lap Time: {currentLapMeta.duration_seconds.toFixed(2)}s</span>
                            )}
                            {currentLapMeta.max_distance_m && (
                                <span className="ml-3">Distance: {currentLapMeta.max_distance_m.toFixed(0)}m</span>
                            )}
                        </div>
                    )}

                    {/* Lap Selector */}
                    <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-gray-300">Lap:</label>
                        <select
                            value={selectedLap ?? ''}
                            onChange={(e) => setSelectedLap(Number(e.target.value))}
                            className="bg-[#242324] border border-[#2C2C2B] text-white text-sm rounded-md focus:ring-red-500 focus:border-red-500 block p-2"
                            disabled={loadingLap}
                        >
                            {availableLaps.map((lap: any) => (
                                <option key={lap.lap} value={lap.lap}>
                                    Lap {lap.lap}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            {/* Insights Section */}
            {insights.length > 0 && (
                <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-4">
                    <h2 className="text-lg font-semibold mb-3">AI Insights</h2>
                    <ul className="space-y-2">
                        {insights.map((insight, idx) => (
                            <li key={idx} className="text-sm text-gray-300 flex items-start gap-2">
                                <span className="text-yellow-500 mt-1">⚠️</span>
                                <span>{insight}</span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {loadingLap ? (
                <div className="text-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600 mx-auto mb-4"></div>
                    <p className="text-gray-400">Loading lap data...</p>
                </div>
            ) : (
                <>
                    {/* Main Graphs Stack */}
                    <div className="flex flex-col gap-6">
                        <Graph
                            title="Speed (km/h)"
                            data={[speedTrace]}
                            height={400}
                            turns={barberTurns}
                            mistakes={speedMistakes}
                            yAxisLabel="Speed (km/h)"
                        />

                        <Graph
                            title="Gear"
                            data={[gearTrace]}
                            height={200}
                            turns={barberTurns}
                            yAxisLabel="Gear"
                        />

                        <Graph
                            title="Throttle (%)"
                            data={[throttleTrace]}
                            height={300}
                            turns={barberTurns}
                            yAxisLabel="Throttle (%)"
                        />

                        <Graph
                            title="Brake Pressure (bar)"
                            data={[brakeTrace]}
                            height={300}
                            turns={barberTurns}
                            yAxisLabel="Brake Pressure (bar)"
                        />

                        <Graph
                            title="Steering Angle (°)"
                            data={[steeringTrace]}
                            height={300}
                            turns={barberTurns}
                            yAxisLabel="Steering Angle (°)"
                        />
                    </div>
                </>
            )}
        </div>
    );
}
