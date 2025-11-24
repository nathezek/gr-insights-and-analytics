'use client';

import { useEffect, useState, useCallback } from 'react';
import Image from 'next/image';
import { ArrowUp, ArrowDown } from 'lucide-react';
import dynamic from 'next/dynamic';
import Graph from '@/components/Graph';
import { useRouter } from 'next/navigation';
import { getSessionLaps, getLapData, getMistakeAnalysis } from '@/lib/api';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

type DashboardTab = 'telemetry' | 'mistakes' | 'comparison' | 'sectors' | 'export';

export default function DashboardPage() {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [availableLaps, setAvailableLaps] = useState<any[]>([]);
    const [selectedLap, setSelectedLap] = useState<number | null>(null);
    const [lapData, setLapData] = useState<any[]>([]);
    const [insights, setInsights] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadingLap, setLoadingLap] = useState(false);
    const [activeTab, setActiveTab] = useState<DashboardTab>('telemetry');
    const [mistakeAnalysis, setMistakeAnalysis] = useState<any>(null);
    const [loadingMistakes, setLoadingMistakes] = useState(false);
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

    // Load mistake analysis when switching to mistakes tab or changing lap
    useEffect(() => {
        if (activeTab === 'mistakes' && sessionId && selectedLap !== null) {
            setLoadingMistakes(true);
            getMistakeAnalysis(sessionId, selectedLap)
                .then(response => {
                    setMistakeAnalysis(response.data);
                    setLoadingMistakes(false);
                })
                .catch(error => {
                    console.error('Failed to load mistake analysis:', error);
                    setLoadingMistakes(false);
                });
        }
    }, [activeTab, sessionId, selectedLap]);

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

            {/* Sub-Tab Navigation */}
            <div className="border-b border-[#2C2C2B]">
                <nav className="flex items-center gap-1">
                    {[
                        { id: 'telemetry' as DashboardTab, label: 'Telemetry Overview' },
                        { id: 'mistakes' as DashboardTab, label: 'Mistake Analysis' },
                        { id: 'comparison' as DashboardTab, label: 'Lap Comparison' },
                        { id: 'sectors' as DashboardTab, label: 'Sector Analysis' },
                        { id: 'export' as DashboardTab, label: 'Export & Details' },
                    ].map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`px-4 py-3 text-sm font-medium transition-colors relative ${activeTab === tab.id
                                ? 'text-white'
                                : 'text-neutral-500 hover:text-neutral-300'
                                }`}
                        >
                            {tab.label}
                            {activeTab === tab.id && (
                                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white" />
                            )}
                        </button>
                    ))}
                </nav>
            </div>

            {loadingLap ? (
                <div className="text-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600 mx-auto mb-4"></div>
                    <p className="text-gray-400">Loading lap data...</p>
                </div>
            ) : (
                <>
                    {/* Telemetry Overview Tab */}
                    {activeTab === 'telemetry' && (
                        <>
                            {/* Insights Section */}
                            {insights.length > 0 && (
                                <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-6">
                                    <h2 className="text-lg font-semibold mb-4">AI Insights</h2>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                        {insights.map((insight, idx) => {
                                            const isNegative = insight.includes('too slow') ||
                                                insight.includes('below expected') ||
                                                insight.includes('mistakes detected') ||
                                                insight.includes('mistake rate');
                                            const isPositive = insight.includes('too fast') ||
                                                insight.includes('above expected');

                                            return (
                                                <div key={idx} className="flex items-center gap-2">
                                                    {isNegative && <ArrowDown size={16} className="text-red-500 shrink-0" />}
                                                    {isPositive && <ArrowUp size={16} className="text-green-500 shrink-0" />}
                                                    <span className="text-sm text-neutral-200">{insight}</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

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

                    {/* Mistake Analysis Tab */}
                    {activeTab === 'mistakes' && (
                        <div className="space-y-6">
                            {loadingMistakes ? (
                                <div className="text-center py-12">
                                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-600 mx-auto mb-4"></div>
                                    <p className="text-gray-400">Loading mistake analysis...</p>
                                </div>
                            ) : mistakeAnalysis ? (
                                <>
                                    {/* Statistics Cards */}
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                        <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-4">
                                            <div className="text-sm text-neutral-400 mb-1">Total Mistakes</div>
                                            <div className="text-2xl font-bold text-white">
                                                {mistakeAnalysis.statistics.mistake_count.toLocaleString()}
                                            </div>
                                            <div className="text-xs text-neutral-500 mt-1">
                                                {mistakeAnalysis.statistics.mistake_percentage.toFixed(1)}% of lap
                                            </div>
                                        </div>
                                        <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-4">
                                            <div className="text-sm text-neutral-400 mb-1">Too Slow</div>
                                            <div className="text-2xl font-bold text-red-500">
                                                {mistakeAnalysis.statistics.too_slow_count.toLocaleString()}
                                            </div>
                                            <div className="text-xs text-neutral-500 mt-1">
                                                <ArrowDown size={12} className="inline mr-1" />
                                                Below expected
                                            </div>
                                        </div>
                                        <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-4">
                                            <div className="text-sm text-neutral-400 mb-1">Too Fast</div>
                                            <div className="text-2xl font-bold text-green-500">
                                                {mistakeAnalysis.statistics.too_fast_count.toLocaleString()}
                                            </div>
                                            <div className="text-xs text-neutral-500 mt-1">
                                                <ArrowUp size={12} className="inline mr-1" />
                                                Above expected
                                            </div>
                                        </div>
                                        <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-4">
                                            <div className="text-sm text-neutral-400 mb-1">Avg Error</div>
                                            <div className="text-2xl font-bold text-white">
                                                {mistakeAnalysis.statistics.avg_error_kmh.toFixed(1)}
                                            </div>
                                            <div className="text-xs text-neutral-500 mt-1">km/h deviation</div>
                                        </div>
                                    </div>

                                    {/* Speed Error Heatmap */}
                                    {mistakeAnalysis.full_lap_data && mistakeAnalysis.full_lap_data.length > 0 && (
                                        <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-6">
                                            <h3 className="text-lg font-semibold mb-4">Speed Error Analysis</h3>
                                            <Plot
                                                data={[
                                                    {
                                                        x: mistakeAnalysis.full_lap_data.map((p: any) => p.Laptrigger_lapdist_dls),
                                                        y: mistakeAnalysis.full_lap_data.map((p: any) => p.predicted_speed),
                                                        mode: 'lines',
                                                        name: 'AI Expected Speed',
                                                        line: { color: '#22c55e', dash: 'dash', width: 2 },
                                                        type: 'scatter',
                                                        hovertemplate: '<b>Expected:</b> %{y:.1f} km/h<br><b>Distance:</b> %{x:.0f}m<extra></extra>'
                                                    },
                                                    {
                                                        x: mistakeAnalysis.full_lap_data.map((p: any) => p.Laptrigger_lapdist_dls),
                                                        y: mistakeAnalysis.full_lap_data.map((p: any) => p.speed),
                                                        mode: 'lines',
                                                        name: 'Actual Speed',
                                                        line: { color: '#3b82f6', width: 2 },
                                                        type: 'scatter',
                                                        hovertemplate: '<b>Actual:</b> %{y:.1f} km/h<br><b>Distance:</b> %{x:.0f}m<extra></extra>'
                                                    },
                                                    ...(mistakeAnalysis.mistake_points && mistakeAnalysis.mistake_points.length > 0 ? [{
                                                        x: mistakeAnalysis.mistake_points.map((p: any) => p.Laptrigger_lapdist_dls),
                                                        y: mistakeAnalysis.mistake_points.map((p: any) => p.speed_error),
                                                        mode: 'markers' as const,
                                                        name: 'Speed Error',
                                                        marker: {
                                                            size: 6,
                                                            color: mistakeAnalysis.mistake_points.map((p: any) => p.speed_error),
                                                            colorscale: [
                                                                [0, '#ef4444'],
                                                                [0.5, '#fbbf24'],
                                                                [1, '#22c55e']
                                                            ] as [number, string][],
                                                            showscale: true,
                                                            colorbar: {
                                                                title: { text: 'Error (km/h)' },
                                                                thickness: 15,
                                                                len: 0.7
                                                            }
                                                        },
                                                        type: 'scatter' as const,
                                                        yaxis: 'y2',
                                                        hovertemplate: '<b>Error:</b> %{y:.1f} km/h<br><b>Distance:</b> %{x:.0f}m<extra></extra>'
                                                    }] : [])
                                                ]}
                                                layout={{
                                                    paper_bgcolor: '#242324',
                                                    plot_bgcolor: '#191818',
                                                    font: { color: '#fafafa' },
                                                    height: 500,
                                                    margin: { l: 60, r: 60, t: 40, b: 60 },
                                                    xaxis: {
                                                        title: { text: 'Distance (m)' },
                                                        gridcolor: '#2C2C2B',
                                                        color: '#fafafa'
                                                    },
                                                    yaxis: {
                                                        title: { text: 'Speed (km/h)' },
                                                        gridcolor: '#2C2C2B',
                                                        color: '#fafafa'
                                                    },
                                                    yaxis2: {
                                                        title: { text: 'Error (km/h)' },
                                                        overlaying: 'y',
                                                        side: 'right',
                                                        gridcolor: '#2C2C2B',
                                                        color: '#fafafa'
                                                    },
                                                    legend: {
                                                        x: 0.01,
                                                        y: 0.99,
                                                        bgcolor: 'rgba(36, 35, 36, 0.8)',
                                                        bordercolor: '#2C2C2B',
                                                        borderwidth: 1
                                                    },
                                                    hovermode: 'closest'
                                                }}
                                                config={{ responsive: true, displayModeBar: true }}
                                                style={{ width: '100%' }}
                                            />
                                        </div>
                                    )}

                                    {/* Top Contributing Features */}
                                    {mistakeAnalysis.feature_importance && mistakeAnalysis.feature_importance.length > 0 && (
                                        <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-6">
                                            <h3 className="text-lg font-semibold mb-4">Top Contributing Features</h3>
                                            <p className="text-sm text-neutral-400 mb-6">Top 10 Most Important Telemetry Features</p>
                                            <Plot
                                                data={[
                                                    {
                                                        x: mistakeAnalysis.feature_importance.map((f: any) => f.importance),
                                                        y: mistakeAnalysis.feature_importance.map((f: any) => f.feature),
                                                        type: 'bar',
                                                        orientation: 'h',
                                                        marker: {
                                                            color: mistakeAnalysis.feature_importance.map((f: any) => f.importance),
                                                            colorscale: [
                                                                [0, '#a855f7'],
                                                                [0.5, '#3b82f6'],
                                                                [1, '#eab308']
                                                            ],
                                                            showscale: false
                                                        }
                                                    }
                                                ]}
                                                layout={{
                                                    paper_bgcolor: '#242324',
                                                    plot_bgcolor: '#191818',
                                                    font: { color: '#fafafa' },
                                                    height: 400,
                                                    margin: { l: 150, r: 40, t: 20, b: 60 },
                                                    xaxis: {
                                                        title: { text: 'Importance Score' },
                                                        gridcolor: '#2C2C2B',
                                                        color: '#fafafa'
                                                    },
                                                    yaxis: {
                                                        gridcolor: '#2C2C2B',
                                                        color: '#fafafa',
                                                        automargin: true
                                                    },
                                                    hovermode: 'closest'
                                                }}
                                                config={{ responsive: true, displayModeBar: false }}
                                                style={{ width: '100%' }}
                                            />
                                        </div>
                                    )}
                                </>
                            ) : (
                                <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-6">
                                    <p className="text-neutral-400 text-center py-12">
                                        No mistake analysis data available for this lap.
                                    </p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Lap Comparison Tab */}
                    {activeTab === 'comparison' && (
                        <div className="space-y-6">
                            <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-6">
                                <h2 className="text-xl font-semibold mb-6">Lap Comparison</h2>
                                <p className="text-neutral-400 text-center py-12">
                                    Lap comparison view coming soon...
                                    <br />
                                    Compare multiple laps side-by-side with delta analysis.
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Sector Analysis Tab */}
                    {activeTab === 'sectors' && (
                        <div className="space-y-6">
                            <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-6">
                                <h2 className="text-xl font-semibold mb-6">Sector Analysis</h2>
                                <p className="text-neutral-400 text-center py-12">
                                    Sector analysis coming soon...
                                    <br />
                                    Break down performance by track sectors and corners.
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Export & Details Tab */}
                    {activeTab === 'export' && (
                        <div className="space-y-6">
                            <div className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-6">
                                <h2 className="text-xl font-semibold mb-6">Export & Details</h2>
                                <p className="text-neutral-400 text-center py-12">
                                    Export options and session details coming soon...
                                    <br />
                                    Download processed data, generate reports, and view session metadata.
                                </p>
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
