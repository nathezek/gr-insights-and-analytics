'use client';

import dynamic from 'next/dynamic';
import { useMemo } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface Turn {
    name: string;
    start: number;
    end: number;
}

interface Mistake {
    x: number;
    y: number;
    text?: string;
}

interface GraphProps {
    data: any[];
    layout?: any;
    title?: string;
    height?: number;
    turns?: Turn[];
    mistakes?: Mistake[];
    yAxisLabel?: string;
}

const Graph = ({ data, layout, title, height = 300, turns = [], mistakes = [], yAxisLabel = '' }: GraphProps) => {
    const defaultLayout = useMemo(() => {
        const shapes = turns.map(turn => ({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: turn.start,
            x1: turn.end,
            y0: 0,
            y1: 1,
            fillcolor: 'rgba(255, 0, 0, 0.15)',
            line: { width: 0 },
            layer: 'below'
        }));

        const annotations = turns.map(turn => ({
            x: (turn.start + turn.end) / 2,
            y: 1,
            xref: 'x',
            yref: 'paper',
            text: turn.name,
            showarrow: false,
            font: { color: '#ffcc00', size: 10 },
            yanchor: 'top',
            yshift: -5
        }));

        return {
            autosize: true,
            height: height,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
                color: '#a0a0a0',
                family: 'Arial, sans-serif',
            },
            margin: { t: 40, r: 20, b: 40, l: 60 },
            title: title ? { text: title, font: { color: '#fff', size: 16 } } : undefined,
            xaxis: {
                gridcolor: '#2C2C2B',
                zerolinecolor: '#2C2C2B',
                title: { text: 'Distance (m)', font: { color: '#a0a0a0' } }
            },
            yaxis: {
                gridcolor: '#2C2C2B',
                zerolinecolor: '#2C2C2B',
                title: yAxisLabel ? { text: yAxisLabel, font: { color: '#a0a0a0' } } : undefined
            },
            shapes: shapes,
            annotations: annotations,
            hovermode: 'closest',
            ...layout,
        };
    }, [layout, title, height, turns, yAxisLabel]);

    // Add mistake markers if provided
    const plotData = mistakes.length > 0
        ? [
            ...data,
            {
                x: mistakes.map(m => m.x),
                y: mistakes.map(m => m.y),
                type: 'scatter',
                mode: 'markers',
                name: 'Mistakes',
                marker: {
                    color: '#ff6b6b',
                    size: 10,
                    symbol: 'circle',
                    line: {
                        color: '#fff',
                        width: 2
                    }
                },
                hovertemplate: mistakes.map(m => m.text || 'Mistake').join('<br>'),
                showlegend: false
            }
        ]
        : data;

    return (
        <div
            className="bg-[#242324] border border-[#2C2C2B] rounded-lg p-4"
            style={{ height: `${height}px` }}
        >
            <Plot
                data={plotData}
                layout={defaultLayout}
                config={{ responsive: true, displayModeBar: true }}
                style={{ width: '100%', height: '100%' }}
            />
        </div>
    );
};

export default Graph;
