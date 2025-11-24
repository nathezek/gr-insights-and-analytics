'use client';

import dynamic from 'next/dynamic';
import { useMemo } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface Turn {
    name: string;
    start: number;
    end: number;
}

interface GraphProps {
    data: any[];
    layout?: any;
    title?: string;
    height?: number;
    turns?: Turn[];
}

const Graph = ({ data, layout, title, height = 300, turns = [] }: GraphProps) => {
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
            margin: { t: 40, r: 20, b: 40, l: 40 },
            title: title ? { text: title, font: { color: '#fff', size: 16 } } : undefined,
            xaxis: {
                gridcolor: '#2C2C2B',
                zerolinecolor: '#2C2C2B',
            },
            yaxis: {
                gridcolor: '#2C2C2B',
                zerolinecolor: '#2C2C2B',
            },
            shapes: shapes,
            annotations: annotations,
            ...layout,
        };
    }, [layout, title, height, turns]);

    return (
        <div
            className="w-full bg-[#242324] border border-[#2C2C2B] rounded-lg p-2"
            style={{ height: `${height}px` }}
        >
            <Plot
                data={data}
                layout={defaultLayout}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
                config={{ responsive: true, displayModeBar: false }}
            />
        </div>
    );
};

export default Graph;
