'use client';

import dynamic from 'next/dynamic';
import { useMemo } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface GraphProps {
    data: any[];
    layout?: any;
    title?: string;
    height?: number;
}

const Graph = ({ data, layout, title, height = 300 }: GraphProps) => {
    const defaultLayout = useMemo(() => ({
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
        ...layout,
    }), [layout, title, height]);

    return (
        <div className="w-full h-full bg-[#242324] border border-[#2C2C2B] rounded-lg p-2">
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
