import React from 'react';

// Custom SVG funnel — recharts FunnelChart API is unstable across minor versions
// This implementation is pure SVG and always works.
const FunnelChartComponent = ({ chart }) => {
    const { config, data } = chart;
    if (!data?.length) return null;

    const keys = Object.keys(data[0]);
    const nameKey = config?.nameKey || config?.labelKey || keys[0];
    const valueKey = config?.valueKey || keys.find(k => k !== nameKey && typeof data[0][k] === 'number') || keys[1];

    const sorted = [...data]
        .map(item => ({ name: String(item[nameKey]), value: Number(item[valueKey]) || 0 }))
        .sort((a, b) => b.value - a.value);

    const maxVal = sorted[0]?.value || 1;

    const PALETTE = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe', '#e0e7ff', '#ede9fe', '#ddd6fe', '#c4b5fd'];

    const formatValue = (v) => {
        if (v >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
        if (v >= 1000) return `${(v / 1000).toFixed(1)}K`;
        return v.toLocaleString();
    };

    return (
        <div className="h-full flex flex-col justify-center gap-1 px-4 py-2 overflow-y-auto">
            {sorted.map((item, idx) => {
                const pct = (item.value / maxVal) * 100;
                const convRate = idx > 0 ? ((item.value / maxVal) * 100).toFixed(0) : 100;
                return (
                    <div key={idx} className="flex flex-col gap-0.5">
                        <div className="flex items-center justify-between text-xs mb-0.5">
                            <span className="font-medium text-gray-700 truncate max-w-[45%]">{item.name}</span>
                            <div className="flex items-center gap-2">
                                <span className="text-gray-500">{formatValue(item.value)}</span>
                                {idx > 0 && (
                                    <span className="text-xs text-indigo-500 font-semibold">{convRate}%</span>
                                )}
                            </div>
                        </div>
                        <div className="relative h-7 rounded-md overflow-hidden bg-gray-100">
                            <div
                                className="absolute inset-y-0 left-0 rounded-md transition-all duration-500"
                                style={{
                                    width: `${pct}%`,
                                    backgroundColor: PALETTE[idx % PALETTE.length],
                                }}
                            />
                            <div className="absolute inset-0 flex items-center px-2">
                                <span className="text-xs font-semibold text-white drop-shadow" style={{ textShadow: '0 1px 2px rgba(0,0,0,0.4)' }}>
                                    {pct.toFixed(0)}%
                                </span>
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default FunnelChartComponent;
