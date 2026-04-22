import React from 'react';
import {
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    ResponsiveContainer, Tooltip, Legend
} from 'recharts';

const COLORS = ['#6366f1', '#3b82f6', '#10b981', '#f59e0b', '#ec4899'];

const RadarChartComponent = ({ chart }) => {
    const { config, data } = chart;
    if (!data?.length) return null;

    const keys = Object.keys(data[0]);
    const subjectKey = config?.subjectKey || keys[0];
    const valueKeys = config?.valueKeys || keys.filter((k, i) => i > 0 && typeof data[0][k] === 'number');

    const effectiveValueKeys = valueKeys.length > 0 ? valueKeys : [keys[1]];

    const CustomTooltip = ({ active, payload }) => {
        if (!active || !payload?.length) return null;
        const subject = payload[0]?.payload?.[subjectKey];
        return (
            <div className="bg-white border border-gray-200 rounded-xl shadow-xl p-3">
                <p className="font-semibold text-gray-900 text-sm mb-2">{subject}</p>
                {payload.map((p, i) => (
                    <p key={i} className="text-xs" style={{ color: p.color }}>
                        {p.name}: <span className="font-semibold">{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}</span>
                    </p>
                ))}
            </div>
        );
    };

    return (
        <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="65%" data={data}>
                <PolarGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                <PolarAngleAxis
                    dataKey={subjectKey}
                    tick={{ fontSize: 11, fill: '#64748b' }}
                />
                <PolarRadiusAxis
                    angle={30}
                    tick={{ fontSize: 10, fill: '#94a3b8' }}
                    stroke="#e2e8f0"
                />
                {effectiveValueKeys.map((key, i) => (
                    <Radar
                        key={key}
                        name={key.replace(/_/g, ' ')}
                        dataKey={key}
                        stroke={COLORS[i % COLORS.length]}
                        fill={COLORS[i % COLORS.length]}
                        fillOpacity={effectiveValueKeys.length === 1 ? 0.25 : 0.15}
                        strokeWidth={2}
                        dot={{ r: 3, fill: COLORS[i % COLORS.length] }}
                    />
                ))}
                <Tooltip content={<CustomTooltip />} />
                {effectiveValueKeys.length > 1 && (
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                )}
            </RadarChart>
        </ResponsiveContainer>
    );
};

export default RadarChartComponent;
