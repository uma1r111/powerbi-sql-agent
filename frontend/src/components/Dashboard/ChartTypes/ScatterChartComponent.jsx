import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label } from 'recharts';

const ScatterChartComponent = ({ chart }) => {
    const { config, data } = chart;
    if (!data?.length) return null;

    const keys = Object.keys(data[0]);
    const xKey = config?.xAxis || keys[0];
    const yKey = config?.yAxis || keys[1];
    const zKey = config?.zAxis || (keys.length > 2 ? keys[2] : null);
    const color = config?.color || '#6366f1';

    const chartData = data.map(row => ({
        x: Number(row[xKey]) || 0,
        y: Number(row[yKey]) || 0,
        ...(zKey ? { z: Math.max(1, Number(row[zKey]) || 1) } : {}),
        _label: row[keys[0]],
    }));

    const fmt = (v) => typeof v === 'number' ? v.toLocaleString() : v;

    const CustomTooltip = ({ active, payload }) => {
        if (!active || !payload?.length) return null;
        const d = payload[0]?.payload;
        return (
            <div className="bg-white border border-gray-200 rounded-xl shadow-xl p-3">
                {d._label && String(d._label) !== String(d.x) && (
                    <p className="font-semibold text-gray-900 text-sm mb-2">{d._label}</p>
                )}
                <p className="text-xs text-gray-500">{xKey.replace(/_/g, ' ')}: <span className="font-semibold text-gray-800">{fmt(d?.x)}</span></p>
                <p className="text-xs text-gray-500">{yKey.replace(/_/g, ' ')}: <span className="font-semibold text-gray-800">{fmt(d?.y)}</span></p>
                {zKey && <p className="text-xs text-gray-500">{zKey.replace(/_/g, ' ')}: <span className="font-semibold text-gray-800">{fmt(d?.z)}</span></p>}
            </div>
        );
    };

    return (
        <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="x" type="number" name={xKey} stroke="#94a3b8" style={{ fontSize: '11px' }}>
                    <Label value={xKey.replace(/_/g, ' ')} position="bottom" offset={-10}
                        style={{ fontSize: '11px', fill: '#64748b', textTransform: 'capitalize' }} />
                </XAxis>
                <YAxis dataKey="y" type="number" name={yKey} stroke="#94a3b8" style={{ fontSize: '11px' }}>
                    <Label value={yKey.replace(/_/g, ' ')} angle={-90} position="insideLeft" offset={15}
                        style={{ fontSize: '11px', fill: '#64748b', textTransform: 'capitalize' }} />
                </YAxis>
                {zKey && <ZAxis dataKey="z" range={[40, 500]} name={zKey} />}
                <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: '#e2e8f0' }} />
                <Scatter
                    data={chartData}
                    fill={color}
                    fillOpacity={0.75}
                    stroke={color}
                    strokeWidth={1}
                />
            </ScatterChart>
        </ResponsiveContainer>
    );
};

export default ScatterChartComponent;
