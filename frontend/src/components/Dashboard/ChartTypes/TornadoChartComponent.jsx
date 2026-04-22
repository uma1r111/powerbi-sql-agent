import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, ReferenceLine, Cell, Legend
} from 'recharts';

const TornadoChartComponent = ({ chart }) => {
    const { config, data } = chart;
    if (!data?.length) return null;

    const keys = Object.keys(data[0]);
    const categoryKey = config?.categoryKey || config?.xAxis || keys[0];
    const leftKey = config?.leftKey || keys[1];
    const rightKey = config?.rightKey || (keys.length > 2 ? keys[2] : keys[1]);
    const hasTwoSeries = rightKey !== leftKey && keys.length > 2;

    const chartData = data.map(item => ({
        [categoryKey]: item[categoryKey],
        _left: hasTwoSeries ? -Math.abs(Number(item[leftKey]) || 0) : -Math.abs(Number(item[leftKey]) || 0),
        _right: hasTwoSeries ? Math.abs(Number(item[rightKey]) || 0) : Math.abs(Number(item[leftKey]) || 0),
        _rawLeft: Number(item[leftKey]) || 0,
        _rawRight: Number(item[rightKey]) || 0,
    }));

    const sorted = [...chartData].sort((a, b) => Math.abs(b._right) - Math.abs(a._right));

    const absMax = Math.max(
        ...data.map(d => Math.abs(Number(d[leftKey]) || 0)),
        ...(hasTwoSeries ? data.map(d => Math.abs(Number(d[rightKey]) || 0)) : [])
    );

    const tickFmt = (v) => {
        const abs = Math.abs(v);
        if (abs >= 1000000) return `${(abs / 1000000).toFixed(1)}M`;
        if (abs >= 1000) return `${(abs / 1000).toFixed(0)}K`;
        return abs.toLocaleString();
    };

    const CustomTooltip = ({ active, payload }) => {
        if (!active || !payload?.length) return null;
        const d = payload[0]?.payload;
        return (
            <div className="bg-white border border-gray-200 rounded-xl shadow-xl p-3">
                <p className="font-semibold text-gray-900 text-sm mb-2">{d[categoryKey]}</p>
                <div className="space-y-1">
                    <p className="text-xs" style={{ color: '#6366f1' }}>
                        {leftKey.replace(/_/g, ' ')}: <span className="font-bold">{d._rawLeft?.toLocaleString()}</span>
                    </p>
                    {hasTwoSeries && (
                        <p className="text-xs" style={{ color: '#10b981' }}>
                            {rightKey.replace(/_/g, ' ')}: <span className="font-bold">{d._rawRight?.toLocaleString()}</span>
                        </p>
                    )}
                </div>
            </div>
        );
    };

    return (
        <ResponsiveContainer width="100%" height="100%">
            <BarChart
                data={sorted}
                layout="vertical"
                margin={{ top: 10, right: 30, left: 100, bottom: 20 }}
                barGap={0}
            >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                <XAxis
                    type="number"
                    domain={[-absMax * 1.15, absMax * 1.15]}
                    tickFormatter={tickFmt}
                    stroke="#94a3b8"
                    style={{ fontSize: '11px' }}
                />
                <YAxis
                    dataKey={categoryKey}
                    type="category"
                    width={95}
                    stroke="#94a3b8"
                    style={{ fontSize: '11px' }}
                    tick={{ fill: '#475569' }}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(99,102,241,0.05)' }} />
                <ReferenceLine x={0} stroke="#334155" strokeWidth={2} />
                <Legend
                    wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
                    formatter={(value) => value === '_left' ? leftKey.replace(/_/g, ' ') : rightKey.replace(/_/g, ' ')}
                />
                <Bar dataKey="_left" name="_left" fill="#6366f1" radius={[0, 4, 4, 0]} maxBarSize={28} />
                {hasTwoSeries && (
                    <Bar dataKey="_right" name="_right" fill="#10b981" radius={[4, 0, 0, 4]} maxBarSize={28} />
                )}
            </BarChart>
        </ResponsiveContainer>
    );
};

export default TornadoChartComponent;
