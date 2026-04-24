import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const COLORS = ['#6366f1','#3b82f6','#10b981','#f59e0b','#ec4899','#06b6d4','#8b5cf6','#f43f5e','#a855f7','#14b8a6'];

const DonutChartComponent = ({ chart, selectedKey, selectedValue, onSelect }) => {
    const { config, data } = chart;
    if (!data?.length) return null;

    const allKeys = Object.keys(data[0]);
    const labelKey = config?.labelKey || allKeys[0];
    const valueKey = config?.valueKey || allKeys.find(k => k !== labelKey && typeof data[0][k] === 'number') || allKeys[1];
    const colors = config?.colors || COLORS;

    const total = data.reduce((sum, item) => sum + Number(item[valueKey] || 0), 0);
    const hasSelection = !!selectedValue;

    const formatTotal = (val) => {
        if (val >= 1000000) return `${(val / 1000000).toFixed(1)}M`;
        if (val >= 1000) return `${(val / 1000).toFixed(1)}K`;
        return val.toLocaleString();
    };

    const formatValue = (val) => {
        if (typeof val !== 'number') return val;
        if (val >= 1000000) return `${(val / 1000000).toFixed(2)}M`;
        if (val >= 1000) return val.toLocaleString();
        return val % 1 !== 0 ? val.toFixed(2) : val.toLocaleString();
    };

    const CustomTooltip = ({ active, payload }) => {
        if (!active || !payload?.length) return null;
        const dp = payload[0];
        const original = data.find(item => String(item[labelKey]) === String(dp.name));
        const value = original ? Number(original[valueKey] || 0) : Number(dp.value || 0);
        const pct = total > 0 ? ((value / total) * 100).toFixed(1) : '0';
        return (
            <div className="bg-white border border-gray-200 rounded-xl shadow-xl p-3 max-w-[180px]">
                <p className="font-semibold text-gray-900 text-sm truncate mb-1">{dp.name}</p>
                <p className="text-base font-bold text-gray-800">{formatValue(value)}</p>
                <div className="flex items-center gap-1 mt-1">
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: dp.payload?.fill || colors[0] }} />
                    <p className="text-xs text-gray-500">{pct}% of total</p>
                </div>
            </div>
        );
    };

    const chartData = data.map(item => ({
        name: String(item[labelKey]),
        value: Number(item[valueKey] || 0),
    }));

    const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
        if (percent < 0.06) return null;
        const RADIAN = Math.PI / 180;
        const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
        const x = cx + radius * Math.cos(-midAngle * RADIAN);
        const y = cy + radius * Math.sin(-midAngle * RADIAN);
        return (
            <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central"
                fontSize={11} fontWeight="600">
                {`${(percent * 100).toFixed(0)}%`}
            </text>
        );
    };

    return (
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                    <Pie
                        data={chartData}
                        cx="50%"
                        cy="44%"
                        innerRadius="38%"
                        outerRadius="62%"
                        dataKey="value"
                        fill={colors[0]}
                        labelLine={false}
                        label={renderCustomizedLabel}
                        isAnimationActive={false}
                        onClick={(_, index) => onSelect?.(labelKey, data[index]?.[labelKey])}
                        style={{ cursor: onSelect ? 'pointer' : 'default' }}
                    >
                        {chartData.map((entry, index) => (
                            <Cell
                                key={`cell-${index}`}
                                fill={colors[index % colors.length]}
                                opacity={!hasSelection ? 1 : String(entry.name) === selectedValue ? 1 : 0.2}
                                stroke={hasSelection && String(entry.name) === selectedValue ? '#1d4ed8' : 'none'}
                                strokeWidth={2}
                            />
                        ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                        verticalAlign="bottom"
                        height={46}
                        iconType="circle"
                        iconSize={8}
                        wrapperStyle={{ fontSize: '11px', paddingTop: '2px' }}
                        formatter={(value) => value.length > 18 ? value.slice(0, 18) + '…' : value}
                    />
                </PieChart>
            </ResponsiveContainer>
            {/* Center label — rendered as HTML overlay so it never interferes with Pie rendering */}
            <div style={{
                position: 'absolute', top: '44%', left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center', pointerEvents: 'none',
            }}>
                <div style={{ fontSize: '18px', fontWeight: '700', color: '#1e293b', lineHeight: 1.1 }}>
                    {formatTotal(total)}
                </div>
                <div style={{ fontSize: '9px', color: '#64748b', fontWeight: '600', letterSpacing: '1px', marginTop: '2px' }}>
                    TOTAL
                </div>
            </div>
        </div>
    );
};

export default DonutChartComponent;
