// frontend/src/components/Dashboard/ChartTypes/PieChartComponent.jsx

import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const PieChartComponent = ({ chart, selectedKey, selectedValue, onSelect }) => {
    const { config, data } = chart;

    const allKeys = Object.keys(data[0] || {});
    const labelKey = config?.labelKey || allKeys[0];
    const valueKey = config?.valueKey || allKeys.find(k => k !== labelKey && typeof data[0][k] === 'number') || allKeys[1];

    const colors = config?.colors || [
        '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981',
        '#6366f1', '#14b8a6', '#f43f5e', '#a855f7', '#06b6d4'
    ];

    const total = data.reduce((sum, item) => sum + Number(item[valueKey] || 0), 0);
    const hasSelection = !!selectedValue;

    const getCellOpacity = (entry) => {
        if (!hasSelection) return 1;
        return String(entry.name) === selectedValue ? 1 : 0.25;
    };

    const getCellStroke = (entry) => {
        if (!hasSelection) return 'none';
        return String(entry.name) === selectedValue ? '#1d4ed8' : 'none';
    };

    const formatValue = (val) => {
        if (typeof val !== 'number') return val;
        if (val >= 1000000) return `$${(val / 1000000).toFixed(2)}M`;
        if (val >= 1000) return `$${val.toLocaleString()}`;
        if (val % 1 !== 0) return val.toFixed(2);
        return val.toLocaleString();
    };

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const dp = payload[0];
            const original = data.find(item => item[labelKey] === dp.name);
            const value = original ? Number(original[valueKey] || 0) : Number(dp.value || 0);
            const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
            return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg max-w-xs">
                    <p className="font-semibold text-gray-900 break-words mb-1">{dp.name}</p>
                    <p className="text-sm text-gray-800 font-medium">{formatValue(value)}</p>
                    <p className="text-xs text-gray-500">{percentage}% of total</p>
                </div>
            );
        }
        return null;
    };

    const handleClick = (_, index) => {
        if (onSelect) {
            const originalItem = data[index];
            onSelect(labelKey, originalItem[labelKey]);
        }
    };

    const chartData = data.map(item => ({
        name: item[labelKey],
        value: Number(item[valueKey] || 0),
    }));

    const renderLabel = ({ percent }) => {
        const pct = (percent * 100).toFixed(0);
        return pct > 5 ? `${pct}%` : '';
    };

    return (
        <ResponsiveContainer width="100%" height="100%">
            <PieChart margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                <Pie
                    data={chartData}
                    cx="50%"
                    cy="45%"
                    labelLine={false}
                    label={renderLabel}
                    outerRadius="70%"
                    dataKey="value"
                    fill={colors[0]}
                    isAnimationActive={false}
                    onClick={handleClick}
                    style={{ cursor: onSelect ? 'pointer' : 'default' }}
                >
                    {chartData.map((entry, index) => (
                        <Cell
                            key={`cell-${index}`}
                            fill={colors[index % colors.length]}
                            opacity={getCellOpacity(entry)}
                            stroke={getCellStroke(entry)}
                            strokeWidth={hasSelection && String(entry.name) === selectedValue ? 2 : 0}
                        />
                    ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend
                    verticalAlign="bottom"
                    align="center"
                    height={60}
                    iconType="circle"
                    wrapperStyle={{ fontSize: '11px', paddingTop: '8px', lineHeight: '16px' }}
                    formatter={(value) => value.length > 20 ? value.substring(0, 20) + '...' : value}
                />
            </PieChart>
        </ResponsiveContainer>
    );
};

export default PieChartComponent;
