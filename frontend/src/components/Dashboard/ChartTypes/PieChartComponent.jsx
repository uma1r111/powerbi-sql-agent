// frontend/src/components/Dashboard/ChartTypes/PieChartComponent.jsx

import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const PieChartComponent = ({ chart, onSliceClick }) => {
    const { config, data } = chart;

    // Auto-detect columns
    const allKeys = Object.keys(data[0] || {});
    const labelKey = config?.labelKey || allKeys[0];
    const valueKey = config?.valueKey || allKeys.find(k => k !== labelKey && typeof data[0][k] === 'number') || allKeys[1];

    console.log('🥧 PieChart data:', { labelKey, valueKey, firstRow: data[0] });

    const colors = config?.colors || [
        '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981',
        '#6366f1', '#14b8a6', '#f43f5e', '#a855f7', '#06b6d4'
    ];

    // Calculate total for percentage
    const total = data.reduce((sum, item) => sum + Number(item[valueKey] || 0), 0);

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const dataPoint = payload[0];
            const originalData = data.find(item => item[labelKey] === dataPoint.name);
            const value = originalData ? Number(originalData[valueKey] || 0) : Number(dataPoint.value || 0);
            const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;

            // Format the value nicely
            const formatValue = (val) => {
                if (typeof val === 'number') {
                    if (val >= 1000000) {
                        return `$${(val / 1000000).toFixed(2)}M`;
                    } else if (val >= 1000) {
                        return `$${val.toLocaleString()}`;
                    } else if (val % 1 !== 0) {
                        return val.toFixed(2);
                    } else {
                        return val.toLocaleString();
                    }
                }
                return val;
            };

            return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg max-w-xs">
                    <p className="font-semibold text-gray-900 break-words mb-2">{dataPoint.name}</p>
                    <div className="space-y-1">
                        <p className="text-sm text-gray-800 font-medium">
                            {formatValue(value)}
                        </p>
                        <p className="text-xs text-gray-600">
                            {percentage}% of total
                        </p>
                        {/* Show all other fields except label and value */}
                        {originalData && Object.entries(originalData).map(([key, val]) => {
                            if (key !== labelKey && key !== valueKey) {
                                return (
                                    <p key={key} className="text-xs text-gray-500">
                                        {key}: {String(val)}
                                    </p>
                                );
                            }
                            return null;
                        })}
                    </div>
                </div>
            );
        }
        return null;
    };

    const handleClick = (dataPoint, index) => {
        if (onSliceClick) {
            const originalItem = data[index];
            onSliceClick(labelKey, originalItem[labelKey]);
        }
    };

    // Transform data for recharts
    const chartData = data.map(item => ({
        name: item[labelKey],
        value: Number(item[valueKey] || 0),
    }));

    // Custom label to show only percentage on slices
    const renderLabel = ({ percent }) => {
        const percentValue = (percent * 100).toFixed(0);
        return percentValue > 5 ? `${percentValue}%` : '';
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
                    fill="#8884d8"
                    dataKey="value"
                    onClick={handleClick}
                    style={{ cursor: onSliceClick ? 'pointer' : 'default' }}
                >
                    {chartData.map((entry, index) => (
                        <Cell
                            key={`cell-${index}`}
                            fill={colors[index % colors.length]}
                            className="transition-opacity hover:opacity-80"
                        />
                    ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend
                    verticalAlign="bottom"
                    align="center"
                    height={60}
                    iconType="circle"
                    wrapperStyle={{
                        fontSize: '11px',
                        paddingTop: '8px',
                        lineHeight: '16px'
                    }}
                    formatter={(value) => {
                        return value.length > 20 ? value.substring(0, 20) + '...' : value;
                    }}
                />
            </PieChart>
        </ResponsiveContainer>
    );
};

export default PieChartComponent;