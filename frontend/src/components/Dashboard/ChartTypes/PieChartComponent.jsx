// frontend/src/components/Dashboard/ChartTypes/PieChartComponent.jsx

import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const PieChartComponent = ({ chart, onSliceClick }) => {
    const { config, data } = chart;
    const labelKey = config?.labelKey || Object.keys(data[0])[0];
    const valueKey = config?.valueKey || Object.keys(data[0])[1];
    const colors = config?.colors || [
        '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981',
        '#6366f1', '#14b8a6', '#f43f5e', '#a855f7', '#06b6d4'
    ];

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0];

            // Calculate total safely
            const total = payload[0].payload?.payload
                ? payload[0].payload.payload.reduce((sum, item) => sum + Number(item[valueKey] || 0), 0)
                : 0;

            const percentage = total > 0 ? ((Number(data.value) / total) * 100).toFixed(1) : 0;

            return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                    <p className="font-semibold text-gray-900">{data.name}</p>
                    <p className="text-sm text-gray-600">
                        Value: <span className="font-medium">{typeof data.value === 'number' ? data.value.toLocaleString() : data.value}</span>
                    </p>
                    {total > 0 && (
                        <p className="text-sm text-gray-600">
                            Share: <span className="font-medium">{percentage}%</span>
                        </p>
                    )}
                </div>
            );
        }
        return null;
    };

    const handleClick = (data, index) => {
        if (onSliceClick) {
            onSliceClick(labelKey, data[labelKey]);
        }
    };

    // Transform data for recharts
    const chartData = data.map(item => ({
        name: item[labelKey],
        value: Number(item[valueKey] || 0)
    }));

    return (
        <ResponsiveContainer width="100%" height="100%">
            <PieChart>
                <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
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
                    height={36}
                    formatter={(value) => <span className="text-sm">{value}</span>}
                />
            </PieChart>
        </ResponsiveContainer>
    );
};

export default PieChartComponent;