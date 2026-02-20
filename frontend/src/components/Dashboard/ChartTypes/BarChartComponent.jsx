// frontend/src/components/Dashboard/ChartTypes/BarChartComponent.jsx

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const BarChartComponent = ({ chart }) => {
    const { config, data } = chart;
    const isHorizontal = config?.horizontal || false;
    const xAxis = config?.xAxis || Object.keys(data[0])[0];
    const yAxis = config?.yAxis || Object.keys(data[0])[1];
    const color = config?.color || '#3b82f6';

    // Generate colors for multiple bars
    const colors = [
        '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981',
        '#6366f1', '#14b8a6', '#f43f5e', '#a855f7', '#06b6d4'
    ];

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                    <p className="font-semibold text-gray-900">
                        {isHorizontal ? data[yAxis] : data[xAxis]}
                    </p>
                    <p className="text-sm text-gray-600">
                        {isHorizontal ? xAxis : yAxis}: {' '}
                        <span className="font-medium">
                            {typeof payload[0].value === 'number'
                                ? payload[0].value.toLocaleString()
                                : payload[0].value}
                        </span>
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <ResponsiveContainer width="100%" height="100%">
            {isHorizontal ? (
                <BarChart data={data} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" stroke="#6b7280" />
                    <YAxis
                        dataKey={yAxis}
                        type="category"
                        width={90}
                        stroke="#6b7280"
                        style={{ fontSize: '12px' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey={xAxis} radius={[0, 4, 4, 0]}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                        ))}
                    </Bar>
                </BarChart>
            ) : (
                <BarChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 50 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                        dataKey={xAxis}
                        stroke="#6b7280"
                        angle={-45}
                        textAnchor="end"
                        height={80}
                        style={{ fontSize: '11px' }}
                    />
                    <YAxis stroke="#6b7280" />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey={yAxis} fill={color} radius={[4, 4, 0, 0]}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                        ))}
                    </Bar>
                </BarChart>
            )}
        </ResponsiveContainer>
    );
};

export default BarChartComponent;