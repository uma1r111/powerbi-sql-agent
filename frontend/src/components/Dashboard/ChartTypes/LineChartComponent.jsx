// frontend/src/components/Dashboard/ChartTypes/LineChartComponent.jsx

import React from 'react';
import {
    LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer
} from 'recharts';

const LineChartComponent = ({ chart }) => {
    const { config, data } = chart;
    const xAxis = config?.xAxis || Object.keys(data[0])[0];
    const yAxis = config?.yAxis || Object.keys(data[0])[1];
    const color = config?.color || '#3b82f6';
    const curved = config?.curved || false;
    const isArea = chart.type === 'area';

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                    <p className="font-semibold text-gray-900">{data[xAxis]}</p>
                    <p className="text-sm text-gray-600">
                        {yAxis}: {' '}
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
            {isArea ? (
                <AreaChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                    <defs>
                        <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={color} stopOpacity={0.8} />
                            <stop offset="95%" stopColor={color} stopOpacity={0.1} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                        dataKey={xAxis}
                        stroke="#6b7280"
                        angle={-45}
                        textAnchor="end"
                        height={60}
                        style={{ fontSize: '11px' }}
                    />
                    <YAxis stroke="#6b7280" />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                        type={curved ? "monotone" : "linear"}
                        dataKey={yAxis}
                        stroke={color}
                        strokeWidth={2}
                        fill="url(#colorGradient)"
                    />
                </AreaChart>
            ) : (
                <LineChart data={data} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                        dataKey={xAxis}
                        stroke="#6b7280"
                        angle={-45}
                        textAnchor="end"
                        height={60}
                        style={{ fontSize: '11px' }}
                    />
                    <YAxis stroke="#6b7280" />
                    <Tooltip content={<CustomTooltip />} />
                    <Line
                        type={curved ? "monotone" : "linear"}
                        dataKey={yAxis}
                        stroke={color}
                        strokeWidth={3}
                        dot={{ fill: color, r: 4 }}
                        activeDot={{ r: 6 }}
                    />
                </LineChart>
            )}
        </ResponsiveContainer>
    );
};

export default LineChartComponent;