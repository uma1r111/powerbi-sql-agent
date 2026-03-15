// frontend/src/components/Dashboard/ChartTypes/LineChartComponent.jsx

import React from 'react';
import {
    LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

const LineChartComponent = ({ chart, selectedKey, selectedValue, onSelect }) => {
    const { config, data } = chart;
    const xAxis = config?.xAxis || Object.keys(data[0])[0];
    const yAxis = config?.yAxis || Object.keys(data[0])[1];
    const color = config?.color || '#3b82f6';
    const curved = config?.curved || false;
    const isArea = chart.type === 'area';

    const isSelectable = !!onSelect;
    const hasSelection = !!selectedValue;

    const handleClick = (e) => {
        if (!isSelectable || !e?.activePayload) return;
        const d = e.activePayload[0].payload;
        onSelect(xAxis, d[xAxis]);
    };

    // Custom dot: highlight selected, dim others
    const renderDot = (props) => {
        const { cx, cy, payload } = props;
        const isSelected = hasSelection && String(payload[xAxis]) === selectedValue;
        const isDimmed = hasSelection && !isSelected;
        return (
            <circle
                key={`dot-${cx}-${cy}`}
                cx={cx}
                cy={cy}
                r={isSelected ? 7 : 4}
                fill={isSelected ? '#1d4ed8' : color}
                stroke={isSelected ? '#fff' : 'none'}
                strokeWidth={isSelected ? 2 : 0}
                opacity={isDimmed ? 0.2 : 1}
                style={{ cursor: isSelectable ? 'pointer' : 'default' }}
            />
        );
    };

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                    <p className="font-semibold text-gray-900">{d[xAxis]}</p>
                    <p className="text-sm text-gray-600">
                        {yAxis}:{' '}
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

    const sharedProps = {
        data,
        onClick: handleClick,
        style: { cursor: isSelectable ? 'pointer' : 'default' },
    };

    return (
        <ResponsiveContainer width="100%" height="100%">
            {isArea ? (
                <AreaChart {...sharedProps} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                    <defs>
                        <linearGradient id={`colorGradient-${chart.chart_id}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={color} stopOpacity={0.8} />
                            <stop offset="95%" stopColor={color} stopOpacity={0.1} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey={xAxis} stroke="#6b7280" angle={-45} textAnchor="end" height={60} style={{ fontSize: '11px' }} />
                    <YAxis stroke="#6b7280" />
                    <Tooltip content={<CustomTooltip />} />
                    {hasSelection && (
                        <ReferenceLine x={selectedValue} stroke="#1d4ed8" strokeWidth={2} strokeDasharray="4 2" />
                    )}
                    <Area
                        type={curved ? 'monotone' : 'linear'}
                        dataKey={yAxis}
                        stroke={color}
                        strokeWidth={2}
                        fill={`url(#colorGradient-${chart.chart_id})`}
                        dot={renderDot}
                        activeDot={{ r: 7, fill: '#1d4ed8' }}
                    />
                </AreaChart>
            ) : (
                <LineChart {...sharedProps} margin={{ top: 10, right: 30, left: 20, bottom: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey={xAxis} stroke="#6b7280" angle={-45} textAnchor="end" height={60} style={{ fontSize: '11px' }} />
                    <YAxis stroke="#6b7280" />
                    <Tooltip content={<CustomTooltip />} />
                    {hasSelection && (
                        <ReferenceLine x={selectedValue} stroke="#1d4ed8" strokeWidth={2} strokeDasharray="4 2" />
                    )}
                    <Line
                        type={curved ? 'monotone' : 'linear'}
                        dataKey={yAxis}
                        stroke={hasSelection ? `${color}55` : color}
                        strokeWidth={3}
                        dot={renderDot}
                        activeDot={{ r: 7, fill: '#1d4ed8' }}
                    />
                </LineChart>
            )}
        </ResponsiveContainer>
    );
};

export default LineChartComponent;
