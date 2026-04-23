// frontend/src/components/Dashboard/ChartTypes/BarChartComponent.jsx

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const PALETTE = [
    '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981',
    '#6366f1', '#14b8a6', '#f43f5e', '#a855f7', '#06b6d4'
];

const BarChartComponent = ({ chart, selectedKey, selectedValue, onSelect }) => {
    const { config, data } = chart;
    const isHorizontal = config?.horizontal || false;
    const xAxis = config?.xAxis || Object.keys(data[0])[0];
    const yAxis = config?.yAxis || Object.keys(data[0])[1];
    const color = config?.color || '#3b82f6';

    // If a custom color is configured (anything other than the default), use single-color mode
    const isCustomColor = !!(config?.color && config.color !== '#3b82f6');

    const isSelectable = !!onSelect;
    const hasSelection = !!selectedValue;

    const valueCol = isHorizontal ? xAxis : yAxis;
    const hasNumericData = data.some(row => typeof row[valueCol] === 'number' && !isNaN(row[valueCol]));
    if (!hasNumericData) {
        const cols = Object.keys(data[0]);
        return (
            <div className="h-full overflow-auto">
                <table className="w-full text-sm border-collapse">
                    <thead className="bg-gray-50 sticky top-0">
                        <tr>
                            {cols.map(col => (
                                <th key={col} className="px-3 py-2 text-left text-xs font-semibold text-gray-600 uppercase border-b border-gray-200">
                                    {col.replace(/_/g, ' ')}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                        {data.map((row, i) => (
                            <tr key={i} className="hover:bg-gray-50">
                                {cols.map(col => (
                                    <td key={col} className="px-3 py-2 text-gray-800 whitespace-nowrap">
                                        {row[col] ?? '—'}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
    }

    const barLabelKey = isHorizontal ? yAxis : xAxis;

    const getCellOpacity = (entry) => {
        if (!hasSelection) return 1;
        return String(entry[barLabelKey]) === selectedValue ? 1 : 0.25;
    };

    const getCellStroke = (entry) => {
        if (!hasSelection) return 'none';
        return String(entry[barLabelKey]) === selectedValue ? '#1d4ed8' : 'none';
    };

    // Respect configured color; fall back to multi-color palette when no custom color
    const getCellFill = (index) => {
        if (isCustomColor) return color;
        return PALETTE[index % PALETTE.length];
    };

    const handleClick = (barData) => {
        if (!isSelectable) return;
        onSelect(barLabelKey, barData[barLabelKey]);
    };

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '10px 14px', boxShadow: '0 4px 12px rgba(0,0,0,.12)' }}>
                    <p style={{ fontWeight: '600', color: '#1e293b', margin: '0 0 4px' }}>{isHorizontal ? d[yAxis] : d[xAxis]}</p>
                    <p style={{ fontSize: '12px', color: '#64748b', margin: 0 }}>
                        {isHorizontal ? xAxis : yAxis}:{' '}
                        <span style={{ fontWeight: '600', color: '#1e293b' }}>
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
                <BarChart
                    data={data}
                    layout="vertical"
                    margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
                >
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
                    <Bar
                        dataKey={xAxis}
                        radius={[0, 4, 4, 0]}
                        onClick={(barData) => handleClick(barData)}
                        style={{ cursor: isSelectable ? 'pointer' : 'default' }}
                    >
                        {data.map((entry, index) => (
                            <Cell
                                key={`cell-${index}`}
                                fill={getCellFill(index)}
                                opacity={getCellOpacity(entry)}
                                stroke={getCellStroke(entry)}
                                strokeWidth={2}
                            />
                        ))}
                    </Bar>
                </BarChart>
            ) : (
                <BarChart
                    data={data}
                    margin={{ top: 10, right: 30, left: 20, bottom: 50 }}
                >
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
                    <Bar
                        dataKey={yAxis}
                        radius={[4, 4, 0, 0]}
                        onClick={(barData) => handleClick(barData)}
                        style={{ cursor: isSelectable ? 'pointer' : 'default' }}
                    >
                        {data.map((entry, index) => (
                            <Cell
                                key={`cell-${index}`}
                                fill={getCellFill(index)}
                                opacity={getCellOpacity(entry)}
                                stroke={getCellStroke(entry)}
                                strokeWidth={2}
                            />
                        ))}
                    </Bar>
                </BarChart>
            )}
        </ResponsiveContainer>
    );
};

export default BarChartComponent;
