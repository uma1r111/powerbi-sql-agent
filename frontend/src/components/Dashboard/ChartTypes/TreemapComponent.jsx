import React, { useState } from 'react';
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';

const COLORS = ['#6366f1', '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#8b5cf6', '#f43f5e', '#a855f7', '#14b8a6'];

const CustomContent = (props) => {
    const { x, y, width, height, index, name, value } = props;
    if (!width || !height || width < 10 || height < 10) return null;

    const fill = COLORS[index % COLORS.length];
    const showText = width > 45 && height > 28;
    const showValue = width > 60 && height > 48;

    return (
        <g>
            <rect
                x={x + 1}
                y={y + 1}
                width={Math.max(0, width - 2)}
                height={Math.max(0, height - 2)}
                rx={4}
                ry={4}
                style={{ fill, fillOpacity: 0.85, stroke: '#fff', strokeWidth: 2 }}
            />
            {showText && (
                <text
                    x={x + width / 2}
                    y={y + height / 2 - (showValue ? 8 : 0)}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="#fff"
                    fontSize={Math.min(Math.max(10, width / 8), 14)}
                    fontWeight="600"
                    style={{ pointerEvents: 'none' }}
                >
                    {name && name.length > Math.floor(width / 9)
                        ? name.slice(0, Math.floor(width / 9)) + '…'
                        : name}
                </text>
            )}
            {showValue && (
                <text
                    x={x + width / 2}
                    y={y + height / 2 + 12}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="rgba(255,255,255,0.8)"
                    fontSize={Math.min(Math.max(9, width / 10), 12)}
                    style={{ pointerEvents: 'none' }}
                >
                    {typeof value === 'number'
                        ? value >= 1000000
                            ? `${(value / 1000000).toFixed(1)}M`
                            : value >= 1000
                                ? `${(value / 1000).toFixed(1)}K`
                                : value.toLocaleString()
                        : value}
                </text>
            )}
        </g>
    );
};

const TreemapComponent = ({ chart }) => {
    const { config, data } = chart;
    const [hoveredName, setHoveredName] = useState(null);
    if (!data?.length) return null;

    const keys = Object.keys(data[0]);
    const nameKey = config?.nameKey || config?.labelKey || keys[0];
    const valueKey = config?.valueKey || keys.find(k => k !== nameKey && typeof data[0][k] === 'number') || keys[1];

    const chartData = data.map(item => ({
        name: String(item[nameKey]),
        size: Math.max(1, Number(item[valueKey]) || 0),
    }));

    const CustomTooltip = ({ active, payload }) => {
        if (!active || !payload?.length) return null;
        const d = payload[0]?.payload;
        const total = chartData.reduce((s, i) => s + i.size, 0);
        const pct = total > 0 ? ((d.size / total) * 100).toFixed(1) : '0';
        return (
            <div className="bg-white border border-gray-200 rounded-xl shadow-xl p-3">
                <p className="font-semibold text-gray-900 text-sm mb-1">{d.name}</p>
                <p className="text-sm font-bold text-gray-800">{d.size?.toLocaleString()}</p>
                <p className="text-xs text-gray-500 mt-0.5">{pct}% of total</p>
            </div>
        );
    };

    return (
        <ResponsiveContainer width="100%" height="100%">
            <Treemap
                data={chartData}
                dataKey="size"
                aspectRatio={4 / 3}
                stroke="#fff"
                content={<CustomContent />}
            >
                <Tooltip content={<CustomTooltip />} />
            </Treemap>
        </ResponsiveContainer>
    );
};

export default TreemapComponent;
