// frontend/src/components/Dashboard/ChartTypes/KPICard.jsx

import React, { useMemo } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, Tooltip } from 'recharts';

const KPICard = ({ chart, theme }) => {
    const { config, data, title } = chart;
    const t = theme || {
        surface: '#fff', text: '#1e293b', textSub: '#475569', textMuted: '#94a3b8',
        accent: '#6366f1', accentLight: '#eef2ff', border: '#e2e8f0', bg: '#f8fafc',
    };

    // Extract main numeric value from data or config
    const rawValue = useMemo(() => {
        if (config?.value !== undefined && config.value !== null) return config.value;
        if (!data?.length) return null;
        const firstRow = data[0];
        const keys = Object.keys(firstRow);
        // Prefer columns named total, count, sum, value, amount, revenue
        const preferred = ['total', 'count', 'sum', 'value', 'amount', 'revenue', 'avg', 'average'];
        const key = keys.find(k => preferred.some(p => k.toLowerCase().includes(p))) || keys[0];
        const v = firstRow[key];
        // Parse string numbers (e.g. PostgreSQL numeric type comes as string)
        if (typeof v === 'string' && !isNaN(parseFloat(v))) return parseFloat(v);
        return v;
    }, [config, data]);

    const label = config?.label || title || 'Metric';
    const trend = config?.trend;
    const accentColor = config?.color || t.accent;

    // Build sparkline data from query results (last N values)
    const sparkData = useMemo(() => {
        if (!data || data.length < 2) return null;
        const cols = Object.keys(data[0]);
        // Find a numeric column for sparkline
        const numCol = cols.find(k => {
            const v = data[0][k];
            return typeof v === 'number' || (typeof v === 'string' && !isNaN(parseFloat(v)));
        });
        if (!numCol) return null;
        return data.slice(-12).map((row, i) => ({
            i,
            v: typeof row[numCol] === 'string' ? parseFloat(row[numCol]) : row[numCol],
        }));
    }, [data]);

    const formatValue = (val) => {
        if (val === null || val === undefined) return '—';
        const num = typeof val === 'string' ? parseFloat(val) : val;
        if (typeof num !== 'number' || isNaN(num)) return String(val);
        if (num >= 1_000_000) return `$${(num / 1_000_000).toFixed(2)}M`;
        if (num >= 10_000) return num.toLocaleString(undefined, { maximumFractionDigits: 0 });
        if (Number.isInteger(num)) return num.toLocaleString();
        return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
    };

    const trendColor = !trend ? t.textMuted : trend > 0 ? '#10b981' : '#ef4444';
    const TrendIcon = !trend ? Minus : trend > 0 ? TrendingUp : TrendingDown;

    return (
        <div style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            padding: '16px 20px 12px',
            background: t.surface,
            position: 'relative',
            overflow: 'hidden',
        }}>
            {/* Accent gradient strip */}
            <div style={{
                position: 'absolute',
                top: 0, left: 0, right: 0,
                height: '3px',
                background: `linear-gradient(90deg, ${accentColor}, ${accentColor}88)`,
            }} />

            {/* Label */}
            <p style={{
                fontSize: '11px',
                fontWeight: '600',
                color: t.textMuted,
                letterSpacing: '0.8px',
                textTransform: 'uppercase',
                margin: '0 0 8px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
            }}>
                {label}
            </p>

            {/* Value row */}
            <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flex: 1 }}>
                <div>
                    <div style={{
                        fontSize: 'clamp(22px, 4vw, 36px)',
                        fontWeight: '800',
                        color: t.text,
                        lineHeight: 1,
                        letterSpacing: '-0.5px',
                        marginBottom: trend !== undefined ? '6px' : '0',
                    }}>
                        {formatValue(rawValue)}
                    </div>

                    {trend !== undefined && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <TrendIcon style={{ width: '14px', height: '14px', color: trendColor }} />
                            <span style={{ fontSize: '12px', fontWeight: '600', color: trendColor }}>
                                {trend > 0 ? '+' : ''}{trend}%
                            </span>
                            <span style={{ fontSize: '11px', color: t.textMuted }}>vs last period</span>
                        </div>
                    )}
                </div>

                {/* Mini sparkline */}
                {sparkData && sparkData.length >= 3 && (
                    <div style={{ width: '80px', height: '40px', flexShrink: 0 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={sparkData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
                                <defs>
                                    <linearGradient id={`kpi-spark-${chart.chart_id}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={accentColor} stopOpacity={0.4} />
                                        <stop offset="100%" stopColor={accentColor} stopOpacity={0.02} />
                                    </linearGradient>
                                </defs>
                                <Tooltip content={() => null} />
                                <Area
                                    type="monotone"
                                    dataKey="v"
                                    stroke={accentColor}
                                    strokeWidth={1.5}
                                    fill={`url(#kpi-spark-${chart.chart_id})`}
                                    dot={false}
                                    activeDot={false}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* Accent icon when no sparkline */}
                {(!sparkData || sparkData.length < 3) && (
                    <div style={{
                        width: '40px', height: '40px',
                        borderRadius: '12px',
                        background: `${accentColor}15`,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        flexShrink: 0,
                    }}>
                        <TrendingUp style={{ width: '18px', height: '18px', color: accentColor }} />
                    </div>
                )}
            </div>

            {/* Row count badge */}
            {data?.length > 1 && (
                <p style={{
                    fontSize: '10px',
                    color: t.textMuted,
                    margin: '6px 0 0',
                    opacity: 0.6,
                }}>
                    {data.length} data points
                </p>
            )}
        </div>
    );
};

export default KPICard;
