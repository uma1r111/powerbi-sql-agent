// frontend/src/components/Dashboard/ChartCard.jsx

import React, { useState, useRef, useMemo } from 'react';
import { X, Maximize2, Minimize2, Download, FileText, Image as ImageIcon, Settings2, RefreshCw } from 'lucide-react';
import { toPng } from 'html-to-image';
import KPICard from './ChartTypes/KPICard';
import BarChartComponent from './ChartTypes/BarChartComponent';
import LineChartComponent from './ChartTypes/LineChartComponent';
import PieChartComponent from './ChartTypes/PieChartComponent';
import DonutChartComponent from './ChartTypes/DonutChartComponent';
import ScatterChartComponent from './ChartTypes/ScatterChartComponent';
import RadarChartComponent from './ChartTypes/RadarChartComponent';
import FunnelChartComponent from './ChartTypes/FunnelChartComponent';
import TreemapComponent from './ChartTypes/TreemapComponent';
import TornadoChartComponent from './ChartTypes/TornadoChartComponent';
import DataTable from './ChartTypes/DataTable';

const CHART_TYPE_LABELS = {
    kpi: 'KPI',
    bar: 'Bar',
    line: 'Line',
    area: 'Area',
    pie: 'Pie',
    donut: 'Donut',
    scatter: 'Scatter',
    radar: 'Radar',
    funnel: 'Funnel',
    treemap: 'Treemap',
    tornado: 'Tornado',
    table: 'Table',
};

const TYPE_BADGE_COLOR = {
    kpi: '#6366f1',
    bar: '#3b82f6',
    line: '#0ea5e9',
    area: '#06b6d4',
    pie: '#ec4899',
    donut: '#a855f7',
    scatter: '#f59e0b',
    radar: '#10b981',
    funnel: '#8b5cf6',
    treemap: '#f43f5e',
    tornado: '#14b8a6',
    table: '#64748b',
};

const ChartCard = ({ chart, activeFilter, onFilterSelect, onRemove, onChartUpdate, theme }) => {
    const [isMaximized, setIsMaximized] = useState(false);
    const [showDownloadMenu, setShowDownloadMenu] = useState(false);
    const [isCapturing, setIsCapturing] = useState(false);
    const [showEditPanel, setShowEditPanel] = useState(false);
    const [editTitle, setEditTitle] = useState(chart.title);
    const [editColor, setEditColor] = useState(chart.config?.color || '#6366f1');
    const [editType, setEditType] = useState(chart.type);
    const chartRef = useRef(null);

    const t = theme || { surface: '#fff', border: '#e2e8f0', text: '#1e293b', textSub: '#64748b', textMuted: '#94a3b8', accent: '#6366f1', accentLight: '#eef2ff', shadow: '0 1px 3px rgba(0,0,0,0.08)' };

    const isSource   = activeFilter?.sourceChartId === chart.chart_id;
    const isFiltered = !!(activeFilter && !isSource);

    const hasMatchingKey = useMemo(() => {
        if (!activeFilter || !chart.data?.length) return false;
        return chart.data.some(row =>
            Object.keys(row).some(k => k.toLowerCase() === activeFilter.key.toLowerCase())
        );
    }, [activeFilter, chart.data]);

    const selectedKey   = (activeFilter && hasMatchingKey) ? activeFilter.key : null;
    const selectedValue = (activeFilter && hasMatchingKey) ? String(activeFilter.value) : null;
    const onSelect      = (key, value) => onFilterSelect?.(chart.chart_id, key, value);

    // Render chart — when edit panel is open, apply live preview overrides (color + type)
    const renderChart = (c = chart, livePreview = false) => {
        const previewChart = livePreview && showEditPanel
            ? { ...c, type: editType, config: { ...c.config, color: editColor } }
            : c;
        const props = { chart: previewChart, selectedKey, selectedValue, onSelect, theme: t };
        switch (previewChart.type) {
            case 'kpi':      return <KPICard {...props} />;
            case 'bar':      return <BarChartComponent {...props} />;
            case 'line':     return <LineChartComponent {...props} />;
            case 'area':     return <LineChartComponent {...props} />;
            case 'pie':      return <PieChartComponent {...props} />;
            case 'donut':    return <DonutChartComponent {...props} />;
            case 'scatter':  return <ScatterChartComponent {...props} />;
            case 'radar':    return <RadarChartComponent {...props} />;
            case 'funnel':   return <FunnelChartComponent {...props} />;
            case 'treemap':  return <TreemapComponent {...props} />;
            case 'tornado':  return <TornadoChartComponent {...props} />;
            case 'table':    return <DataTable {...props} />;
            default:
                return (
                    <div className="h-full flex items-center justify-center" style={{ color: t.textMuted }}>
                        <p className="text-sm">Unsupported type: {previewChart.type}</p>
                    </div>
                );
        }
    };

    const handleDownloadCSV = () => {
        try {
            const data = chart.data;
            if (!data?.length) { alert('No data to download'); return; }
            const headers = Object.keys(data[0]);
            const csv = [
                headers.join(','),
                ...data.map(row => headers.map(h => {
                    const v = row[h];
                    return typeof v === 'string' && (v.includes(',') || v.includes('"'))
                        ? `"${v.replace(/"/g, '""')}"` : v;
                }).join(','))
            ].join('\n');
            const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }));
            const a = Object.assign(document.createElement('a'), { href: url, download: `${chart.title.replace(/[^a-z0-9]/gi, '_')}.csv` });
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            setShowDownloadMenu(false);
        } catch (e) { console.error(e); }
    };

    const handleDownloadImage = async () => {
        if (!chartRef.current) return;
        setShowDownloadMenu(false);
        setIsCapturing(true);
        await new Promise(r => setTimeout(r, 150));
        try {
            const dataUrl = await toPng(chartRef.current, { backgroundColor: t.surface, pixelRatio: 2 });
            const a = Object.assign(document.createElement('a'), { href: dataUrl, download: `${chart.title.replace(/[^a-z0-9]/gi, '_')}.png` });
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } catch (e) { console.error(e); }
        finally { setIsCapturing(false); }
    };

    const handleSaveEdit = () => {
        if (onChartUpdate) {
            onChartUpdate(chart.chart_id, {
                title: editTitle,
                type: editType,
                config: { ...chart.config, color: editColor }
            });
        }
        setShowEditPanel(false);
    };

    const badgeColor = TYPE_BADGE_COLOR[chart.type] || '#64748b';

    const ringStyle = isSource
        ? `0 0 0 2px ${t.accent}`
        : (isFiltered && hasMatchingKey)
            ? `0 0 0 1px ${t.accent}40`
            : 'none';

    const cardStyle = {
        background: t.surface,
        border: `1px solid ${t.border}`,
        boxShadow: `${t.shadow}, ${ringStyle !== 'none' ? ringStyle : ''}`,
        outline: ringStyle !== 'none' ? ringStyle : undefined,
    };

    const headerStyle = {
        borderBottom: `1px solid ${t.border}`,
        background: t.surface,
    };

    const COLORS_PALETTE = ['#6366f1','#3b82f6','#10b981','#f59e0b','#ec4899','#06b6d4','#8b5cf6','#f43f5e','#14b8a6','#f97316'];

    return (
        <>
            <div
                ref={chartRef}
                className={`rounded-xl h-full flex flex-col transition-all duration-200 ${isMaximized ? 'hidden' : ''}`}
                style={cardStyle}
            >
                {/* Header */}
                <div className="flex items-center justify-between px-3 py-2.5 rounded-t-xl shrink-0" style={headerStyle}>
                    <div className="flex items-center gap-2 min-w-0">
                        <span
                            className="shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider"
                            style={{ background: `${badgeColor}18`, color: badgeColor }}
                        >
                            {CHART_TYPE_LABELS[chart.type] || chart.type}
                        </span>
                        <h3 className="font-semibold text-sm truncate" style={{ color: t.text }}>
                            {chart.title}
                        </h3>
                        {isSource && (
                            <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded-full font-semibold"
                                style={{ background: `${t.accent}20`, color: t.accent }}>
                                filtering
                            </span>
                        )}
                        {isFiltered && hasMatchingKey && (
                            <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded-full"
                                style={{ background: `${t.accent}12`, color: t.accent }}>
                                filtered
                            </span>
                        )}
                    </div>

                    <div className={`flex items-center gap-0.5 shrink-0 ${isCapturing ? 'opacity-0' : ''}`}>
                        {onChartUpdate && (
                            <button onClick={() => setShowEditPanel(!showEditPanel)}
                                className="p-1.5 rounded-lg transition-colors hover:opacity-70"
                                style={{ color: t.textMuted }}
                                title="Edit chart">
                                <Settings2 className="w-3.5 h-3.5" />
                            </button>
                        )}
                        <button onClick={() => setIsMaximized(true)}
                            className="p-1.5 rounded-lg transition-colors hover:opacity-70"
                            style={{ color: t.textMuted }}
                            title="Expand">
                            <Maximize2 className="w-3.5 h-3.5" />
                        </button>
                        <div className="relative">
                            <button onClick={() => setShowDownloadMenu(!showDownloadMenu)}
                                className="p-1.5 rounded-lg transition-colors hover:opacity-70"
                                style={{ color: t.textMuted }}
                                title="Download">
                                <Download className="w-3.5 h-3.5" />
                            </button>
                            {showDownloadMenu && (
                                <>
                                    <div className="fixed inset-0 z-40" onClick={() => setShowDownloadMenu(false)} />
                                    <div className="absolute right-0 mt-1 w-44 rounded-xl shadow-xl border py-1 z-50"
                                        style={{ background: t.surface, borderColor: t.border }}>
                                        <button onClick={handleDownloadCSV}
                                            className="w-full flex items-center gap-2 px-3 py-2 text-sm transition-colors hover:opacity-70"
                                            style={{ color: t.text }}>
                                            <FileText className="w-4 h-4" /> Download CSV
                                        </button>
                                        <button onClick={handleDownloadImage}
                                            className="w-full flex items-center gap-2 px-3 py-2 text-sm transition-colors hover:opacity-70"
                                            style={{ color: t.text }}>
                                            <ImageIcon className="w-4 h-4" /> Download PNG
                                        </button>
                                    </div>
                                </>
                            )}
                        </div>
                        {onRemove && (
                            <button onClick={() => onRemove(chart.chart_id)}
                                className="p-1.5 rounded-lg transition-colors hover:opacity-70"
                                style={{ color: '#ef4444' }}
                                title="Remove">
                                <X className="w-3.5 h-3.5" />
                            </button>
                        )}
                    </div>
                </div>

                {/* Inline edit panel */}
                {showEditPanel && (
                    <div className="px-3 py-2 border-b shrink-0" style={{ borderColor: t.border, background: t.surfaceHover || t.surface }}>
                        <div className="flex items-center gap-2 flex-wrap">
                            <input
                                value={editTitle}
                                onChange={e => setEditTitle(e.target.value)}
                                className="flex-1 text-sm px-2 py-1 rounded-lg border focus:outline-none"
                                style={{ background: t.surface, color: t.text, borderColor: t.border }}
                                placeholder="Chart title"
                            />
                            <select value={editType} onChange={e => setEditType(e.target.value)}
                                style={{ background: t.surface, color: t.text, border: `1px solid ${t.border}`, borderRadius: '6px', padding: '4px 8px', fontSize: '12px' }}>
                                {Object.entries(CHART_TYPE_LABELS).filter(([id]) => id !== 'kpi').map(([id, label]) => (
                                    <option key={id} value={id}>{label}</option>
                                ))}
                            </select>
                            <div className="flex gap-1 items-center">
                                <span style={{ fontSize: '10px', color: t.textMuted, marginRight: '2px' }}>Color:</span>
                                {COLORS_PALETTE.map(c => (
                                    <button key={c}
                                        onClick={() => setEditColor(c)}
                                        title={editColor === c ? 'Selected (preview active)' : 'Preview'}
                                        className="w-5 h-5 rounded-full border-2 transition-transform hover:scale-125"
                                        style={{ background: c, borderColor: editColor === c ? '#fff' : 'transparent', boxShadow: editColor === c ? `0 0 0 2px ${c}` : 'none' }} />
                                ))}
                            </div>
                            <button onClick={handleSaveEdit}
                                className="text-xs px-2 py-1 rounded-lg font-semibold"
                                style={{ background: t.accent, color: '#fff' }}>
                                Save
                            </button>
                            <button onClick={() => setShowEditPanel(false)}
                                className="text-xs px-2 py-1 rounded-lg"
                                style={{ color: t.textMuted }}>
                                Cancel
                            </button>
                        </div>
                    </div>
                )}

                {/* Chart — live preview when edit panel is open */}
                <div className="flex-1 p-3 overflow-hidden min-h-0">
                    {renderChart(chart, true)}
                </div>

                {/* Footer */}
                {chart.type !== 'kpi' && chart.data && !isCapturing && (
                    <div className="px-3 py-1.5 border-t rounded-b-xl shrink-0 flex items-center justify-between"
                        style={{ borderColor: t.border }}>
                        <span className="text-xs" style={{ color: t.textMuted }}>
                            {chart.data.length.toLocaleString()} {chart.data.length === 1 ? 'row' : 'rows'}
                        </span>
                        {activeFilter && hasMatchingKey && !isSource && (
                            <span className="text-xs font-medium" style={{ color: t.accent }}>filtered</span>
                        )}
                    </div>
                )}
            </div>

            {/* Maximized overlay */}
            {isMaximized && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-6"
                    style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}>
                    <div className="rounded-2xl shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col"
                        style={{ background: t.surface, border: `1px solid ${t.border}` }}>
                        <div className="flex items-center justify-between px-6 py-4 border-b shrink-0"
                            style={{ borderColor: t.border }}>
                            <div className="flex items-center gap-3">
                                <span className="text-xs font-bold px-2 py-1 rounded uppercase"
                                    style={{ background: `${badgeColor}18`, color: badgeColor }}>
                                    {CHART_TYPE_LABELS[chart.type]}
                                </span>
                                <h3 className="font-bold text-lg" style={{ color: t.text }}>{chart.title}</h3>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="relative">
                                    <button onClick={() => setShowDownloadMenu(!showDownloadMenu)}
                                        className="p-2 rounded-lg transition-colors hover:opacity-70"
                                        style={{ color: t.textMuted }}>
                                        <Download className="w-5 h-5" />
                                    </button>
                                    {showDownloadMenu && (
                                        <>
                                            <div className="fixed inset-0 z-40" onClick={() => setShowDownloadMenu(false)} />
                                            <div className="absolute right-0 mt-1 w-44 rounded-xl shadow-xl border py-1 z-50"
                                                style={{ background: t.surface, borderColor: t.border }}>
                                                <button onClick={handleDownloadCSV}
                                                    className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:opacity-70"
                                                    style={{ color: t.text }}>
                                                    <FileText className="w-4 h-4" /> CSV
                                                </button>
                                                <button onClick={handleDownloadImage}
                                                    className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:opacity-70"
                                                    style={{ color: t.text }}>
                                                    <ImageIcon className="w-4 h-4" /> PNG
                                                </button>
                                            </div>
                                        </>
                                    )}
                                </div>
                                <button onClick={() => setIsMaximized(false)}
                                    className="p-2 rounded-lg transition-colors hover:opacity-70"
                                    style={{ color: t.textMuted }}>
                                    <Minimize2 className="w-5 h-5" />
                                </button>
                            </div>
                        </div>
                        <div className="flex-1 p-6 overflow-hidden min-h-0">
                            {renderChart(chart)}
                        </div>
                        {chart.type !== 'kpi' && chart.data && (
                            <div className="px-6 py-3 border-t text-sm"
                                style={{ borderColor: t.border, color: t.textMuted }}>
                                {chart.data.length.toLocaleString()} rows
                            </div>
                        )}
                    </div>
                </div>
            )}
        </>
    );
};

export default ChartCard;
