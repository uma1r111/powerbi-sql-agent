// frontend/src/components/Dashboard/ChartCard.jsx

import React, { useState, useRef, useMemo } from 'react';
import { X, Maximize2, Minimize2, Download, FileText, Image as ImageIcon } from 'lucide-react';
import html2canvas from 'html2canvas';
import KPICard from './ChartTypes/KPICard';
import BarChartComponent from './ChartTypes/BarChartComponent';
import LineChartComponent from './ChartTypes/LineChartComponent';
import PieChartComponent from './ChartTypes/PieChartComponent';
import DataTable from './ChartTypes/DataTable';

const ChartCard = ({ chart, activeFilter, onFilterSelect, onRemove }) => {
    const [isMaximized, setIsMaximized] = useState(false);
    const [showDownloadMenu, setShowDownloadMenu] = useState(false);
    const [isCapturing, setIsCapturing] = useState(false);
    const chartRef = useRef(null);

    // Is this chart the source of the active filter?
    const isSource = activeFilter?.sourceChartId === chart.chart_id;
    // Is there an active filter from a different chart?
    const isFiltered = !!(activeFilter && !isSource);

    // For non-source charts: filter data to matching rows only
    const displayChart = useMemo(() => {
        if (!activeFilter || isSource || !chart.data?.length) return chart;
        const filtered = chart.data.filter(row =>
            Object.entries(row).some(([k, v]) =>
                k.toLowerCase() === activeFilter.key.toLowerCase() &&
                String(v) === String(activeFilter.value)
            )
        );
        // If no rows match the filter key exists in this chart's data — don't filter
        const hasKey = chart.data.some(row =>
            Object.keys(row).some(k => k.toLowerCase() === activeFilter.key.toLowerCase())
        );
        if (!hasKey) return chart;
        return filtered.length > 0 ? { ...chart, data: filtered } : chart;
    }, [activeFilter, chart, isSource]);

    // Selection state passed to chart components (only the source chart gets this)
    const selectedKey = isSource ? activeFilter.key : null;
    const selectedValue = isSource ? String(activeFilter.value) : null;
    const onSelect = (key, value) => onFilterSelect?.(chart.chart_id, key, value);

    const renderChart = (chartData = displayChart) => {
        switch (chartData.type) {
            case 'kpi':
                return <KPICard chart={chartData} />;
            case 'bar':
                return <BarChartComponent chart={chartData} selectedKey={selectedKey} selectedValue={selectedValue} onSelect={onSelect} />;
            case 'line':
            case 'area':
                return <LineChartComponent chart={chartData} selectedKey={selectedKey} selectedValue={selectedValue} onSelect={onSelect} />;
            case 'pie':
                return <PieChartComponent chart={chartData} selectedKey={selectedKey} selectedValue={selectedValue} onSelect={onSelect} />;
            case 'table':
                return <DataTable chart={chartData} />;
            default:
                return (
                    <div className="h-full flex items-center justify-center text-gray-500">
                        Unsupported chart type: {chartData.type}
                    </div>
                );
        }
    };

    const handleDownloadCSV = () => {
        try {
            const data = chart.data;
            if (!data || data.length === 0) { alert('No data available to download'); return; }
            const headers = Object.keys(data[0]);
            const csvContent = [
                headers.join(','),
                ...data.map(row => headers.map(h => {
                    const value = row[h];
                    if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                        return `"${value.replace(/"/g, '""')}"`;
                    }
                    return value;
                }).join(','))
            ].join('\n');
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${chart.title.replace(/[^a-z0-9]/gi, '_')}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            setShowDownloadMenu(false);
        } catch (error) {
            console.error('❌ CSV download error:', error);
            alert('Failed to download chart data as CSV');
        }
    };

    const handleDownloadImage = async () => {
        try {
            if (!chartRef.current) { alert('Chart not ready for download'); return; }

            setShowDownloadMenu(false);
            setIsCapturing(true);
            await new Promise(resolve => setTimeout(resolve, 150));

            const element = chartRef.current;
            const svgEl = element.querySelector('svg');

            const triggerDownload = (blob) => {
                if (!blob) { alert('Failed to generate image'); return; }
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${chart.title.replace(/[^a-z0-9]/gi, '_')}.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                console.log('✅ Chart downloaded as image');
            };

            if (svgEl) {
                // Recharts renders SVG — html2canvas can't handle it.
                // Serialize the SVG and draw it onto a canvas instead.
                const rect = element.getBoundingClientRect();
                const svgRect = svgEl.getBoundingClientRect();
                const scale = 2;

                const svgClone = svgEl.cloneNode(true);
                svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
                svgClone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
                svgClone.setAttribute('width', svgRect.width);
                svgClone.setAttribute('height', svgRect.height);

                const svgStr = new XMLSerializer().serializeToString(svgClone);
                const svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
                const svgUrl = URL.createObjectURL(svgBlob);

                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = rect.width * scale;
                    canvas.height = rect.height * scale;
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#ffffff';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = '#111827';
                    ctx.font = `bold ${14 * scale}px "Segoe UI", system-ui, sans-serif`;
                    ctx.fillText(chart.title, 16 * scale, 28 * scale);
                    const offsetX = (svgRect.left - rect.left) * scale;
                    const offsetY = (svgRect.top - rect.top) * scale;
                    ctx.drawImage(img, offsetX, offsetY, svgRect.width * scale, svgRect.height * scale);
                    URL.revokeObjectURL(svgUrl);
                    setIsCapturing(false);
                    canvas.toBlob(triggerDownload, 'image/png');
                };
                img.onerror = () => {
                    URL.revokeObjectURL(svgUrl);
                    setIsCapturing(false);
                    alert('Failed to generate image');
                };
                img.src = svgUrl;
            } else {
                // KPI card — no SVG, html2canvas works fine
                const canvas = await html2canvas(element, {
                    backgroundColor: '#ffffff', scale: 2, logging: false, useCORS: true, allowTaint: true,
                });
                setIsCapturing(false);
                canvas.toBlob(triggerDownload, 'image/png');
            }
        } catch (error) {
            console.error('❌ Image download error:', error);
            setIsCapturing(false);
            alert('Failed to download chart as image. Please try again.');
        }
    };

    const toggleMaximize = () => setIsMaximized(!isMaximized);

    // Border/ring style: blue if source (filtering others), gray-blue if being filtered
    const cardBorderClass = isSource
        ? 'ring-2 ring-blue-500'
        : isFiltered
            ? 'ring-1 ring-blue-200'
            : '';

    return (
        <>
            {/* Normal card */}
            <div
                ref={chartRef}
                className={`bg-white rounded-lg shadow-md border border-gray-200 h-full flex flex-col transition-all ${cardBorderClass} ${isMaximized ? 'hidden' : ''}`}
            >
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 shrink-0">
                    <div className="flex items-center gap-2 min-w-0">
                        <h3 className="font-semibold text-gray-900 text-sm truncate">{chart.title}</h3>
                        {isSource && (
                            <span className="shrink-0 text-xs bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full font-medium">
                                filtering
                            </span>
                        )}
                        {isFiltered && displayChart.data?.length !== chart.data?.length && (
                            <span className="shrink-0 text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded-full">
                                {displayChart.data?.length} / {chart.data?.length}
                            </span>
                        )}
                    </div>

                    <div className={`flex items-center gap-2 shrink-0 ${isCapturing ? 'opacity-0' : 'opacity-100'}`}>
                        <button onClick={toggleMaximize} className="p-1 hover:bg-gray-100 rounded transition-colors" title="Maximize">
                            <Maximize2 className="w-4 h-4 text-gray-600" />
                        </button>

                        <div className="relative">
                            <button
                                onClick={() => setShowDownloadMenu(!showDownloadMenu)}
                                className="p-1 hover:bg-gray-100 rounded transition-colors"
                                title="Download"
                            >
                                <Download className="w-4 h-4 text-gray-600" />
                            </button>
                            {showDownloadMenu && (
                                <>
                                    <div className="fixed inset-0 z-40" onClick={() => setShowDownloadMenu(false)} />
                                    <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50">
                                        <button onClick={handleDownloadCSV} className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors">
                                            <FileText className="w-4 h-4" /> Download as CSV
                                        </button>
                                        <button onClick={handleDownloadImage} className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors">
                                            <ImageIcon className="w-4 h-4" /> Download as PNG
                                        </button>
                                    </div>
                                </>
                            )}
                        </div>

                        {onRemove && (
                            <button onClick={() => onRemove(chart.chart_id)} className="p-1 hover:bg-red-100 rounded transition-colors" title="Remove">
                                <X className="w-4 h-4 text-red-600" />
                            </button>
                        )}
                    </div>
                </div>

                {/* Chart Content */}
                <div className="flex-1 p-4 overflow-hidden min-h-0">
                    {renderChart()}
                </div>

                {chart.type !== 'kpi' && chart.data && !isCapturing && (
                    <div className="px-4 py-2 border-t border-gray-100 text-xs text-gray-400 shrink-0">
                        {displayChart.data?.length ?? chart.data.length} {chart.data.length === 1 ? 'row' : 'rows'}
                        {activeFilter && !isSource && (
                            <span className="ml-1 text-blue-500">
                                (filtered from {chart.data.length})
                            </span>
                        )}
                    </div>
                )}
            </div>

            {/* Maximized modal */}
            {isMaximized && (
                <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-lg shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col">
                        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
                            <h3 className="font-semibold text-gray-900 text-lg">{chart.title}</h3>
                            <div className="flex items-center gap-2">
                                <div className="relative">
                                    <button onClick={() => setShowDownloadMenu(!showDownloadMenu)} className="p-2 hover:bg-gray-100 rounded transition-colors" title="Download">
                                        <Download className="w-5 h-5 text-gray-600" />
                                    </button>
                                    {showDownloadMenu && (
                                        <>
                                            <div className="fixed inset-0 z-40" onClick={() => setShowDownloadMenu(false)} />
                                            <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50">
                                                <button onClick={handleDownloadCSV} className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors">
                                                    <FileText className="w-4 h-4" /> Download as CSV
                                                </button>
                                                <button onClick={handleDownloadImage} className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors">
                                                    <ImageIcon className="w-4 h-4" /> Download as PNG
                                                </button>
                                            </div>
                                        </>
                                    )}
                                </div>
                                <button onClick={toggleMaximize} className="p-2 hover:bg-gray-100 rounded transition-colors" title="Minimize">
                                    <Minimize2 className="w-5 h-5 text-gray-600" />
                                </button>
                            </div>
                        </div>
                        <div className="flex-1 p-6 overflow-auto">
                            {renderChart(chart)}
                        </div>
                        {chart.type !== 'kpi' && chart.data && (
                            <div className="px-6 py-3 border-t border-gray-100 text-sm text-gray-500">
                                {chart.data.length} {chart.data.length === 1 ? 'row' : 'rows'}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </>
    );
};

export default ChartCard;
