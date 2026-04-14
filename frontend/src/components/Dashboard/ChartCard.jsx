// frontend/src/components/Dashboard/ChartCard.jsx

import React, { useState, useRef, useMemo } from 'react';
import { X, Maximize2, Minimize2, Download, FileText, Image as ImageIcon } from 'lucide-react';
import { toPng } from 'html-to-image';
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

    // Does this chart's data contain the filter key column?
    const hasMatchingKey = useMemo(() => {
        if (!activeFilter || !chart.data?.length) return false;
        return chart.data.some(row =>
            Object.keys(row).some(k => k.toLowerCase() === activeFilter.key.toLowerCase())
        );
    }, [activeFilter, chart.data]);

    // Power BI style: pass selectedKey/selectedValue to ANY chart that shares the filter
    // column — that chart's component handles dimming non-selected elements.
    // Charts that don't share the column are unaffected (null passed → no visual change).
    const selectedKey = (activeFilter && hasMatchingKey) ? activeFilter.key : null;
    const selectedValue = (activeFilter && hasMatchingKey) ? String(activeFilter.value) : null;

    // displayChart is always the full dataset — dimming is done visually inside each
    // chart component, not by removing rows.
    const displayChart = chart;

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
                return <DataTable chart={chartData} selectedKey={selectedKey} selectedValue={selectedValue} onSelect={onSelect} />;
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

            const dataUrl = await toPng(chartRef.current, {
                backgroundColor: '#ffffff',
                pixelRatio: 2,
            });

            setIsCapturing(false);
            const a = document.createElement('a');
            a.href = dataUrl;
            a.download = `${chart.title.replace(/[^a-z0-9]/gi, '_')}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } catch (error) {
            console.error('❌ Image download error:', error);
            setIsCapturing(false);
            alert('Failed to download chart as image. Please try again.');
        }
    };

    const toggleMaximize = () => setIsMaximized(!isMaximized);

    // Border: blue ring on source chart; light ring on charts that share the filter column
    const cardBorderClass = isSource
        ? 'ring-2 ring-blue-500'
        : (isFiltered && hasMatchingKey)
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
                        {isFiltered && hasMatchingKey && (
                            <span className="shrink-0 text-xs bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded-full">
                                filtered
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
                        {chart.data.length} {chart.data.length === 1 ? 'row' : 'rows'}
                        {activeFilter && hasMatchingKey && !isSource && (
                            <span className="ml-1 text-blue-500">• filtered</span>
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
