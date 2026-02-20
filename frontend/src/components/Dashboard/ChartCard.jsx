// frontend/src/components/Dashboard/ChartCard.jsx

import React, { useState } from 'react';
import { X, Maximize2, Minimize2, Download } from 'lucide-react';
import KPICard from './ChartTypes/KPICard';
import BarChartComponent from './ChartTypes/BarChartComponent';
import LineChartComponent from './ChartTypes/LineChartComponent';
import PieChartComponent from './ChartTypes/PieChartComponent';
import DataTable from './ChartTypes/DataTable';

const ChartCard = ({ chart, onRemove, onSliceClick }) => {
    const [isMaximized, setIsMaximized] = useState(false);

    const renderChart = () => {
        switch (chart.type) {
            case 'kpi':
                return <KPICard chart={chart} />;
            case 'bar':
                return <BarChartComponent chart={chart} />;
            case 'line':
            case 'area':
                return <LineChartComponent chart={chart} />;
            case 'pie':
                return <PieChartComponent chart={chart} onSliceClick={onSliceClick} />;
            case 'table':
                return <DataTable chart={chart} />;
            default:
                return (
                    <div className="h-full flex items-center justify-center text-gray-500">
                        Unsupported chart type: {chart.type}
                    </div>
                );
        }
    };

    const handleDownload = () => {
        try {
            // Convert chart data to CSV
            const data = chart.data;
            if (!data || data.length === 0) return;

            const headers = Object.keys(data[0]);
            const csvContent = [
                headers.join(','),
                ...data.map(row => headers.map(h => row[h]).join(','))
            ].join('\n');

            // Create download link
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${chart.title.replace(/[^a-z0-9]/gi, '_')}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            console.log('✅ Chart downloaded as CSV');
        } catch (error) {
            console.error('❌ Download error:', error);
            alert('Failed to download chart data');
        }
    };

    const toggleMaximize = () => {
        setIsMaximized(!isMaximized);
    };

    return (
        <>
            {/* Normal chart */}
            <div className={`bg-white rounded-lg shadow-md border border-gray-200 h-full flex flex-col ${isMaximized ? 'hidden' : ''}`}>
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
                    <h3 className="font-semibold text-gray-900 text-sm">{chart.title}</h3>
                    <div className="flex items-center gap-2">
                        {/* Maximize button */}
                        <button
                            onClick={toggleMaximize}
                            className="p-1 hover:bg-gray-100 rounded transition-colors"
                            title="Maximize"
                        >
                            <Maximize2 className="w-4 h-4 text-gray-600" />
                        </button>
                        {/* Download button */}
                        <button
                            onClick={handleDownload}
                            className="p-1 hover:bg-gray-100 rounded transition-colors"
                            title="Download as CSV"
                        >
                            <Download className="w-4 h-4 text-gray-600" />
                        </button>
                        {/* Remove button */}
                        {onRemove && (
                            <button
                                onClick={() => onRemove(chart.chart_id)}
                                className="p-1 hover:bg-red-100 rounded transition-colors"
                                title="Remove"
                            >
                                <X className="w-4 h-4 text-red-600" />
                            </button>
                        )}
                    </div>
                </div>

                {/* Chart Content */}
                <div className="flex-1 p-4 overflow-hidden">
                    {renderChart()}
                </div>

                {/* Footer */}
                {chart.type !== 'kpi' && chart.data && (
                    <div className="px-4 py-2 border-t border-gray-100 text-xs text-gray-500">
                        {chart.data.length} {chart.data.length === 1 ? 'row' : 'rows'}
                    </div>
                )}
            </div>

            {/* Maximized modal */}
            {isMaximized && (
                <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-lg shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col">
                        {/* Modal Header */}
                        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
                            <h3 className="font-semibold text-gray-900 text-lg">{chart.title}</h3>
                            <div className="flex items-center gap-2">
                                {/* Download button */}
                                <button
                                    onClick={handleDownload}
                                    className="p-2 hover:bg-gray-100 rounded transition-colors"
                                    title="Download as CSV"
                                >
                                    <Download className="w-5 h-5 text-gray-600" />
                                </button>
                                {/* Minimize button */}
                                <button
                                    onClick={toggleMaximize}
                                    className="p-2 hover:bg-gray-100 rounded transition-colors"
                                    title="Minimize"
                                >
                                    <Minimize2 className="w-5 h-5 text-gray-600" />
                                </button>
                            </div>
                        </div>

                        {/* Modal Content */}
                        <div className="flex-1 p-6 overflow-auto">
                            {renderChart()}
                        </div>

                        {/* Modal Footer */}
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