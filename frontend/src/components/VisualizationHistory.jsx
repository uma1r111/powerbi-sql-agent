// frontend/src/components/VisualizationHistory.jsx

import React, { useState } from 'react';
import { BarChart2, PieChart, LineChart, Table, Hash, Search, Clock, X, Plus } from 'lucide-react';
import ChartCard from './Dashboard/ChartCard';

const VisualizationHistory = ({ charts, onRestoreChart, onClearHistory }) => {
    const [search, setSearch] = useState('');
    const [selectedChart, setSelectedChart] = useState(null);

    const getChartIcon = (type) => {
        switch (type) {
            case 'bar': return <BarChart2 className="w-4 h-4" />;
            case 'pie': return <PieChart className="w-4 h-4" />;
            case 'line':
            case 'area': return <LineChart className="w-4 h-4" />;
            case 'table': return <Table className="w-4 h-4" />;
            case 'kpi': return <Hash className="w-4 h-4" />;
            default: return <BarChart2 className="w-4 h-4" />;
        }
    };

    const filteredCharts = charts
        .filter(chart =>
            chart.title.toLowerCase().includes(search.toLowerCase()) ||
            chart.query?.toLowerCase().includes(search.toLowerCase())
        )
        .reverse(); // Most recent first

    return (
        <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-7xl mx-auto">
                <div className="mb-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-lg font-bold text-gray-900">Visualization History</h3>
                            <p className="text-sm text-gray-400 mt-0.5">
                                All charts generated in this conversation ({charts.length} total)
                            </p>
                        </div>
                        {charts.length > 0 && (
                            <button
                                onClick={onClearHistory}
                                className="text-sm text-red-600 hover:text-red-700 font-medium"
                            >
                                Clear History
                            </button>
                        )}
                    </div>
                </div>

                <div className="relative mb-5">
                    <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                    <input
                        type="text"
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="Search visualizations..."
                        className="w-full pl-9 pr-4 py-2.5 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                    />
                </div>

                {filteredCharts.length === 0 ? (
                    <div className="text-center py-16">
                        <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                            <BarChart2 className="w-7 h-7 text-gray-300" />
                        </div>
                        <p className="text-sm text-gray-500">
                            {search ? 'No visualizations match your search.' : 'No visualizations yet. Ask questions to generate charts!'}
                        </p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {filteredCharts.map((chart, index) => {
                            const timeStr = chart.created_at
                                ? new Date(chart.created_at).toLocaleString()
                                : 'Unknown time';

                            return (
                                <div
                                    key={`${chart.chart_id}-${index}`}
                                    className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow"
                                >
                                    {/* Preview Header */}
                                    <div className="p-3 border-b border-gray-100 bg-gray-50">
                                        <div className="flex items-start justify-between gap-2 mb-2">
                                            <div className="flex items-center gap-2 flex-1 min-w-0">
                                                <div className={`p-1.5 rounded ${chart.type === 'kpi' ? 'bg-purple-100 text-purple-600' :
                                                        chart.type === 'bar' ? 'bg-blue-100 text-blue-600' :
                                                            chart.type === 'pie' ? 'bg-pink-100 text-pink-600' :
                                                                chart.type === 'line' || chart.type === 'area' ? 'bg-green-100 text-green-600' :
                                                                    'bg-gray-100 text-gray-600'
                                                    }`}>
                                                    {getChartIcon(chart.type)}
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <h4 className="text-sm font-semibold text-gray-900 truncate">
                                                        {chart.title}
                                                    </h4>
                                                    <p className="text-xs text-gray-500">{chart.type.toUpperCase()}</p>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => onRestoreChart(chart)}
                                                className="shrink-0 p-1 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                                                title="Add to Dashboard"
                                            >
                                                <Plus className="w-3 h-3" />
                                            </button>
                                        </div>

                                        {chart.query && (
                                            <p className="text-xs text-gray-600 italic truncate">
                                                "{chart.query}"
                                            </p>
                                        )}

                                        <div className="flex items-center gap-1 mt-1 text-xs text-gray-400">
                                            <Clock className="w-3 h-3" />
                                            <span>{timeStr}</span>
                                        </div>
                                    </div>

                                    {/* Chart Preview */}
                                    <div
                                        className="h-48 p-3 cursor-pointer hover:bg-gray-50 transition-colors"
                                        onClick={() => setSelectedChart(chart)}
                                    >
                                        <ChartCard chart={chart} />
                                    </div>

                                    {/* Data Info */}
                                    {chart.data && chart.data.length > 0 && (
                                        <div className="px-3 py-2 bg-gray-50 border-t border-gray-100 text-xs text-gray-500">
                                            {chart.data.length} rows • {Object.keys(chart.data[0]).length} columns
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}

                {/* Fullscreen Modal */}
                {selectedChart && (
                    <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center p-4">
                        <div className="bg-white rounded-lg shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col">
                            {/* Modal Header */}
                            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
                                <div>
                                    <h3 className="font-semibold text-gray-900 text-lg">{selectedChart.title}</h3>
                                    {selectedChart.query && (
                                        <p className="text-sm text-gray-500 italic mt-1">"{selectedChart.query}"</p>
                                    )}
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={() => {
                                            onRestoreChart(selectedChart);
                                            setSelectedChart(null);
                                        }}
                                        className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                                    >
                                        <Plus className="w-4 h-4" />
                                        Add to Dashboard
                                    </button>
                                    <button
                                        onClick={() => setSelectedChart(null)}
                                        className="p-2 hover:bg-gray-100 rounded transition-colors"
                                    >
                                        <X className="w-5 h-5 text-gray-600" />
                                    </button>
                                </div>
                            </div>

                            {/* Modal Content */}
                            <div className="flex-1 p-6 overflow-auto">
                                <ChartCard chart={selectedChart} />
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default VisualizationHistory;