// frontend/src/components/Dashboard/DashboardContainer.jsx

import React, { useState, useEffect } from 'react';
import { RefreshCw, Loader2, AlertCircle } from 'lucide-react';
import ChartCard from './ChartCard';
import FilterPanel from './FilterPanel';

const DashboardContainer = ({ sessionId = 'default' }) => {
    const [dashboard, setDashboard] = useState(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const [removedChartIds, setRemovedChartIds] = useState(new Set());

    useEffect(() => {
        console.log('📊 DashboardContainer mounted with sessionId:', sessionId);
        loadDashboard();
    }, [sessionId]);

    // Listen for refresh events from chat
    useEffect(() => {
        const handleRefresh = () => {
            console.log('🔄 Dashboard refresh triggered by query');
            loadDashboard();
        };

        window.addEventListener('refreshDashboard', handleRefresh);
        return () => window.removeEventListener('refreshDashboard', handleRefresh);
    }, []);

    const loadDashboard = async () => {
        try {
            setLoading(true);
            setError(null);
            const token = localStorage.getItem('token');

            console.log('🔄 Fetching dashboard...');

            // Use /current to get existing dashboard (with query-added charts)
            const response = await fetch(
                `http://localhost:8000/api/dashboard/current?session_id=${sessionId}`,
                {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('✅ Dashboard loaded:', data);

            if (data.dashboard) {
                // Filter out charts that user has removed
                const filteredCharts = data.dashboard.charts.filter(
                    chart => !removedChartIds.has(chart.chart_id)
                );

                setDashboard({
                    ...data.dashboard,
                    charts: filteredCharts
                });

                console.log(`📊 Dashboard has ${filteredCharts.length} charts (${removedChartIds.size} removed by user)`);
            }

        } catch (err) {
            console.error('❌ Dashboard load error:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const refreshDashboard = async () => {
        setRefreshing(true);
        await loadDashboard();
        setRefreshing(false);
    };

    const removeChart = async (chartId) => {
        try {
            console.log('🗑️ Removing chart:', chartId);

            // Add to removed set (persists across refreshes)
            setRemovedChartIds(prev => new Set([...prev, chartId]));

            // Update UI immediately
            setDashboard(prev => ({
                ...prev,
                charts: prev.charts.filter(c => c.chart_id !== chartId)
            }));

            // Also remove from backend
            const token = localStorage.getItem('token');
            await fetch('http://localhost:8000/api/dashboard/remove-chart', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    chart_id: chartId
                })
            });

            console.log('✅ Chart removed');
        } catch (error) {
            console.error('❌ Error removing chart:', error);
        }
    };

    const applyFilter = async (filterKey, filterValue) => {
        try {
            const token = localStorage.getItem('token');

            const response = await fetch('http://localhost:8000/api/dashboard/cross-filter', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    filter_key: filterKey,
                    filter_value: filterValue
                })
            });

            if (!response.ok) throw new Error('Failed to apply filter');

            const data = await response.json();

            // Filter out removed charts
            const filteredCharts = data.dashboard.charts.filter(
                chart => !removedChartIds.has(chart.chart_id)
            );

            setDashboard({
                ...data.dashboard,
                charts: filteredCharts
            });
        } catch (error) {
            console.error('Error applying filter:', error);
        }
    };

    const clearFilters = async () => {
        try {
            const token = localStorage.getItem('token');

            const response = await fetch(
                `http://localhost:8000/api/dashboard/clear-filters?session_id=${sessionId}`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                }
            );

            if (!response.ok) throw new Error('Failed to clear filters');

            const data = await response.json();

            // Filter out removed charts
            const filteredCharts = data.dashboard.charts.filter(
                chart => !removedChartIds.has(chart.chart_id)
            );

            setDashboard({
                ...data.dashboard,
                charts: filteredCharts
            });
        } catch (error) {
            console.error('Error clearing filters:', error);
        }
    };

    // Loading state
    if (loading) {
        return (
            <div className="flex items-center justify-center h-full bg-gray-50">
                <div className="text-center">
                    <Loader2 className="w-12 h-12 text-blue-500 animate-spin mx-auto mb-4" />
                    <p className="text-gray-600">Loading dashboard...</p>
                </div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className="flex items-center justify-center h-full bg-gray-50">
                <div className="text-center max-w-md">
                    <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">Failed to Load Dashboard</h3>
                    <p className="text-sm text-gray-600 mb-4">{error}</p>
                    <button
                        onClick={loadDashboard}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                    >
                        Try Again
                    </button>
                </div>
            </div>
        );
    }

    // Empty dashboard state
    if (!dashboard || !dashboard.charts || dashboard.charts.length === 0) {
        return (
            <div className="flex items-center justify-center h-full bg-gray-50">
                <div className="text-center">
                    <div className="text-6xl mb-4">📊</div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">No Charts Yet</h3>
                    <p className="text-gray-600 mb-4 max-w-sm">
                        Start asking questions in the chat to generate visualizations. Charts will appear here automatically.
                    </p>
                    <button
                        onClick={loadDashboard}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                    >
                        Generate Default Dashboard
                    </button>
                </div>
            </div>
        );
    }

    // Dashboard with charts
    return (
        <div className="h-full flex flex-col p-6 bg-gray-50 overflow-auto">
            {/* Header */}
            <div className="flex items-center justify-between mb-4 shrink-0">
                <div>
                    <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
                    <p className="text-sm text-gray-500">{dashboard.charts.length} charts</p>
                </div>
                <button
                    onClick={refreshDashboard}
                    disabled={refreshing}
                    className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
                >
                    <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </div>

            {/* Filter Panel */}
            {dashboard.filters && Object.keys(dashboard.filters).length > 0 && (
                <FilterPanel
                    filters={dashboard.filters}
                    onRemoveFilter={(key) => {
                        const newFilters = { ...dashboard.filters };
                        delete newFilters[key];
                        if (Object.keys(newFilters).length === 0) {
                            clearFilters();
                        }
                    }}
                    onClearAll={clearFilters}
                />
            )}

            {/* Charts Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 auto-rows-[300px]">
                {dashboard.charts.map((chart) => {
                    console.log('🎨 Rendering chart:', chart.title, chart.type);

                    return (
                        <div
                            key={chart.chart_id}
                            className={`
                ${chart.type === 'kpi' ? 'md:col-span-1' : 'md:col-span-2'}
                ${chart.type === 'table' ? 'row-span-2' : 'row-span-1'}
              `}
                        >
                            <ChartCard
                                chart={chart}
                                onRemove={removeChart}
                                onSliceClick={(filterKey, filterValue) => applyFilter(filterKey, filterValue)}
                            />
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default DashboardContainer;