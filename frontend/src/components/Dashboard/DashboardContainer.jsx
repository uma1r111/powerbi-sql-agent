// frontend/src/components/Dashboard/DashboardContainer.jsx
// Compatible with react-grid-layout v2.2.2

import React, { useState, useEffect } from 'react';
import { RefreshCw, Loader2, AlertCircle } from 'lucide-react';
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import ChartCard from './ChartCard';
import FilterPanel from './FilterPanel';

const DashboardContainer = ({ sessionId = 'default', onChartsLoaded }) => {
    const [dashboard, setDashboard] = useState(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const [removedChartIds, setRemovedChartIds] = useState(new Set());

    useEffect(() => {
        console.log('📊 DashboardContainer mounted with sessionId:', sessionId);
        loadDashboard();
    }, [sessionId]);

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
                const filteredCharts = data.dashboard.charts.filter(
                    chart => !removedChartIds.has(chart.chart_id)
                );

                setDashboard({
                    ...data.dashboard,
                    charts: filteredCharts
                });

                if (onChartsLoaded && filteredCharts.length > 0) {
                    onChartsLoaded(filteredCharts);
                }

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

            setRemovedChartIds(prev => new Set([...prev, chartId]));
            setDashboard(prev => ({
                ...prev,
                charts: prev.charts.filter(c => c.chart_id !== chartId)
            }));

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

    // Prepare layout for react-grid-layout v2
    const layout = dashboard.charts.map((chart, index) => ({
        i: chart.chart_id,
        x: chart.position?.x || (index % 3) * 4,
        y: chart.position?.y || Math.floor(index / 3) * 6,
        w: chart.position?.w || (chart.type === 'kpi' ? 3 : 6),
        h: chart.position?.h || (chart.type === 'kpi' ? 4 : 8),
        minW: chart.type === 'kpi' ? 2 : 4,
        minH: chart.type === 'kpi' ? 3 : 6,
    }));

    return (
        <div className="h-full flex flex-col p-6 bg-gray-50">
            <div className="flex items-center justify-between mb-4 shrink-0">
                <div>
                    <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
                    <p className="text-sm text-gray-500">{dashboard.charts.length} charts • Drag to move, resize from corners</p>
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

            <div className="flex-1 overflow-auto">
                <GridLayout
                    className="layout"
                    layout={layout}
                    cols={12}
                    rowHeight={50}
                    width={1200}
                    isDraggable={true}
                    isResizable={true}
                    compactType="vertical"
                    preventCollision={false}
                >
                    {dashboard.charts.map((chart) => (
                        <div key={chart.chart_id} className="dashboard-grid-item">
                            <ChartCard
                                chart={chart}
                                onRemove={removeChart}
                                onSliceClick={(filterKey, filterValue) => applyFilter(filterKey, filterValue)}
                            />
                        </div>
                    ))}
                </GridLayout>
            </div>
        </div>
    );
};

export default DashboardContainer;