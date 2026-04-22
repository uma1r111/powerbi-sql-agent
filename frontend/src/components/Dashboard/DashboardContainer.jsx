// frontend/src/components/Dashboard/DashboardContainer.jsx

import React, { useState, useEffect, useRef } from 'react';
import { RefreshCw, Loader2, AlertCircle, X } from 'lucide-react';
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
    // Cross-filter state: { sourceChartId, key, value } | null
    const [activeFilter, setActiveFilter] = useState(null);
    // Container width for GridLayout
    const [gridWidth, setGridWidth] = useState(800);
    const observerRef = useRef(null);
    // Use a ref so the event listener always calls the latest refreshFromCurrent
    // without needing to re-register on every render
    const refreshFromCurrentRef = useRef(null);

    useEffect(() => {
        loadDashboard();
    }, [sessionId]);

    // Callback ref — fires when the grid div actually mounts (after loading finishes)
    const gridContainerRef = React.useCallback((node) => {
        if (observerRef.current) {
            observerRef.current.disconnect();
            observerRef.current = null;
        }
        if (node) {
            setGridWidth(node.getBoundingClientRect().width || 800);
            const observer = new ResizeObserver(entries => {
                for (const entry of entries) {
                    if (entry.contentRect.width > 0) {
                        setGridWidth(entry.contentRect.width);
                    }
                }
            });
            observer.observe(node);
            observerRef.current = observer;
        }
    }, []);

    useEffect(() => {
        const handleRefresh = () => refreshFromCurrentRef.current?.();
        window.addEventListener('refreshDashboard', handleRefresh);
        return () => window.removeEventListener('refreshDashboard', handleRefresh);
    }, []);

    const loadDashboard = async () => {
        try {
            setLoading(true);
            setError(null);
            setActiveFilter(null);
            const token = sessionStorage.getItem('token');

            // ── Step 1: Try loading existing dashboard from Redis first ──
            const currentResponse = await fetch(
                `http://localhost:8000/api/dashboard/current?session_id=${sessionId}`,
                { headers: { 'Authorization': `Bearer ${token}` } }
            );

            if (currentResponse.ok) {
                const currentData = await currentResponse.json();

                if (currentData?.dashboard?.charts?.length > 0) {
                    // Redis has data — use it directly, no regeneration needed
                    const filteredCharts = currentData.dashboard.charts.filter(
                        chart => !removedChartIds.has(chart.chart_id)
                    );
                    setDashboard({ ...currentData.dashboard, charts: filteredCharts });
                    if (onChartsLoaded && filteredCharts.length > 0) onChartsLoaded(filteredCharts);
                    console.log(`✅ Dashboard loaded from Redis cache (${filteredCharts.length} charts)`);
                    setLoading(false);
                    return;
                }
            }

            // ── Step 2: Nothing in Redis — generate fresh dashboard ──
            console.log('📊 No cached dashboard found — generating fresh...');
            const initialResponse = await fetch(
                `http://localhost:8000/api/dashboard/initial?session_id=${sessionId}`,
                { headers: { 'Authorization': `Bearer ${token}` } }
            );

            if (!initialResponse.ok) throw new Error(`HTTP ${initialResponse.status}: ${initialResponse.statusText}`);

            const data = await initialResponse.json();
            if (data.dashboard) {
                const filteredCharts = data.dashboard.charts.filter(
                    chart => !removedChartIds.has(chart.chart_id)
                );
                setDashboard({ ...data.dashboard, charts: filteredCharts });
                if (onChartsLoaded && filteredCharts.length > 0) onChartsLoaded(filteredCharts);
                console.log(`✅ Fresh dashboard generated (${filteredCharts.length} charts)`);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Pull the cached dashboard without regenerating it (used after adding/restoring charts)
    refreshFromCurrentRef.current = async () => {
        try {
            const token = sessionStorage.getItem('token');
            const response = await fetch(
                `http://localhost:8000/api/dashboard/current?session_id=${sessionId}`,
                { headers: { 'Authorization': `Bearer ${token}` } }
            );
            if (!response.ok) return;
            const data = await response.json();
            if (data.dashboard?.charts?.length) {
                const filteredCharts = data.dashboard.charts.filter(
                    chart => !removedChartIds.has(chart.chart_id)
                );
                setDashboard({ ...data.dashboard, charts: filteredCharts });
                if (onChartsLoaded && filteredCharts.length > 0) onChartsLoaded(filteredCharts);
            }
        } catch (err) {
            console.error('Failed to refresh dashboard from current:', err);
        }
    };

    const refreshDashboard = async () => {
        setRefreshing(true);
        await loadDashboard();
        setRefreshing(false);
    };

    const removeChart = async (chartId) => {
        try {
            setRemovedChartIds(prev => new Set([...prev, chartId]));
            setDashboard(prev => ({
                ...prev,
                charts: prev.charts.filter(c => c.chart_id !== chartId)
            }));
            if (activeFilter?.sourceChartId === chartId) setActiveFilter(null);
            const token = sessionStorage.getItem('token');
            await fetch('http://localhost:8000/api/dashboard/remove-chart', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, chart_id: chartId })
            });
        } catch (error) {
            console.error('❌ Error removing chart:', error);
        }
    };

    const handleChartFilter = (sourceChartId, key, value) => {
        setActiveFilter(prev =>
            prev?.sourceChartId === sourceChartId &&
            prev?.key === key &&
            String(prev?.value) === String(value)
                ? null
                : { sourceChartId, key, value }
        );
    };

    const clearFilters = async () => {
        setActiveFilter(null);
        try {
            const token = sessionStorage.getItem('token');
            const response = await fetch(
                `http://localhost:8000/api/dashboard/clear-filters?session_id=${sessionId}`,
                { method: 'POST', headers: { 'Authorization': `Bearer ${token}` } }
            );
            if (!response.ok) throw new Error('Failed to clear filters');
            const data = await response.json();
            const filteredCharts = data.dashboard.charts.filter(
                chart => !removedChartIds.has(chart.chart_id)
            );
            setDashboard({ ...data.dashboard, charts: filteredCharts });
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
                    <button onClick={loadDashboard} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
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
                        Start asking questions in the chat to generate visualizations.
                    </p>
                    <button onClick={loadDashboard} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                        Generate Default Dashboard
                    </button>
                </div>
            </div>
        );
    }

    const layout = dashboard.charts.map((chart, index) => ({
        i: chart.chart_id,
        x: chart.position?.x ?? (chart.type === 'kpi' ? (index % 4) * 3 : (index % 2) * 6),
        y: chart.position?.y ?? Math.floor(index / (chart.type === 'kpi' ? 4 : 2)) * 6,
        w: chart.position?.w ?? (chart.type === 'kpi' ? 3 : 6),
        h: chart.position?.h ?? (chart.type === 'kpi' ? 4 : 7),
        minW: chart.type === 'kpi' ? 2 : 4,
        minH: chart.type === 'kpi' ? 3 : 5,
    }));

    const activeFilterSource = activeFilter
        ? dashboard.charts.find(c => c.chart_id === activeFilter.sourceChartId)
        : null;

    return (
        <div className="h-full flex flex-col p-6 bg-gray-50">
            {/* Header */}
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

            {/* Active cross-filter banner */}
            {activeFilter && (
                <div className="flex items-center gap-3 mb-3 px-4 py-2.5 bg-blue-50 border border-blue-200 rounded-lg shrink-0">
                    <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                    <span className="text-sm text-blue-800 font-medium">
                        Filtering by <span className="font-bold">{String(activeFilter.value)}</span>
                        {activeFilterSource && (
                            <span className="font-normal text-blue-600"> from {activeFilterSource.title}</span>
                        )}
                    </span>
                    <span className="text-xs text-blue-500 ml-1">— click the same element again to clear</span>
                    <button
                        onClick={() => setActiveFilter(null)}
                        className="ml-auto p-1 hover:bg-blue-100 rounded-full transition-colors"
                        title="Clear filter"
                    >
                        <X className="w-4 h-4 text-blue-600" />
                    </button>
                </div>
            )}

            {dashboard.filters && Object.keys(dashboard.filters).length > 0 && (
                <FilterPanel
                    filters={dashboard.filters}
                    onRemoveFilter={(key) => {
                        const newFilters = { ...dashboard.filters };
                        delete newFilters[key];
                        if (Object.keys(newFilters).length === 0) clearFilters();
                    }}
                    onClearAll={clearFilters}
                />
            )}

            <div ref={gridContainerRef} className="flex-1 overflow-auto">
                <GridLayout
                    className="layout"
                    layout={layout}
                    cols={12}
                    rowHeight={60}
                    width={gridWidth}
                    isDraggable={true}
                    isResizable={true}
                    compactType="vertical"
                    preventCollision={false}
                    margin={[12, 12]}
                >
                    {dashboard.charts.map((chart) => (
                        <div key={chart.chart_id} className="dashboard-grid-item">
                            <ChartCard
                                chart={chart}
                                activeFilter={activeFilter}
                                onFilterSelect={handleChartFilter}
                                onRemove={removeChart}
                            />
                        </div>
                    ))}
                </GridLayout>
            </div>
        </div>
    );
};

export default DashboardContainer;