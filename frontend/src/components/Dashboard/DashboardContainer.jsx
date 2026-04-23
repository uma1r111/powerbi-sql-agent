// frontend/src/components/Dashboard/DashboardContainer.jsx

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { RefreshCw, Loader2, AlertCircle, X, Plus, Wand2 } from 'lucide-react';
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import ChartCard from './ChartCard';
import FilterPanel from './FilterPanel';
import ManualChartBuilder from './ManualChartBuilder';
import { useTheme } from '../../contexts/ThemeContext';

const API = 'http://localhost:8000/api';

const DashboardContainer = ({ sessionId = 'default', onChartsLoaded, onChartBuilderAdded }) => {
    const { theme: t } = useTheme();

    const [dashboard, setDashboard] = useState(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);
    const [removedChartIds, setRemovedChartIds] = useState(new Set());
    const [activeFilter, setActiveFilter] = useState(null);
    const [gridWidth, setGridWidth] = useState(800);
    const [containerHeight, setContainerHeight] = useState(600);
    const [showBuilder, setShowBuilder] = useState(false);

    const observerRef = useRef(null);
    const refreshFromCurrentRef = useRef(null);

    useEffect(() => { loadDashboard(); }, [sessionId]);

    const gridContainerRef = React.useCallback((node) => {
        observerRef.current?.disconnect();
        observerRef.current = null;
        if (node) {
            const rect = node.getBoundingClientRect();
            setGridWidth(rect.width || 800);
            setContainerHeight(rect.height || 600);
            const ro = new ResizeObserver(entries => {
                for (const e of entries) {
                    if (e.contentRect.width > 0) setGridWidth(e.contentRect.width);
                    if (e.contentRect.height > 0) setContainerHeight(e.contentRect.height);
                }
            });
            ro.observe(node);
            observerRef.current = ro;
        }
    }, []);

    useEffect(() => {
        const h = () => refreshFromCurrentRef.current?.();
        window.addEventListener('refreshDashboard', h);
        return () => window.removeEventListener('refreshDashboard', h);
    }, []);

    const loadDashboard = async () => {
        setLoading(true);
        setError(null);
        setActiveFilter(null);
        try {
            const token = sessionStorage.getItem('token');

            const cached = await fetch(`${API}/dashboard/current?session_id=${sessionId}`,
                { headers: { Authorization: `Bearer ${token}` } });

            if (cached.ok) {
                const data = await cached.json();
                if (data?.dashboard?.charts?.length > 0) {
                    const charts = data.dashboard.charts.filter(c => !removedChartIds.has(c.chart_id));
                    setDashboard({ ...data.dashboard, charts });
                    onChartsLoaded?.(charts);
                    setLoading(false);
                    return;
                }
            }

            const fresh = await fetch(`${API}/dashboard/initial?session_id=${sessionId}`,
                { headers: { Authorization: `Bearer ${token}` } });
            if (!fresh.ok) throw new Error(`HTTP ${fresh.status}`);
            const data = await fresh.json();
            if (data.dashboard) {
                const charts = data.dashboard.charts.filter(c => !removedChartIds.has(c.chart_id));
                setDashboard({ ...data.dashboard, charts });
                onChartsLoaded?.(charts);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    refreshFromCurrentRef.current = async () => {
        try {
            const token = sessionStorage.getItem('token');
            const res = await fetch(`${API}/dashboard/current?session_id=${sessionId}`,
                { headers: { Authorization: `Bearer ${token}` } });
            if (!res.ok) return;
            const data = await res.json();
            if (data.dashboard?.charts?.length) {
                const charts = data.dashboard.charts.filter(c => !removedChartIds.has(c.chart_id));
                setDashboard({ ...data.dashboard, charts });
                onChartsLoaded?.(charts);
            }
        } catch { }
    };

    const refreshDashboard = async () => {
        setRefreshing(true);
        await loadDashboard();
        setRefreshing(false);
    };

    const removeChart = async (chartId) => {
        setRemovedChartIds(prev => new Set([...prev, chartId]));
        setDashboard(prev => ({ ...prev, charts: prev.charts.filter(c => c.chart_id !== chartId) }));
        if (activeFilter?.sourceChartId === chartId) setActiveFilter(null);
        try {
            const token = sessionStorage.getItem('token');
            await fetch(`${API}/dashboard/remove-chart`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, chart_id: chartId }),
            });
        } catch { }
    };

    const handleChartFilter = (sourceChartId, key, value) => {
        setActiveFilter(prev =>
            prev?.sourceChartId === sourceChartId && prev?.key === key && String(prev?.value) === String(value)
                ? null : { sourceChartId, key, value }
        );
    };

    const clearFilters = async () => {
        setActiveFilter(null);
        try {
            const token = sessionStorage.getItem('token');
            const res = await fetch(`${API}/dashboard/clear-filters?session_id=${sessionId}`,
                { method: 'POST', headers: { Authorization: `Bearer ${token}` } });
            if (res.ok) {
                const data = await res.json();
                const charts = data.dashboard.charts.filter(c => !removedChartIds.has(c.chart_id));
                setDashboard({ ...data.dashboard, charts });
            }
        } catch { }
    };

    const handleLayoutChange = async (layout) => {
        try {
            const token = sessionStorage.getItem('token');
            for (const item of layout) {
                await fetch(`${API}/dashboard/update-position`, {
                    method: 'POST',
                    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        chart_id: item.i,
                        position: { x: item.x, y: item.y, w: item.w, h: item.h },
                    }),
                });
            }
        } catch { }
    };

    const handleChartUpdate = async (chartId, updates) => {
        try {
            const token = sessionStorage.getItem('token');
            const res = await fetch(`${API}/dashboard/update-chart`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, chart_id: chartId, updates }),
            });
            if (res.ok) {
                setDashboard(prev => ({
                    ...prev,
                    charts: prev.charts.map(c =>
                        c.chart_id === chartId
                            ? { ...c, title: updates.title ?? c.title, type: updates.type ?? c.type, config: { ...c.config, ...(updates.config || {}) } }
                            : c
                    ),
                }));
            }
        } catch { }
    };

    // ── Layout calculation ──────────────────────────────────────────────────────
    // Build grid layout with a 2-column arrangement for regular charts
    const layout = useMemo(() => {
        if (!dashboard?.charts) return [];
        let regularCount = 0;
        return dashboard.charts.map((chart, idx) => {
            if (chart.position?.x !== undefined && chart.position?.y !== undefined) {
                return {
                    i: chart.chart_id,
                    x: chart.position.x, y: chart.position.y,
                    w: chart.position.w ?? (chart.type === 'kpi' ? 3 : 6),
                    h: chart.position.h ?? (chart.type === 'kpi' ? 2 : 4),
                    minW: chart.type === 'kpi' ? 2 : 3,
                    minH: chart.type === 'kpi' ? 2 : 3,
                };
            }
            // No saved position — auto-arrange in 2-column grid
            if (chart.type === 'kpi') {
                return {
                    i: chart.chart_id,
                    x: (idx % 4) * 3, y: 0,
                    w: 3, h: 2, minW: 2, minH: 2,
                };
            }
            // Regular charts: alternate left (x=0) and right (x=6) columns
            const col = regularCount % 2;          // 0 = left, 1 = right
            const row = Math.floor(regularCount / 2);
            regularCount++;
            return {
                i: chart.chart_id,
                x: col * 6, y: row * 4,
                w: 6, h: 4, minW: 3, minH: 3,
            };
        });
    }, [dashboard?.charts]);

    // rowHeight targets MIN_VISIBLE_ROWS so each chart stays readable.
    // When there are more charts the grid scrolls rather than squishing everything.
    const rowHeight = useMemo(() => {
        if (!containerHeight) return 62;
        const MIN_VISIBLE_ROWS = 8;
        const margins = MIN_VISIBLE_ROWS * 10;
        const available = Math.max(0, containerHeight - 16 - margins);
        return Math.max(52, Math.floor(available / MIN_VISIBLE_ROWS));
    }, [containerHeight]);

    // ── Render states ───────────────────────────────────────────────────────────

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full" style={{ background: t.bg }}>
                <div className="text-center">
                    <div className="relative mx-auto w-16 h-16 mb-5">
                        <div className="w-16 h-16 rounded-2xl flex items-center justify-center"
                            style={{ background: t.accentLight }}>
                            <Loader2 className="w-8 h-8 animate-spin" style={{ color: t.accent }} />
                        </div>
                    </div>
                    <p className="font-semibold text-base" style={{ color: t.text }}>Loading Dashboard</p>
                    <p className="text-sm mt-1" style={{ color: t.textMuted }}>Fetching your insights…</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center justify-center h-full" style={{ background: t.bg }}>
                <div className="text-center max-w-sm">
                    <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
                        style={{ background: '#fef2f2' }}>
                        <AlertCircle className="w-7 h-7" style={{ color: '#dc2626' }} />
                    </div>
                    <p className="font-bold text-base mb-1" style={{ color: t.text }}>Failed to Load</p>
                    <p className="text-sm mb-4" style={{ color: t.textMuted }}>{error}</p>
                    <button onClick={loadDashboard}
                        className="px-5 py-2 rounded-xl text-sm font-semibold"
                        style={{ background: t.accent, color: '#fff' }}>
                        Try Again
                    </button>
                </div>
            </div>
        );
    }

    if (!dashboard?.charts?.length) {
        return (
            <div className="flex items-center justify-center h-full" style={{ background: t.bg }}>
                <div className="text-center max-w-sm">
                    <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5"
                        style={{ background: t.accentLight }}>
                        <Wand2 className="w-8 h-8" style={{ color: t.accent }} />
                    </div>
                    <p className="font-bold text-xl mb-2" style={{ color: t.text }}>No Charts Yet</p>
                    <p className="text-sm mb-5" style={{ color: t.textMuted }}>
                        Ask the AI assistant a question or build a chart manually to populate your dashboard.
                    </p>
                    <div className="flex items-center justify-center gap-3">
                        <button onClick={loadDashboard}
                            className="px-4 py-2 rounded-xl text-sm font-semibold border transition-all"
                            style={{ borderColor: t.border, color: t.textSub, background: t.surface }}>
                            Generate Default
                        </button>
                        <button onClick={() => setShowBuilder(true)}
                            className="px-4 py-2 rounded-xl text-sm font-semibold flex items-center gap-2"
                            style={{ background: t.accent, color: '#fff' }}>
                            <Plus className="w-4 h-4" /> Build Chart
                        </button>
                    </div>
                </div>

                {/* Chart Builder Panel */}
                {showBuilder && (
                    <div className="fixed inset-y-0 right-0 w-80 flex flex-col shadow-2xl z-50"
                        style={{ background: t.surface, borderLeft: `1px solid ${t.border}` }}>
                        <ManualChartBuilder
                            sessionId={sessionId}
                            onAddChart={(payload) => onChartBuilderAdded?.(payload)}
                            onClose={() => setShowBuilder(false)}
                            theme={t}
                        />
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="h-full flex" style={{ background: t.bg }}>
            {/* Main Dashboard Area */}
            <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
                {/* Toolbar */}
                <div className="flex items-center justify-between px-5 py-2.5 border-b shrink-0"
                    style={{ background: t.header, borderColor: t.border }}>
                    <div>
                        <h2 className="text-sm font-bold" style={{ color: t.text }}>Dashboard</h2>
                        <p className="text-xs" style={{ color: t.textMuted }}>
                            {dashboard.charts.length} chart{dashboard.charts.length !== 1 ? 's' : ''} · Drag to rearrange
                        </p>
                    </div>

                    <div className="flex items-center gap-2">
                        {activeFilter && (
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold"
                                style={{ background: `${t.accent}15`, color: t.accent, border: `1px solid ${t.accent}30` }}>
                                <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: t.accent }} />
                                Filtering: {String(activeFilter.value)}
                                <button onClick={() => setActiveFilter(null)} className="ml-1 hover:opacity-70">
                                    <X className="w-3 h-3" />
                                </button>
                            </div>
                        )}

                        <button
                            onClick={() => setShowBuilder(!showBuilder)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all"
                            style={{
                                background: showBuilder ? t.accent : t.accentLight,
                                color: showBuilder ? '#fff' : t.accentText || t.accent,
                                border: `1px solid ${showBuilder ? t.accent : t.border}`,
                            }}>
                            <Plus className="w-3.5 h-3.5" />
                            Add Chart
                        </button>

                        <button
                            onClick={refreshDashboard}
                            disabled={refreshing}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all disabled:opacity-50"
                            style={{ background: t.surface, color: t.textSub, borderColor: t.border }}>
                            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
                            Refresh
                        </button>
                    </div>
                </div>

                {/* Filter panel */}
                {dashboard.filters && Object.keys(dashboard.filters).length > 0 && (
                    <FilterPanel
                        filters={dashboard.filters}
                        onRemoveFilter={(key) => {
                            const f = { ...dashboard.filters };
                            delete f[key];
                            if (!Object.keys(f).length) clearFilters();
                        }}
                        onClearAll={clearFilters}
                    />
                )}

                {/* Grid — fills remaining height, scrolls when charts overflow */}
                <div ref={gridContainerRef} style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '8px' }}>
                    <GridLayout
                        className="layout"
                        layout={layout}
                        cols={12}
                        rowHeight={rowHeight}
                        width={gridWidth}
                        isDraggable
                        isResizable
                        compactType="vertical"
                        preventCollision={false}
                        margin={[10, 10]}
                        onLayoutChange={handleLayoutChange}
                    >
                        {dashboard.charts.map((chart) => (
                            <div key={chart.chart_id}>
                                <ChartCard
                                    chart={chart}
                                    activeFilter={activeFilter}
                                    onFilterSelect={handleChartFilter}
                                    onRemove={removeChart}
                                    onChartUpdate={handleChartUpdate}
                                    theme={t}
                                />
                            </div>
                        ))}
                    </GridLayout>
                </div>
            </div>

            {/* Chart Builder Side Panel */}
            {showBuilder && (
                <div
                    className="shrink-0 border-l overflow-hidden flex flex-col"
                    style={{
                        width: '340px',
                        background: t.surface,
                        borderColor: t.border,
                    }}>
                    <ManualChartBuilder
                        sessionId={sessionId}
                        onAddChart={(payload) => onChartBuilderAdded?.(payload)}
                        onClose={() => setShowBuilder(false)}
                        theme={t}
                    />
                </div>
            )}
        </div>
    );
};

export default DashboardContainer;
