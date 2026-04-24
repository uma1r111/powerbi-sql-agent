// frontend/src/components/Dashboard/DashboardContainer.jsx

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { RefreshCw, Loader2, AlertCircle, X, Plus, Wand2, Download, FileText, Image as ImageIcon } from 'lucide-react';
import { toPng } from 'html-to-image';
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
    const [gridWidth, setGridWidth] = useState(0);        // 0 = not yet measured
    const [containerHeight, setContainerHeight] = useState(600);
    const [showBuilder, setShowBuilder] = useState(false);
    const [exporting, setExporting] = useState(false);
    const [showExportMenu, setShowExportMenu] = useState(false);
    // Prevent the RGL "slide-in" animation on first render: items animate from
    // their default positions to their grid positions while the container measures
    // itself.  We suppress transitions until the grid width has been set at least once.
    const [gridReady, setGridReady] = useState(false);

    const observerRef = useRef(null);
    const refreshFromCurrentRef = useRef(null);
    const gridRef = useRef(null);

    useEffect(() => { loadDashboard(); }, [sessionId]);

    const gridContainerRef = React.useCallback((node) => {
        observerRef.current?.disconnect();
        observerRef.current = null;
        if (node) {
            const rect = node.getBoundingClientRect();
            if (rect.width > 0) {
                setGridWidth(rect.width);
                setContainerHeight(rect.height || 600);
                setGridReady(true);
            }
            const ro = new ResizeObserver(entries => {
                for (const e of entries) {
                    if (e.contentRect.width > 0) {
                        setGridWidth(e.contentRect.width);
                        setGridReady(true);
                    }
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
                const cachedCharts = data?.dashboard?.charts || [];
                // Only use cache when the dashboard is "high quality":
                // - at least 4 charts, AND
                // - contains at least one non-KPI visualisation
                // A dashboard with only 2-3 KPIs (or all-KPI) is considered stale/incomplete
                // and is discarded so /initial regenerates a proper dashboard.
                const hasNonKPI = cachedCharts.some(c => c.type !== 'kpi');
                const isGoodCache = cachedCharts.length >= 4 && hasNonKPI;
                if (isGoodCache) {
                    const charts = cachedCharts.filter(c => !removedChartIds.has(c.chart_id));
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

    const exportDashboardPNG = async () => {
        if (!gridRef.current) return;
        setExporting(true);
        setShowExportMenu(false);
        const overridden = [];
        try {
            await new Promise(r => setTimeout(r, 100));

            const node = gridRef.current;
            const fullH = node.scrollHeight;
            const fullW = node.scrollWidth;

            // Walk up the DOM and temporarily remove all overflow/height clipping so
            // html-to-image sees the full grid content, not just the scrolled viewport.
            let el = node;
            while (el && el !== document.body) {
                overridden.push({
                    el,
                    overflow: el.style.overflow,
                    overflowY: el.style.overflowY,
                    overflowX: el.style.overflowX,
                    height: el.style.height,
                    maxHeight: el.style.maxHeight,
                    minHeight: el.style.minHeight,
                    flex: el.style.flex,
                });
                el.style.overflow = 'visible';
                el.style.maxHeight = 'none';
                el.style.minHeight = '0';
                if (el === node) {
                    // Force the scroll container to its full content size
                    el.style.height = `${fullH}px`;
                    el.style.flex = 'none';
                } else {
                    el.style.height = 'auto';
                }
                el = el.parentElement;
            }

            await new Promise(r => setTimeout(r, 100));

            const dataUrl = await toPng(node, { backgroundColor: t.bg, pixelRatio: 2, width: fullW, height: fullH });

            const a = Object.assign(document.createElement('a'), {
                href: dataUrl,
                download: `dashboard-${new Date().toISOString().slice(0,10)}.png`
            });
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
        } catch (e) { console.error(e); }
        finally {
            // Always restore — even on error
            for (const s of overridden) {
                s.el.style.overflow = s.overflow;
                s.el.style.overflowY = s.overflowY;
                s.el.style.overflowX = s.overflowX;
                s.el.style.height = s.height;
                s.el.style.maxHeight = s.maxHeight;
                s.el.style.minHeight = s.minHeight;
                s.el.style.flex = s.flex;
            }
            setExporting(false);
        }
    };

    const exportAllCSV = () => {
        if (!dashboard?.charts?.length) return;
        setShowExportMenu(false);
        const sections = dashboard.charts
            .filter(c => c.data?.length)
            .map(c => {
                const headers = Object.keys(c.data[0]);
                const rows = c.data.map(row =>
                    headers.map(h => {
                        const v = row[h];
                        return typeof v === 'string' && (v.includes(',') || v.includes('"'))
                            ? `"${v.replace(/"/g, '""')}"` : (v ?? '');
                    }).join(',')
                );
                return `# ${c.title}\n${headers.join(',')}\n${rows.join('\n')}`;
            });
        const blob = new Blob([sections.join('\n\n')], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = Object.assign(document.createElement('a'), {
            href: url,
            download: `dashboard-export-${new Date().toISOString().slice(0,10)}.csv`
        });
        document.body.appendChild(a); a.click();
        document.body.removeChild(a); URL.revokeObjectURL(url);
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
    // Build grid layout: KPIs 3-per-row (w=4), regular charts 2-per-row (w=6)
    const layout = useMemo(() => {
        if (!dashboard?.charts) return [];
        let regularCount = 0;
        let kpiCount = 0;
        return dashboard.charts.map((chart) => {
            if (chart.position?.x !== undefined && chart.position?.y !== undefined) {
                return {
                    i: chart.chart_id,
                    x: chart.position.x, y: chart.position.y,
                    w: chart.position.w ?? (chart.type === 'kpi' ? 4 : 6),
                    h: chart.position.h ?? (chart.type === 'kpi' ? 2 : 4),
                    minW: chart.type === 'kpi' ? 2 : 3,
                    minH: chart.type === 'kpi' ? 2 : 3,
                };
            }
            // No saved position — auto-arrange
            if (chart.type === 'kpi') {
                const col = kpiCount % 3;
                kpiCount++;
                return {
                    i: chart.chart_id,
                    x: col * 4, y: 0,
                    w: 4, h: 2, minW: 2, minH: 2,
                };
            }
            // Regular charts: alternate left (x=0) and right (x=6) columns
            const col = regularCount % 2;
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
                            {dashboard.charts.length} chart{dashboard.charts.length !== 1 ? 's' : ''} • Drag to rearrange
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

                        {/* Export menu */}
                        <div className="relative">
                            <button
                                onClick={() => setShowExportMenu(!showExportMenu)}
                                disabled={exporting}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all disabled:opacity-50"
                                style={{ background: t.surface, color: t.textSub, borderColor: t.border }}>
                                <Download className={`w-3.5 h-3.5 ${exporting ? 'animate-pulse' : ''}`} />
                                Export
                            </button>
                            {showExportMenu && (
                                <>
                                    <div className="fixed inset-0 z-40" onClick={() => setShowExportMenu(false)} />
                                    <div className="absolute right-0 mt-1 w-44 rounded-xl shadow-xl border py-1 z-50"
                                        style={{ background: t.surface, borderColor: t.border }}>
                                        <button onClick={exportDashboardPNG}
                                            className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:opacity-70"
                                            style={{ color: t.text }}>
                                            <ImageIcon className="w-4 h-4" /> Export as PNG
                                        </button>
                                        <button onClick={exportAllCSV}
                                            className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:opacity-70"
                                            style={{ color: t.text }}>
                                            <FileText className="w-4 h-4" /> Export All as CSV
                                        </button>
                                    </div>
                                </>
                            )}
                        </div>

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
                <div ref={node => { gridContainerRef(node); if (node) gridRef.current = node; }} style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden', padding: '8px' }}>
                    {/* Suppress the RGL slide-in animation completely.
                        Items slide because the grid width changes from 0→real on first measure.
                        Disabling transitions here removes the glitch without affecting drag UX
                        (react-grid-layout already sets transition:none during active drags). */}
                    <style>{`
                        .react-grid-item { transition: none !important; }
                        .react-grid-item.react-grid-placeholder { display: none !important; }
                    `}</style>
                    {gridReady && (
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
                    )}
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
