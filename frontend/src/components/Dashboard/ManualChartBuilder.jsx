// frontend/src/components/Dashboard/ManualChartBuilder.jsx
// Power BI-style chart builder panel

import React, { useState, useEffect, useMemo } from 'react';
import {
    X, Play, Plus, BarChart2, TrendingUp, PieChart, Circle,
    Grid3x3, Layers, Filter, Target, Activity, Loader2, CheckCircle2,
    Table2, Wand2, Code2
} from 'lucide-react';
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

const API = 'http://localhost:8000/api';

const CHART_TYPES = [
    { id: 'bar',      label: 'Bar',      icon: BarChart2,  desc: 'Compare values across categories' },
    { id: 'line',     label: 'Line',     icon: TrendingUp, desc: 'Show trends over time' },
    { id: 'area',     label: 'Area',     icon: Activity,   desc: 'Magnitude of change over time' },
    { id: 'donut',    label: 'Donut',    icon: Circle,     desc: 'Proportional composition' },
    { id: 'pie',      label: 'Pie',      icon: PieChart,   desc: 'Simple part-to-whole' },
    { id: 'scatter',  label: 'Scatter',  icon: Target,     desc: 'Correlation between two metrics' },
    { id: 'funnel',   label: 'Funnel',   icon: Filter,     desc: 'Stages in a process/pipeline' },
    { id: 'treemap',  label: 'Treemap',  icon: Grid3x3,    desc: 'Hierarchical proportions' },
    { id: 'radar',    label: 'Radar',    icon: Layers,     desc: 'Multi-dimensional comparison' },
    { id: 'tornado',  label: 'Tornado',  icon: BarChart2,  desc: 'Compare two groups side by side' },
    { id: 'table',    label: 'Table',    icon: Table2,     desc: 'Detailed tabular data' },
];

const ACCENT_COLORS = [
    '#6366f1','#3b82f6','#10b981','#f59e0b','#ec4899',
    '#06b6d4','#8b5cf6','#f43f5e','#14b8a6','#f97316',
];

// Defines which columns each chart type needs and what types they must be.
// numeric: true  → only numeric columns shown
// numeric: false → all columns shown (categorical preferred)
// z field        → third column (tornado only)
// null           → table chart needs no column selection
const CHART_COL_CONFIG = {
    bar:     { x: { label: 'Category',     numeric: false }, y: { label: 'Value',        numeric: true  } },
    line:    { x: { label: 'X / Date',     numeric: false }, y: { label: 'Value',        numeric: true  } },
    area:    { x: { label: 'X / Date',     numeric: false }, y: { label: 'Value',        numeric: true  } },
    donut:   { x: { label: 'Label',        numeric: false }, y: { label: 'Value',        numeric: true  } },
    pie:     { x: { label: 'Label',        numeric: false }, y: { label: 'Value',        numeric: true  } },
    scatter: { x: { label: 'X (numeric)',  numeric: true  }, y: { label: 'Y (numeric)',  numeric: true  } },
    funnel:  { x: { label: 'Stage',        numeric: false }, y: { label: 'Value',        numeric: true  } },
    treemap: { x: { label: 'Name',         numeric: false }, y: { label: 'Size',         numeric: true  } },
    radar:   { x: { label: 'Subject',      numeric: false }, y: { label: 'Value',        numeric: true  } },
    tornado: { x: { label: 'Category',     numeric: false }, y: { label: 'Left Value',   numeric: true  },
               z: { label: 'Right Value',  numeric: true  } },
    table:   null,
};

const ManualChartBuilder = ({ sessionId, onAddChart, onClose, theme }) => {
    const t = theme || {
        surface: '#fff', surfaceHover: '#f8fafc', border: '#e2e8f0',
        text: '#1e293b', textSub: '#64748b', textMuted: '#94a3b8',
        accent: '#6366f1', accentLight: '#eef2ff', accentText: '#4f46e5',
        success: '#10b981', danger: '#ef4444',
    };

    const [selectedType, setSelectedType] = useState('bar');
    const [mode, setMode] = useState('sql');
    const [sql, setSql] = useState('');
    const [tables, setTables] = useState({});
    const [selectedTable, setSelectedTable] = useState('');
    const [xCol, setXCol] = useState('');
    const [yCol, setYCol] = useState('');
    const [zCol, setZCol] = useState(''); // tornado right-side value
    const [chartTitle, setChartTitle] = useState('');
    const [chartColor, setChartColor] = useState('#6366f1');

    const [executing, setExecuting] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [adding, setAdding] = useState(false);
    const [added, setAdded] = useState(false);

    // Load schema on mount
    useEffect(() => {
        const token = sessionStorage.getItem('token');
        fetch(`${API}/schema`, { headers: { Authorization: `Bearer ${token}` } })
            .then(r => r.json())
            .then(d => d.success && setTables(d.schema?.tables || {}))
            .catch(() => {});
    }, []);

    // When chart type changes, reset column selections — prior selections may be incompatible
    useEffect(() => {
        setXCol('');
        setYCol('');
        setZCol('');
        setResult(null);
        setError(null);
    }, [selectedType]);

    // Derived column lists from schema
    const tableCols   = selectedTable ? (tables[selectedTable]?.columns || []) : [];
    const tableTypes  = selectedTable ? (tables[selectedTable]?.types   || {}) : {};
    const isNum = (col) => /int|float|numeric|decimal|real|double|money/.test((tableTypes[col] || '').toLowerCase());
    const numericCols = tableCols.filter(isNum);

    const colCfg  = CHART_COL_CONFIG[selectedType];
    const xOptions = colCfg ? (colCfg.x.numeric ? numericCols : tableCols) : tableCols;
    const yOptions = colCfg ? (colCfg.y.numeric ? numericCols : tableCols) : tableCols;
    const zOptions = colCfg?.z ? (colCfg.z.numeric ? numericCols : tableCols) : [];

    // Auto-generate SQL from table + column picks
    useEffect(() => {
        if (mode !== 'table' || !selectedTable) return;

        if (selectedType === 'table') {
            setSql(`SELECT *\nFROM ${selectedTable}\nLIMIT 50`);
            return;
        }
        if (!xCol || !yCol) return;

        if (selectedType === 'scatter') {
            setSql(`SELECT ${xCol}, ${yCol}\nFROM ${selectedTable}\nLIMIT 200`);
            return;
        }
        if (selectedType === 'tornado') {
            if (!zCol) return;
            setSql(
                `SELECT ${xCol},\n  SUM(${yCol}) AS left_val,\n  SUM(${zCol}) AS right_val\nFROM ${selectedTable}\nGROUP BY ${xCol}\nORDER BY left_val DESC\nLIMIT 15`
            );
            return;
        }

        setSql(
            `SELECT ${xCol}, SUM(${yCol}) AS total\nFROM ${selectedTable}\nGROUP BY ${xCol}\nORDER BY total DESC\nLIMIT 20`
        );
    }, [mode, selectedTable, selectedType, xCol, yCol, zCol]);

    const runQuery = async () => {
        if (!sql.trim()) return;
        setExecuting(true);
        setError(null);
        setResult(null);
        setAdded(false);
        try {
            const token = sessionStorage.getItem('token');
            const res = await fetch(`${API}/execute-query`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ sql }),
            });
            const data = await res.json();
            if (!res.ok || !data.success) {
                setError(data.detail || data.error || 'Query failed');
            } else {
                setResult(data);
            }
        } catch {
            setError('Network error — is the backend running?');
        } finally {
            setExecuting(false);
        }
    };

    const previewChart = useMemo(() => {
        if (!result?.data?.length) return null;
        const keys = Object.keys(result.data[0]);
        const xKey = xCol && keys.includes(xCol) ? xCol : keys[0];
        const yKey = yCol && keys.includes(yCol) && yCol !== xKey
            ? yCol
            : keys.find(k => k !== xKey && typeof result.data[0][k] === 'number') || keys[1] || keys[0];
        const zKey = zCol && keys.includes(zCol) ? zCol
            : keys.find(k => k !== xKey && k !== yKey && typeof result.data[0][k] === 'number') || yKey;

        return {
            chart_id: 'preview',
            title: chartTitle || 'Preview',
            type: selectedType,
            data: result.data,
            config: {
                xAxis: xKey, yAxis: yKey,
                labelKey: xKey, valueKey: yKey,
                subjectKey: xKey, nameKey: xKey,
                valueKeys: [yKey], categoryKey: xKey,
                leftKey: yKey, rightKey: zKey,
                color: chartColor, horizontal: false,
                colors: ACCENT_COLORS,
            },
        };
    }, [result, selectedType, xCol, yCol, zCol, chartTitle, chartColor]);

    const addToDashboard = async () => {
        if (!result?.data?.length || adding) return;
        setAdding(true);
        try {
            const token = sessionStorage.getItem('token');
            const keys = Object.keys(result.data[0]);
            const xKey = xCol && keys.includes(xCol) ? xCol : keys[0];
            const yKey = yCol && yCol !== xKey && keys.includes(yCol)
                ? yCol
                : keys.find(k => k !== xKey && typeof result.data[0][k] === 'number') || keys[1] || keys[0];
            const zKey = zCol && zCol !== xKey && zCol !== yKey && keys.includes(zCol)
                ? zCol
                : keys.find(k => k !== xKey && k !== yKey && typeof result.data[0][k] === 'number') || yKey;

            const chartDef = {
                title: chartTitle || `${selectedType} — ${new Date().toLocaleTimeString()}`,
                type: selectedType, data: result.data, sql, query: 'Manual Chart',
                config: {
                    xAxis: xKey, yAxis: yKey,
                    labelKey: xKey, valueKey: yKey,
                    subjectKey: xKey, nameKey: xKey,
                    valueKeys: [yKey], categoryKey: xKey,
                    leftKey: yKey, rightKey: zKey,
                    color: chartColor, colors: ACCENT_COLORS,
                    horizontal: selectedType === 'bar' ? false : undefined,
                },
            };

            const res = await fetch(`${API}/dashboard/add-manual-chart`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, chart: chartDef }),
            });
            if (res.ok) {
                window.dispatchEvent(new CustomEvent('refreshDashboard'));
                setAdded(true);
                if (onAddChart) onAddChart({ chart: chartDef, sql });
                setTimeout(() => setAdded(false), 3000);
            }
        } catch (e) { console.error(e); }
        finally { setAdding(false); }
    };

    const renderPreviewChart = () => {
        if (!previewChart) return null;
        const props = { chart: previewChart, selectedKey: null, selectedValue: null, onSelect: null };
        switch (selectedType) {
            case 'bar':     return <BarChartComponent {...props} />;
            case 'line':
            case 'area':    return <LineChartComponent {...props} />;
            case 'pie':     return <PieChartComponent {...props} />;
            case 'donut':   return <DonutChartComponent {...props} />;
            case 'scatter': return <ScatterChartComponent {...props} />;
            case 'funnel':  return <FunnelChartComponent {...props} />;
            case 'treemap': return <TreemapComponent {...props} />;
            case 'radar':   return <RadarChartComponent {...props} />;
            case 'tornado': return <TornadoChartComponent {...props} />;
            case 'table':   return <DataTable {...props} />;
            default:        return null;
        }
    };

    // Column pickers in the Configuration section (after query runs) — filtered by chart type
    const renderPostQueryColPickers = () => {
        if (!result?.data?.length || !colCfg) return null;
        const keys = Object.keys(result.data[0]);
        const numKeys = keys.filter(k => typeof result.data[0][k] === 'number');
        const xOpts = colCfg.x.numeric ? numKeys : keys;
        const yOpts = colCfg.y.numeric ? numKeys : keys;
        const zOpts = colCfg.z ? (colCfg.z.numeric ? numKeys : keys) : [];

        return (
            <div className={`grid gap-2 ${colCfg.z ? 'grid-cols-1' : 'grid-cols-2'}`}>
                <div>
                    <label style={labelStyle}>{colCfg.x.label}</label>
                    <select value={xCol} onChange={e => setXCol(e.target.value)} style={inputStyle}>
                        <option value="">Auto</option>
                        {xOpts.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                </div>
                <div>
                    <label style={labelStyle}>{colCfg.y.label}</label>
                    <select value={yCol} onChange={e => setYCol(e.target.value)} style={inputStyle}>
                        <option value="">Auto</option>
                        {yOpts.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                </div>
                {colCfg.z && (
                    <div>
                        <label style={labelStyle}>{colCfg.z.label}</label>
                        <select value={zCol} onChange={e => setZCol(e.target.value)} style={inputStyle}>
                            <option value="">Auto</option>
                            {zOpts.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                    </div>
                )}
            </div>
        );
    };

    const inputStyle = {
        background: t.surfaceHover || t.surface, border: `1px solid ${t.border}`,
        color: t.text, borderRadius: '8px', padding: '6px 10px',
        fontSize: '13px', outline: 'none', width: '100%',
    };
    const labelStyle = {
        fontSize: '11px', fontWeight: '600', color: t.textSub,
        textTransform: 'uppercase', letterSpacing: '0.05em',
        marginBottom: '4px', display: 'block',
    };

    return (
        <div className="flex flex-col h-full" style={{ background: t.surface, color: t.text }}>
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b shrink-0" style={{ borderColor: t.border }}>
                <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: t.accentLight || '#eef2ff' }}>
                        <Wand2 className="w-4 h-4" style={{ color: t.accent }} />
                    </div>
                    <div>
                        <p className="font-bold text-sm" style={{ color: t.text }}>Chart Builder</p>
                        <p className="text-xs" style={{ color: t.textMuted }}>Create custom visualizations</p>
                    </div>
                </div>
                <button onClick={onClose} className="p-1.5 rounded-lg hover:opacity-70 transition-opacity" style={{ color: t.textMuted }}>
                    <X className="w-4 h-4" />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto">
                {/* ── Section 1: Chart Type ─────────────────────────── */}
                <div className="px-4 pt-4 pb-3 border-b" style={{ borderColor: t.border }}>
                    <p className="text-xs font-bold uppercase tracking-wider mb-3" style={{ color: t.textSub }}>Chart Type</p>
                    <div className="grid grid-cols-4 gap-1.5">
                        {CHART_TYPES.map(({ id, label, icon: Icon, desc }) => (
                            <button
                                key={id}
                                onClick={() => setSelectedType(id)}
                                title={desc}
                                className="flex flex-col items-center gap-1 p-2 rounded-lg border transition-all text-center"
                                style={{
                                    background: selectedType === id ? t.accentLight : 'transparent',
                                    borderColor: selectedType === id ? t.accent : t.border,
                                    color: selectedType === id ? t.accentText || t.accent : t.textSub,
                                }}>
                                <Icon className="w-4 h-4" />
                                <span style={{ fontSize: '10px', fontWeight: '600' }}>{label}</span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* ── Section 2: Data Source ────────────────────────── */}
                <div className="px-4 pt-4 pb-3 border-b" style={{ borderColor: t.border }}>
                    <p className="text-xs font-bold uppercase tracking-wider mb-3" style={{ color: t.textSub }}>Data Source</p>

                    {/* Mode toggle */}
                    <div className="flex rounded-lg overflow-hidden border mb-3" style={{ borderColor: t.border }}>
                        {[['sql', <Code2 className="w-3 h-3" />, 'SQL'], ['table', <Table2 className="w-3 h-3" />, 'Table Picker']].map(([m, icon, lbl]) => (
                            <button key={m} onClick={() => setMode(m)}
                                className="flex-1 flex items-center justify-center gap-1.5 py-1.5 text-xs font-semibold transition-all"
                                style={{ background: mode === m ? t.accent : 'transparent', color: mode === m ? '#fff' : t.textSub }}>
                                {icon} {lbl}
                            </button>
                        ))}
                    </div>

                    {/* Table Picker controls */}
                    {mode === 'table' && Object.keys(tables).length > 0 && (
                        <div className="space-y-2 mb-3">
                            <div>
                                <label style={labelStyle}>Table</label>
                                <select
                                    value={selectedTable}
                                    onChange={e => { setSelectedTable(e.target.value); setXCol(''); setYCol(''); setZCol(''); }}
                                    style={inputStyle}>
                                    <option value="">Select table…</option>
                                    {Object.keys(tables).map(tbl => <option key={tbl} value={tbl}>{tbl}</option>)}
                                </select>
                            </div>

                            {/* Table chart: no column pickers needed */}
                            {selectedTable && selectedType === 'table' && (
                                <p className="text-xs px-1 italic" style={{ color: t.textMuted }}>
                                    All columns will be fetched automatically.
                                </p>
                            )}

                            {/* Per-chart column pickers */}
                            {selectedTable && colCfg && (
                                <div className={`grid gap-2 ${colCfg.z ? 'grid-cols-1' : 'grid-cols-2'}`}>
                                    <div>
                                        <label style={labelStyle}>{colCfg.x.label}</label>
                                        <select value={xCol} onChange={e => setXCol(e.target.value)} style={inputStyle}>
                                            <option value="">Pick column…</option>
                                            {xOptions.map(c => (
                                                <option key={c} value={c}>{c}{tableTypes[c] ? ` · ${tableTypes[c]}` : ''}</option>
                                            ))}
                                        </select>
                                        {colCfg.x.numeric && xOptions.length === 0 && (
                                            <p className="text-xs mt-1" style={{ color: t.danger || '#ef4444' }}>No numeric columns found</p>
                                        )}
                                    </div>
                                    <div>
                                        <label style={labelStyle}>{colCfg.y.label}</label>
                                        <select value={yCol} onChange={e => setYCol(e.target.value)} style={inputStyle}>
                                            <option value="">Pick column…</option>
                                            {yOptions.map(c => (
                                                <option key={c} value={c}>{c}{tableTypes[c] ? ` · ${tableTypes[c]}` : ''}</option>
                                            ))}
                                        </select>
                                        {colCfg.y.numeric && yOptions.length === 0 && (
                                            <p className="text-xs mt-1" style={{ color: t.danger || '#ef4444' }}>No numeric columns found</p>
                                        )}
                                    </div>
                                    {colCfg.z && (
                                        <div>
                                            <label style={labelStyle}>{colCfg.z.label}</label>
                                            <select value={zCol} onChange={e => setZCol(e.target.value)} style={inputStyle}>
                                                <option value="">Pick column…</option>
                                                {zOptions.map(c => (
                                                    <option key={c} value={c}>{c}{tableTypes[c] ? ` · ${tableTypes[c]}` : ''}</option>
                                                ))}
                                            </select>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* SQL editor (always visible) */}
                    <div>
                        <label style={labelStyle}>{mode === 'sql' ? 'SQL Query' : 'Generated SQL'}</label>
                        <textarea
                            value={sql}
                            onChange={e => setSql(e.target.value)}
                            placeholder="SELECT category, SUM(revenue) AS total FROM orders GROUP BY category ORDER BY total DESC LIMIT 10"
                            rows={mode === 'sql' ? 5 : 4}
                            style={{ ...inputStyle, resize: 'vertical', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.5' }}
                        />
                    </div>

                    <button
                        onClick={runQuery}
                        disabled={executing || !sql.trim()}
                        className="mt-2 w-full flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-50"
                        style={{ background: t.accent, color: '#fff' }}>
                        {executing
                            ? <><Loader2 className="w-4 h-4 animate-spin" /> Running…</>
                            : <><Play className="w-4 h-4" /> Run Query</>}
                    </button>

                    {error && (
                        <div className="mt-2 text-xs p-2 rounded-lg" style={{ background: '#fef2f2', color: '#dc2626', border: '1px solid #fecaca' }}>
                            {error}
                        </div>
                    )}
                    {result && !error && (
                        <p className="mt-2 text-xs" style={{ color: t.success || '#10b981' }}>
                            ✓ {result.data?.length} rows returned
                        </p>
                    )}
                </div>

                {/* ── Section 3: Configuration ──────────────────────── */}
                <div className="px-4 pt-4 pb-3 border-b" style={{ borderColor: t.border }}>
                    <p className="text-xs font-bold uppercase tracking-wider mb-3" style={{ color: t.textSub }}>Configuration</p>
                    <div className="space-y-3">
                        <div>
                            <label style={labelStyle}>Chart Title</label>
                            <input value={chartTitle} onChange={e => setChartTitle(e.target.value)}
                                placeholder="Enter chart title…" style={inputStyle} />
                        </div>

                        {renderPostQueryColPickers()}

                        <div>
                            <label style={labelStyle}>Color</label>
                            <div className="flex gap-1.5 flex-wrap">
                                {ACCENT_COLORS.map(c => (
                                    <button key={c} onClick={() => setChartColor(c)}
                                        className="w-6 h-6 rounded-full border-2 transition-transform hover:scale-110"
                                        style={{ background: c, borderColor: chartColor === c ? t.text : 'transparent' }} />
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── Section 4: Preview ────────────────────────────── */}
                {previewChart && (
                    <div className="px-4 pt-4 pb-3">
                        <p className="text-xs font-bold uppercase tracking-wider mb-3" style={{ color: t.textSub }}>Preview</p>
                        <div className="rounded-xl border overflow-hidden"
                            style={{ height: '240px', borderColor: t.border, background: t.surfaceHover || t.surface }}>
                            {renderPreviewChart()}
                        </div>
                    </div>
                )}
            </div>

            {/* Footer CTA */}
            <div className="px-4 py-3 border-t shrink-0" style={{ borderColor: t.border }}>
                <button
                    onClick={addToDashboard}
                    disabled={!result?.data?.length || adding || added}
                    className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-bold transition-all disabled:opacity-50"
                    style={{ background: added ? (t.success || '#10b981') : t.accent, color: '#fff' }}>
                    {adding
                        ? <><Loader2 className="w-4 h-4 animate-spin" /> Adding…</>
                        : added
                            ? <><CheckCircle2 className="w-4 h-4" /> Added to Dashboard!</>
                            : <><Plus className="w-4 h-4" /> Add to Dashboard</>}
                </button>
            </div>
        </div>
    );
};

export default ManualChartBuilder;
