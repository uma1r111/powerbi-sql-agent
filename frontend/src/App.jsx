import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Send, LogOut, MessageSquare, Database, Clock, User, Menu,
  BarChart2, MessageCircle, Search, Trash2, Plus, Edit2, Check, X,
  Code, Copy, ChevronDown, PieChart, Palette, Sparkles, TrendingUp,
  Activity, ChevronRight
} from 'lucide-react';
import axios from 'axios';
import DashboardContainer from './components/Dashboard/DashboardContainer';
import VisualizationHistory from './components/VisualizationHistory';
import { ThemeProvider, useTheme } from './contexts/ThemeContext';

const API_URL = 'http://localhost:8000/api';

const DEFAULT_WELCOME_MESSAGE = {
  type: 'ai',
  content: "Hello! I'm IntelliQuery AI. Ask me anything about your data — I can run queries, generate insights, and create visualizations.",
  timestamp: new Date().toISOString()
};

const DEFAULT_CONVERSATION = {
  id: 1,
  title: 'New Conversation',
  messages: [DEFAULT_WELCOME_MESSAGE],
  charts: [],
  lastUpdated: new Date().toISOString()
};

// ─── Login Page ───────────────────────────────────────────────────────────────
const LoginPage = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await axios.post(`${API_URL}/login`, { email, password });
      sessionStorage.setItem('token', res.data.token);
      sessionStorage.setItem('user', JSON.stringify(res.data.user));
      onLogin(res.data.user);
    } catch {
      setError('Invalid credentials. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4"
      style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)' }}>
      {/* Background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-96 h-96 rounded-full blur-3xl opacity-20 -top-20 -left-20"
          style={{ background: 'radial-gradient(circle, #6366f1, transparent)' }} />
        <div className="absolute w-96 h-96 rounded-full blur-3xl opacity-20 -bottom-20 -right-20"
          style={{ background: 'radial-gradient(circle, #3b82f6, transparent)' }} />
      </div>

      <div className="relative w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-4 shadow-2xl"
            style={{ background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)' }}>
            <Sparkles className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white tracking-tight">IntelliQuery</h1>
          <p className="text-slate-400 mt-2 text-sm">AI-Powered Business Intelligence Platform</p>
        </div>

        <div className="rounded-2xl p-8 shadow-2xl"
          style={{ background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(255,255,255,0.08)', backdropFilter: 'blur(20px)' }}>
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Email address</label>
              <input type="email" value={email} onChange={e => setEmail(e.target.value)} required
                className="w-full px-4 py-3 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 transition-all"
                placeholder="you@intelliquery.com"
                style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(255,255,255,0.1)', focusRingColor: '#6366f1' }} />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Password</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} required
                className="w-full px-4 py-3 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 transition-all"
                placeholder="••••••••"
                style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(255,255,255,0.1)' }} />
            </div>
            {error && (
              <div className="text-sm text-red-300 px-4 py-3 rounded-xl"
                style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)' }}>
                {error}
              </div>
            )}
            <button type="submit" disabled={loading}
              className="w-full py-3 rounded-xl font-bold text-white text-sm transition-all disabled:opacity-50"
              style={{ background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)' }}>
              {loading ? 'Signing in…' : 'Sign In →'}
            </button>
          </form>

          <div className="mt-6 pt-5" style={{ borderTop: '1px solid rgba(255,255,255,0.08)' }}>
            <p className="text-xs text-slate-500 text-center mb-3">Quick Demo Access</p>
            <div className="flex justify-center gap-2">
              {['sameed', 'izma', 'umair'].map(name => (
                <button key={name}
                  onClick={() => { setEmail(`${name}@intelliquery.com`); setPassword('1234'); }}
                  className="text-xs px-3 py-1.5 rounded-lg capitalize transition-all hover:opacity-80"
                  style={{ background: 'rgba(99,102,241,0.15)', color: '#a5b4fc', border: '1px solid rgba(99,102,241,0.2)' }}>
                  {name}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─── Chat Panel ───────────────────────────────────────────────────────────────
const ChatPanel = ({ messages, setMessages, scrollToIndex, sessionId, onChartCreated }) => {
  const { theme: t } = useTheme();
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const messageRefs = useRef({});

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);
  useEffect(() => {
    if (scrollToIndex !== null && messageRefs.current[scrollToIndex])
      messageRefs.current[scrollToIndex].scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [scrollToIndex]);

  const sendQuery = async () => {
    if (!input.trim() || loading) return;
    const q = input;
    setMessages(prev => [...prev, { type: 'user', content: q, timestamp: new Date().toISOString() }]);
    setInput('');
    setLoading(true);
    try {
      const token = sessionStorage.getItem('token');
      const res = await axios.post(`${API_URL}/query`,
        { question: q, session_id: sessionId || 'default' },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      const ai = {
        type: 'ai',
        content: res.data.explanation || 'Query processed successfully.',
        sql: res.data.sql,
        results: res.data.results || [],
        execution_time: res.data.execution_time,
        chart: res.data.chart,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, ai]);
      if (res.data.chart && onChartCreated)
        onChartCreated({ ...res.data.chart, query: q, created_at: new Date().toISOString() });
      if (res.data.chart)
        setTimeout(() => window.dispatchEvent(new CustomEvent('refreshDashboard')), 500);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setMessages(prev => [...prev, {
        type: 'error',
        content: typeof detail === 'string' ? detail : 'Something went wrong. Please try again.',
        timestamp: new Date().toISOString()
      }]);
    } finally { setLoading(false); }
  };

  const SUGGESTIONS = ['Top 10 customers by revenue', 'Monthly sales trend', 'Products low in stock'];

  return (
    <div className="flex flex-col h-full rounded-2xl overflow-hidden"
      style={{ background: t.surface, border: `1px solid ${t.border}`, boxShadow: t.shadowMd }}>
      {/* Header */}
      <div className="px-4 py-3 shrink-0 flex items-center gap-3"
        style={{ background: `linear-gradient(135deg, ${t.accent} 0%, ${t.accentHover || t.accent} 100%)` }}>
        <div className="w-8 h-8 rounded-xl flex items-center justify-center"
          style={{ background: 'rgba(255,255,255,0.2)' }}>
          <Sparkles className="w-4 h-4 text-white" />
        </div>
        <div>
          <p className="text-white text-sm font-bold">AI Assistant</p>
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.7)' }}>Powered by Llama 3.3 · 70B</p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3" style={{ minHeight: 0 }}>
        {messages.map((msg, idx) => (
          <div key={idx} ref={el => { messageRefs.current[idx] = el; }}
            className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[90%] rounded-2xl px-3 py-2.5 text-sm ${msg.type === 'user' ? 'rounded-tr-sm' : 'rounded-tl-sm'}`}
              style={{
                background: msg.type === 'user'
                  ? t.accent
                  : msg.type === 'error'
                    ? 'rgba(239,68,68,0.08)'
                    : t.surfaceHover || t.surface,
                color: msg.type === 'user' ? '#fff' : msg.type === 'error' ? '#dc2626' : t.text,
                border: msg.type === 'error' ? '1px solid rgba(239,68,68,0.2)' : msg.type !== 'user' ? `1px solid ${t.border}` : 'none',
              }}>
              {msg.type === 'error' ? (
                <div className="flex items-start gap-2">
                  <span className="shrink-0 mt-0.5">⚠</span>
                  <p>{msg.content}</p>
                </div>
              ) : <p style={{ lineHeight: '1.5' }}>{msg.content}</p>}

              {msg.results?.length > 0 && (
                <div className="mt-2 overflow-x-auto rounded-lg"
                  style={{ border: `1px solid ${t.border}` }}>
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr style={{ background: t.surfaceHover || t.bg }}>
                        {Object.keys(msg.results[0]).map(k => (
                          <th key={k} className="text-left px-2 py-1.5 font-semibold whitespace-nowrap"
                            style={{ color: t.textSub, borderBottom: `1px solid ${t.border}` }}>
                            {k}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {msg.results.slice(0, 5).map((row, i) => (
                        <tr key={i} style={{ borderBottom: `1px solid ${t.border}` }}>
                          {Object.values(row).map((val, j) => (
                            <td key={j} className="px-2 py-1.5 whitespace-nowrap"
                              style={{ color: t.text }}>
                              {val !== null ? String(val) : '—'}
                            </td>
                          ))}
                        </tr>
                      ))}
                      {msg.results.length > 5 && (
                        <tr>
                          <td colSpan={Object.keys(msg.results[0]).length}
                            className="px-2 py-1.5 text-center text-xs" style={{ color: t.textMuted }}>
                            +{msg.results.length - 5} more rows
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}
              {msg.chart && (
                <p className="text-xs mt-2 flex items-center gap-1" style={{ color: msg.type === 'user' ? 'rgba(255,255,255,0.8)' : t.accent }}>
                  <BarChart2 className="w-3 h-3" /> Visualization added to dashboard
                </p>
              )}
              {msg.execution_time > 0 && (
                <p className="text-xs mt-1 opacity-60">{msg.execution_time.toFixed(2)}s</p>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-2"
              style={{ background: t.surfaceHover || t.surface, border: `1px solid ${t.border}` }}>
              <div className="flex gap-1">
                {[0, 1, 2].map(i => (
                  <div key={i} className="w-1.5 h-1.5 rounded-full animate-bounce"
                    style={{ background: t.accent, animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
              <span className="text-xs" style={{ color: t.textMuted }}>Thinking…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggestion chips */}
      <div className="px-3 py-2 flex gap-1.5 flex-wrap shrink-0"
        style={{ borderTop: `1px solid ${t.border}` }}>
        {SUGGESTIONS.map(s => (
          <button key={s} onClick={() => setInput(s)}
            className="text-xs px-2.5 py-1 rounded-full transition-all hover:opacity-80"
            style={{ background: t.accentLight, color: t.accentText || t.accent, border: `1px solid ${t.border}` }}>
            {s}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="px-3 pb-3 shrink-0">
        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); } }}
            placeholder="Ask about your data…"
            rows={1}
            disabled={loading}
            className="flex-1 px-3 py-2.5 text-sm rounded-xl resize-none focus:outline-none focus:ring-2 transition-all disabled:opacity-50"
            style={{
              background: t.surfaceHover || t.bg,
              border: `1px solid ${t.border}`,
              color: t.text,
              maxHeight: '100px',
              minHeight: '40px',
            }} />
          <button onClick={sendQuery} disabled={loading || !input.trim()}
            className="p-2.5 rounded-xl font-semibold transition-all disabled:opacity-40 shrink-0"
            style={{ background: t.accent, color: '#fff' }}>
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

// ─── SQL Viewer ───────────────────────────────────────────────────────────────
const HistoryTab = ({ messages, onClickQuery }) => {
  const { theme: t } = useTheme();
  const [search, setSearch] = useState('');
  const [expandedSql, setExpandedSql] = useState(null);

  const history = messages
    .map((m, i) => ({ ...m, index: i }))
    .filter(m => m.type === 'user')
    .filter(m => m.content.toLowerCase().includes(search.toLowerCase()))
    .reverse();

  const copy = (text) => { navigator.clipboard.writeText(text); };

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: t.bg }}>
      <div className="max-w-3xl mx-auto">
        <h3 className="text-lg font-bold mb-1" style={{ color: t.text }}>Query History</h3>
        <p className="text-sm mb-5" style={{ color: t.textMuted }}>All queries in this conversation</p>

        <div className="relative mb-5">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2" style={{ color: t.textMuted }} />
          <input type="text" value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search queries…"
            className="w-full pl-9 pr-4 py-2.5 text-sm rounded-xl focus:outline-none"
            style={{ background: t.surface, border: `1px solid ${t.border}`, color: t.text }} />
        </div>

        {history.length === 0 ? (
          <div className="text-center py-16">
            <Clock className="w-10 h-10 mx-auto mb-3" style={{ color: t.border }} />
            <p className="text-sm" style={{ color: t.textMuted }}>
              {search ? 'No matching queries.' : 'No queries yet. Ask something in the chat!'}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {history.map((msg, i) => {
              const ai = messages[msg.index + 1];
              const timeStr = new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
              const expanded = expandedSql === msg.index;
              return (
                <div key={msg.index} className="rounded-xl overflow-hidden transition-shadow"
                  style={{ background: t.surface, border: `1px solid ${t.border}`, boxShadow: t.shadow }}>
                  <div className="p-4">
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                          <span className="text-xs font-bold px-2 py-0.5 rounded-full"
                            style={{ background: t.accentLight, color: t.accentText || t.accent }}>
                            #{history.length - i}
                          </span>
                          <span className="text-xs" style={{ color: t.textMuted }}>{timeStr}</span>
                          {ai?.results?.length > 0 && (
                            <span className="text-xs px-2 py-0.5 rounded-full"
                              style={{ background: 'rgba(16,185,129,0.1)', color: '#10b981' }}>
                              {ai.results.length} rows
                            </span>
                          )}
                          {ai?.chart && (
                            <span className="text-xs px-2 py-0.5 rounded-full"
                              style={{ background: 'rgba(139,92,246,0.1)', color: '#8b5cf6' }}>
                              📊 Chart
                            </span>
                          )}
                        </div>
                        <p className="text-sm font-semibold" style={{ color: t.text }}>{msg.content}</p>
                      </div>
                      <button onClick={() => onClickQuery(msg.index)} style={{ color: t.textMuted }}>
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                    {ai?.sql && (
                      <div className="mt-3">
                        <div className="flex items-center justify-between mb-2">
                          <button onClick={() => setExpandedSql(expanded ? null : msg.index)}
                            className="flex items-center gap-1.5 text-xs font-medium transition-colors"
                            style={{ color: t.textSub }}>
                            <Code className="w-3.5 h-3.5" />
                            {expanded ? 'Hide SQL' : 'View SQL'}
                            <ChevronDown className={`w-3 h-3 transition-transform ${expanded ? 'rotate-180' : ''}`} />
                          </button>
                          {expanded && (
                            <button onClick={() => copy(ai.sql)}
                              className="flex items-center gap-1 text-xs transition-colors"
                              style={{ color: t.textMuted }}>
                              <Copy className="w-3 h-3" /> Copy
                            </button>
                          )}
                        </div>
                        {expanded ? (
                          <div className="rounded-xl p-3 overflow-x-auto"
                            style={{ background: '#0f172a' }}>
                            <pre className="text-xs font-mono text-green-400 whitespace-pre-wrap">{ai.sql}</pre>
                          </div>
                        ) : (
                          <div className="rounded-lg px-3 py-2 text-xs font-mono truncate"
                            style={{ background: t.surfaceHover || t.bg, color: t.textMuted }}>
                            {ai.sql.slice(0, 80)}…
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

// ─── Theme Selector ───────────────────────────────────────────────────────────
const ThemeSelector = () => {
  const { theme: t, themeId, setTheme, themes } = useTheme();
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 rounded-xl text-sm transition-all"
        style={{ background: open ? t.sidebarBgActive : 'transparent', color: t.sidebarText }}>
        <Palette className="w-4 h-4" />
        <span>Theme</span>
        <div className="ml-auto flex gap-1">
          {Object.values(themes).map(th => (
            <div key={th.id} className="w-2.5 h-2.5 rounded-full border"
              style={{
                background: th.preview?.[2] || th.accent,
                borderColor: themeId === th.id ? t.sidebarTextActive : 'transparent',
                transform: themeId === th.id ? 'scale(1.3)' : 'scale(1)',
              }} />
          ))}
        </div>
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute bottom-full left-0 right-0 mb-2 rounded-xl shadow-2xl z-50 overflow-hidden"
            style={{ background: t.surface, border: `1px solid ${t.border}` }}>
            {Object.values(themes).map(th => (
              <button key={th.id}
                onClick={() => { setTheme(th.id); setOpen(false); }}
                className="w-full flex items-center gap-3 px-3 py-2.5 text-sm transition-all hover:opacity-80"
                style={{
                  background: themeId === th.id ? t.accentLight : 'transparent',
                  color: themeId === th.id ? t.accentText || t.accent : t.text,
                }}>
                <div className="flex gap-1 shrink-0">
                  {th.preview?.map((c, i) => (
                    <div key={i} className="w-3 h-3 rounded-full" style={{ background: c }} />
                  ))}
                </div>
                <span className="font-medium">{th.name}</span>
                {themeId === th.id && <Check className="w-3.5 h-3.5 ml-auto" />}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

// ─── Main Dashboard Shell ─────────────────────────────────────────────────────
const Dashboard = ({ user, onLogout }) => {
  const { theme: t } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeNav, setActiveNav] = useState('dashboard');
  const [conversations, setConversations] = useState([DEFAULT_CONVERSATION]);
  const [activeConvId, setActiveConvId] = useState(1);
  const [scrollToIndex, setScrollToIndex] = useState(null);
  const [editingConvId, setEditingConvId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  const [sessionLoading, setSessionLoading] = useState(true);

  const saveTimerRef = useRef(null);

  const saveSession = useCallback((convs, activeId) => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(async () => {
      try {
        const token = sessionStorage.getItem('token');
        await axios.post(`${API_URL}/session/save`,
          { conversations: convs, active_conv_id: activeId },
          { headers: { Authorization: `Bearer ${token}` } }
        );
      } catch { }
    }, 1500);
  }, []);

  useEffect(() => {
    const load = async () => {
      try {
        const token = sessionStorage.getItem('token');
        const res = await axios.get(`${API_URL}/session/load`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (res.data.success && !res.data.first_login && res.data.session?.conversations?.length) {
          const { conversations: saved, active_conv_id } = res.data.session;
          setConversations(saved);
          setActiveConvId(active_conv_id || saved[0].id);
        }
      } catch { }
      finally { setSessionLoading(false); }
    };
    load();
  }, []);

  useEffect(() => {
    if (!sessionLoading) saveSession(conversations, activeConvId);
  }, [conversations, activeConvId, sessionLoading, saveSession]);

  const activeConv = conversations.find(c => c.id === activeConvId);

  const setMessages = (updater) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id !== activeConvId) return conv;
      const msgs = typeof updater === 'function' ? updater(conv.messages) : updater;
      let title = conv.title;
      if (title === 'New Conversation') {
        const first = msgs.find(m => m.type === 'user');
        if (first) title = first.content.slice(0, 40) + (first.content.length > 40 ? '…' : '');
      }
      return { ...conv, messages: msgs, lastUpdated: new Date().toISOString(), title };
    }));
  };

  const handleChartCreated = (chart) => {
    setConversations(prev => prev.map(c =>
      c.id === activeConvId ? { ...c, charts: [...(c.charts || []), chart] } : c
    ));
  };

  const handleChartBuilderAdded = ({ chart, sql }) => {
    const now = new Date().toISOString();
    const chartWithMeta = { ...chart, query: chart.title, created_at: now };
    // Add to visualization history
    handleChartCreated(chartWithMeta);
    // Add to query history as a synthetic user + ai message pair
    setMessages(prev => [
      ...prev,
      { type: 'user', content: `[Chart Builder] ${chart.title}`, timestamp: now },
      { type: 'ai', content: 'Chart created via Chart Builder and added to dashboard.', sql, results: chart.data?.slice(0, 100) || [], chart: chartWithMeta, timestamp: now },
    ]);
  };

  const handleChartsLoaded = (charts) => {
    setConversations(prev => prev.map(c => {
      if (c.id !== activeConvId || c.charts?.length) return c;
      return {
        ...c, charts: charts.map(ch => ({
          ...ch, query: 'Initial Dashboard', created_at: ch.created_at || new Date().toISOString()
        }))
      };
    }));
  };

  const createNewConversation = () => {
    const id = Math.max(...conversations.map(c => c.id)) + 1;
    const conv = {
      id, title: 'New Conversation',
      messages: [{ ...DEFAULT_WELCOME_MESSAGE, timestamp: new Date().toISOString() }],
      charts: [], lastUpdated: new Date().toISOString()
    };
    setConversations(prev => [...prev, conv]);
    setActiveConvId(id);
    setActiveNav('dashboard');
  };

  const deleteConversation = (id) => {
    if (conversations.length === 1) return;
    setConversations(prev => prev.filter(c => c.id !== id));
    if (activeConvId === id)
      setActiveConvId(conversations.find(c => c.id !== id).id);
  };

  const startEdit = (id, title) => { setEditingConvId(id); setEditTitle(title); };
  const saveTitle = (id) => {
    setConversations(prev => prev.map(c => c.id === id ? { ...c, title: editTitle.trim() || 'New Conversation' } : c));
    setEditingConvId(null);
  };

  const handleHistoryClick = (idx) => { setActiveNav('dashboard'); setScrollToIndex(idx); };

  useEffect(() => {
    if (scrollToIndex !== null) {
      const t = setTimeout(() => setScrollToIndex(null), 500);
      return () => clearTimeout(t);
    }
  }, [scrollToIndex]);

  if (sessionLoading) {
    return (
      <div className="flex h-screen items-center justify-center" style={{ background: t.bg }}>
        <div className="text-center">
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-4"
            style={{ background: t.accentLight }}>
            <Activity className="w-6 h-6 animate-pulse" style={{ color: t.accent }} />
          </div>
          <p className="font-semibold" style={{ color: t.text }}>Restoring your session…</p>
          <p className="text-sm mt-1" style={{ color: t.textMuted }}>Loading your workspace</p>
        </div>
      </div>
    );
  }

  const NAV_ITEMS = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart2 },
    { id: 'history', label: 'Query History', icon: Clock },
    { id: 'visualizations', label: 'Chart Gallery', icon: TrendingUp },
  ];

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: t.bg, fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
      {/* ── Sidebar ── */}
      {sidebarOpen && (
        <div className="w-60 flex flex-col shrink-0"
          style={{ background: t.sidebar, borderRight: `1px solid ${t.sidebarBorder}` }}>
          {/* Logo */}
          <div className="px-5 py-4 shrink-0" style={{ borderBottom: `1px solid ${t.sidebarBorder}` }}>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center shadow-lg"
                style={{ background: t.logoGradient }}>
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-sm" style={{ color: t.text }}>IntelliQuery</h1>
                <p className="text-xs" style={{ color: t.textMuted }}>Business Intelligence</p>
              </div>
            </div>
          </div>

          {/* Nav */}
          <nav className="flex-1 p-3 overflow-y-auto space-y-0.5">
            <button onClick={createNewConversation}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-sm font-semibold mb-4 transition-all"
              style={{ background: t.accent, color: '#fff' }}>
              <Plus className="w-4 h-4" /> New Chat
            </button>

            <p className="text-xs font-bold uppercase tracking-wider px-2 mb-2"
              style={{ color: t.textMuted }}>Navigation</p>
            {NAV_ITEMS.map(item => {
              const active = activeNav === item.id;
              return (
                <button key={item.id} onClick={() => setActiveNav(item.id)}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-sm transition-all"
                  style={{
                    background: active ? t.sidebarBgActive : 'transparent',
                    color: active ? t.sidebarTextActive : t.sidebarText,
                    fontWeight: active ? '600' : '400',
                  }}>
                  <item.icon className="w-4 h-4 shrink-0" />
                  {item.label}
                  {active && <div className="ml-auto w-1.5 h-1.5 rounded-full" style={{ background: t.sidebarTextActive }} />}
                </button>
              );
            })}

            <p className="text-xs font-bold uppercase tracking-wider px-2 mt-5 mb-2"
              style={{ color: t.textMuted }}>Conversations</p>
            <div className="space-y-0.5">
              {conversations
                .sort((a, b) => new Date(b.lastUpdated) - new Date(a.lastUpdated))
                .map(conv => (
                  <div key={conv.id} className="group relative">
                    {editingConvId === conv.id ? (
                      <div className="flex items-center gap-1 px-2 py-1.5 rounded-xl"
                        style={{ background: t.sidebarBgActive }}>
                        <input value={editTitle} onChange={e => setEditTitle(e.target.value)}
                          onKeyDown={e => e.key === 'Enter' && saveTitle(conv.id)}
                          className="flex-1 px-2 py-0.5 text-xs rounded-lg focus:outline-none"
                          style={{ background: t.surface, color: t.text, border: `1px solid ${t.accent}` }}
                          autoFocus />
                        <button onClick={() => saveTitle(conv.id)} style={{ color: t.success || '#10b981' }}>
                          <Check className="w-3 h-3" />
                        </button>
                        <button onClick={() => setEditingConvId(null)} style={{ color: t.danger || '#ef4444' }}>
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ) : (
                      <button onClick={() => setActiveConvId(conv.id)}
                        className="w-full flex items-center gap-2 px-3 py-2 rounded-xl text-sm transition-all"
                        style={{
                          background: activeConvId === conv.id ? t.sidebarBgActive : 'transparent',
                          color: activeConvId === conv.id ? t.sidebarTextActive : t.sidebarText,
                          fontWeight: activeConvId === conv.id ? '600' : '400',
                        }}>
                        <MessageSquare className="w-3.5 h-3.5 shrink-0" />
                        <span className="truncate flex-1 text-left text-xs">{conv.title}</span>
                        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button onClick={e => { e.stopPropagation(); startEdit(conv.id, conv.title); }}
                            className="p-0.5 rounded hover:opacity-70" style={{ color: t.textMuted }}>
                            <Edit2 className="w-2.5 h-2.5" />
                          </button>
                          {conversations.length > 1 && (
                            <button onClick={e => { e.stopPropagation(); deleteConversation(conv.id); }}
                              className="p-0.5 rounded hover:opacity-70" style={{ color: t.danger || '#ef4444' }}>
                              <Trash2 className="w-2.5 h-2.5" />
                            </button>
                          )}
                        </div>
                      </button>
                    )}
                  </div>
                ))}
            </div>
          </nav>

          {/* Sidebar footer */}
          <div className="p-3 shrink-0 space-y-1" style={{ borderTop: `1px solid ${t.sidebarBorder}` }}>
            <ThemeSelector />
            <div className="flex items-center gap-2.5 px-3 py-2 rounded-xl"
              style={{ background: t.sidebarBgActive }}>
              <div className="w-7 h-7 rounded-full flex items-center justify-center shrink-0"
                style={{ background: t.accent }}>
                <span className="text-white text-xs font-bold">
                  {user.full_name?.charAt(0).toUpperCase()}
                </span>
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-semibold truncate" style={{ color: t.sidebarTextActive }}>{user.full_name}</p>
                <p className="text-xs truncate" style={{ color: t.textMuted }}>{user.email}</p>
              </div>
              <button onClick={onLogout} className="p-1 rounded-lg hover:opacity-70 shrink-0"
                style={{ color: t.danger || '#ef4444' }} title="Sign out">
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Main Area ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex items-center justify-between px-5 py-3 border-b shrink-0"
          style={{ background: t.header, borderColor: t.headerBorder }}>
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1.5 rounded-lg transition-all hover:opacity-70"
              style={{ color: t.textMuted }}>
              <Menu className="w-5 h-5" />
            </button>
            <div>
              <h2 className="text-sm font-bold" style={{ color: t.text }}>
                {activeNav === 'history' ? 'Query History' : activeNav === 'visualizations' ? 'Chart Gallery' : 'Analytics Dashboard'}
              </h2>
              <p className="text-xs" style={{ color: t.textMuted }}>
                {activeConv?.title !== 'New Conversation' ? activeConv?.title : 'Ready for insights'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-full"
              style={{ background: t.accentLight, color: t.textSub }}>
              <Clock className="w-3 h-3" />
              {new Date().toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex flex-1 overflow-hidden">
          {activeNav === 'history' && (
            <HistoryTab messages={activeConv?.messages || []} onClickQuery={handleHistoryClick} />
          )}
          {activeNav === 'visualizations' && (
            <VisualizationHistory
              charts={activeConv?.charts || []}
              onRestoreChart={async (chart) => {
                try {
                  const token = sessionStorage.getItem('token');
                  await fetch(`${API_URL}/dashboard/add-chart`, {
                    method: 'POST',
                    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      session_id: `session-${activeConvId}`,
                      query: chart.query || chart.title || '',
                      sql: '',
                      result: { success: true, data: chart.data || [] }
                    })
                  });
                  window.dispatchEvent(new CustomEvent('refreshDashboard'));
                } catch { }
              }}
              onClearHistory={() => setConversations(prev => prev.map(c =>
                c.id === activeConvId ? { ...c, charts: [] } : c
              ))}
            />
          )}

          {/* Dashboard — always mounted, hidden when not active */}
          <div className={`flex-1 overflow-hidden${activeNav === 'dashboard' ? '' : ' hidden'}`}>
            <DashboardContainer
              sessionId={`session-${activeConvId}`}
              onChartsLoaded={handleChartsLoaded}
              onChartBuilderAdded={handleChartBuilderAdded}
            />
          </div>

          {/* Chat panel */}
          <div className="w-96 shrink-0 p-4 overflow-hidden" style={{ borderLeft: `1px solid ${t.border}` }}>
            <ChatPanel
              messages={activeConv?.messages || []}
              setMessages={setMessages}
              scrollToIndex={scrollToIndex}
              sessionId={`session-${activeConvId}`}
              onChartCreated={handleChartCreated}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

// ─── App Root ─────────────────────────────────────────────────────────────────
const AppInner = () => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const saved = sessionStorage.getItem('user');
    if (saved) setUser(JSON.parse(saved));
  }, []);

  const handleLogout = () => {
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('user');
    setUser(null);
  };

  if (!user) return <LoginPage onLogin={setUser} />;
  return <Dashboard user={user} onLogout={handleLogout} />;
};

export default function App() {
  return (
    <ThemeProvider>
      <AppInner />
    </ThemeProvider>
  );
}
