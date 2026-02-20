import React, { useState, useEffect, useRef } from 'react';
import { Send, LogOut, MessageSquare, Database, Clock, User, Menu, BarChart2, MessageCircle, Search, Trash2, Plus, Edit2, Check, X, Code, Copy, ChevronDown } from 'lucide-react';
import axios from 'axios';
import DashboardContainer from './components/Dashboard/DashboardContainer';

const API_URL = 'http://localhost:8000/api';

// ─── Login Page ──────────────────────────────────────────────────────
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
      const response = await axios.post(`${API_URL}/login`, { email, password });
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
      onLogin(response.data.user);
    } catch (err) {
      setError('Invalid credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-600 rounded-2xl mb-4 shadow-lg shadow-blue-600/30">
            <Database className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-white">IntelliQuery</h1>
          <p className="text-slate-400 mt-1">AI-Powered Business Intelligence</p>
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-2xl p-8 shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 bg-slate-900 border border-slate-600 text-white rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-slate-500"
                placeholder="you@intelliquery.com"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-slate-900 border border-slate-600 text-white rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-slate-500"
                placeholder="••••••••"
                required
              />
            </div>
            {error && (
              <div className="bg-red-900/40 border border-red-800 text-red-300 px-4 py-3 rounded-xl text-sm">{error}</div>
            )}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 text-white py-3 rounded-xl font-semibold disabled:opacity-50 transition-colors"
            >
              {loading ? 'Signing in...' : 'Sign In'}
            </button>
          </form>

          <div className="mt-6 pt-5 border-t border-slate-700">
            <p className="text-slate-500 text-xs text-center mb-2">Demo Accounts</p>
            <div className="flex justify-center gap-3">
              {['sameed', 'izma', 'umair'].map((name) => (
                <button
                  key={name}
                  onClick={() => {
                    setEmail(`${name}@intelliquery.com`);
                    setPassword('1234');
                  }}
                  className="text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-1.5 rounded-lg transition-colors capitalize"
                >
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

// ─── Chat Panel (right side) ─────────────────────────────────────────
const ChatPanel = ({ messages, setMessages, scrollToIndex, sessionId }) => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const messageRefs = useRef({});

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    if (scrollToIndex !== null && messageRefs.current[scrollToIndex]) {
      messageRefs.current[scrollToIndex].scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [scrollToIndex]);

  const sendQuery = async () => {
    if (!input.trim() || loading) return;
    const userMsg = { type: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${API_URL}/query`,
        { question: input, session_id: sessionId || 'default' },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      const aiMsg = {
        type: 'ai',
        content: response.data.explanation || 'Query processed successfully.',
        sql: response.data.sql,
        results: response.data.results || [],
        execution_time: response.data.execution_time,
        chart: response.data.chart,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiMsg]);

      if (response.data.chart) {
        console.log('📊 Chart created, triggering dashboard refresh');
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('refreshDashboard'));
        }, 500);
      }

    } catch (error) {
      setMessages(prev => [
        ...prev,
        { type: 'error', content: error.response?.data?.detail || 'Query failed.', timestamp: new Date() }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="bg-blue-600 px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
            <MessageCircle className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-white text-sm font-semibold">AI Assistant</p>
            <p className="text-blue-200 text-xs">Ask me about your data</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3" style={{ minHeight: 0 }}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            ref={(el) => { messageRefs.current[idx] = el; }}
            className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[92%] rounded-lg px-3 py-2 text-sm ${msg.type === 'user'
                ? 'bg-blue-600 text-white'
                : msg.type === 'error'
                  ? 'bg-red-50 border border-red-200 text-red-700'
                  : 'bg-gray-50 border border-gray-200 text-gray-800'
                }`}
            >
              {/* Text response */}
              <p>{msg.content}</p>

              {/* Table Display */}
              {msg.results && msg.results.length > 0 && (
                <div className="mt-2 overflow-x-auto">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="bg-gray-100">
                        {Object.keys(msg.results[0]).map((key) => (
                          <th key={key} className="text-left px-2 py-1 font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {msg.results.map((row, i) => (
                        <tr key={i} className="border-b border-gray-100">
                          {Object.values(row).map((val, j) => (
                            <td key={j} className="px-2 py-1 text-gray-700 whitespace-nowrap">
                              {val !== null ? String(val) : 'NULL'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Chart notification */}
              {msg.chart && (
                <p className="text-xs text-blue-600 mt-2 italic">
                  📊 Visualization added to dashboard
                </p>
              )}

              {/* Execution time */}
              {msg.execution_time && msg.execution_time > 0 && (
                <p className="text-xs text-gray-400 mt-1">Executed in {msg.execution_time.toFixed(2)}s</p>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 flex items-center gap-2">
              <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600"></div>
              <span className="text-xs text-gray-500">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-gray-100 bg-gray-50 shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendQuery()}
            placeholder="Ask about your data..."
            disabled={loading}
            className="flex-1 px-3 py-2 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 bg-white"
          />
          <button
            onClick={sendQuery}
            disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 text-white px-3 py-2 rounded-lg disabled:opacity-50 transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        <div className="flex gap-1.5 mt-2 flex-wrap">
          {['Top 10 customers', 'Monthly sales trends', 'Products low in stock'].map((q) => (
            <button
              key={q}
              onClick={() => setInput(q)}
              className="text-xs bg-white border border-gray-200 text-gray-500 px-2 py-0.5 rounded-full hover:bg-gray-50 transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

// ─── Conversation History Tab with Expandable SQL ────────────────────────────────────────
const HistoryTab = ({ messages, onClickQuery }) => {
  const [search, setSearch] = useState('');
  const [expandedSql, setExpandedSql] = useState(null);

  const queryHistory = messages
    .map((msg, idx) => ({ ...msg, index: idx }))
    .filter((msg) => msg.type === 'user')
    .filter((msg) => msg.content.toLowerCase().includes(search.toLowerCase()))
    .reverse();

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert('SQL copied to clipboard!');
  };

  const toggleSql = (index) => {
    setExpandedSql(expandedSql === index ? null : index);
  };

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-3xl mx-auto">
        <div className="mb-6">
          <h3 className="text-lg font-bold text-gray-900">Conversation History</h3>
          <p className="text-sm text-gray-400 mt-0.5">All queries in this conversation</p>
        </div>

        <div className="relative mb-5">
          <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search queries..."
            className="w-full pl-9 pr-4 py-2.5 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
          />
        </div>

        {queryHistory.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center mx-auto mb-4">
              <Clock className="w-7 h-7 text-gray-300" />
            </div>
            <p className="text-sm text-gray-500">
              {search ? 'No queries match your search.' : 'No queries yet. Ask something in the chat!'}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {queryHistory.map((msg, i) => {
              const aiResponse = messages[msg.index + 1];
              const hasResults = aiResponse?.results && aiResponse.results.length > 0;
              const timeStr = new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
              const isSqlExpanded = expandedSql === msg.index;

              return (
                <div
                  key={msg.index}
                  className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow"
                >
                  {/* Query Header */}
                  <div className="p-4">
                    <div className="flex items-start justify-between gap-3 mb-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs font-bold bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full">
                            #{queryHistory.length - i}
                          </span>
                          <span className="text-xs text-gray-400">{timeStr}</span>
                          {hasResults && (
                            <span className="text-xs bg-green-50 text-green-600 px-2 py-0.5 rounded-full">
                              {aiResponse.results.length} rows
                            </span>
                          )}
                          {aiResponse?.chart && (
                            <span className="text-xs bg-purple-50 text-purple-600 px-2 py-0.5 rounded-full">
                              📊 Chart
                            </span>
                          )}
                        </div>
                        <p className="text-sm font-semibold text-gray-900">{msg.content}</p>
                      </div>
                      <button
                        onClick={() => onClickQuery(msg.index)}
                        className="text-gray-400 hover:text-blue-500 transition-colors shrink-0"
                        title="Jump to query"
                      >
                        <svg width="20" height="20" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M6 3l5 5-5 5" />
                        </svg>
                      </button>
                    </div>

                    {/* SQL Section */}
                    {aiResponse?.sql && (
                      <div className="mt-3">
                        <div className="flex items-center justify-between mb-2">
                          <button
                            onClick={() => toggleSql(msg.index)}
                            className="flex items-center gap-2 text-xs font-medium text-gray-600 hover:text-blue-600 transition-colors"
                          >
                            <Code className="w-3.5 h-3.5" />
                            <span>{isSqlExpanded ? 'Hide SQL' : 'View SQL'}</span>
                            <ChevronDown className={`w-3.5 h-3.5 transition-transform ${isSqlExpanded ? 'rotate-180' : ''}`} />
                          </button>
                          {isSqlExpanded && (
                            <button
                              onClick={() => copyToClipboard(aiResponse.sql)}
                              className="flex items-center gap-1 text-xs font-medium text-gray-600 hover:text-blue-600 transition-colors"
                            >
                              <Copy className="w-3.5 h-3.5" />
                              <span>Copy</span>
                            </button>
                          )}
                        </div>

                        {/* Expandable SQL */}
                        {isSqlExpanded && (
                          <div className="bg-gray-900 rounded-lg p-3 overflow-x-auto">
                            <pre className="text-xs text-green-400 font-mono whitespace-pre-wrap">
                              {aiResponse.sql}
                            </pre>
                          </div>
                        )}

                        {/* SQL Preview (when collapsed) */}
                        {!isSqlExpanded && (
                          <div className="bg-gray-50 rounded px-3 py-2">
                            <p className="text-xs text-gray-500 font-mono truncate">
                              {aiResponse.sql}
                            </p>
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

// ─── Dashboard Layout ────────────────────────────────────────────────
const Dashboard = ({ user, onLogout }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeNav, setActiveNav] = useState('dashboard');
  const [conversations, setConversations] = useState([
    {
      id: 1,
      title: 'New Conversation',
      messages: [
        {
          type: 'ai',
          content: "Hello! I'm IntelliQuery AI. Ask me anything about your data — I can run queries, generate insights, and explain results.",
          timestamp: new Date()
        }
      ],
      lastUpdated: new Date()
    }
  ]);
  const [activeConvId, setActiveConvId] = useState(1);
  const [scrollToIndex, setScrollToIndex] = useState(null);
  const [editingConvId, setEditingConvId] = useState(null);
  const [editTitle, setEditTitle] = useState('');

  const activeConv = conversations.find(c => c.id === activeConvId);

  const setMessages = (updater) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === activeConvId) {
        const newMessages = typeof updater === 'function' ? updater(conv.messages) : updater;
        let newTitle = conv.title;
        if (newTitle === 'New Conversation') {
          const firstUserMsg = newMessages.find(m => m.type === 'user');
          if (firstUserMsg) {
            newTitle = firstUserMsg.content.slice(0, 40) + (firstUserMsg.content.length > 40 ? '...' : '');
          }
        }
        return { ...conv, messages: newMessages, lastUpdated: new Date(), title: newTitle };
      }
      return conv;
    }));
  };

  const createNewConversation = () => {
    const newId = Math.max(...conversations.map(c => c.id)) + 1;
    const newConv = {
      id: newId,
      title: 'New Conversation',
      messages: [
        {
          type: 'ai',
          content: "Hello! I'm IntelliQuery AI. Ask me anything about your data — I can run queries, generate insights, and explain results.",
          timestamp: new Date()
        }
      ],
      lastUpdated: new Date()
    };
    setConversations(prev => [...prev, newConv]);
    setActiveConvId(newId);
    setActiveNav('dashboard');
  };

  const deleteConversation = (convId) => {
    if (conversations.length === 1) return;
    setConversations(prev => prev.filter(c => c.id !== convId));
    if (activeConvId === convId) {
      setActiveConvId(conversations.find(c => c.id !== convId).id);
    }
  };

  const startEditTitle = (convId, currentTitle) => {
    setEditingConvId(convId);
    setEditTitle(currentTitle);
  };

  const saveTitle = (convId) => {
    setConversations(prev => prev.map(c => c.id === convId ? { ...c, title: editTitle.trim() || 'New Conversation' } : c));
    setEditingConvId(null);
  };

  const handleHistoryClick = (msgIndex) => {
    setActiveNav('dashboard');
    setScrollToIndex(msgIndex);
  };

  useEffect(() => {
    if (scrollToIndex !== null) {
      const timer = setTimeout(() => setScrollToIndex(null), 500);
      return () => clearTimeout(timer);
    }
  }, [scrollToIndex]);

  return (
    <div className="flex h-screen bg-gray-50" style={{ fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
      {/* Sidebar */}
      {sidebarOpen && (
        <div className="w-56 bg-white border-r border-gray-200 flex flex-col shrink-0">
          <div className="p-5 border-b border-gray-100">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Database className="w-4 h-4 text-white" />
              </div>
              <h1 className="text-base font-bold text-gray-900">IntelliQuery</h1>
            </div>
            <p className="text-xs text-gray-400 mt-1 ml-10">Business Intelligence</p>
          </div>

          <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto">
            <button
              onClick={createNewConversation}
              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors bg-blue-600 hover:bg-blue-500 text-white font-semibold mb-3"
            >
              <Plus className="w-4 h-4" />
              New Chat
            </button>

            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-3 mb-2 mt-1">Main</p>
            {[
              { id: 'dashboard', label: 'Dashboard', icon: BarChart2 },
              { id: 'history', label: 'Conversation History', icon: Clock },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveNav(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${activeNav === item.id
                  ? 'bg-blue-50 text-blue-600 font-semibold'
                  : 'text-gray-600 hover:bg-gray-50'
                  }`}
              >
                <item.icon className="w-4 h-4" />
                {item.label}
              </button>
            ))}

            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-3 mb-2 mt-4">Chats</p>
            <div className="space-y-0.5">
              {conversations.sort((a, b) => b.lastUpdated - a.lastUpdated).map((conv) => (
                <div key={conv.id} className="group relative">
                  {editingConvId === conv.id ? (
                    <div className="flex items-center gap-1 px-2 py-1">
                      <input
                        type="text"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && saveTitle(conv.id)}
                        className="flex-1 px-2 py-1 text-xs border border-blue-500 rounded focus:outline-none"
                        autoFocus
                      />
                      <button onClick={() => saveTitle(conv.id)} className="text-green-600 hover:text-green-700">
                        <Check className="w-3 h-3" />
                      </button>
                      <button onClick={() => setEditingConvId(null)} className="text-red-600 hover:text-red-700">
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setActiveConvId(conv.id)}
                      className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${activeConvId === conv.id
                        ? 'bg-blue-50 text-blue-600 font-semibold'
                        : 'text-gray-600 hover:bg-gray-50'
                        }`}
                    >
                      <MessageSquare className="w-3.5 h-3.5 shrink-0" />
                      <span className="truncate flex-1 text-left">{conv.title}</span>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => { e.stopPropagation(); startEditTitle(conv.id, conv.title); }}
                          className="text-gray-400 hover:text-gray-600 p-0.5"
                        >
                          <Edit2 className="w-3 h-3" />
                        </button>
                        {conversations.length > 1 && (
                          <button
                            onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id); }}
                            className="text-gray-400 hover:text-red-600 p-0.5"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                    </button>
                  )}
                </div>
              ))}
            </div>
          </nav>

          <div className="p-3 border-t border-gray-100">
            <div className="flex items-center gap-3 px-3 py-2 mb-1">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <User className="w-4 h-4 text-blue-600" />
              </div>
              <div>
                <p className="text-sm font-semibold text-gray-900">{user.full_name}</p>
                <p className="text-xs text-gray-400">{user.email}</p>
              </div>
            </div>
            <button
              onClick={onLogout}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-500 hover:bg-red-50 rounded-lg transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 flex flex-col min-w-0">
        <div className="bg-white border-b border-gray-200 px-5 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-gray-400 hover:text-gray-600 p-1 rounded-lg hover:bg-gray-50"
            >
              <Menu className="w-5 h-5" />
            </button>
            <h2 className="text-lg font-bold text-gray-900">
              {activeNav === 'history' ? 'Conversation History' : 'Dashboard'}
            </h2>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Clock className="w-3.5 h-3.5" />
            <span>{new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}</span>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {activeNav === 'history' ? (
            <HistoryTab messages={activeConv?.messages || []} onClickQuery={handleHistoryClick} />
          ) : (
            <div className="flex-1 overflow-hidden">
              <DashboardContainer sessionId={`session-${activeConvId}`} />
            </div>
          )}

          <div className="w-96 shrink-0 p-4 overflow-hidden">
            <ChatPanel
              messages={activeConv?.messages || []}
              setMessages={setMessages}
              scrollToIndex={scrollToIndex}
              sessionId={`session-${activeConvId}`}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
  };

  if (!user) return <LoginPage onLogin={setUser} />;
  return <Dashboard user={user} onLogout={handleLogout} />;
}