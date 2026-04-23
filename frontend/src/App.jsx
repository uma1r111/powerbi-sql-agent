import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Send, LogOut, MessageSquare, Database, Clock, User, Menu,
  BarChart2, Search, Trash2, Plus, Edit2, Check, X,
  Code, Copy, ChevronDown, Palette, Sparkles, TrendingUp,
  Activity, ChevronRight, Eye, EyeOff, Lock, Mail, Bell, BookOpen,
  Shield, FileText
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

// ─── Login Page — Desktop Split-Screen ───────────────────────────────────────
const LoginPage = ({ onLogin }) => {
  const [tab, setTab] = useState('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [showConfirmPwd, setShowConfirmPwd] = useState(false);

  const handleSignIn = async (e) => {
    e.preventDefault();
    setLoading(true); setError('');
    try {
      const res = await axios.post(`${API_URL}/login`, { email, password });
      sessionStorage.setItem('token', res.data.token);
      sessionStorage.setItem('user', JSON.stringify(res.data.user));
      onLogin(res.data.user);
    } catch { setError('Invalid credentials. Please check your email and password.'); }
    finally { setLoading(false); }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) { setError('Passwords do not match.'); return; }
    if (password.length < 4) { setError('Password must be at least 4 characters.'); return; }
    setLoading(true); setError('');
    try {
      await axios.post(`${API_URL}/register`, { email, password, full_name: fullName });
      setTab('signin');
      setSuccessMsg('Account created! Sign in with your credentials.');
      setPassword(''); setConfirmPassword('');
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed. Please try again.');
    } finally { setLoading(false); }
  };

  const FEATURES = [
    { icon: MessageSquare, text: 'Natural language to SQL — ask in plain English', col: '#6366f1' },
    { icon: BarChart2,     text: '12+ interactive chart types with cross-filtering', col: '#d4af37' },
    { icon: Database,      text: 'Live database analysis across all your tables',    col: '#06b6d4' },
    { icon: BookOpen,      text: 'RAG-powered document Q&A with source citations',  col: '#10b981' },
  ];

  const STATS = [
    { label: 'Revenue', val: '$2.4M', chg: '+12%', chgCol: '#10b981' },
    { label: 'Orders',  val: '1,847', chg: '+8%',  chgCol: '#10b981' },
    { label: 'Queries', val: '98.4%', chg: 'Acc.', chgCol: '#d4af37' },
  ];

  // Shared styles
  const fieldBox = {
    display: 'flex', alignItems: 'center', gap: '12px',
    background: 'rgba(255,255,255,0.04)',
    border: '1px solid rgba(212,175,55,0.14)',
    borderRadius: '14px', padding: '0 18px',
    transition: 'border-color .2s, box-shadow .2s',
  };
  const fieldInput = {
    flex: 1, background: 'transparent', border: 'none', outline: 'none',
    color: '#e8d5a3', fontSize: '15px', padding: '15px 0',
    caretColor: '#d4af37',
  };
  const fieldIcon = { width: '17px', height: '17px', color: 'rgba(212,175,55,0.45)', flexShrink: 0 };

  return (
    <div style={{
      minHeight: '100vh', display: 'flex',
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
      background: '#080d1a',
    }}>
      <style>{`
        @keyframes iqShimmer { 0%{background-position:-200% center} 100%{background-position:200% center} }
        @keyframes iqPulse { 0%,100%{opacity:.22;transform:scale(1)} 50%{opacity:.46;transform:scale(1.04)} }
        @keyframes iqFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
        @keyframes iqFadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
        .iq-form-card { animation: iqFadeUp .5s ease forwards; }
        .iq-shimmer-btn {
          background: linear-gradient(90deg,#a07820 0%,#d4af37 25%,#f5e060 50%,#d4af37 75%,#a07820 100%);
          background-size: 200% auto; transition: all .3s ease;
        }
        .iq-shimmer-btn:hover:not(:disabled) {
          animation: iqShimmer 1.4s linear infinite;
          box-shadow: 0 12px 40px rgba(212,175,55,.45), 0 0 0 1px rgba(212,175,55,.25);
          transform: translateY(-2px);
        }
        .iq-shimmer-btn:active:not(:disabled) { transform: translateY(0); }
        .iq-field:focus-within {
          border-color: rgba(212,175,55,.4) !important;
          box-shadow: 0 0 0 3px rgba(212,175,55,.07);
        }
        .iq-demo:hover { background: rgba(212,175,55,.15) !important; transform: translateY(-1px); }
        .iq-tab:hover { color: rgba(212,175,55,.9) !important; }
        .iq-link:hover { color: #d4af37 !important; }
      `}</style>

      {/* ══ LEFT PANEL ══════════════════════════════════════════════════════ */}
      <div style={{
        width: '480px', flexShrink: 0,
        background: 'linear-gradient(160deg,#0a0f1e 0%,#080d18 100%)',
        borderRight: '1px solid rgba(212,175,55,0.07)',
        padding: '56px 48px',
        display: 'flex', flexDirection: 'column',
        position: 'relative', overflow: 'hidden',
      }}>
        {/* Decorative grid */}
        <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0, opacity: 0.03, pointerEvents: 'none' }}>
          <defs><pattern id="lgrid" width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M50 0L0 0 0 50" fill="none" stroke="#d4af37" strokeWidth=".7"/>
          </pattern></defs>
          <rect width="100%" height="100%" fill="url(#lgrid)" />
        </svg>
        {/* Glow blobs */}
        <div style={{ position:'absolute',top:'10%',left:'-10%',width:'400px',height:'400px',borderRadius:'50%',background:'radial-gradient(circle,rgba(25,55,160,.1) 0%,transparent 70%)',animation:'iqPulse 10s ease-in-out infinite',pointerEvents:'none' }} />
        <div style={{ position:'absolute',bottom:'5%',right:'-10%',width:'320px',height:'320px',borderRadius:'50%',background:'radial-gradient(circle,rgba(212,175,55,.07) 0%,transparent 70%)',animation:'iqPulse 13s ease-in-out infinite 3s',pointerEvents:'none' }} />

        {/* Logo + Brand */}
        <div style={{ marginBottom: '40px' }}>
          <div style={{
            width: '80px', height: '80px', borderRadius: '24px',
            background: 'linear-gradient(145deg,#c29020 0%,#e8c840 40%,#f5e060 55%,#d4af37 75%,#a07820 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px',
            boxShadow: '0 16px 48px rgba(212,175,55,.38),0 0 0 1px rgba(212,175,55,.18),inset 0 1px 0 rgba(255,255,255,.3)',
            animation: 'iqFloat 5.5s ease-in-out infinite',
            position: 'relative',
          }}>
            <svg width="50" height="37" viewBox="0 0 50 37" fill="none">
              <rect x="1" y="1" width="8" height="35" rx="3" fill="#0a0f1e"/>
              <circle cx="35" cy="18" r="14" stroke="#0a0f1e" strokeWidth="7" fill="none"/>
              <line x1="44" y1="27" x2="51" y2="36" stroke="#0a0f1e" strokeWidth="7" strokeLinecap="round"/>
            </svg>
            <div style={{ position:'absolute',top:'8px',right:'9px',width:'6px',height:'6px',borderRadius:'50%',background:'rgba(255,255,255,.5)' }} />
          </div>

          <h1 style={{
            fontSize: '30px', fontWeight: '900', letterSpacing: '6px',
            color: '#e8d595', margin: '0 0 8px',
            textShadow: '0 0 40px rgba(212,175,55,.25)',
          }}>INTELLIQUERY</h1>
          <p style={{ fontSize: '12px', letterSpacing: '2.5px', color: 'rgba(190,160,100,.55)', margin: 0, textTransform: 'uppercase' }}>
            AI-Driven Analytics Suite
          </p>
        </div>

        {/* Tagline */}
        <p style={{ fontSize: '16px', lineHeight: '1.75', color: 'rgba(200,180,140,.65)', marginBottom: '36px' }}>
          Transform natural language into actionable data insights with enterprise-grade AI analytics.
        </p>

        {/* Features */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px', marginBottom: '40px' }}>
          {FEATURES.map(({ icon: Icon, text, col }) => (
            <div key={text} style={{ display: 'flex', alignItems: 'flex-start', gap: '14px' }}>
              <div style={{
                width: '36px', height: '36px', borderRadius: '10px', flexShrink: 0,
                background: `${col}18`, border: `1px solid ${col}28`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Icon style={{ width: '16px', height: '16px', color: col }} />
              </div>
              <p style={{ fontSize: '14px', color: 'rgba(210,185,140,.7)', lineHeight: '1.5', margin: '6px 0 0' }}>{text}</p>
            </div>
          ))}
        </div>

        {/* Mini KPI preview */}
        <div style={{
          borderRadius: '16px', padding: '18px',
          background: 'rgba(255,255,255,.03)', border: '1px solid rgba(212,175,55,.08)',
          marginBottom: '12px',
        }}>
          <p style={{ fontSize: '11px', color: 'rgba(190,160,100,.4)', letterSpacing: '1.5px', textTransform: 'uppercase', marginBottom: '14px' }}>
            Live Analytics Preview
          </p>
          <div style={{ display: 'flex', gap: '10px', marginBottom: '14px' }}>
            {STATS.map(({ label, val, chg, chgCol }) => (
              <div key={label} style={{
                flex: 1, borderRadius: '12px', padding: '12px',
                background: 'rgba(255,255,255,.04)', border: '1px solid rgba(255,255,255,.05)',
              }}>
                <p style={{ fontSize: '11px', color: 'rgba(200,180,140,.45)', margin: '0 0 4px' }}>{label}</p>
                <p style={{ fontSize: '18px', fontWeight: '700', color: '#e8d595', margin: '0 0 2px' }}>{val}</p>
                <p style={{ fontSize: '11px', fontWeight: '600', color: chgCol, margin: 0 }}>{chg}</p>
              </div>
            ))}
          </div>
          {/* Mini bar chart */}
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '3px', height: '36px' }}>
            {[55,72,48,88,65,80,58,92,70,85,78,100].map((h, i) => (
              <div key={i} style={{
                flex: 1, borderRadius: '3px 3px 0 0',
                height: `${h}%`,
                background: `rgba(212,175,55,${.2 + h / 400})`,
              }} />
            ))}
          </div>
        </div>

        <div style={{ flex: 1 }} />
        <p style={{ fontSize: '12px', color: 'rgba(140,115,70,.3)', marginTop: '24px' }}>
          © 2026 IntelliQuery BI Suite
        </p>
      </div>

      {/* ══ RIGHT PANEL ═════════════════════════════════════════════════════ */}
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '48px 40px',
        background: 'linear-gradient(160deg,#0c1525 0%,#080d1a 100%)',
        position: 'relative', overflow: 'hidden',
      }}>
        {/* Subtle right-panel glow */}
        <div style={{ position:'absolute',top:'30%',right:'20%',width:'300px',height:'300px',borderRadius:'50%',background:'radial-gradient(circle,rgba(80,100,220,.06) 0%,transparent 70%)',pointerEvents:'none' }} />

        <div className="iq-form-card" style={{ width: '100%', maxWidth: '460px' }}>
          {/* Card */}
          <div style={{
            background: 'rgba(10,18,42,.78)',
            backdropFilter: 'blur(32px)', WebkitBackdropFilter: 'blur(32px)',
            border: '1px solid rgba(212,175,55,.14)',
            borderRadius: '28px', padding: '44px 44px 36px',
            boxShadow: '0 32px 100px rgba(0,0,0,.65),inset 0 1px 0 rgba(212,175,55,.1),inset 0 -1px 0 rgba(0,0,0,.4)',
          }}>
            <h2 style={{
              fontSize: '22px', fontWeight: '700', color: '#e8d5a3',
              margin: '0 0 6px', letterSpacing: '.3px',
            }}>
              {tab === 'signin' ? 'Welcome back' : 'Create your account'}
            </h2>
            <p style={{ fontSize: '14px', color: 'rgba(190,160,100,.5)', margin: '0 0 28px' }}>
              {tab === 'signin' ? 'Sign in to your IntelliQuery workspace' : 'Get started with IntelliQuery today'}
            </p>

            {/* Tab switcher */}
            <div style={{
              display: 'flex', background: 'rgba(255,255,255,.03)',
              border: '1px solid rgba(212,175,55,.1)', borderRadius: '14px',
              padding: '4px', marginBottom: '26px',
            }}>
              {[{ id: 'signin', label: 'Sign In' }, { id: 'register', label: 'Request Access' }].map(({ id, label }) => (
                <button key={id} className="iq-tab"
                  onClick={() => { setTab(id); setError(''); setSuccessMsg(''); }}
                  style={{
                    flex: 1, padding: '10px 14px', borderRadius: '10px',
                    fontSize: '13px', fontWeight: '600', cursor: 'pointer', border: 'none',
                    background: tab === id
                      ? 'linear-gradient(135deg,rgba(212,175,55,.22) 0%,rgba(212,175,55,.10) 100%)'
                      : 'transparent',
                    color: tab === id ? '#d4af37' : 'rgba(180,150,90,.45)',
                    boxShadow: tab === id ? '0 0 0 1px rgba(212,175,55,.2)' : 'none',
                    transition: 'all .2s',
                  }}>
                  {label}
                </button>
              ))}
            </div>

            {/* Alerts */}
            {successMsg && (
              <div style={{ background:'rgba(16,185,129,.09)',border:'1px solid rgba(16,185,129,.22)',borderRadius:'12px',padding:'12px 16px',fontSize:'14px',color:'#34d399',marginBottom:'18px' }}>
                {successMsg}
              </div>
            )}
            {error && (
              <div style={{ background:'rgba(239,68,68,.08)',border:'1px solid rgba(239,68,68,.2)',borderRadius:'12px',padding:'12px 16px',fontSize:'14px',color:'#f87171',marginBottom:'18px' }}>
                {error}
              </div>
            )}

            {/* Sign In form */}
            {tab === 'signin' && (
              <form onSubmit={handleSignIn} style={{ display:'flex',flexDirection:'column',gap:'14px' }}>
                <div className="iq-field" style={fieldBox}>
                  <Mail style={fieldIcon} />
                  <input type="email" value={email} onChange={e=>setEmail(e.target.value)} required
                    placeholder="your@email.com" style={fieldInput} />
                </div>
                <div className="iq-field" style={fieldBox}>
                  <Lock style={fieldIcon} />
                  <input type={showPwd?'text':'password'} value={password} onChange={e=>setPassword(e.target.value)} required
                    placeholder="Enter your password" style={fieldInput} />
                  <button type="button" onClick={()=>setShowPwd(!showPwd)}
                    style={{ background:'none',border:'none',cursor:'pointer',padding:'4px',color:'rgba(212,175,55,.38)',display:'flex' }}>
                    {showPwd ? <EyeOff style={{width:'16px',height:'16px'}}/> : <Eye style={{width:'16px',height:'16px'}}/>}
                  </button>
                </div>
                <div style={{ textAlign:'right',marginTop:'-4px' }}>
                  <button type="button" className="iq-link"
                    style={{ background:'none',border:'none',cursor:'pointer',color:'rgba(212,175,55,.44)',fontSize:'13px',transition:'color .2s' }}>
                    Forgot Password?
                  </button>
                </div>
                <button type="submit" disabled={loading} className="iq-shimmer-btn"
                  style={{
                    width:'100%',padding:'16px',border:'none',borderRadius:'14px',
                    fontSize:'14px',fontWeight:'700',letterSpacing:'1.5px',textTransform:'uppercase',
                    color:'#0a0f1e',cursor:loading?'not-allowed':'pointer',opacity:loading?.65:1,marginTop:'6px',
                  }}>
                  {loading?'Authenticating…':'Sign In'}
                </button>
                <p style={{ textAlign:'center',fontSize:'13px',color:'rgba(180,150,90,.4)',margin:'6px 0 0' }}>
                  Don't have an account?{' '}
                  <button type="button" onClick={()=>setTab('register')} className="iq-link"
                    style={{ background:'none',border:'none',cursor:'pointer',color:'rgba(212,175,55,.7)',fontWeight:'600',fontSize:'13px',transition:'color .2s' }}>
                    Request Access
                  </button>
                </p>
              </form>
            )}

            {/* Register form */}
            {tab === 'register' && (
              <form onSubmit={handleRegister} style={{ display:'flex',flexDirection:'column',gap:'14px' }}>
                <div className="iq-field" style={fieldBox}>
                  <User style={fieldIcon} />
                  <input type="text" value={fullName} onChange={e=>setFullName(e.target.value)} required
                    placeholder="Full name" style={fieldInput} />
                </div>
                <div className="iq-field" style={fieldBox}>
                  <Mail style={fieldIcon} />
                  <input type="email" value={email} onChange={e=>setEmail(e.target.value)} required
                    placeholder="your@email.com" style={fieldInput} />
                </div>
                <div className="iq-field" style={fieldBox}>
                  <Lock style={fieldIcon} />
                  <input type={showPwd?'text':'password'} value={password} onChange={e=>setPassword(e.target.value)} required
                    placeholder="Choose a password" style={fieldInput} />
                  <button type="button" onClick={()=>setShowPwd(!showPwd)}
                    style={{ background:'none',border:'none',cursor:'pointer',padding:'4px',color:'rgba(212,175,55,.38)',display:'flex' }}>
                    {showPwd ? <EyeOff style={{width:'16px',height:'16px'}}/> : <Eye style={{width:'16px',height:'16px'}}/>}
                  </button>
                </div>
                <div className="iq-field" style={fieldBox}>
                  <Lock style={fieldIcon} />
                  <input type={showConfirmPwd?'text':'password'} value={confirmPassword} onChange={e=>setConfirmPassword(e.target.value)} required
                    placeholder="Confirm password" style={fieldInput} />
                  <button type="button" onClick={()=>setShowConfirmPwd(!showConfirmPwd)}
                    style={{ background:'none',border:'none',cursor:'pointer',padding:'4px',color:'rgba(212,175,55,.38)',display:'flex' }}>
                    {showConfirmPwd ? <EyeOff style={{width:'16px',height:'16px'}}/> : <Eye style={{width:'16px',height:'16px'}}/>}
                  </button>
                </div>
                <button type="submit" disabled={loading} className="iq-shimmer-btn"
                  style={{
                    width:'100%',padding:'16px',border:'none',borderRadius:'14px',
                    fontSize:'14px',fontWeight:'700',letterSpacing:'1.5px',textTransform:'uppercase',
                    color:'#0a0f1e',cursor:loading?'not-allowed':'pointer',opacity:loading?.65:1,marginTop:'6px',
                  }}>
                  {loading?'Creating Account…':'Request Access'}
                </button>
              </form>
            )}

            {/* Demo accounts */}
            <div style={{ marginTop:'26px',paddingTop:'22px',borderTop:'1px solid rgba(212,175,55,.09)' }}>
              <p style={{ fontSize:'11px',color:'rgba(180,150,90,.35)',textAlign:'center',letterSpacing:'2px',textTransform:'uppercase',marginBottom:'13px' }}>
                Demo Accounts
              </p>
              <div style={{ display:'flex',justifyContent:'center',gap:'10px' }}>
                {['sameed','izma','umair'].map(name => (
                  <button key={name} className="iq-demo"
                    onClick={() => { setTab('signin'); setEmail(`${name}@intelliquery.com`); setPassword('1234'); }}
                    style={{
                      padding:'7px 16px',borderRadius:'10px',
                      background:'rgba(212,175,55,.07)',border:'1px solid rgba(212,175,55,.15)',
                      color:'rgba(212,175,55,.7)',fontSize:'13px',fontWeight:'600',cursor:'pointer',
                      textTransform:'capitalize',transition:'all .2s',
                    }}>
                    {name}
                  </button>
                ))}
              </div>
            </div>

            <p style={{ marginTop:'22px',textAlign:'center',fontSize:'12px',color:'rgba(140,115,70,.3)' }}>
              © 2026 IntelliQuery BI Suite &nbsp;·&nbsp; Terms &nbsp;·&nbsp; Privacy
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─── Chat Panel ───────────────────────────────────────────────────────────────
const ChatPanel = ({ messages, setMessages, scrollToIndex, sessionId, onChartCreated, hideHeader }) => {
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

  const addChartToDashboard = async (chart) => {
    try {
      const token = sessionStorage.getItem('token');
      await axios.post(`${API_URL}/dashboard/add-manual-chart`,
        { session_id: sessionId, chart },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setTimeout(() => window.dispatchEvent(new CustomEvent('refreshDashboard')), 300);
    } catch (err) { console.error('Failed to add chart:', err); }
  };

  const sendQuery = async (queryOverride) => {
    const q = queryOverride || input;
    if (!q.trim() || loading) return;
    setMessages(prev => [...prev, { type: 'user', content: q, timestamp: new Date().toISOString() }]);
    if (!queryOverride) setInput('');
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
        sql: res.data.sql, results: res.data.results || [],
        execution_time: res.data.execution_time, chart: res.data.chart,
        sessionId, timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, ai]);
      if (res.data.chart && onChartCreated)
        onChartCreated({ ...res.data.chart, query: q, created_at: new Date().toISOString() });
      if (res.data.chart?.auto_added || res.data.full_dashboard_generated)
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
    <div className="flex flex-col h-full overflow-hidden" style={{ background: t.surface }}>
      {!hideHeader && (
        <div className="px-4 py-3 shrink-0 flex items-center gap-3"
          style={{ background: `linear-gradient(135deg,${t.accent} 0%,${t.accentHover||t.accent} 100%)` }}>
          <div className="w-8 h-8 rounded-xl flex items-center justify-center" style={{ background: 'rgba(255,255,255,.2)' }}>
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-white text-sm font-bold">AI Assistant</p>
            <p className="text-xs" style={{ color: 'rgba(255,255,255,.7)' }}>Powered by Llama 3.3 · 70B</p>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-3 space-y-3" style={{ minHeight: 0 }}>
        {messages.map((msg, idx) => (
          <div key={idx} ref={el => { messageRefs.current[idx] = el; }}
            className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[90%] rounded-2xl px-3 py-2.5 text-sm ${msg.type === 'user' ? 'rounded-tr-sm' : 'rounded-tl-sm'}`}
              style={{
                background: msg.type==='user' ? t.accent : msg.type==='error' ? 'rgba(239,68,68,.08)' : t.surfaceHover||t.surface,
                color: msg.type==='user' ? '#fff' : msg.type==='error' ? '#dc2626' : t.text,
                border: msg.type==='error' ? '1px solid rgba(239,68,68,.2)' : msg.type!=='user' ? `1px solid ${t.border}` : 'none',
              }}>
              {msg.type === 'error'
                ? <div className="flex items-start gap-2"><span className="shrink-0 mt-0.5">⚠</span><p>{msg.content}</p></div>
                : <p style={{ lineHeight: '1.5' }}>{msg.content}</p>}
              {msg.results?.length > 0 && (
                <div className="mt-2 overflow-x-auto rounded-lg" style={{ border:`1px solid ${t.border}` }}>
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr style={{ background: t.surfaceHover||t.bg }}>
                        {Object.keys(msg.results[0]).map(k => (
                          <th key={k} className="text-left px-2 py-1.5 font-semibold whitespace-nowrap"
                            style={{ color: t.textSub, borderBottom: `1px solid ${t.border}` }}>{k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {msg.results.slice(0,5).map((row,i) => (
                        <tr key={i} style={{ borderBottom:`1px solid ${t.border}` }}>
                          {Object.values(row).map((val,j) => (
                            <td key={j} className="px-2 py-1.5 whitespace-nowrap" style={{ color: t.text }}>
                              {val!==null ? String(val) : '—'}
                            </td>
                          ))}
                        </tr>
                      ))}
                      {msg.results.length > 5 && (
                        <tr><td colSpan={Object.keys(msg.results[0]).length} className="px-2 py-1.5 text-center text-xs" style={{ color: t.textMuted }}>+{msg.results.length-5} more rows</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}
              {msg.chart && !msg.chart.auto_added && (
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-xs" style={{ color: t.textMuted }}>Chart ready</span>
                  <button onClick={() => addChartToDashboard(msg.chart)}
                    className="text-xs px-2 py-1 rounded-lg font-semibold"
                    style={{ background: t.accent, color: '#fff' }}>Add to Dashboard</button>
                </div>
              )}
              {msg.chart?.auto_added && (
                <p className="text-xs mt-2 flex items-center gap-1"
                  style={{ color: msg.type==='user' ? 'rgba(255,255,255,.8)' : t.accent }}>
                  <BarChart2 className="w-3 h-3" /> Added to dashboard
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
              style={{ background: t.surfaceHover||t.surface, border:`1px solid ${t.border}` }}>
              <div className="flex gap-1">
                {[0,1,2].map(i => (
                  <div key={i} className="w-1.5 h-1.5 rounded-full animate-bounce"
                    style={{ background: t.accent, animationDelay:`${i*.15}s` }} />
                ))}
              </div>
              <span className="text-xs" style={{ color: t.textMuted }}>Thinking…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="px-3 py-2 flex gap-1.5 flex-wrap shrink-0" style={{ borderTop:`1px solid ${t.border}` }}>
        {SUGGESTIONS.map(s => (
          <button key={s} onClick={() => setInput(s)}
            className="text-xs px-2.5 py-1 rounded-full transition-all hover:opacity-80"
            style={{ background: t.accentLight, color: t.accentText||t.accent, border:`1px solid ${t.border}` }}>
            {s}
          </button>
        ))}
      </div>

      <div className="px-3 pb-3 shrink-0">
        <div className="flex gap-2 items-end">
          <textarea
            value={input} onChange={e=>setInput(e.target.value)}
            onKeyDown={e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendQuery();} }}
            placeholder="Ask about your data…" rows={1} disabled={loading}
            className="flex-1 px-3 py-2.5 text-sm rounded-xl resize-none focus:outline-none focus:ring-2 transition-all disabled:opacity-50"
            style={{ background:t.surfaceHover||t.bg,border:`1px solid ${t.border}`,color:t.text,maxHeight:'100px',minHeight:'40px' }} />
          <button onClick={()=>sendQuery()} disabled={loading||!input.trim()}
            className="p-2.5 rounded-xl font-semibold transition-all disabled:opacity-40 shrink-0"
            style={{ background: t.accent, color: '#fff' }}>
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

// ─── SQL History Tab ──────────────────────────────────────────────────────────
const HistoryTab = ({ messages, onClickQuery }) => {
  const { theme: t } = useTheme();
  const [search, setSearch] = useState('');
  const [expandedSql, setExpandedSql] = useState(null);

  const history = messages
    .map((m,i) => ({...m,index:i}))
    .filter(m => m.type==='user')
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
          <input type="text" value={search} onChange={e=>setSearch(e.target.value)}
            placeholder="Search queries…"
            className="w-full pl-9 pr-4 py-2.5 text-sm rounded-xl focus:outline-none"
            style={{ background:t.surface,border:`1px solid ${t.border}`,color:t.text }} />
        </div>
        {history.length===0 ? (
          <div className="text-center py-16">
            <Clock className="w-10 h-10 mx-auto mb-3" style={{ color: t.border }} />
            <p className="text-sm" style={{ color: t.textMuted }}>
              {search ? 'No matching queries.' : 'No queries yet. Ask something in the chat!'}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {history.map((msg,i) => {
              const ai = messages[msg.index+1];
              const timeStr = new Date(msg.timestamp).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
              const expanded = expandedSql===msg.index;
              return (
                <div key={msg.index} className="rounded-xl overflow-hidden"
                  style={{ background:t.surface,border:`1px solid ${t.border}`,boxShadow:t.shadow }}>
                  <div className="p-4">
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                          <span className="text-xs font-bold px-2 py-0.5 rounded-full"
                            style={{ background:t.accentLight,color:t.accentText||t.accent }}>#{history.length-i}</span>
                          <span className="text-xs" style={{ color:t.textMuted }}>{timeStr}</span>
                          {ai?.results?.length > 0 && (
                            <span className="text-xs px-2 py-0.5 rounded-full" style={{ background:'rgba(16,185,129,.1)',color:'#10b981' }}>{ai.results.length} rows</span>
                          )}
                          {ai?.chart && (
                            <span className="text-xs px-2 py-0.5 rounded-full" style={{ background:'rgba(139,92,246,.1)',color:'#8b5cf6' }}>📊 Chart</span>
                          )}
                        </div>
                        <p className="text-sm font-semibold" style={{ color:t.text }}>{msg.content}</p>
                      </div>
                      <button onClick={()=>onClickQuery(msg.index)} style={{ color:t.textMuted }}>
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                    {ai?.sql && (
                      <div className="mt-3">
                        <div className="flex items-center justify-between mb-2">
                          <button onClick={()=>setExpandedSql(expanded?null:msg.index)}
                            className="flex items-center gap-1.5 text-xs font-medium" style={{ color:t.textSub }}>
                            <Code className="w-3.5 h-3.5" />
                            {expanded?'Hide SQL':'View SQL'}
                            <ChevronDown className={`w-3 h-3 transition-transform ${expanded?'rotate-180':''}`} />
                          </button>
                          {expanded && (
                            <button onClick={()=>copy(ai.sql)} className="flex items-center gap-1 text-xs" style={{ color:t.textMuted }}>
                              <Copy className="w-3 h-3" /> Copy
                            </button>
                          )}
                        </div>
                        {expanded ? (
                          <div className="rounded-xl p-3 overflow-x-auto" style={{ background:'#0f172a' }}>
                            <pre className="text-xs font-mono text-green-400 whitespace-pre-wrap">{ai.sql}</pre>
                          </div>
                        ) : (
                          <div className="rounded-lg px-3 py-2 text-xs font-mono truncate"
                            style={{ background:t.surfaceHover||t.bg,color:t.textMuted }}>{ai.sql.slice(0,80)}…</div>
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
      <button onClick={()=>setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 rounded-xl text-sm transition-all"
        style={{ background:open?t.sidebarBgActive:'transparent',color:t.sidebarText }}>
        <Palette className="w-4 h-4" />
        <span>Theme</span>
        <div className="ml-auto flex gap-1">
          {Object.values(themes).map(th => (
            <div key={th.id} className="w-2.5 h-2.5 rounded-full border"
              style={{ background:th.preview?.[2]||th.accent,borderColor:themeId===th.id?t.sidebarTextActive:'transparent',transform:themeId===th.id?'scale(1.3)':'scale(1)' }} />
          ))}
        </div>
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={()=>setOpen(false)} />
          <div className="absolute bottom-full left-0 right-0 mb-2 rounded-xl shadow-2xl z-50 overflow-hidden"
            style={{ background:t.surface,border:`1px solid ${t.border}` }}>
            {Object.values(themes).map(th => (
              <button key={th.id} onClick={()=>{setTheme(th.id);setOpen(false);}}
                className="w-full flex items-center gap-3 px-3 py-2.5 text-sm transition-all hover:opacity-80"
                style={{ background:themeId===th.id?t.accentLight:'transparent',color:themeId===th.id?t.accentText||t.accent:t.text }}>
                <div className="flex gap-1 shrink-0">
                  {th.preview?.map((c,i) => <div key={i} className="w-3 h-3 rounded-full" style={{ background:c }} />)}
                </div>
                <span className="font-medium">{th.name}</span>
                {themeId===th.id && <Check className="w-3.5 h-3.5 ml-auto" />}
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
  const [chatOpen, setChatOpen] = useState(false);
  const [topQuery, setTopQuery] = useState('');

  const [convPages, setConvPages] = useState({ 1: [{ id: 1, name: 'Page 1' }] });
  const [convActivePage, setConvActivePage] = useState({ 1: 1 });
  const [renamingPageId, setRenamingPageId] = useState(null);
  const [renamePageText, setRenamePageText] = useState('');

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
      c.id === activeConvId ? { ...c, charts: [...(c.charts||[]), chart] } : c
    ));
  };

  const handleChartBuilderAdded = ({ chart, sql }) => {
    const now = new Date().toISOString();
    const chartWithMeta = { ...chart, query: chart.title, created_at: now };
    handleChartCreated(chartWithMeta);
    setMessages(prev => [
      ...prev,
      { type:'user', content:`[Chart Builder] ${chart.title}`, timestamp:now },
      { type:'ai', content:'Chart created via Chart Builder and added to dashboard.', sql, results:chart.data?.slice(0,100)||[], chart:chartWithMeta, timestamp:now },
    ]);
  };

  const handleChartsLoaded = (charts) => {
    setConversations(prev => prev.map(c => {
      if (c.id !== activeConvId || c.charts?.length) return c;
      return { ...c, charts: charts.map(ch => ({ ...ch, query:'Initial Dashboard', created_at:ch.created_at||new Date().toISOString() })) };
    }));
  };

  const createNewConversation = () => {
    const id = Math.max(...conversations.map(c => c.id)) + 1;
    const conv = { id, title:'New Conversation', messages:[{...DEFAULT_WELCOME_MESSAGE, timestamp:new Date().toISOString()}], charts:[], lastUpdated:new Date().toISOString() };
    setConversations(prev => [...prev, conv]);
    setActiveConvId(id);
    setActiveNav('dashboard');
    setConvPages(prev => ({...prev,[id]:[{id:1,name:'Page 1'}]}));
    setConvActivePage(prev => ({...prev,[id]:1}));
  };

  const deleteConversation = (id) => {
    if (conversations.length === 1) return;
    setConversations(prev => prev.filter(c => c.id !== id));
    if (activeConvId === id) setActiveConvId(conversations.find(c => c.id !== id).id);
  };

  const startEdit = (id, title) => { setEditingConvId(id); setEditTitle(title); };
  const saveTitle = (id) => {
    setConversations(prev => prev.map(c => c.id===id ? {...c, title:editTitle.trim()||'New Conversation'} : c));
    setEditingConvId(null);
  };

  const handleHistoryClick = (idx) => { setActiveNav('dashboard'); setScrollToIndex(idx); };

  const addPage = (convId) => {
    const pages = convPages[convId]||[{id:1,name:'Page 1'}];
    const newId = Math.max(...pages.map(p=>p.id))+1;
    setConvPages(prev => ({...prev,[convId]:[...pages,{id:newId,name:`Page ${newId}`}]}));
    setConvActivePage(prev => ({...prev,[convId]:newId}));
  };
  const deletePage = (convId, pageId) => {
    const pages = convPages[convId]||[{id:1,name:'Page 1'}];
    if (pages.length<=1) return;
    const remaining = pages.filter(p=>p.id!==pageId);
    setConvPages(prev => ({...prev,[convId]:remaining}));
    if ((convActivePage[convId]||1)===pageId) setConvActivePage(prev => ({...prev,[convId]:remaining[0].id}));
  };
  const switchPage = (convId, pageId) => setConvActivePage(prev => ({...prev,[convId]:pageId}));
  const startRenamePage = (pageId, name) => { setRenamingPageId(pageId); setRenamePageText(name); };
  const savePageRename = (convId, pageId) => {
    if (renamePageText.trim()) {
      setConvPages(prev => ({...prev,[convId]:(prev[convId]||[]).map(p=>p.id===pageId?{...p,name:renamePageText.trim()}:p)}));
    }
    setRenamingPageId(null);
  };

  useEffect(() => {
    if (scrollToIndex !== null) {
      const timer = setTimeout(() => setScrollToIndex(null), 500);
      return () => clearTimeout(timer);
    }
  }, [scrollToIndex]);

  // Send query from top NL bar
  const handleTopQuery = async (q) => {
    if (!q.trim()) return;
    setTopQuery('');
    setChatOpen(true);
    setActiveNav('dashboard');
    setMessages(prev => [...prev, { type:'user', content:q, timestamp:new Date().toISOString() }]);
    try {
      const token = sessionStorage.getItem('token');
      const res = await axios.post(`${API_URL}/query`,
        { question:q, session_id:dashSessionId },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      const ai = { type:'ai', content:res.data.explanation||'Query processed successfully.', sql:res.data.sql, results:res.data.results||[], execution_time:res.data.execution_time, chart:res.data.chart, sessionId:dashSessionId, timestamp:new Date().toISOString() };
      setMessages(prev => [...prev, ai]);
      if (res.data.chart) handleChartCreated({...res.data.chart, query:q, created_at:new Date().toISOString()});
      if (res.data.chart?.auto_added||res.data.full_dashboard_generated)
        setTimeout(() => window.dispatchEvent(new CustomEvent('refreshDashboard')), 500);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setMessages(prev => [...prev, { type:'error', content:typeof detail==='string'?detail:'Something went wrong.', timestamp:new Date().toISOString() }]);
    }
  };

  const activePages = convPages[activeConvId]||[{id:1,name:'Page 1'}];
  const activeDashPage = convActivePage[activeConvId]||1;
  const dashSessionId = `session-${activeConvId}-p${activeDashPage}`;

  if (sessionLoading) {
    return (
      <div className="flex h-screen items-center justify-center" style={{ background: t.bg }}>
        <div className="text-center">
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-4" style={{ background:t.accentLight }}>
            <Activity className="w-6 h-6 animate-pulse" style={{ color:t.accent }} />
          </div>
          <p className="font-semibold" style={{ color:t.text }}>Restoring your session…</p>
          <p className="text-sm mt-1" style={{ color:t.textMuted }}>Loading your workspace</p>
        </div>
      </div>
    );
  }

  const NAV_ITEMS = [
    { id:'dashboard',     label:'Dashboard',      icon:BarChart2 },
    { id:'history',       label:'Query History',  icon:Clock },
    { id:'visualizations',label:'Chart Gallery',  icon:TrendingUp },
    { id:'data',          label:'Data Sources',   icon:Database },
    { id:'reports',       label:'Reports',        icon:FileText },
    { id:'security',      label:'Security (RLS)', icon:Shield },
  ];

  const sidebarW = sidebarOpen ? 220 : 56;

  return (
    <div style={{ display:'flex',height:'100vh',overflow:'hidden',background:t.bg,fontFamily:"'Segoe UI',system-ui,sans-serif" }}>
      <style>{`
        @keyframes slideInRight{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}
        .iq-nav:hover{opacity:.85!important}
        .iq-conv:hover .iq-cact{opacity:1!important}
      `}</style>

      {/* ── Sidebar ── */}
      <aside style={{
        width:`${sidebarW}px`,flexShrink:0,display:'flex',flexDirection:'column',
        background:t.sidebar,borderRight:`1px solid ${t.sidebarBorder}`,
        transition:'width .2s ease',overflow:'hidden',
      }}>
        {/* Logo */}
        <div style={{
          display:'flex',alignItems:'center',
          gap:sidebarOpen?'10px':'0',justifyContent:sidebarOpen?'flex-start':'center',
          padding:sidebarOpen?'14px 14px 12px':'14px 8px 12px',
          borderBottom:`1px solid ${t.sidebarBorder}`,flexShrink:0,overflow:'hidden',
        }}>
          <div style={{ width:'32px',height:'32px',borderRadius:'10px',flexShrink:0,background:'linear-gradient(135deg,#c9a227 0%,#e8c840 100%)',display:'flex',alignItems:'center',justifyContent:'center',boxShadow:'0 4px 14px rgba(212,175,55,.32)' }}>
            <span style={{ color:'#0a0f1e',fontWeight:'900',fontSize:'12px',letterSpacing:'.5px' }}>IQ</span>
          </div>
          {sidebarOpen && (
            <div style={{ overflow:'hidden',minWidth:0 }}>
              <h1 style={{ fontSize:'12px',fontWeight:'800',color:t.text,letterSpacing:'1.5px',margin:0,whiteSpace:'nowrap' }}>INTELLIQUERY</h1>
              <p style={{ fontSize:'10px',color:t.textMuted,margin:0 }}>BI Suite</p>
            </div>
          )}
        </div>

        {/* Nav */}
        <nav style={{ flex:1,padding:'10px 8px',overflowY:'auto',display:'flex',flexDirection:'column',gap:'2px' }}>
          <button onClick={createNewConversation} className="iq-nav"
            title={!sidebarOpen?'New Chat':undefined}
            style={{ width:'100%',display:'flex',alignItems:'center',gap:sidebarOpen?'8px':'0',justifyContent:sidebarOpen?'flex-start':'center',padding:sidebarOpen?'8px 12px':'8px',borderRadius:'10px',border:'none',cursor:'pointer',background:t.accent,color:'#fff',marginBottom:'10px',fontSize:'12px',fontWeight:'600',transition:'opacity .15s' }}>
            <Plus style={{ width:'14px',height:'14px',flexShrink:0 }} />
            {sidebarOpen && 'New Chat'}
          </button>

          {sidebarOpen && <p style={{ fontSize:'9px',fontWeight:'700',color:t.textMuted,letterSpacing:'1.5px',padding:'2px 8px 8px',textTransform:'uppercase' }}>Workspace</p>}

          {NAV_ITEMS.map(item => {
            const active = activeNav===item.id;
            return (
              <button key={item.id} onClick={()=>setActiveNav(item.id)} className="iq-nav"
                title={!sidebarOpen?item.label:undefined}
                style={{ width:'100%',display:'flex',alignItems:'center',gap:sidebarOpen?'8px':'0',justifyContent:sidebarOpen?'flex-start':'center',padding:sidebarOpen?'8px 12px':'8px',borderRadius:'10px',border:'none',cursor:'pointer',background:active?t.sidebarBgActive:'transparent',color:active?t.sidebarTextActive:t.sidebarText,fontSize:'12px',fontWeight:active?'600':'400',transition:'background .15s,opacity .15s' }}>
                <item.icon style={{ width:'15px',height:'15px',flexShrink:0 }} />
                {sidebarOpen && <span style={{ whiteSpace:'nowrap' }}>{item.label}</span>}
                {sidebarOpen && active && <div style={{ marginLeft:'auto',width:'5px',height:'5px',borderRadius:'50%',background:t.sidebarTextActive }} />}
              </button>
            );
          })}

          <div style={{ height:'1px',background:t.sidebarBorder,margin:'8px 4px' }} />

          {/* Ask AI */}
          <button onClick={()=>{setChatOpen(true);setActiveNav('dashboard');}} className="iq-nav"
            title={!sidebarOpen?'Ask AI':undefined}
            style={{ width:'100%',display:'flex',alignItems:'center',gap:sidebarOpen?'8px':'0',justifyContent:sidebarOpen?'flex-start':'center',padding:sidebarOpen?'8px 12px':'8px',borderRadius:'10px',border:'1px solid rgba(212,175,55,.2)',cursor:'pointer',background:'linear-gradient(135deg,rgba(212,175,55,.12) 0%,rgba(212,175,55,.05) 100%)',color:'#c9a227',fontSize:'12px',fontWeight:'600',transition:'opacity .15s' }}>
            <Sparkles style={{ width:'15px',height:'15px',flexShrink:0 }} />
            {sidebarOpen && 'Ask AI'}
          </button>

          {sidebarOpen && (
            <>
              <p style={{ fontSize:'9px',fontWeight:'700',color:t.textMuted,letterSpacing:'1.5px',padding:'12px 8px 6px',textTransform:'uppercase' }}>Recent</p>
              {conversations
                .sort((a,b)=>new Date(b.lastUpdated)-new Date(a.lastUpdated))
                .slice(0,8)
                .map(conv => (
                  <div key={conv.id} className="iq-conv" style={{ position:'relative' }}>
                    {editingConvId===conv.id ? (
                      <div style={{ display:'flex',alignItems:'center',gap:'4px',padding:'6px 8px',borderRadius:'10px',background:t.sidebarBgActive }}>
                        <input value={editTitle} onChange={e=>setEditTitle(e.target.value)}
                          onKeyDown={e=>e.key==='Enter'&&saveTitle(conv.id)} autoFocus
                          style={{ flex:1,padding:'2px 6px',fontSize:'11px',borderRadius:'6px',background:t.surface,color:t.text,border:`1px solid ${t.accent}`,outline:'none' }} />
                        <button onClick={()=>saveTitle(conv.id)} style={{ background:'none',border:'none',cursor:'pointer',color:'#10b981',padding:'2px' }}><Check style={{width:'11px',height:'11px'}}/></button>
                        <button onClick={()=>setEditingConvId(null)} style={{ background:'none',border:'none',cursor:'pointer',color:'#ef4444',padding:'2px' }}><X style={{width:'11px',height:'11px'}}/></button>
                      </div>
                    ) : (
                      <button onClick={()=>setActiveConvId(conv.id)}
                        style={{ width:'100%',display:'flex',alignItems:'center',gap:'6px',padding:'7px 10px',borderRadius:'10px',border:'none',cursor:'pointer',background:activeConvId===conv.id?t.sidebarBgActive:'transparent',color:activeConvId===conv.id?t.sidebarTextActive:t.sidebarText,fontSize:'11px',fontWeight:activeConvId===conv.id?'600':'400' }}>
                        <MessageSquare style={{ width:'12px',height:'12px',flexShrink:0 }} />
                        <span style={{ flex:1,textAlign:'left',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap' }}>{conv.title}</span>
                        <div className="iq-cact" style={{ display:'flex',gap:'2px',opacity:0,transition:'opacity .15s' }}>
                          <button onClick={e=>{e.stopPropagation();startEdit(conv.id,conv.title);}} style={{ background:'none',border:'none',cursor:'pointer',padding:'2px',color:t.textMuted }}><Edit2 style={{width:'10px',height:'10px'}}/></button>
                          {conversations.length>1 && <button onClick={e=>{e.stopPropagation();deleteConversation(conv.id);}} style={{ background:'none',border:'none',cursor:'pointer',padding:'2px',color:'#ef4444' }}><Trash2 style={{width:'10px',height:'10px'}}/></button>}
                        </div>
                      </button>
                    )}
                  </div>
                ))}
            </>
          )}
        </nav>

        {/* Footer */}
        <div style={{ padding:'8px',borderTop:`1px solid ${t.sidebarBorder}`,flexShrink:0 }}>
          {sidebarOpen && <ThemeSelector />}
          <div style={{ display:'flex',alignItems:'center',gap:'8px',padding:sidebarOpen?'8px 10px':'8px',borderRadius:'10px',background:t.sidebarBgActive,marginTop:'4px',justifyContent:sidebarOpen?'flex-start':'center',overflow:'hidden' }}>
            <div style={{ width:'28px',height:'28px',borderRadius:'50%',flexShrink:0,background:t.accent,display:'flex',alignItems:'center',justifyContent:'center' }}>
              <span style={{ color:'#fff',fontSize:'11px',fontWeight:'700' }}>{user.full_name?.charAt(0).toUpperCase()}</span>
            </div>
            {sidebarOpen && (
              <>
                <div style={{ minWidth:0,flex:1 }}>
                  <p style={{ fontSize:'11px',fontWeight:'600',color:t.sidebarTextActive,margin:0,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap' }}>{user.full_name}</p>
                  <p style={{ fontSize:'9px',color:t.textMuted,margin:0,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap' }}>{user.email}</p>
                </div>
                <button onClick={onLogout} style={{ background:'none',border:'none',cursor:'pointer',padding:'4px',color:'#ef4444' }}>
                  <LogOut style={{ width:'13px',height:'13px' }} />
                </button>
              </>
            )}
          </div>
        </div>
      </aside>

      {/* ── Main Area ── */}
      <div style={{ flex:1,display:'flex',flexDirection:'column',minWidth:0,overflow:'hidden' }}>

        {/* Top Bar */}
        <header style={{ display:'flex',alignItems:'center',gap:'12px',padding:'0 16px',height:'52px',flexShrink:0,background:t.header,borderBottom:`1px solid ${t.headerBorder||t.border}` }}>
          <button onClick={()=>setSidebarOpen(!sidebarOpen)} style={{ background:'none',border:'none',cursor:'pointer',padding:'6px',color:t.textMuted,borderRadius:'8px',display:'flex' }}>
            <Menu style={{ width:'18px',height:'18px' }} />
          </button>

          {/* Breadcrumbs */}
          <div style={{ display:'flex',alignItems:'center',gap:'5px',flexShrink:0 }}>
            <span style={{ fontSize:'12px',color:t.textMuted }}>My Workspace</span>
            {activeConv?.title && activeConv.title!=='New Conversation' && (
              <>
                <ChevronRight style={{ width:'11px',height:'11px',color:t.border }} />
                <span style={{ fontSize:'12px',fontWeight:'600',color:t.text,maxWidth:'160px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap' }}>{activeConv.title}</span>
              </>
            )}
          </div>

          {/* NL Query bar */}
          <div style={{ flex:1,maxWidth:'560px',margin:'0 auto' }}>
            <div style={{ display:'flex',alignItems:'center',gap:'8px',background:t.surfaceHover||t.surface,border:`1px solid ${t.border}`,borderRadius:'12px',padding:'0 14px' }}>
              <Search style={{ width:'15px',height:'15px',color:t.textMuted,flexShrink:0 }} />
              <input type="text" value={topQuery} onChange={e=>setTopQuery(e.target.value)}
                onKeyDown={e=>{ if(e.key==='Enter'&&topQuery.trim()) handleTopQuery(topQuery); }}
                placeholder="Ask anything about your data…"
                style={{ flex:1,background:'transparent',border:'none',outline:'none',color:t.text,fontSize:'13px',padding:'10px 0' }} />
              <kbd style={{ fontSize:'10px',padding:'2px 6px',borderRadius:'4px',background:t.accentLight,color:t.textMuted,border:`1px solid ${t.border}` }}>↵</kbd>
            </div>
          </div>

          {/* Right controls */}
          <div style={{ display:'flex',alignItems:'center',gap:'6px',flexShrink:0 }}>
            <div title="Knowledge Base — RAG Active" style={{ display:'flex',alignItems:'center',gap:'5px',padding:'5px 10px',borderRadius:'8px',background:t.accentLight,border:`1px solid ${t.border}`,fontSize:'11px',color:t.textSub,cursor:'default' }}>
              <BookOpen style={{ width:'13px',height:'13px' }} />
              <div style={{ width:'6px',height:'6px',borderRadius:'50%',background:'#10b981',boxShadow:'0 0 6px rgba(16,185,129,.6)' }} />
            </div>
            <button style={{ background:'none',border:'none',cursor:'pointer',padding:'6px',borderRadius:'8px',color:t.textMuted,display:'flex' }}>
              <Bell style={{ width:'16px',height:'16px' }} />
            </button>
            <div title={`${user.full_name} · ${user.email}`} style={{ width:'30px',height:'30px',borderRadius:'50%',cursor:'pointer',background:t.accent,display:'flex',alignItems:'center',justifyContent:'center',boxShadow:`0 0 0 2px ${t.border}` }}>
              <span style={{ color:'#fff',fontSize:'12px',fontWeight:'700' }}>{user.full_name?.charAt(0).toUpperCase()}</span>
            </div>
          </div>
        </header>

        {/* Content */}
        <div style={{ flex:1,display:'flex',overflow:'hidden',position:'relative' }}>

          {activeNav==='history' && <HistoryTab messages={activeConv?.messages||[]} onClickQuery={handleHistoryClick} />}

          {activeNav==='visualizations' && (
            <VisualizationHistory
              charts={activeConv?.charts||[]}
              onRestoreChart={async (chart) => {
                try {
                  const token = sessionStorage.getItem('token');
                  await fetch(`${API_URL}/dashboard/add-chart`,{method:'POST',headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},body:JSON.stringify({session_id:dashSessionId,query:chart.query||chart.title||'',sql:'',result:{success:true,data:chart.data||[]}})});
                  window.dispatchEvent(new CustomEvent('refreshDashboard'));
                } catch {}
              }}
              onClearHistory={()=>setConversations(prev=>prev.map(c=>c.id===activeConvId?{...c,charts:[]}:c))}
            />
          )}

          {/* Placeholder pages */}
          {(activeNav==='data'||activeNav==='reports'||activeNav==='security') && (
            <div style={{ flex:1,display:'flex',alignItems:'center',justifyContent:'center',background:t.bg }}>
              <div style={{ textAlign:'center' }}>
                <div style={{ width:'48px',height:'48px',borderRadius:'14px',background:t.accentLight,display:'flex',alignItems:'center',justifyContent:'center',margin:'0 auto 16px' }}>
                  {activeNav==='data' && <Database style={{width:'22px',height:'22px',color:t.accent}}/>}
                  {activeNav==='reports' && <FileText style={{width:'22px',height:'22px',color:t.accent}}/>}
                  {activeNav==='security' && <Shield style={{width:'22px',height:'22px',color:t.accent}}/>}
                </div>
                <p style={{ fontWeight:'700',color:t.text,marginBottom:'6px' }}>
                  {activeNav==='data'?'Data Sources':activeNav==='reports'?'Reports':'Security (RLS)'}
                </p>
                <p style={{ fontSize:'13px',color:t.textMuted }}>Coming soon</p>
              </div>
            </div>
          )}

          {/* Dashboard — always mounted */}
          <div style={{ display:activeNav==='dashboard'?'flex':'none',flex:1,flexDirection:'column',overflow:'hidden' }}>
            <div style={{ flex:1,overflow:'hidden' }}>
              <DashboardContainer
                sessionId={dashSessionId}
                onChartsLoaded={handleChartsLoaded}
                onChartBuilderAdded={handleChartBuilderAdded}
              />
            </div>

            {/* Power BI page tabs */}
            <div style={{ display:'flex',alignItems:'center',borderTop:`1px solid ${t.border}`,overflowX:'auto',background:t.header,minHeight:'36px',paddingLeft:'12px',flexShrink:0 }}>
              {activePages.map(page => {
                const isActive = page.id===activeDashPage;
                const isRenaming = renamingPageId===page.id;
                return (
                  <div key={page.id} style={{ position:'relative',display:'flex',alignItems:'center',flexShrink:0,marginRight:'2px' }}>
                    <div style={{ display:'flex',alignItems:'center',gap:'4px',padding:'6px 12px',borderRadius:'6px 6px 0 0',cursor:'pointer',background:isActive?t.bg:'transparent',color:isActive?t.text:t.textMuted,fontWeight:isActive?'600':'400',fontSize:'11.5px',border:isActive?`1px solid ${t.border}`:'1px solid transparent',borderBottom:isActive?`1px solid ${t.bg}`:'1px solid transparent',marginBottom:isActive?'-1px':'0',userSelect:'none' }}
                      onClick={()=>!isRenaming&&switchPage(activeConvId,page.id)}
                      onDoubleClick={()=>startRenamePage(page.id,page.name)}>
                      {isRenaming ? (
                        <input autoFocus value={renamePageText} onChange={e=>setRenamePageText(e.target.value)}
                          onKeyDown={e=>{if(e.key==='Enter')savePageRename(activeConvId,page.id);if(e.key==='Escape')setRenamingPageId(null);}}
                          onBlur={()=>savePageRename(activeConvId,page.id)}
                          onClick={e=>e.stopPropagation()}
                          style={{ width:'64px',fontSize:'11px',background:'transparent',border:'none',outline:'none',color:t.text }} />
                      ) : <span>{page.name}</span>}
                      {activePages.length>1&&!isRenaming && (
                        <button onClick={e=>{e.stopPropagation();deletePage(activeConvId,page.id);}}
                          style={{ background:'none',border:'none',cursor:'pointer',padding:'1px',color:t.textMuted,opacity:0 }}>
                          <X style={{width:'9px',height:'9px'}}/>
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
              <button onClick={()=>addPage(activeConvId)} style={{ width:'20px',height:'20px',borderRadius:'5px',marginLeft:'4px',display:'flex',alignItems:'center',justifyContent:'center',background:t.accentLight,border:'none',cursor:'pointer',color:t.textMuted }}>
                <Plus style={{width:'11px',height:'11px'}}/>
              </button>
              <span style={{ marginLeft:'10px',fontSize:'10px',color:t.textMuted,opacity:.45,whiteSpace:'nowrap' }}>Double-click to rename</span>
            </div>
          </div>

          {/* ── Ask AI Drawer (no background blur on main content) ── */}
          {chatOpen && (
            <div style={{
              position:'absolute',right:0,top:0,bottom:0,width:'340px',
              background:t.surface,borderLeft:`1px solid ${t.border}`,
              zIndex:50,display:'flex',flexDirection:'column',
              boxShadow:'-16px 0 60px rgba(0,0,0,.22)',
              animation:'slideInRight .22s ease',
            }}>
              {/* Drawer header */}
              <div style={{ display:'flex',alignItems:'center',justifyContent:'space-between',padding:'14px 16px',flexShrink:0,background:`linear-gradient(135deg,${t.accent} 0%,${t.accentHover||t.accent} 100%)` }}>
                <div style={{ display:'flex',alignItems:'center',gap:'10px' }}>
                  <div style={{ width:'30px',height:'30px',borderRadius:'8px',background:'rgba(255,255,255,.2)',display:'flex',alignItems:'center',justifyContent:'center' }}>
                    <Sparkles style={{ width:'14px',height:'14px',color:'#fff' }} />
                  </div>
                  <div>
                    <p style={{ color:'#fff',fontSize:'13px',fontWeight:'700',margin:0 }}>Ask AI</p>
                    <p style={{ color:'rgba(255,255,255,.65)',fontSize:'10px',margin:0 }}>Llama 3.3 · 70B</p>
                  </div>
                </div>
                <button onClick={()=>setChatOpen(false)} style={{ background:'rgba(255,255,255,.15)',border:'none',borderRadius:'6px',cursor:'pointer',padding:'5px',color:'#fff',display:'flex' }}>
                  <X style={{ width:'14px',height:'14px' }} />
                </button>
              </div>

              <div style={{ flex:1,overflow:'hidden',display:'flex',flexDirection:'column' }}>
                <ChatPanel
                  messages={activeConv?.messages||[]}
                  setMessages={setMessages}
                  scrollToIndex={scrollToIndex}
                  sessionId={dashSessionId}
                  onChartCreated={handleChartCreated}
                  hideHeader={true}
                />
              </div>
            </div>
          )}
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
    sessionStorage.removeItem('token'); sessionStorage.removeItem('user'); setUser(null);
  };
  if (!user) return <LoginPage onLogin={setUser} />;
  return <Dashboard user={user} onLogout={handleLogout} />;
};

export default function App() {
  return <ThemeProvider><AppInner /></ThemeProvider>;
}
