import React, { createContext, useContext, useState, useEffect } from 'react';

export const THEMES = {
    light: {
        id: 'light',
        name: 'Light Pro',
        preview: ['#f8fafc', '#ffffff', '#6366f1'],
        bg: '#f1f5f9',
        surface: '#ffffff',
        surfaceHover: '#f8fafc',
        sidebar: '#ffffff',
        sidebarBorder: '#e2e8f0',
        sidebarText: '#475569',
        sidebarTextActive: '#6366f1',
        sidebarBgActive: '#eef2ff',
        header: '#ffffff',
        headerBorder: '#e2e8f0',
        border: '#e2e8f0',
        text: '#1e293b',
        textSub: '#64748b',
        textMuted: '#94a3b8',
        accent: '#6366f1',
        accentHover: '#4f46e5',
        accentLight: '#eef2ff',
        accentText: '#4f46e5',
        success: '#10b981',
        danger: '#ef4444',
        warning: '#f59e0b',
        shadow: '0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)',
        shadowMd: '0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06)',
        chartColors: ['#6366f1', '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#8b5cf6', '#f43f5e'],
        kpiGradient: 'linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)',
        logoGradient: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
    },
    dark: {
        id: 'dark',
        name: 'Dark Mode',
        preview: ['#0f172a', '#1e293b', '#6366f1'],
        bg: '#0f172a',
        surface: '#1e293b',
        surfaceHover: '#243148',
        sidebar: '#1e293b',
        sidebarBorder: '#334155',
        sidebarText: '#94a3b8',
        sidebarTextActive: '#a5b4fc',
        sidebarBgActive: 'rgba(99,102,241,0.15)',
        header: '#1e293b',
        headerBorder: '#334155',
        border: '#334155',
        text: '#f1f5f9',
        textSub: '#94a3b8',
        textMuted: '#64748b',
        accent: '#818cf8',
        accentHover: '#6366f1',
        accentLight: 'rgba(99,102,241,0.15)',
        accentText: '#a5b4fc',
        success: '#34d399',
        danger: '#f87171',
        warning: '#fbbf24',
        shadow: '0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)',
        shadowMd: '0 4px 6px rgba(0,0,0,0.4), 0 2px 4px rgba(0,0,0,0.3)',
        chartColors: ['#818cf8', '#60a5fa', '#34d399', '#fbbf24', '#f472b6', '#22d3ee', '#a78bfa', '#fb7185'],
        kpiGradient: 'linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(99,102,241,0.05) 100%)',
        logoGradient: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
    },
    executive: {
        id: 'executive',
        name: 'Executive',
        preview: ['#020617', '#0d1b2e', '#0ea5e9'],
        bg: '#020617',
        surface: '#0d1b2e',
        surfaceHover: '#112240',
        sidebar: '#060e1f',
        sidebarBorder: '#0e2240',
        sidebarText: '#7ea8c9',
        sidebarTextActive: '#38bdf8',
        sidebarBgActive: 'rgba(14,165,233,0.12)',
        header: '#0d1b2e',
        headerBorder: '#0e2240',
        border: '#0e2240',
        text: '#e2e8f0',
        textSub: '#7ea8c9',
        textMuted: '#4a7899',
        accent: '#0ea5e9',
        accentHover: '#0284c7',
        accentLight: 'rgba(14,165,233,0.12)',
        accentText: '#38bdf8',
        success: '#10b981',
        danger: '#f87171',
        warning: '#fbbf24',
        shadow: '0 1px 3px rgba(0,0,0,0.6), 0 1px 2px rgba(0,0,0,0.5)',
        shadowMd: '0 4px 6px rgba(0,0,0,0.5), 0 2px 4px rgba(0,0,0,0.4)',
        chartColors: ['#0ea5e9', '#38bdf8', '#10b981', '#8b5cf6', '#f59e0b', '#6366f1', '#06b6d4', '#ec4899'],
        kpiGradient: 'linear-gradient(135deg, rgba(14,165,233,0.15) 0%, rgba(14,165,233,0.05) 100%)',
        logoGradient: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
    },
    rose: {
        id: 'rose',
        name: 'Rose Gold',
        preview: ['#fff1f2', '#ffffff', '#e11d48'],
        bg: '#fff1f2',
        surface: '#ffffff',
        surfaceHover: '#fff5f6',
        sidebar: '#ffffff',
        sidebarBorder: '#fecdd3',
        sidebarText: '#9f1239',
        sidebarTextActive: '#e11d48',
        sidebarBgActive: '#fff1f2',
        header: '#ffffff',
        headerBorder: '#fecdd3',
        border: '#fecdd3',
        text: '#1c0b0f',
        textSub: '#6b2138',
        textMuted: '#9f1239',
        accent: '#e11d48',
        accentHover: '#be123c',
        accentLight: '#fff1f2',
        accentText: '#be123c',
        success: '#059669',
        danger: '#dc2626',
        warning: '#d97706',
        shadow: '0 1px 3px rgba(225,29,72,0.08), 0 1px 2px rgba(0,0,0,0.04)',
        shadowMd: '0 4px 6px rgba(225,29,72,0.08), 0 2px 4px rgba(0,0,0,0.05)',
        chartColors: ['#e11d48', '#f43f5e', '#f97316', '#8b5cf6', '#0ea5e9', '#10b981', '#fbbf24', '#6366f1'],
        kpiGradient: 'linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%)',
        logoGradient: 'linear-gradient(135deg, #e11d48 0%, #be123c 100%)',
    },
};

const ThemeContext = createContext(null);

export const ThemeProvider = ({ children }) => {
    const [themeId, setThemeId] = useState(() => localStorage.getItem('iq-theme') || 'light');
    const theme = THEMES[themeId] || THEMES.light;

    const setTheme = (id) => {
        setThemeId(id);
        localStorage.setItem('iq-theme', id);
    };

    return (
        <ThemeContext.Provider value={{ theme, themeId, setTheme, themes: THEMES }}>
            {children}
        </ThemeContext.Provider>
    );
};

export const useTheme = () => {
    const ctx = useContext(ThemeContext);
    if (!ctx) throw new Error('useTheme must be inside ThemeProvider');
    return ctx;
};

export default ThemeContext;
