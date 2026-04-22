import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ChartRecommender:
    """
    Smart chart type recommendation following Power BI best-practice guidelines.

    Supported types: kpi, line, area, bar, pie, donut, scatter, treemap,
                     funnel, radar, tornado, table
    """

    # ── Keyword → type mapping ──────────────────────────────────────────
    _KW = {
        'line':     ['trend', 'over time', 'monthly', 'yearly', 'daily', 'weekly', 'timeline', 'history', 'growth', 'progression'],
        'area':     ['cumulative', 'stacked', 'total over time', 'magnitude', 'fill', 'area under'],
        'bar':      ['top', 'bottom', 'compare', 'comparison', 'best', 'worst', 'ranking', 'vs', 'highest', 'lowest', 'most', 'least'],
        'pie':      ['share', 'percentage', 'proportion', 'portion', 'slice'],
        'donut':    ['distribution', 'breakdown', 'composition', 'by category', 'split', 'makeup'],
        'scatter':  ['correlation', 'relationship', 'versus', 'impact', 'effect', 'regression', 'cluster'],
        'treemap':  ['hierarchy', 'nested', 'sub-category', 'sub category', 'tree', 'map', 'portfolio'],
        'funnel':   ['pipeline', 'funnel', 'stage', 'conversion', 'drop off', 'dropoff', 'process', 'flow'],
        'radar':    ['performance', 'multi-dimension', 'profile', 'spider', 'radar', 'capabilities', 'skills', 'attributes'],
        'tornado':  ['sensitivity', 'impact analysis', 'drivers', 'factors', 'waterfall', 'bridge'],
        'table':    ['list', 'show all', 'details', 'full data', 'all records', 'raw', 'report'],
        'kpi':      ['total', 'sum', 'count', 'average', 'how many', 'how much', 'metric', 'kpi', 'indicator'],
    }

    def recommend_chart_type(self, query: str, data: List[Dict], query_result_count: int = 0) -> str:
        """Recommend the most appropriate chart type for the given query + data."""
        if not data:
            return 'table'

        query_lower = query.lower()
        columns = list(data[0].keys())
        row_count = len(data)

        # ── Single-value → KPI ──────────────────────────────────────────
        if row_count == 1 and len(columns) == 1:
            return 'kpi'
        if row_count == 1 and len(columns) <= 2:
            numeric_vals = [v for v in data[0].values() if isinstance(v, (int, float))]
            if len(numeric_vals) >= 1:
                return 'kpi'

        # ── Keyword match (priority order) ─────────────────────────────
        for chart_type, keywords in self._KW.items():
            if any(kw in query_lower for kw in keywords):
                if self._data_supports(chart_type, data, columns):
                    return chart_type

        # ── Data-structure inference ────────────────────────────────────
        return self._infer(data, columns, row_count)

    # ── Data compatibility checks ────────────────────────────────────────

    def _data_supports(self, chart_type: str, data: List[Dict], columns: List[str]) -> bool:
        if chart_type == 'kpi':
            return len(data) == 1

        if chart_type in ('line', 'area'):
            return any(self._is_date(c) for c in columns)

        if chart_type == 'scatter':
            numeric = [c for c in columns if self._is_numeric(c, data)]
            return len(numeric) >= 2

        if chart_type in ('pie', 'donut', 'funnel'):
            return len(columns) >= 2 and len(data) <= 30

        if chart_type == 'treemap':
            return len(columns) >= 2 and self._has_numeric(columns, data)

        if chart_type == 'radar':
            numeric = [c for c in columns if self._is_numeric(c, data)]
            return len(numeric) >= 2

        if chart_type == 'tornado':
            return len(columns) >= 2

        return True

    def _infer(self, data: List[Dict], columns: List[str], row_count: int) -> str:
        """Infer chart type purely from data structure."""
        date_cols = [c for c in columns if self._is_date(c)]
        numeric_cols = [c for c in columns if self._is_numeric(c, data)]

        # Time-series
        if date_cols and len(columns) == 2:
            return 'line'
        if date_cols and numeric_cols:
            return 'area' if row_count > 30 else 'line'

        # Multiple numeric columns with a label → radar or bar
        if len(numeric_cols) >= 3 and len(data) <= 15:
            return 'radar'

        # Two columns, few rows → donut
        if len(columns) == 2 and 2 <= row_count <= 8:
            return 'donut'

        # Two columns, moderate rows → bar
        if len(columns) == 2 and 2 <= row_count <= 20:
            return 'bar'

        # Many rows → table
        if row_count > 50:
            return 'table'

        # Hierarchical-looking data (both categorical + numeric)
        if len(numeric_cols) >= 1 and row_count > 20:
            return 'treemap'

        # Default
        return 'bar'

    # ── Column type helpers ──────────────────────────────────────────────

    def _is_date(self, col: str) -> bool:
        return any(kw in col.lower() for kw in ['date', 'time', 'month', 'year', 'day', 'week', 'period', 'quarter'])

    def _is_numeric(self, col: str, data: List[Dict]) -> bool:
        numeric_kw = ['amount', 'total', 'revenue', 'sales', 'profit', 'quantity', 'count',
                      'price', 'cost', 'value', 'rate', 'sum', 'avg', 'average', 'num',
                      'score', 'percent', 'pct', 'ratio', 'units', 'volume']
        if any(kw in col.lower() for kw in numeric_kw):
            return True
        try:
            val = data[0].get(col)
            return isinstance(val, (int, float)) and not isinstance(val, bool)
        except Exception:
            return False

    def _has_numeric(self, columns: List[str], data: List[Dict]) -> bool:
        return any(self._is_numeric(c, data) for c in columns)

    # ── Config extraction ────────────────────────────────────────────────

    def extract_chart_config(self, data: List[Dict], chart_type: str, query: str) -> Dict[str, Any]:
        """Build a complete chart configuration dictionary."""
        if not data:
            return {'type': chart_type, 'title': self._title(query, chart_type), 'data': data, 'config': {}}

        columns = list(data[0].keys())
        base = {
            'type': chart_type,
            'title': self._title(query, chart_type),
            'data': data,
            'config': {}
        }

        dispatch = {
            'kpi':      self._cfg_kpi,
            'line':     lambda d, c, cfg: self._cfg_line_area(d, c, cfg, 'line'),
            'area':     lambda d, c, cfg: self._cfg_line_area(d, c, cfg, 'area'),
            'bar':      self._cfg_bar,
            'pie':      self._cfg_pie_donut,
            'donut':    self._cfg_pie_donut,
            'scatter':  self._cfg_scatter,
            'treemap':  self._cfg_treemap,
            'funnel':   self._cfg_funnel,
            'radar':    self._cfg_radar,
            'tornado':  self._cfg_tornado,
            'table':    self._cfg_table,
        }

        handler = dispatch.get(chart_type, self._cfg_table)
        try:
            return handler(data, columns, base)
        except Exception as e:
            logger.warning(f"Config extraction failed for {chart_type}: {e}")
            return self._cfg_table(data, columns, base)

    # ── Per-type config builders ─────────────────────────────────────────

    def _cfg_kpi(self, data, columns, cfg):
        col = columns[0]
        value = data[0][col]
        cfg['config'] = {
            'value': value,
            'label': col.replace('_', ' ').title(),
            'format': 'number' if isinstance(value, (int, float)) else 'text',
        }
        if len(columns) > 1:
            cfg['config']['trend'] = data[0][columns[1]]
        return cfg

    def _cfg_line_area(self, data, columns, cfg, chart_type):
        date_col = next((c for c in columns if self._is_date(c)), columns[0])
        value_col = next(
            (c for c in columns if c != date_col and self._is_numeric(c, data)),
            columns[1] if len(columns) > 1 else columns[0]
        )
        cfg['config'] = {
            'xAxis': date_col,
            'yAxis': value_col,
            'xLabel': date_col.replace('_', ' ').title(),
            'yLabel': value_col.replace('_', ' ').title(),
            'color': '#6366f1',
            'curved': True,
        }
        return cfg

    def _cfg_bar(self, data, columns, cfg):
        # category column: prefer name/category/label columns
        cat_col = columns[0]
        for c in columns:
            if any(kw in c.lower() for kw in ['name', 'category', 'label', 'type', 'region', 'country', 'city']):
                cat_col = c
                break

        val_col = next(
            (c for c in columns if c != cat_col and self._is_numeric(c, data)),
            None
        )
        if val_col is None:
            return self._cfg_table(data, columns, cfg)

        query_lower = cfg.get('title', '').lower()
        is_horizontal = any(kw in query_lower for kw in ['top', 'bottom', 'ranking', 'best', 'worst', 'highest', 'lowest'])

        cfg['config'] = {
            'xAxis': val_col if is_horizontal else cat_col,
            'yAxis': cat_col if is_horizontal else val_col,
            'xLabel': (val_col if is_horizontal else cat_col).replace('_', ' ').title(),
            'yLabel': (cat_col if is_horizontal else val_col).replace('_', ' ').title(),
            'color': '#6366f1',
            'horizontal': is_horizontal,
        }
        return cfg

    def _cfg_pie_donut(self, data, columns, cfg):
        label_col = columns[0]
        val_col = next(
            (c for c in columns if c != label_col and self._is_numeric(c, data)),
            columns[1] if len(columns) > 1 else columns[0]
        )
        cfg['config'] = {
            'labelKey': label_col,
            'valueKey': val_col,
            'colors': ['#6366f1', '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#8b5cf6', '#f43f5e'],
        }
        return cfg

    def _cfg_scatter(self, data, columns, cfg):
        numeric = [c for c in columns if self._is_numeric(c, data)]
        x, y = (numeric[0], numeric[1]) if len(numeric) >= 2 else (columns[0], columns[1] if len(columns) > 1 else columns[0])
        z = numeric[2] if len(numeric) >= 3 else None
        cfg['config'] = {
            'xAxis': x,
            'yAxis': y,
            'xLabel': x.replace('_', ' ').title(),
            'yLabel': y.replace('_', ' ').title(),
            'color': '#6366f1',
        }
        if z:
            cfg['config']['zAxis'] = z
        return cfg

    def _cfg_treemap(self, data, columns, cfg):
        name_col = next((c for c in columns if any(kw in c.lower() for kw in ['name', 'category', 'label'])), columns[0])
        val_col = next((c for c in columns if c != name_col and self._is_numeric(c, data)), columns[1] if len(columns) > 1 else columns[0])
        cfg['config'] = {
            'nameKey': name_col,
            'valueKey': val_col,
        }
        return cfg

    def _cfg_funnel(self, data, columns, cfg):
        name_col = columns[0]
        val_col = next((c for c in columns if c != name_col and self._is_numeric(c, data)), columns[1] if len(columns) > 1 else columns[0])
        cfg['config'] = {
            'nameKey': name_col,
            'valueKey': val_col,
        }
        return cfg

    def _cfg_radar(self, data, columns, cfg):
        subject_col = next(
            (c for c in columns if not self._is_numeric(c, data)),
            columns[0]
        )
        value_keys = [c for c in columns if c != subject_col and self._is_numeric(c, data)]
        cfg['config'] = {
            'subjectKey': subject_col,
            'valueKeys': value_keys,
        }
        return cfg

    def _cfg_tornado(self, data, columns, cfg):
        cat_col = next(
            (c for c in columns if not self._is_numeric(c, data)),
            columns[0]
        )
        numeric_cols = [c for c in columns if c != cat_col and self._is_numeric(c, data)]
        cfg['config'] = {
            'categoryKey': cat_col,
            'leftKey': numeric_cols[0] if numeric_cols else columns[1],
            'rightKey': numeric_cols[1] if len(numeric_cols) >= 2 else (numeric_cols[0] if numeric_cols else columns[1]),
        }
        return cfg

    def _cfg_table(self, data, columns, cfg):
        cfg['type'] = 'table'
        cfg['config'] = {
            'columns': [c.replace('_', ' ').title() for c in columns],
            'sortable': True,
            'pagination': len(data) > 20,
        }
        return cfg

    # ── Title generation ─────────────────────────────────────────────────

    def _title(self, query: str, chart_type: str) -> str:
        stop = {'show', 'me', 'get', 'find', 'what', 'is', 'the', 'a', 'an', 'of', 'for', 'by', 'give', 'display'}
        words = [w for w in query.lower().split() if w not in stop]
        title = ' '.join(words).strip()
        return title.title() if title else chart_type.replace('_', ' ').title()


# Singleton
chart_recommender = ChartRecommender()

__all__ = ['ChartRecommender', 'chart_recommender']
