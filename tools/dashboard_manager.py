# tools/dashboard_manager.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from tools.chart_recommender import chart_recommender
from tools.sql_tools import sql_executor
from database.redis_client import RedisClient

logger = logging.getLogger(__name__)

# Dashboard TTL: 7 days (matches session TTL so dashboards persist with sessions)
DASHBOARD_TTL_SECONDS = 7 * 24 * 3600


class DashboardManager:
    """Universal dashboard manager — adapts to any PostgreSQL schema."""

    def __init__(self):
        self.redis = RedisClient.get_instance()
        self._fallback_store = {}
        self.schema_cache = {}
        logger.info(f"DashboardManager initialized — Redis: {'connected' if self.redis.is_connected else 'fallback'}")

    # ── Storage helpers ──────────────────────────────────────────────────

    def _get_dashboard(self, session_id: str) -> Optional[Dict]:
        if self.redis.is_connected:
            data = self.redis.get(self.redis.dashboard_key(session_id))
            if data is not None:
                self.redis.refresh_ttl(self.redis.dashboard_key(session_id))
            return data
        return self._fallback_store.get(session_id)

    def _save_dashboard(self, session_id: str, dashboard: Dict) -> None:
        if self.redis.is_connected:
            self.redis.set(self.redis.dashboard_key(session_id), dashboard, ttl=DASHBOARD_TTL_SECONDS)
        else:
            self._fallback_store[session_id] = dashboard

    def _delete_dashboard(self, session_id: str) -> None:
        if self.redis.is_connected:
            self.redis.delete(self.redis.dashboard_key(session_id))
        self._fallback_store.pop(session_id, None)

    def _dashboard_exists(self, session_id: str) -> bool:
        if self.redis.is_connected:
            return self.redis.exists(self.redis.dashboard_key(session_id))
        return session_id in self._fallback_store

    # ── Cache management ─────────────────────────────────────────────────

    def clear_cache(self):
        self.schema_cache = {}
        self._fallback_store = {}
        logger.info("DashboardManager cache cleared")

    # ── Schema detection ─────────────────────────────────────────────────

    def _get_database_schema(self) -> Dict[str, Any]:
        if 'schema' in self.schema_cache:
            return self.schema_cache['schema']
        try:
            tables_result = sql_executor.execute_query("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            if not tables_result['success']:
                return {}

            schema = {'tables': {}}
            for row in tables_result['data']:
                tname = row['table_name']
                col_result = sql_executor.execute_query(f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{tname}'
                    ORDER BY ordinal_position
                """)
                if col_result['success']:
                    schema['tables'][tname] = {
                        'columns': [c['column_name'] for c in col_result['data']],
                        'types': {c['column_name']: c['data_type'] for c in col_result['data']},
                    }

            self.schema_cache['schema'] = schema
            logger.info(f"Schema detected: {len(schema['tables'])} tables")
            return schema
        except Exception as e:
            logger.error(f"Schema detection failed: {e}")
            return {}

    def get_schema(self) -> Dict[str, Any]:
        """Public schema accessor for the API."""
        return self._get_database_schema()

    # ── Column analysis ──────────────────────────────────────────────────

    _NUMERIC_TYPES = {'integer', 'bigint', 'numeric', 'real', 'double precision', 'decimal', 'money', 'float', 'smallint'}
    _DATE_TYPES    = {'date', 'timestamp', 'timestamp without time zone', 'timestamp with time zone', 'time'}
    _TEXT_TYPES    = {'character varying', 'varchar', 'text', 'char', 'character'}

    _SKIP_SUFFIXES = ('_id', '_no', '_code', '_num', '_number', '_via', '_rank', '_order', '_index', '_pos', '_key', '_ref')
    _SKIP_EXACT    = {'id', 'rank', 'order', 'index', 'position', 'sort', 'sequence'}

    def _numeric_cols(self, cols, types):
        return [c for c in cols
                if any(nt in types.get(c, '').lower() for nt in self._NUMERIC_TYPES)
                and c.lower() not in self._SKIP_EXACT
                and not any(c.lower().endswith(s) for s in self._SKIP_SUFFIXES)]

    def _date_cols(self, cols, types):
        return [c for c in cols if any(dt in types.get(c, '').lower() for dt in self._DATE_TYPES)]

    def _cat_cols(self, cols, types):
        return [c for c in cols
                if any(tt in types.get(c, '').lower() for tt in self._TEXT_TYPES)
                and 'description' not in c.lower()
                and 'note' not in c.lower()
                and 'address' not in c.lower()]

    def _col_stats(self, table, col):
        try:
            r = sql_executor.execute_query(
                f"SELECT COUNT({col}) as n, COUNT(DISTINCT {col}) as d FROM {table}"
            )
            if r['success'] and r.get('data'):
                return int(r['data'][0].get('n', 0)), int(r['data'][0].get('d', 0))
        except Exception:
            pass
        return 0, 0

    def _best_cat_col(self, table, cat_cols, min_d=3, max_d=25, min_rows=10):
        best, best_score = (None, 0, 0), 0
        for col in cat_cols:
            n, d = self._col_stats(table, col)
            if d < min_d or d > max_d or n < min_rows:
                continue
            score = n / d if d else 0
            if score > best_score:
                best_score, best = score, (col, n, d)
        # relaxed fallback — any distinct count in range
        if best[0] is None:
            for col in cat_cols:
                n, d = self._col_stats(table, col)
                if min_d <= d <= 30 and n >= min_rows:
                    score = n / d if d else 0
                    if score > best_score:
                        best_score, best = score, (col, n, d)
        return best

    # ── Insightful query generation ───────────────────────────────────────

    def _generate_smart_queries(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate business-insight-focused queries.
        Aims to surface trends, top performers, distributions, and KPIs
        that highlight things that need attention.
        """
        queries = []
        tables = schema.get('tables', {})
        if not tables:
            return queries

        # Pre-analyse all tables
        analysis = {}
        for tname, info in tables.items():
            cols, types = info['columns'], info['types']
            analysis[tname] = {
                'numeric': self._numeric_cols(cols, types),
                'date':    self._date_cols(cols, types),
                'cat':     self._cat_cols(cols, types),
            }

        # ── Find the primary "fact" table ─────────────────────────────
        fact_table = fact_cat = fact_num = None
        best_score = 0

        for tname, info in analysis.items():
            if not info['cat'] or not info['numeric']:
                continue
            col, n, d = self._best_cat_col(tname, info['cat'])
            if col is None:
                continue
            score = n / d if d else 0
            if score > best_score:
                best_score = score
                fact_table, fact_cat, fact_num = tname, col, info['numeric'][0]

        # ── Insightful KPI cards ──────────────────────────────────────
        kpi_idx = 0
        for tname in list(tables.keys())[:5]:
            info = analysis[tname]
            if info['numeric'] and kpi_idx < 2:
                num_col = info['numeric'][0]
                queries.append({
                    'id':         f'kpi_sum_{tname}',
                    'title':      f'Total {num_col.replace("_", " ").title()}',
                    'sql':        f"SELECT SUM({num_col})::numeric(18,2) as total_{num_col} FROM {tname}",
                    'chart_type': 'kpi',
                    'position':   {'x': kpi_idx * 3, 'y': 0, 'w': 3, 'h': 2},
                })
                kpi_idx += 1
            elif kpi_idx < 3:
                queries.append({
                    'id':         f'kpi_count_{tname}',
                    'title':      f'Total {tname.replace("_", " ").title()}',
                    'sql':        f"SELECT COUNT(*) as total FROM {tname}",
                    'chart_type': 'kpi',
                    'position':   {'x': kpi_idx * 3, 'y': 0, 'w': 3, 'h': 2},
                })
                kpi_idx += 1

        if kpi_idx < 3:
            for tname in list(tables.keys())[:3 - kpi_idx]:
                queries.append({
                    'id':         f'kpi_count_{tname}_extra',
                    'title':      f'Total {tname.replace("_", " ").title()}',
                    'sql':        f"SELECT COUNT(*) as total FROM {tname}",
                    'chart_type': 'kpi',
                    'position':   {'x': kpi_idx * 3, 'y': 0, 'w': 3, 'h': 2},
                })
                kpi_idx += 1

        # ── Revenue/metric trend over time (area chart) ───────────────
        for tname, info in analysis.items():
            if info['date'] and info['numeric']:
                date_col = info['date'][0]
                num_col  = info['numeric'][0]
                queries.append({
                    'id':         f'area_trend_{tname}',
                    'title':      f'{num_col.replace("_", " ").title()} Trend Over Time',
                    'sql':        f"""
                        SELECT DATE_TRUNC('month', {date_col})::date AS period,
                               SUM({num_col})::numeric(18,2) AS total
                        FROM {tname}
                        GROUP BY period
                        ORDER BY period
                        LIMIT 36
                    """,
                    'chart_type': 'area',
                    'position':   {'x': 0, 'y': 4, 'w': 8, 'h': 5},
                })
                break

        # ── Top 10 performers (horizontal bar) ────────────────────────
        if fact_table and fact_cat and fact_num:
            queries.append({
                'id':         f'bar_top_{fact_table}',
                'title':      f'Top 10 {fact_cat.replace("_", " ").title()} by {fact_num.replace("_", " ").title()}',
                'sql':        f"""
                    SELECT {fact_cat}, SUM({fact_num})::numeric(18,2) AS total
                    FROM {fact_table}
                    GROUP BY {fact_cat}
                    ORDER BY total DESC
                    LIMIT 10
                """,
                'chart_type': 'bar',
                'position':   {'x': 8, 'y': 4, 'w': 4, 'h': 5},
            })

            # ── Distribution donut (cross-filterable with bar) ────────
            queries.append({
                'id':         f'donut_{fact_table}_{fact_cat}',
                'title':      f'{fact_cat.replace("_", " ").title()} Distribution',
                'sql':        f"""
                    SELECT {fact_cat}, SUM({fact_num})::numeric(18,2) AS value
                    FROM {fact_table}
                    GROUP BY {fact_cat}
                    ORDER BY value DESC
                    LIMIT 10
                """,
                'chart_type': 'donut',
                'position':   {'x': 0, 'y': 9, 'w': 4, 'h': 5},
            })

        # ── Bottom 5 — need attention ─────────────────────────────────
        if fact_table and fact_cat and fact_num:
            queries.append({
                'id':         f'bar_bottom_{fact_table}',
                'title':      f'Bottom 5 {fact_cat.replace("_", " ").title()} — Needs Attention',
                'sql':        f"""
                    SELECT {fact_cat}, SUM({fact_num})::numeric(18,2) AS total
                    FROM {fact_table}
                    GROUP BY {fact_cat}
                    ORDER BY total ASC
                    LIMIT 5
                """,
                'chart_type': 'bar',
                'position':   {'x': 4, 'y': 9, 'w': 4, 'h': 5},
            })

        # ── Scatter: relationship between two numeric metrics ─────────
        for tname, info in analysis.items():
            if len(info['numeric']) >= 2:
                n1, n2 = info['numeric'][0], info['numeric'][1]
                queries.append({
                    'id':         f'scatter_{tname}',
                    'title':      f'{n1.replace("_"," ").title()} vs {n2.replace("_"," ").title()} Correlation',
                    'sql':        f"""
                        SELECT {n1}, {n2}
                        FROM {tname}
                        WHERE {n1} IS NOT NULL AND {n2} IS NOT NULL
                        LIMIT 200
                    """,
                    'chart_type': 'scatter',
                    'position':   {'x': 8, 'y': 9, 'w': 4, 'h': 5},
                })
                break

        return queries[:9]  # cap at 9 charts for a clean initial layout

    # ── Parallel initial dashboard generation ────────────────────────────

    def generate_initial_dashboard(self, user_email: str, session_id: str) -> Dict[str, Any]:
        """
        Generate default dashboard by auto-detecting schema.
        SQL queries run in parallel for fast load times.
        Result persisted to Redis with 7-day TTL.
        """
        logger.info(f"Generating initial dashboard for {user_email} (session: {session_id})")

        dashboard = {
            'dashboard_id': str(uuid.uuid4()),
            'user_email':   user_email,
            'session_id':   session_id,
            'created_at':   datetime.now().isoformat(),
            'charts':       [],
            'filters':      {},
            'layout':       'grid',
        }

        schema = self._get_database_schema()
        if not schema or not schema.get('tables'):
            logger.warning("No schema detected — returning empty dashboard")
            self._save_dashboard(session_id, dashboard)
            return dashboard

        query_defs = self._generate_smart_queries(schema)
        logger.info(f"Generated {len(query_defs)} smart queries — executing in parallel")

        def run_query(qdef):
            try:
                result = sql_executor.execute_query(qdef['sql'])
                if result['success'] and result.get('data'):
                    cfg = chart_recommender.extract_chart_config(
                        data=result['data'],
                        chart_type=qdef['chart_type'],
                        query=qdef['title'],
                    )
                    return {
                        'chart_id':   qdef['id'],
                        'title':      qdef['title'],
                        'type':       qdef['chart_type'],
                        'data':       result['data'],
                        'config':     cfg.get('config', {}),
                        'sql':        qdef['sql'].strip(),
                        'position':   qdef['position'],
                        'created_at': datetime.now().isoformat(),
                    }
                else:
                    logger.warning(f"No data for: {qdef['title']}")
            except Exception as e:
                logger.error(f"Query failed '{qdef['title']}': {e}")
            return None

        # Run all queries in parallel (up to 8 workers)
        charts_by_id = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(run_query, qdef): qdef['id'] for qdef in query_defs}
            for future in as_completed(futures):
                chart = future.result()
                if chart:
                    charts_by_id[futures[future]] = chart

        # Preserve original order
        for qdef in query_defs:
            chart = charts_by_id.get(qdef['id'])
            if chart:
                dashboard['charts'].append(chart)

        self._save_dashboard(session_id, dashboard)
        logger.info(f"Dashboard saved — {len(dashboard['charts'])} charts")
        return dashboard

    # ── Chart addition ────────────────────────────────────────────────────

    def add_chart_from_query(self, session_id: str, query: str, sql: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not result.get('success') or not result.get('data'):
            return None

        data = result['data']
        chart_type = chart_recommender.recommend_chart_type(query=query, data=data, query_result_count=len(data))
        chart_cfg  = chart_recommender.extract_chart_config(data=data, chart_type=chart_type, query=query)

        chart = {
            'chart_id':   str(uuid.uuid4()),
            'title':      chart_cfg.get('title', query),
            'type':       chart_type,
            'data':       data,
            'config':     chart_cfg.get('config', {}),
            'sql':        sql,
            'query':      query,
            'position':   self._next_position(session_id),
            'created_at': datetime.now().isoformat(),
        }

        dashboard = self._get_dashboard(session_id)
        if dashboard is not None:
            dashboard['charts'].append(chart)
            self._save_dashboard(session_id, dashboard)
            logger.info(f"Added chart: {chart['title']} ({chart_type})")

        return chart

    def add_manual_chart(self, session_id: str, chart_def: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a manually-configured chart (from the chart builder UI)."""
        dashboard = self._get_dashboard(session_id)
        if dashboard is None:
            return None

        chart = {
            'chart_id':   str(uuid.uuid4()),
            'title':      chart_def.get('title', 'Custom Chart'),
            'type':       chart_def.get('type', 'bar'),
            'data':       chart_def.get('data', []),
            'config':     chart_def.get('config', {}),
            'sql':        chart_def.get('sql', ''),
            'query':      chart_def.get('query', 'Manual Chart'),
            'position':   self._next_position(session_id),
            'created_at': datetime.now().isoformat(),
        }

        dashboard['charts'].append(chart)
        self._save_dashboard(session_id, dashboard)
        logger.info(f"Added manual chart: {chart['title']}")
        return chart

    def _next_position(self, session_id: str) -> Dict[str, int]:
        dashboard = self._get_dashboard(session_id)
        if not dashboard or not dashboard.get('charts'):
            return {'x': 0, 'y': 0, 'w': 6, 'h': 5}
        max_y = max(c['position']['y'] + c['position']['h'] for c in dashboard['charts'])
        return {'x': 0, 'y': max_y, 'w': 6, 'h': 5}

    # ── Public dashboard accessors ────────────────────────────────────────

    def get_dashboard(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._get_dashboard(session_id)

    def remove_chart(self, session_id: str, chart_id: str) -> bool:
        dashboard = self._get_dashboard(session_id)
        if not dashboard:
            return False
        before = len(dashboard['charts'])
        dashboard['charts'] = [c for c in dashboard['charts'] if c['chart_id'] != chart_id]
        if len(dashboard['charts']) == before:
            return False
        self._save_dashboard(session_id, dashboard)
        return True

    def update_chart_position(self, session_id: str, chart_id: str, position: Dict[str, int]) -> bool:
        dashboard = self._get_dashboard(session_id)
        if not dashboard:
            return False
        for chart in dashboard['charts']:
            if chart['chart_id'] == chart_id:
                chart['position'] = position
                self._save_dashboard(session_id, dashboard)
                return True
        return False

    def update_chart_config(self, session_id: str, chart_id: str, updates: Dict[str, Any]) -> Optional[Dict]:
        """Update chart title, type, color, or axis config."""
        dashboard = self._get_dashboard(session_id)
        if not dashboard:
            return None
        for chart in dashboard['charts']:
            if chart['chart_id'] == chart_id:
                if 'title' in updates:
                    chart['title'] = updates['title']
                if 'type' in updates:
                    chart['type'] = updates['type']
                if 'config' in updates:
                    chart['config'].update(updates['config'])
                self._save_dashboard(session_id, dashboard)
                return chart
        return None

    def apply_filter(self, session_id: str, filter_key: str, filter_value: Any) -> Dict[str, Any]:
        dashboard = self._get_dashboard(session_id)
        if not dashboard:
            return {}
        dashboard['filters'][filter_key] = filter_value
        for chart in dashboard['charts']:
            try:
                filtered_sql = self._inject_filter(chart['sql'], filter_key, filter_value)
                result = sql_executor.execute_query(filtered_sql)
                if result['success'] and result.get('data'):
                    chart['data'] = result['data']
            except Exception as e:
                logger.error(f"Filter failed on chart {chart['title']}: {e}")
        self._save_dashboard(session_id, dashboard)
        return dashboard

    def _inject_filter(self, sql: str, key: str, value: Any) -> str:
        sql = sql.strip()
        clause = f"{key} = '{value}'"
        upper = sql.upper()
        if 'WHERE' in upper:
            idx = upper.rfind('WHERE')
            after = sql[idx + 5:]
            for kw in ['GROUP BY', 'ORDER BY', 'LIMIT', 'HAVING']:
                ki = after.upper().find(kw)
                if ki != -1:
                    return sql[:idx + 5] + after[:ki] + f' AND {clause} ' + after[ki:]
            return sql + f' AND {clause}'
        for kw in ['GROUP BY', 'ORDER BY', 'LIMIT']:
            ki = upper.find(kw)
            if ki != -1:
                return sql[:ki] + f' WHERE {clause} ' + sql[ki:]
        return sql + f' WHERE {clause}'

    def clear_filters(self, session_id: str) -> Dict[str, Any]:
        dashboard = self._get_dashboard(session_id)
        if not dashboard:
            return {}
        dashboard['filters'] = {}
        for chart in dashboard['charts']:
            try:
                result = sql_executor.execute_query(chart['sql'])
                if result['success'] and result.get('data'):
                    chart['data'] = result['data']
            except Exception as e:
                logger.error(f"Refresh failed on chart {chart['title']}: {e}")
        self._save_dashboard(session_id, dashboard)
        return dashboard

    def clear_dashboard(self, session_id: str) -> bool:
        dashboard = self._get_dashboard(session_id)
        if not dashboard:
            return False
        dashboard['charts'] = []
        dashboard['filters'] = {}
        self._save_dashboard(session_id, dashboard)
        return True


# Singleton
dashboard_manager = DashboardManager()

__all__ = ['DashboardManager', 'dashboard_manager']
