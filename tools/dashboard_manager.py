# tools/dashboard_manager.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from tools.chart_recommender import chart_recommender
from tools.sql_tools import sql_executor
from database.redis_client import RedisClient

logger = logging.getLogger(__name__)


class DashboardManager:
    """Universal dashboard manager that adapts to any database schema"""

    def __init__(self):
        self.redis = RedisClient.get_instance()
        self._fallback_store = {}   # In-memory fallback if Redis is down
        self.schema_cache = {}
        logger.info(f"DashboardManager initialized — Redis: {'connected' if self.redis.is_connected else 'fallback in-memory mode'}")

    # ------------------------------------------------------------------ #
    #  Redis-backed storage helpers                                        #
    # ------------------------------------------------------------------ #

    def _get_dashboard(self, session_id: str) -> Optional[Dict]:
        """Get dashboard from Redis, fall back to in-memory dict."""
        if self.redis.is_connected:
            data = self.redis.get(self.redis.dashboard_key(session_id))
            if data is not None:
                self.redis.refresh_ttl(self.redis.dashboard_key(session_id))
            return data
        return self._fallback_store.get(session_id)

    def _save_dashboard(self, session_id: str, dashboard: Dict) -> None:
        """Save dashboard to Redis, fall back to in-memory dict."""
        if self.redis.is_connected:
            self.redis.set(self.redis.dashboard_key(session_id), dashboard)
        else:
            self._fallback_store[session_id] = dashboard

    def _delete_dashboard(self, session_id: str) -> None:
        """Delete dashboard from Redis and fallback store."""
        if self.redis.is_connected:
            self.redis.delete(self.redis.dashboard_key(session_id))
        self._fallback_store.pop(session_id, None)

    def _dashboard_exists(self, session_id: str) -> bool:
        """Check if a dashboard exists for this session."""
        if self.redis.is_connected:
            return self.redis.exists(self.redis.dashboard_key(session_id))
        return session_id in self._fallback_store

    # ------------------------------------------------------------------ #
    #  Cache management                                                    #
    # ------------------------------------------------------------------ #

    def clear_cache(self):
        """Invalidate schema and dashboard caches (call after schema changes)."""
        self.schema_cache = {}
        self._fallback_store = {}
        logger.info("DashboardManager cache cleared")

    # ------------------------------------------------------------------ #
    #  Schema detection                                                    #
    # ------------------------------------------------------------------ #

    def _get_database_schema(self) -> Dict[str, Any]:
        """
        Auto-detect database schema by querying information_schema.
        Returns dict with tables, columns, and relationships.
        """
        if 'schema' in self.schema_cache:
            return self.schema_cache['schema']

        try:
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """
            tables_result = sql_executor.execute_query(tables_query)

            if not tables_result['success']:
                logger.error("Failed to get tables from database")
                return {}

            schema = {'tables': {}}

            for table_row in tables_result['data']:
                table_name = table_row['table_name']

                columns_query = f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """
                columns_result = sql_executor.execute_query(columns_query)

                if columns_result['success']:
                    schema['tables'][table_name] = {
                        'columns': [col['column_name'] for col in columns_result['data']],
                        'types': {col['column_name']: col['data_type'] for col in columns_result['data']}
                    }

            self.schema_cache['schema'] = schema
            logger.info(f"✅ Detected schema with {len(schema['tables'])} tables")
            return schema

        except Exception as e:
            logger.error(f"Error detecting schema: {e}")
            return {}

    # ------------------------------------------------------------------ #
    #  Column analysis helpers                                             #
    # ------------------------------------------------------------------ #

    def _find_numeric_columns(self, table_name: str, columns: List[str], types: Dict[str, str]) -> List[str]:
        """Find columns with numeric data suitable for aggregation (SUM/AVG)."""
        numeric_types = ['integer', 'bigint', 'numeric', 'real', 'double precision', 'decimal', 'money', 'float']
        skip_suffixes = ('_id', '_no', '_code', '_num', '_number', '_via', '_rank', '_order', '_index', '_pos')
        skip_exact = {'id', 'rank', 'order', 'index', 'position', 'sort'}
        numeric_cols = []

        for col in columns:
            col_lower = col.lower()
            col_type = types.get(col, '').lower()
            if not any(nt in col_type for nt in numeric_types):
                continue
            if col_lower in skip_exact:
                continue
            if any(col_lower.endswith(s) for s in skip_suffixes):
                continue
            numeric_cols.append(col)

        return numeric_cols

    def _find_date_columns(self, columns: List[str], types: Dict[str, str]) -> List[str]:
        """Find date/timestamp columns."""
        date_types = ['date', 'timestamp', 'time']
        date_cols = []

        for col in columns:
            col_type = types.get(col, '').lower()
            if any(dt in col_type for dt in date_types):
                date_cols.append(col)

        return date_cols

    def _find_categorical_columns(self, columns: List[str], types: Dict[str, str]) -> List[str]:
        """Find categorical columns (text, varchar, etc.)"""
        text_types = ['character varying', 'varchar', 'text', 'char']
        cat_cols = []

        for col in columns:
            col_type = types.get(col, '').lower()
            if any(tt in col_type for tt in text_types):
                if 'description' not in col.lower() and 'note' not in col.lower():
                    cat_cols.append(col)

        return cat_cols

    def _get_col_stats(self, table_name: str, col: str):
        """Return (non_null_rows, distinct_count) for a column."""
        try:
            result = sql_executor.execute_query(
                f"SELECT COUNT({col}) as non_null, COUNT(DISTINCT {col}) as distinct_cnt FROM {table_name}"
            )
            if result['success'] and result.get('data'):
                non_null = int(result['data'][0].get('non_null', 0))
                distinct = int(result['data'][0].get('distinct_cnt', 0))
                return non_null, distinct
        except Exception:
            pass
        return 0, 0

    def _find_best_cat_col(self, table_name: str, cat_cols: List[str],
                           min_distinct: int = 3, max_distinct: int = 25,
                           min_ratio: float = 2.0, min_non_null: int = 10):
        """
        Pick the categorical column that gives the most interesting distribution.
        Falls back to relaxed criteria if nothing passes strict check.
        Returns (col_name, non_null, distinct) or (None, 0, 0).
        """
        best = (None, 0, 0)
        best_ratio = 0

        for col in cat_cols:
            non_null, distinct = self._get_col_stats(table_name, col)
            if distinct < min_distinct or distinct > max_distinct:
                continue
            if non_null < min_non_null:
                continue
            ratio = non_null / distinct if distinct else 0
            if ratio < min_ratio:
                continue
            if ratio > best_ratio:
                best_ratio = ratio
                best = (col, non_null, distinct)

        # Relaxed fallback: drop min_ratio requirement
        if best[0] is None:
            for col in cat_cols:
                non_null, distinct = self._get_col_stats(table_name, col)
                if distinct < min_distinct or distinct > max_distinct:
                    continue
                if non_null < min_non_null:
                    continue
                ratio = non_null / distinct if distinct else 0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = (col, non_null, distinct)

        return best

    # ------------------------------------------------------------------ #
    #  Smart query generation                                              #
    # ------------------------------------------------------------------ #

    def _generate_smart_queries(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Auto-generate smart queries based on detected schema.
        PIE and BAR charts come from the SAME table + SAME categorical column
        so that cross-filtering propagates between them.
        """
        queries = []
        tables = schema.get('tables', {})
        if not tables:
            return queries

        table_analysis = {}
        for table_name, table_info in tables.items():
            cols = table_info['columns']
            types = table_info['types']
            table_analysis[table_name] = {
                'numeric': self._find_numeric_columns(table_name, cols, types),
                'date':    self._find_date_columns(cols, types),
                'cat':     self._find_categorical_columns(cols, types),
            }

        # KPI cards: 3 count cards from three tables
        for table_name in list(tables.keys())[:3]:
            queries.append({
                'id':         f'count_{table_name}',
                'title':      f'Total {table_name.replace("_", " ").title()}',
                'sql':        f"SELECT COUNT(*) as total FROM {table_name}",
                'chart_type': 'kpi',
                'position':   {'x': (len(queries) % 4) * 3, 'y': 0, 'w': 3, 'h': 2},
            })

        # Find the best fact table for cross-filterable BAR + PIE
        fact_table = None
        fact_cat_col = None
        fact_numeric_col = None
        best_score = 0

        for table_name, info in table_analysis.items():
            if not info['cat'] or not info['numeric']:
                continue
            col, total, distinct = self._find_best_cat_col(table_name, info['cat'])
            if col is None:
                continue
            score = total / distinct if distinct else 0
            if score > best_score:
                best_score = score
                fact_table = table_name
                fact_cat_col = col
                fact_numeric_col = info['numeric'][0]

        # BAR chart
        if fact_table and fact_cat_col:
            queries.append({
                'id':         f'bar_{fact_table}_{fact_cat_col}',
                'title':      f'{fact_cat_col.replace("_", " ").title()} by {fact_numeric_col.replace("_", " ").title()}',
                'sql':        f"""
                    SELECT {fact_cat_col}, SUM({fact_numeric_col}) as total
                    FROM {fact_table}
                    GROUP BY {fact_cat_col}
                    ORDER BY total DESC
                    LIMIT 10
                """,
                'chart_type': 'bar',
                'position':   {'x': 0, 'y': 4, 'w': 6, 'h': 4},
            })

        # LINE chart: prefer fact table if it has a date col
        line_added = False
        if fact_table and table_analysis[fact_table]['date']:
            date_col = table_analysis[fact_table]['date'][0]
            num_col  = fact_numeric_col
            queries.append({
                'id':         f'line_{fact_table}',
                'title':      f'{num_col.replace("_", " ").title()} Over Time',
                'sql':        f"""
                    SELECT {date_col}::date AS date, SUM({num_col}) as total
                    FROM {fact_table}
                    GROUP BY {date_col}::date
                    ORDER BY date
                    LIMIT 365
                """,
                'chart_type': 'line',
                'position':   {'x': 6, 'y': 4, 'w': 6, 'h': 4},
            })
            line_added = True

        if not line_added:
            for table_name, info in table_analysis.items():
                if info['date'] and info['numeric']:
                    date_col = info['date'][0]
                    num_col  = info['numeric'][0]
                    queries.append({
                        'id':         f'line_{table_name}',
                        'title':      f'{num_col.replace("_", " ").title()} Over Time',
                        'sql':        f"""
                            SELECT {date_col}::date AS date, SUM({num_col}) as total
                            FROM {table_name}
                            GROUP BY {date_col}::date
                            ORDER BY date
                            LIMIT 365
                        """,
                        'chart_type': 'line',
                        'position':   {'x': 6, 'y': 4, 'w': 6, 'h': 4},
                    })
                    break

        # PIE chart: SAME fact table + SAME cat col as bar (enables cross-filtering)
        if fact_table and fact_cat_col:
            queries.append({
                'id':         f'pie_{fact_table}_{fact_cat_col}',
                'title':      f'{fact_cat_col.replace("_", " ").title()} Distribution',
                'sql':        f"""
                    SELECT {fact_cat_col}, SUM({fact_numeric_col}) as value
                    FROM {fact_table}
                    GROUP BY {fact_cat_col}
                    ORDER BY value DESC
                    LIMIT 10
                """,
                'chart_type': 'pie',
                'position':   {'x': 0, 'y': 8, 'w': 6, 'h': 4},
            })

        return queries[:7]

    # ------------------------------------------------------------------ #
    #  Public dashboard API                                                #
    # ------------------------------------------------------------------ #

    def generate_initial_dashboard(self, user_email: str, session_id: str) -> Dict[str, Any]:
        """
        Generate default dashboard by auto-detecting schema.
        Works with ANY PostgreSQL database.
        Result is stored in Redis with 24h TTL.
        """
        logger.info(f"Generating initial dashboard for {user_email} (session: {session_id})")

        dashboard = {
            'dashboard_id': str(uuid.uuid4()),
            'user_email': user_email,
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'charts': [],
            'filters': {},
            'layout': 'grid'
        }

        schema = self._get_database_schema()

        if not schema or not schema.get('tables'):
            logger.warning("⚠️ No schema detected - returning empty dashboard")
            self._save_dashboard(session_id, dashboard)
            return dashboard

        default_queries = self._generate_smart_queries(schema)
        logger.info(f"📊 Generated {len(default_queries)} smart queries")

        for query_def in default_queries:
            try:
                result = sql_executor.execute_query(query_def['sql'])

                if result['success'] and result.get('data'):
                    chart_config = chart_recommender.extract_chart_config(
                        data=result['data'],
                        chart_type=query_def['chart_type'],
                        query=query_def['title']
                    )

                    chart = {
                        'chart_id':   query_def['id'],
                        'title':      query_def['title'],
                        'type':       query_def['chart_type'],
                        'data':       result['data'],
                        'config':     chart_config.get('config', {}),
                        'sql':        query_def['sql'],
                        'position':   query_def['position'],
                        'created_at': datetime.now().isoformat()
                    }

                    dashboard['charts'].append(chart)
                    logger.info(f"✅ Generated chart: {query_def['title']}")
                else:
                    logger.warning(f"⚠️ Query returned no data: {query_def['title']}")

            except Exception as e:
                logger.error(f"❌ Failed to generate chart '{query_def['title']}': {e}")

        self._save_dashboard(session_id, dashboard)
        logger.info(f"✅ Dashboard saved to Redis with {len(dashboard['charts'])} charts")

        return dashboard

    def add_chart_from_query(
        self,
        session_id: str,
        query: str,
        sql: str,
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Add a new chart to dashboard based on a conversation query result."""
        if not result.get('success') or not result.get('data'):
            logger.warning("Cannot create chart - no data")
            return None

        data = result['data']

        chart_type = chart_recommender.recommend_chart_type(
            query=query,
            data=data,
            query_result_count=len(data)
        )

        logger.info(f"📊 Recommended chart type: {chart_type} for query: '{query}'")

        chart_config = chart_recommender.extract_chart_config(
            data=data,
            chart_type=chart_type,
            query=query
        )

        chart = {
            'chart_id':   str(uuid.uuid4()),
            'title':      chart_config.get('title', query),
            'type':       chart_type,
            'data':       data,
            'config':     chart_config.get('config', {}),
            'sql':        sql,
            'query':      query,
            'position':   self._get_next_position(session_id),
            'created_at': datetime.now().isoformat()
        }

        # Load → mutate → save back to Redis atomically
        dashboard = self._get_dashboard(session_id)
        if dashboard is not None:
            dashboard['charts'].append(chart)
            self._save_dashboard(session_id, dashboard)
            logger.info(f"✅ Added chart to dashboard: {chart['title']}")

        return chart

    def _get_next_position(self, session_id: str) -> Dict[str, int]:
        """Calculate position for next chart in grid."""
        dashboard = self._get_dashboard(session_id)

        if not dashboard or not dashboard.get('charts'):
            return {'x': 0, 'y': 0, 'w': 6, 'h': 4}

        max_y = max(
            chart['position']['y'] + chart['position']['h']
            for chart in dashboard['charts']
        )
        return {'x': 0, 'y': max_y, 'w': 6, 'h': 4}

    def get_dashboard(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard for a session."""
        return self._get_dashboard(session_id)

    def remove_chart(self, session_id: str, chart_id: str) -> bool:
        """Remove a chart from dashboard."""
        dashboard = self._get_dashboard(session_id)
        if dashboard is None:
            return False

        original_count = len(dashboard['charts'])
        dashboard['charts'] = [
            chart for chart in dashboard['charts']
            if chart['chart_id'] != chart_id
        ]

        if len(dashboard['charts']) == original_count:
            return False  # Chart not found

        self._save_dashboard(session_id, dashboard)
        logger.info(f"Removed chart {chart_id} from dashboard")
        return True

    def update_chart_position(
        self,
        session_id: str,
        chart_id: str,
        position: Dict[str, int]
    ) -> bool:
        """Update chart position in grid."""
        dashboard = self._get_dashboard(session_id)
        if dashboard is None:
            return False

        for chart in dashboard['charts']:
            if chart['chart_id'] == chart_id:
                chart['position'] = position
                self._save_dashboard(session_id, dashboard)
                return True

        return False

    def apply_filter(
        self,
        session_id: str,
        filter_key: str,
        filter_value: Any
    ) -> Dict[str, Any]:
        """Apply filter to dashboard and re-query all charts."""
        dashboard = self._get_dashboard(session_id)
        if dashboard is None:
            return {}

        dashboard['filters'][filter_key] = filter_value
        logger.info(f"Applied filter: {filter_key} = {filter_value}")

        for chart in dashboard['charts']:
            try:
                filtered_sql = self._add_filter_to_sql(
                    chart['sql'],
                    filter_key,
                    filter_value
                )
                result = sql_executor.execute_query(filtered_sql)

                if result['success'] and result.get('data'):
                    chart['data'] = result['data']
                    logger.info(f"✅ Updated chart: {chart['title']}")

            except Exception as e:
                logger.error(f"Error updating chart {chart['title']}: {e}")

        self._save_dashboard(session_id, dashboard)
        return dashboard

    def _add_filter_to_sql(self, sql: str, filter_key: str, filter_value: Any) -> str:
        """Inject a WHERE clause into an existing SQL query."""
        sql = sql.strip()
        filter_clause = f"{filter_key} = '{filter_value}'"

        if 'WHERE' in sql.upper():
            sql = sql.replace('GROUP BY', f'AND {filter_clause} GROUP BY')
        else:
            if 'GROUP BY' in sql.upper():
                sql = sql.replace('GROUP BY', f'WHERE {filter_clause} GROUP BY')
            elif 'ORDER BY' in sql.upper():
                sql = sql.replace('ORDER BY', f'WHERE {filter_clause} ORDER BY')
            else:
                sql += f' WHERE {filter_clause}'

        return sql

    def clear_filters(self, session_id: str) -> Dict[str, Any]:
        """Clear all filters and re-execute original SQLs."""
        dashboard = self._get_dashboard(session_id)
        if dashboard is None:
            return {}

        dashboard['filters'] = {}

        for chart in dashboard['charts']:
            try:
                result = sql_executor.execute_query(chart['sql'])
                if result['success'] and result.get('data'):
                    chart['data'] = result['data']
            except Exception as e:
                logger.error(f"Error refreshing chart {chart['title']}: {e}")

        self._save_dashboard(session_id, dashboard)
        logger.info("Cleared all filters")
        return dashboard

    def clear_dashboard(self, session_id: str) -> bool:
        """Clear all charts from dashboard."""
        dashboard = self._get_dashboard(session_id)
        if dashboard is None:
            return False

        dashboard['charts'] = []
        dashboard['filters'] = {}
        self._save_dashboard(session_id, dashboard)
        logger.info(f"Cleared dashboard for session {session_id}")
        return True


# Singleton instance
dashboard_manager = DashboardManager()

__all__ = ['DashboardManager', 'dashboard_manager']