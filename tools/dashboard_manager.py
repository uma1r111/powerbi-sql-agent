# tools/dashboard_manager.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from tools.chart_recommender import chart_recommender
from tools.sql_tools import sql_executor

logger = logging.getLogger(__name__)

class DashboardManager:
    """Universal dashboard manager that adapts to any database schema"""
    
    def __init__(self):
        self.active_dashboards = {}
        self.schema_cache = {}
    
    def _get_database_schema(self) -> Dict[str, Any]:
        """
        Auto-detect database schema by querying information_schema
        Returns dict with tables, columns, and relationships
        """
        if 'schema' in self.schema_cache:
            return self.schema_cache['schema']
        
        try:
            # Get all tables
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
            
            # For each table, get columns
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
    
    def _find_numeric_columns(self, table_name: str, columns: List[str], types: Dict[str, str]) -> List[str]:
        """Find columns with numeric data (for aggregation)"""
        numeric_types = ['integer', 'bigint', 'numeric', 'real', 'double precision', 'decimal', 'money']
        numeric_cols = []
        
        for col in columns:
            col_type = types.get(col, '').lower()
            if any(nt in col_type for nt in numeric_types):
                # Exclude ID columns
                if not col.lower().endswith('_id') and 'id' != col.lower():
                    numeric_cols.append(col)
        
        return numeric_cols
    
    def _find_date_columns(self, columns: List[str], types: Dict[str, str]) -> List[str]:
        """Find date/timestamp columns"""
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
                # Exclude very long text fields
                if 'description' not in col.lower() and 'note' not in col.lower():
                    cat_cols.append(col)
        
        return cat_cols
    
    def _generate_smart_queries(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Auto-generate smart queries based on detected schema
        Returns list of query definitions
        """
        queries = []
        tables = schema.get('tables', {})
        
        if not tables:
            return queries
        
        # Strategy: Find the most data-rich tables
        for table_name, table_info in list(tables.items())[:3]:  # Limit to 3 main tables
            columns = table_info['columns']
            types = table_info['types']
            
            numeric_cols = self._find_numeric_columns(table_name, columns, types)
            date_cols = self._find_date_columns(columns, types)
            cat_cols = self._find_categorical_columns(columns, types)
            
            # Query 1: Count of records (KPI)
            queries.append({
                'id': f'count_{table_name}',
                'title': f'Total {table_name.replace("_", " ").title()}',
                'sql': f"SELECT COUNT(*) as total FROM {table_name}",
                'chart_type': 'kpi',
                'position': {'x': len(queries) % 3 * 3, 'y': 0, 'w': 3, 'h': 2}
            })
            
            # Query 2: If has numeric column, sum it
            if numeric_cols:
                main_numeric = numeric_cols[0]
                queries.append({
                    'id': f'sum_{table_name}_{main_numeric}',
                    'title': f'Total {main_numeric.replace("_", " ").title()}',
                    'sql': f"SELECT SUM({main_numeric}) as total FROM {table_name}",
                    'chart_type': 'kpi',
                    'position': {'x': len(queries) % 3 * 3, 'y': 0, 'w': 3, 'h': 2}
                })
            
            # Query 3: If has categorical column + numeric, group by
            if cat_cols and numeric_cols:
                cat_col = cat_cols[0]
                num_col = numeric_cols[0]
                
                queries.append({
                    'id': f'top_{table_name}_{cat_col}',
                    'title': f'Top {cat_col.replace("_", " ").title()} by {num_col.replace("_", " ").title()}',
                    'sql': f"""
                        SELECT {cat_col}, SUM({num_col}) as total
                        FROM {table_name}
                        GROUP BY {cat_col}
                        ORDER BY total DESC
                        LIMIT 10
                    """,
                    'chart_type': 'bar',
                    'position': {'x': 0, 'y': 2, 'w': 6, 'h': 4}
                })
            
            # Query 4: If has date column + numeric, time series
            if date_cols and numeric_cols:
                date_col = date_cols[0]
                num_col = numeric_cols[0]
                
                queries.append({
                    'id': f'trend_{table_name}',
                    'title': f'{num_col.replace("_", " ").title()} Over Time',
                    'sql': f"""
                        SELECT DATE({date_col}) as date, SUM({num_col}) as total
                        FROM {table_name}
                        GROUP BY DATE({date_col})
                        ORDER BY date
                        LIMIT 365
                    """,
                    'chart_type': 'line',
                    'position': {'x': 6, 'y': 2, 'w': 6, 'h': 4}
                })
            
            # Query 5: If has multiple categorical columns, distribution
            if len(cat_cols) >= 2:
                cat_col = cat_cols[0]
                
                queries.append({
                    'id': f'dist_{table_name}_{cat_col}',
                    'title': f'Distribution by {cat_col.replace("_", " ").title()}',
                    'sql': f"""
                        SELECT {cat_col}, COUNT(*) as count
                        FROM {table_name}
                        GROUP BY {cat_col}
                        ORDER BY count DESC
                        LIMIT 10
                    """,
                    'chart_type': 'pie',
                    'position': {'x': 0, 'y': 6, 'w': 6, 'h': 4}
                })
        
        return queries[:6]  # Return max 6 charts
    
    def generate_initial_dashboard(self, user_email: str, session_id: str) -> Dict[str, Any]:
        """
        Generate default dashboard by auto-detecting schema
        Works with ANY database
        """
        logger.info(f"Generating initial dashboard for {user_email}")
        
        dashboard = {
            'dashboard_id': str(uuid.uuid4()),
            'user_email': user_email,
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'charts': [],
            'filters': {},
            'layout': 'grid'
        }
        
        # Auto-detect schema
        schema = self._get_database_schema()
        
        if not schema or not schema.get('tables'):
            logger.warning("⚠️ No schema detected - returning empty dashboard")
            self.active_dashboards[session_id] = dashboard
            return dashboard
        
        # Generate smart queries based on schema
        default_queries = self._generate_smart_queries(schema)
        
        logger.info(f"📊 Generated {len(default_queries)} smart queries")
        
        # Execute queries and generate charts
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
                        'chart_id': query_def['id'],
                        'title': query_def['title'],
                        'type': query_def['chart_type'],
                        'data': result['data'],
                        'config': chart_config.get('config', {}),
                        'sql': query_def['sql'],
                        'position': query_def['position'],
                        'created_at': datetime.now().isoformat()
                    }
                    
                    dashboard['charts'].append(chart)
                    logger.info(f"✅ Generated chart: {query_def['title']}")
                else:
                    logger.warning(f"⚠️ Query returned no data: {query_def['title']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to generate chart '{query_def['title']}': {e}")
        
        self.active_dashboards[session_id] = dashboard
        logger.info(f"✅ Dashboard created with {len(dashboard['charts'])} charts")
        
        return dashboard
    
    def add_chart_from_query(
        self, 
        session_id: str, 
        query: str, 
        sql: str, 
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Add a new chart to dashboard based on query result"""
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
            'chart_id': str(uuid.uuid4()),
            'title': chart_config.get('title', query),
            'type': chart_type,
            'data': data,
            'config': chart_config.get('config', {}),
            'sql': sql,
            'query': query,
            'position': self._get_next_position(session_id),
            'created_at': datetime.now().isoformat()
        }
        
        if session_id in self.active_dashboards:
            self.active_dashboards[session_id]['charts'].append(chart)
            logger.info(f"✅ Added chart to dashboard: {chart['title']}")
        
        return chart
    
    def _get_next_position(self, session_id: str) -> Dict[str, int]:
        """Calculate position for next chart in grid"""
        if session_id not in self.active_dashboards:
            return {'x': 0, 'y': 0, 'w': 6, 'h': 4}
        
        charts = self.active_dashboards[session_id]['charts']
        
        if not charts:
            return {'x': 0, 'y': 0, 'w': 6, 'h': 4}
        
        max_y = max(chart['position']['y'] + chart['position']['h'] for chart in charts)
        return {'x': 0, 'y': max_y, 'w': 6, 'h': 4}
    
    def get_dashboard(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard for a session"""
        return self.active_dashboards.get(session_id)
    
    def remove_chart(self, session_id: str, chart_id: str) -> bool:
        """Remove a chart from dashboard"""
        if session_id not in self.active_dashboards:
            return False
        
        dashboard = self.active_dashboards[session_id]
        dashboard['charts'] = [
            chart for chart in dashboard['charts'] 
            if chart['chart_id'] != chart_id
        ]
        
        logger.info(f"Removed chart {chart_id} from dashboard")
        return True
    
    def update_chart_position(
        self, 
        session_id: str, 
        chart_id: str, 
        position: Dict[str, int]
    ) -> bool:
        """Update chart position in grid"""
        if session_id not in self.active_dashboards:
            return False
        
        dashboard = self.active_dashboards[session_id]
        
        for chart in dashboard['charts']:
            if chart['chart_id'] == chart_id:
                chart['position'] = position
                return True
        
        return False
    
    def apply_filter(
        self, 
        session_id: str, 
        filter_key: str, 
        filter_value: Any
    ) -> Dict[str, Any]:
        """Apply filter to dashboard and re-query all charts"""
        if session_id not in self.active_dashboards:
            return {}
        
        dashboard = self.active_dashboards[session_id]
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
        
        return dashboard
    
    def _add_filter_to_sql(self, sql: str, filter_key: str, filter_value: Any) -> str:
        """Add WHERE clause to SQL query"""
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
        """Clear all filters and refresh dashboard"""
        if session_id not in self.active_dashboards:
            return {}
        
        dashboard = self.active_dashboards[session_id]
        dashboard['filters'] = {}
        
        for chart in dashboard['charts']:
            try:
                result = sql_executor.execute_query(chart['sql'])
                
                if result['success'] and result.get('data'):
                    chart['data'] = result['data']
                    
            except Exception as e:
                logger.error(f"Error refreshing chart {chart['title']}: {e}")
        
        logger.info("Cleared all filters")
        return dashboard
    
    def clear_dashboard(self, session_id: str) -> bool:
        """Clear all charts from dashboard"""
        if session_id in self.active_dashboards:
            self.active_dashboards[session_id]['charts'] = []
            self.active_dashboards[session_id]['filters'] = {}
            logger.info(f"Cleared dashboard for session {session_id}")
            return True
        return False


# Singleton instance
dashboard_manager = DashboardManager()

__all__ = ['DashboardManager', 'dashboard_manager']