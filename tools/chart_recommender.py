import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ChartRecommender:
    """AI-powered chart type recommendation based on query and data structure"""
    
    def __init__(self):
        self.chart_types = {
            'line': ['time', 'trend', 'over time', 'monthly', 'yearly', 'daily', 'weekly'],
            'bar': ['top', 'bottom', 'compare', 'comparison', 'best', 'worst', 'ranking'],
            'pie': ['distribution', 'breakdown', 'by category', 'proportion', 'percentage'],
            'area': ['cumulative', 'stacked', 'total over time'],
            'scatter': ['correlation', 'relationship', 'vs', 'versus'],
            'table': ['list', 'show all', 'details', 'full data'],
            'kpi': ['total', 'sum', 'count', 'average', 'how many', 'how much']
        }
    
    def recommend_chart_type(self, query: str, data: List[Dict], query_result_count: int = 0) -> str:
        """
        Recommend best chart type based on query intent and data structure
        
        Returns: 'line', 'bar', 'pie', 'area', 'scatter', 'table', 'kpi'
        """
        if not data or len(data) == 0:
            return 'table'
        
        query_lower = query.lower()
        columns = list(data[0].keys())
        
        # Single row, single numeric value → KPI Card
        if len(data) == 1 and len(columns) == 1:
            return 'kpi'
        
        # Check for explicit keywords in query
        for chart_type, keywords in self.chart_types.items():
            if any(keyword in query_lower for keyword in keywords):
                # Verify data structure supports this chart type
                if self._validate_chart_type_for_data(chart_type, data):
                    return chart_type
        
        # Analyze data structure
        return self._infer_from_data_structure(data, columns)
    
    def _validate_chart_type_for_data(self, chart_type: str, data: List[Dict]) -> bool:
        """Check if data structure supports the chart type"""
        if not data:
            return False
        
        columns = list(data[0].keys())
        
        if chart_type == 'kpi':
            return len(data) == 1 and len(columns) <= 2
        
        if chart_type in ['line', 'area']:
            # Need time-based column
            return any(self._is_date_column(col) for col in columns)
        
        if chart_type == 'scatter':
            # Need at least 2 numeric columns
            numeric_cols = [col for col in columns if self._is_numeric_column(col, data)]
            return len(numeric_cols) >= 2
        
        return True
    
    def _infer_from_data_structure(self, data: List[Dict], columns: List[str]) -> str:
        """Infer chart type from data structure alone"""
        
        # Check for date columns
        date_columns = [col for col in columns if self._is_date_column(col)]
        
        # Time series data → Line chart
        if date_columns and len(columns) == 2:
            return 'line'
        
        # Many rows with categories → Bar chart
        if len(data) > 2 and len(data) <= 20:
            return 'bar'
        
        # Few categories → Pie chart
        if len(data) <= 6 and len(columns) == 2:
            return 'pie'
        
        # Many rows → Table
        if len(data) > 20:
            return 'table'
        
        # Default to bar chart
        return 'bar'
    
    def _is_date_column(self, column_name: str) -> bool:
        """Check if column name suggests date/time data"""
        date_keywords = ['date', 'time', 'month', 'year', 'day', 'week', 'period']
        return any(keyword in column_name.lower() for keyword in date_keywords)
    
    def _is_numeric_column(self, column_name: str, data: List[Dict]) -> bool:
        """Check if column contains numeric data"""
        numeric_keywords = ['amount', 'total', 'revenue', 'sales', 'quantity', 'count', 'price']
        
        # Check column name
        if any(keyword in column_name.lower() for keyword in numeric_keywords):
            return True
        
        # Check actual data
        try:
            first_value = data[0].get(column_name)
            return isinstance(first_value, (int, float)) or (
                isinstance(first_value, str) and first_value.replace('.', '').replace('-', '').isdigit()
            )
        except:
            return False
    
    def extract_chart_config(self, data: List[Dict], chart_type: str, query: str) -> Dict[str, Any]:
        """
        Extract chart configuration from data
        Returns config with x-axis, y-axis, title, colors, etc.
        """
        if not data:
            return {}
        
        columns = list(data[0].keys())
        
        config = {
            'type': chart_type,
            'title': self._generate_title(query, chart_type),
            'data': data,
            'config': {}
        }
        
        if chart_type == 'kpi':
            return self._config_kpi(data, columns, config)
        
        elif chart_type in ['line', 'area']:
            return self._config_line_area(data, columns, config, chart_type)
        
        elif chart_type == 'bar':
            return self._config_bar(data, columns, config, query)
        
        elif chart_type == 'pie':
            return self._config_pie(data, columns, config)
        
        elif chart_type == 'scatter':
            return self._config_scatter(data, columns, config)
        
        elif chart_type == 'table':
            return self._config_table(data, columns, config)
        
        return config
    
    def _config_kpi(self, data: List[Dict], columns: List[str], config: Dict) -> Dict:
        """Configure KPI card"""
        value_col = columns[0]
        value = data[0][value_col]
        
        config['config'] = {
            'value': value,
            'label': value_col.replace('_', ' ').title(),
            'format': 'number' if isinstance(value, (int, float)) else 'text'
        }
        
        # If there's a second column, use it for trend
        if len(columns) > 1:
            config['config']['trend'] = data[0][columns[1]]
        
        return config
    
    def _config_line_area(self, data: List[Dict], columns: List[str], config: Dict, chart_type: str) -> Dict:
        """Configure line or area chart"""
        # Find date column
        date_col = None
        for col in columns:
            if self._is_date_column(col):
                date_col = col
                break
        
        if not date_col:
            date_col = columns[0]
        
        # Find numeric column
        value_col = None
        for col in columns:
            if col != date_col and self._is_numeric_column(col, data):
                value_col = col
                break
        
        if not value_col:
            value_col = columns[1] if len(columns) > 1 else columns[0]
        
        config['config'] = {
            'xAxis': date_col,
            'yAxis': value_col,
            'xLabel': date_col.replace('_', ' ').title(),
            'yLabel': value_col.replace('_', ' ').title(),
            'color': '#3b82f6',
            'curved': chart_type == 'area'
        }
        
        return config
    
    def _config_bar(self, data: List[Dict], columns: List[str], config: Dict, query: str) -> Dict:
        """Configure bar chart"""
        # Find category column (usually first column or name column)
        category_col = columns[0]
        for col in columns:
            if 'name' in col.lower() or 'category' in col.lower():
                category_col = col
                break
        
        # Find value column — must be a different column AND numeric
        value_col = None
        for col in columns:
            if col != category_col and self._is_numeric_column(col, data):
                value_col = col
                break

        # No numeric column found — fall back to table so the data is still useful
        if value_col is None:
            config['type'] = 'table'
            return self._config_table(data, columns, config)

        # Determine if horizontal (for rankings like "top 10")
        is_horizontal = 'top' in query.lower() or 'bottom' in query.lower() or 'ranking' in query.lower()

        config['config'] = {
            'xAxis': value_col if is_horizontal else category_col,
            'yAxis': category_col if is_horizontal else value_col,
            'xLabel': (value_col if is_horizontal else category_col).replace('_', ' ').title(),
            'yLabel': (category_col if is_horizontal else value_col).replace('_', ' ').title(),
            'color': '#3b82f6',
            'horizontal': is_horizontal
        }

        return config
    
    def _config_pie(self, data: List[Dict], columns: List[str], config: Dict) -> Dict:
        """Configure pie chart"""
        # First column is label, second is value
        label_col = columns[0]
        value_col = columns[1] if len(columns) > 1 else columns[0]
        
        # Find numeric column for value
        for col in columns:
            if col != label_col and self._is_numeric_column(col, data):
                value_col = col
                break
        
        config['config'] = {
            'labelKey': label_col,
            'valueKey': value_col,
            'colors': ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1']
        }
        
        return config
    
    def _config_scatter(self, data: List[Dict], columns: List[str], config: Dict) -> Dict:
        """Configure scatter plot"""
        numeric_cols = [col for col in columns if self._is_numeric_column(col, data)]
        
        if len(numeric_cols) < 2:
            numeric_cols = columns[:2]
        
        config['config'] = {
            'xAxis': numeric_cols[0],
            'yAxis': numeric_cols[1],
            'xLabel': numeric_cols[0].replace('_', ' ').title(),
            'yLabel': numeric_cols[1].replace('_', ' ').title(),
            'color': '#3b82f6'
        }
        
        return config
    
    def _config_table(self, data: List[Dict], columns: List[str], config: Dict) -> Dict:
        """Configure data table"""
        config['config'] = {
            'columns': [col.replace('_', ' ').title() for col in columns],
            'sortable': True,
            'pagination': len(data) > 20
        }
        
        return config
    
    def _generate_title(self, query: str, chart_type: str) -> str:
        """Generate a readable title from the query"""
        # Remove common SQL words
        title = query.lower()
        remove_words = ['show', 'me', 'get', 'find', 'what', 'is', 'the', 'a', 'an']
        
        for word in remove_words:
            title = title.replace(f' {word} ', ' ')
        
        # Capitalize first letter of each word
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title.strip()


# Singleton instance
chart_recommender = ChartRecommender()

__all__ = ['ChartRecommender', 'chart_recommender']