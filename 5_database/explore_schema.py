import psycopg2
import pandas as pd
from connection import get_db_connection

def explore_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("""
        SELECT table_name, table_type 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    tables = cursor.fetchall()
    
    schema_info = {}
    
    for table_name, table_type in tables:
        print(f"\n=== TABLE: {table_name} ===")
        
        # Get column information
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position;
        """, (table_name,))
        
        columns = cursor.fetchall()
        schema_info[table_name] = {
            'columns': columns,
            'sample_data': None,
            'row_count': 0
        }
        
        # Get sample data (first 5 rows)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        sample_data = cursor.fetchall()
        schema_info[table_name]['sample_data'] = sample_data
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        schema_info[table_name]['row_count'] = row_count
        
        print(f"Columns: {len(columns)}, Rows: {row_count}")
        for col_name, data_type, nullable, default in columns:
            print(f"  - {col_name}: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
        
        print("Sample data:")
        for i, row in enumerate(sample_data[:3]):
            print(f"  Row {i+1}: {row}")
    
    return schema_info

if __name__ == "__main__":
    schema_info = explore_database()