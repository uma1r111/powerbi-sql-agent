# database/connection.py
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Establish and return a database connection."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn
    except Exception as e:
        print("Error while connecting to database:", e)
        return None


def test_connection():
    """Test connection, list tables, and print PostgreSQL version."""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()

        # List tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")

        # Show PostgreSQL version
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"PostgreSQL Version: {version[0]}")

        print("Connection successful!")

    except Exception as e:
        print(f"Query failed: {e}")
    finally:
        cursor.close()
        conn.close()


# Run test when this file is executed directly
if __name__ == "__main__":
    test_connection()
