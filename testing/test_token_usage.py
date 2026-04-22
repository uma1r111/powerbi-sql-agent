# testing/test_token_usage.py
"""
Measures token usage for each MCP server tool, resource, and prompt.
Calls the underlying functions directly (no MCP transport needed).

Run from project root:
    python testing/test_token_usage.py
"""

import sys
import json
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.sql_tools import sql_executor
from tools.schema_tools import schema_inspector

# ── Token estimator ────────────────────────────────────────────────────────────
# Gemini counts tokens similarly to GPT: ~4 chars per token for mixed content.
# This is an approximation; use it for relative comparisons between calls.

def estimate_tokens(obj) -> int:
    text = json.dumps(obj) if not isinstance(obj, str) else obj
    return max(1, len(text) // 4)

def json_bytes(obj) -> int:
    text = json.dumps(obj) if not isinstance(obj, str) else obj
    return len(text.encode("utf-8"))


# ── Slim helpers (same logic as mcp_server.py) ─────────────────────────────────

def _slim_sql_result(result):
    if result.get("success"):
        return {
            "success": True,
            "data": result.get("data", []),
            "row_count": result.get("row_count", 0),
            "column_names": result.get("column_names", []),
        }
    return {
        "success": False,
        "error": result.get("error") or result.get("message", "Unknown error"),
    }

def _slim_overview(raw):
    return {
        "success": True,
        "total_tables": raw.get("total_tables", 0),
        "tables": {
            name: info.get("description", "")
            for name, info in raw.get("table_summary", {}).items()
        },
    }

def _slim_multi_table(raw):
    slim = {}
    for tname, ctx in raw.get("tables", {}).items():
        slim[tname] = {
            "description": ctx.get("description", ""),
            "key_fields": ctx.get("key_fields", []),
            "foreign_keys": ctx.get("relationships", {}).get("foreign_keys", {}),
        }
    return {
        "success": True,
        "tables": slim,
        "relationships": raw.get("inter_table_relationships", {}),
    }

NORTHWIND_RULES_OLD = """
=== NORTHWIND DATABASE QUERY RULES ===

You are generating SQL for a PostgreSQL Northwind database. Follow these rules STRICTLY:

## 1. STRING COMPARISONS (CRITICAL)
- NEVER use = for string comparisons
- ALWAYS use ILIKE for case-insensitive matching
  Example: WHERE ship_country ILIKE 'germany'

## 2. REVENUE CALCULATION (MOST COMMON ERROR)
- Revenue formula: SUM(unit_price * quantity * (1 - discount))
- WRONG: SUM(unit_price * quantity)
- CORRECT: SUM(unit_price * quantity * (1 - discount))
- Source table: order_details (NOT orders)

## 3. COMMON TABLE/COLUMN MISTAKES
- orders.country -> orders.ship_country
- customers.name -> customers.company_name
- products.name -> products.product_name
- employees.name -> employees.first_name, employees.last_name

## 4. TERRITORY QUERIES (COMMON JOIN ERROR)
- employees -> employee_territories -> territories -> region

## 5. COUNTRY VALUES
- 'UK' -> 'United Kingdom', 'US' -> 'USA', 'DE' -> 'Germany'

## 6. RESULT LIMITS
- ALWAYS add LIMIT clause unless doing aggregation
- Default limit: 100 rows

## 7. DATE FILTERING
- Use EXTRACT(YEAR FROM order_date) = 1997
- Use order_date::date for date comparisons

## 8. JOIN ORDER
- customers -> orders -> order_details -> products

## 9. BUSINESS LOGIC
- Out of stock: units_in_stock = 0
- Needs reordering: units_in_stock <= reorder_level
- Discontinued: discontinued = 1

## 10. SCHEMA REFERENCE
Available tables: customers, orders, order_details, products, categories,
                 suppliers, employees, shippers, territories, region,
                 employee_territories
"""

NORTHWIND_RULES_NEW = (
    "NORTHWIND SQL RULES:\n"
    "1. String match: use ILIKE, not =. E.g. WHERE ship_country ILIKE 'germany'\n"
    "2. Revenue: SUM(unit_price * quantity * (1 - discount)) from order_details\n"
    "3. Column names: ship_country, company_name, product_name, first_name/last_name\n"
    "4. Countries: full names — 'United Kingdom' not 'UK', 'USA' not 'US'\n"
    "5. Always add LIMIT (default 50) unless using aggregation\n"
    "6. Year filter: EXTRACT(YEAR FROM order_date) = 1997\n"
    "7. Revenue join chain: customers->orders->order_details->products\n"
    "8. Territories: employees->employee_territories->territories->region\n"
    "9. Out of stock: units_in_stock=0 | Reorder: units_in_stock<=reorder_level\n"
    "Tables: customers,orders,order_details,products,categories,suppliers,"
    "employees,shippers,territories,region,employee_territories"
)


# ── Report helpers ─────────────────────────────────────────────────────────────

def row(label, raw_tokens, slim_tokens=None):
    saved = ""
    if slim_tokens is not None:
        diff = raw_tokens - slim_tokens
        pct  = (diff / raw_tokens * 100) if raw_tokens else 0
        saved = f"  slim={slim_tokens:>5}  saved={diff:>4} ({pct:.0f}%)"
    print(f"  {label:<45} raw={raw_tokens:>5}{saved}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MCP SERVER — TOKEN USAGE REPORT")
    print("(~4 chars per token estimate, JSON-serialised payloads)")
    print("=" * 70)

    # ── 1. execute_sql_query ──────────────────────────────────────────────────
    print("\n[ TOOL: execute_sql_query ]\n")

    queries = {
        "SELECT 1 (smoke test)": "SELECT 1 AS ok",
        "Top 5 customers":       "SELECT customer_id, company_name FROM customers LIMIT 5",
        "Revenue by country":    (
            "SELECT ship_country, "
            "SUM(od.unit_price * od.quantity * (1 - od.discount)) AS revenue "
            "FROM orders o JOIN order_details od ON o.order_id = od.order_id "
            "GROUP BY ship_country ORDER BY revenue DESC LIMIT 10"
        ),
        "Bad query (error)":     "SELECT * FROM nonexistent_table",
    }

    for label, sql in queries.items():
        try:
            raw    = sql_executor.execute_query(sql, limit=100)
            slim   = _slim_sql_result(raw)
            row(label, estimate_tokens(raw), estimate_tokens(slim))
        except Exception as e:
            print(f"  {label}: ERROR — {e}")

    # ── 2. get_database_schema — overview ─────────────────────────────────────
    print("\n[ TOOL: get_database_schema (no args = overview) ]\n")
    try:
        raw  = schema_inspector.get_database_overview()
        slim = _slim_overview(raw)
        row("database overview", estimate_tokens(raw), estimate_tokens(slim))
    except Exception as e:
        print(f"  overview: ERROR — {e}")

    # ── 3. get_database_schema — specific tables ──────────────────────────────
    print("\n[ TOOL: get_database_schema (specific tables) ]\n")
    table_combos = {
        "['customers']":                  ["customers"],
        "['customers', 'orders']":        ["customers", "orders"],
        "['orders', 'order_details', 'products']": ["orders", "order_details", "products"],
    }
    for label, tables in table_combos.items():
        try:
            raw  = schema_inspector.get_multiple_tables_context(tables)
            slim = _slim_multi_table(raw)
            row(label, estimate_tokens(raw), estimate_tokens(slim))
        except Exception as e:
            print(f"  {label}: ERROR — {e}")

    # ── 4. northwind_query_rules prompt ───────────────────────────────────────
    print("\n[ PROMPT: northwind_query_rules ]\n")
    row("old prompt", estimate_tokens(NORTHWIND_RULES_OLD),
                      estimate_tokens(NORTHWIND_RULES_NEW))

    # ── 5. Per-field breakdown for a typical SQL success response ──────────────
    print("\n[ BREAKDOWN: fields in a typical SQL success response ]\n")
    try:
        raw = sql_executor.execute_query(
            "SELECT customer_id, company_name, country FROM customers", limit=50
        )
        for field, val in raw.items():
            t = estimate_tokens(val)
            print(f"  {field:<20} ~{t:>4} tokens  ({json_bytes(val):>6} bytes)")
        print(f"  {'TOTAL (raw)':<20} ~{estimate_tokens(raw):>4} tokens")
        print(f"  {'TOTAL (slim)':<20} ~{estimate_tokens(_slim_sql_result(raw)):>4} tokens")
    except Exception as e:
        print(f"  ERROR — {e}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NOTES")
    print("  • Gemini free tier: 1M tokens/min input, 32K output per request")
    print("  • Lower max_rows in execute_sql_query to cut data payload tokens")
    print("  • Call get_database_schema with specific tables, not the overview,")
    print("    when you already know which tables the query needs")
    print("  • The prompt is injected once per conversation, not per query")
    print("=" * 70)


if __name__ == "__main__":
    main()
