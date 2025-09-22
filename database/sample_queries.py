# Organized by complexity and query type
SAMPLE_QUERIES = {
    
    # ============ BASIC QUERIES ============
    "basic_queries": {
        "all_customers": {
            "natural_language": "Show me all customers",
            "sql": "SELECT * FROM customers LIMIT 10;",
            "explanation": "Basic SELECT statement with LIMIT",
            "tables_involved": ["customers"],
            "complexity": "beginner"
        },
        
        "customer_count": {
            "natural_language": "How many customers do we have?",
            "sql": "SELECT COUNT(*) AS total_customers FROM customers;",
            "explanation": "COUNT aggregation function",
            "tables_involved": ["customers"],
            "complexity": "beginner"
        },
        
        "products_list": {
            "natural_language": "Show me all products with their prices",
            "sql": "SELECT product_name, unit_price FROM products ORDER BY unit_price DESC;",
            "explanation": "SELECT specific columns with ORDER BY",
            "tables_involved": ["products"],
            "complexity": "beginner"
        },
        
        "categories_list": {
            "natural_language": "What are all the product categories?",
            "sql": "SELECT category_name, description FROM categories;",
            "explanation": "Simple SELECT with specific columns",
            "tables_involved": ["categories"],
            "complexity": "beginner"
        }
    },
    
    # ============ FILTERED QUERIES ============ 
    "filtered_queries": {
        "customers_by_country": {
            "natural_language": "Show me customers from Germany",
            "sql": "SELECT company_name, contact_name, city FROM customers WHERE country = 'Germany';",
            "explanation": "SELECT with WHERE clause filtering",
            "tables_involved": ["customers"],
            "complexity": "beginner"
        },
        
        "expensive_products": {
            "natural_language": "Show me products that cost more than $50",
            "sql": "SELECT product_name, unit_price FROM products WHERE unit_price > 50 ORDER BY unit_price DESC;",
            "explanation": "Numeric filtering with WHERE and ORDER BY",
            "tables_involved": ["products"],
            "complexity": "beginner"
        },
        
        "out_of_stock": {
            "natural_language": "Which products are out of stock?",
            "sql": "SELECT product_name, units_in_stock FROM products WHERE units_in_stock = 0;",
            "explanation": "Filtering for zero inventory",
            "tables_involved": ["products"],
            "complexity": "beginner"
        },
        
        "orders_1996": {
            "natural_language": "Show me orders from 1996",
            "sql": "SELECT order_id, customer_id, order_date FROM orders WHERE EXTRACT(YEAR FROM order_date) = 1996 LIMIT 20;",
            "explanation": "Date filtering with EXTRACT function",
            "tables_involved": ["orders"],
            "complexity": "intermediate"
        },
        
        "discounted_items": {
            "natural_language": "Show me order items with discounts",
            "sql": "SELECT order_id, product_id, quantity, discount FROM order_details WHERE discount > 0;",
            "explanation": "Filtering for positive discount values",
            "tables_involved": ["order_details"],
            "complexity": "beginner"
        }
    },
    
    # ============ JOIN QUERIES ============
    "join_queries": {
        "orders_with_customers": {
            "natural_language": "Show me orders with customer names",
            "sql": """SELECT o.order_id, c.company_name, o.order_date, o.freight 
                     FROM orders o 
                     JOIN customers c ON o.customer_id = c.customer_id 
                     ORDER BY o.order_date DESC
                     LIMIT 15;""",
            "explanation": "INNER JOIN between orders and customers",
            "tables_involved": ["orders", "customers"],
            "complexity": "intermediate"
        },
        
        "products_with_categories": {
            "natural_language": "Show me products with their categories",
            "sql": """SELECT p.product_name, c.category_name, p.unit_price 
                     FROM products p 
                     JOIN categories c ON p.category_id = c.category_id 
                     ORDER BY c.category_name, p.product_name;""",
            "explanation": "JOIN products with categories",
            "tables_involved": ["products", "categories"],
            "complexity": "intermediate"
        },
        
        "products_with_suppliers": {
            "natural_language": "Show me products with their suppliers",
            "sql": """SELECT p.product_name, s.company_name AS supplier_name, p.unit_price
                     FROM products p
                     JOIN suppliers s ON p.supplier_id = s.supplier_id
                     ORDER BY s.company_name;""",
            "explanation": "JOIN products with suppliers, using column alias",
            "tables_involved": ["products", "suppliers"], 
            "complexity": "intermediate"
        },
        
        "order_details_full": {
            "natural_language": "Show me order details with product names",
            "sql": """SELECT od.order_id, p.product_name, od.quantity, od.unit_price, od.discount
                     FROM order_details od
                     JOIN products p ON od.product_id = p.product_id
                     WHERE od.order_id = 10248;""",
            "explanation": "JOIN order details with products for specific order",
            "tables_involved": ["order_details", "products"],
            "complexity": "intermediate"
        },
        
        "employee_orders": {
            "natural_language": "Show me orders processed by each employee",
            "sql": """SELECT e.first_name, e.last_name, COUNT(o.order_id) AS orders_processed
                     FROM employees e
                     LEFT JOIN orders o ON e.employee_id = o.employee_id
                     GROUP BY e.employee_id, e.first_name, e.last_name
                     ORDER BY orders_processed DESC;""",
            "explanation": "LEFT JOIN with GROUP BY to count orders per employee",
            "tables_involved": ["employees", "orders"],
            "complexity": "advanced"
        }
    },
    
    # ============ AGGREGATION QUERIES ============
    "aggregation_queries": {
        "orders_per_country": {
            "natural_language": "How many orders per country?",
            "sql": """SELECT c.country, COUNT(o.order_id) AS order_count
                     FROM customers c
                     LEFT JOIN orders o ON c.customer_id = o.customer_id
                     GROUP BY c.country
                     ORDER BY order_count DESC;""",
            "explanation": "GROUP BY with COUNT aggregation and LEFT JOIN",
            "tables_involved": ["customers", "orders"],
            "complexity": "advanced"
        },
        
        "revenue_by_product": {
            "natural_language": "Calculate total revenue by product",
            "sql": """SELECT p.product_name, 
                            SUM(od.quantity * od.unit_price * (1 - od.discount)) AS total_revenue
                     FROM order_details od
                     JOIN products p ON od.product_id = p.product_id
                     GROUP BY p.product_id, p.product_name
                     ORDER BY total_revenue DESC
                     LIMIT 10;""",
            "explanation": "Complex calculation with SUM, multiplication, and discount",
            "tables_involved": ["order_details", "products"],
            "complexity": "advanced"
        },
        
        "average_order_value": {
            "natural_language": "What's the average order value?",
            "sql": """SELECT AVG(order_total) AS average_order_value
                     FROM (
                         SELECT order_id, SUM(quantity * unit_price * (1 - discount)) AS order_total
                         FROM order_details
                         GROUP BY order_id
                     ) AS order_totals;""",
            "explanation": "Subquery with AVG and SUM calculations",
            "tables_involved": ["order_details"],
            "complexity": "advanced"
        },
        
        "products_per_category": {
            "natural_language": "How many products are in each category?",
            "sql": """SELECT c.category_name, COUNT(p.product_id) AS product_count
                     FROM categories c
                     LEFT JOIN products p ON c.category_id = p.category_id
                     GROUP BY c.category_id, c.category_name
                     ORDER BY product_count DESC;""",
            "explanation": "COUNT with LEFT JOIN and GROUP BY",
            "tables_involved": ["categories", "products"],
            "complexity": "intermediate"
        },
        
        "monthly_sales_1996": {
            "natural_language": "Show me monthly sales totals for 1996",
            "sql": """SELECT 
                        EXTRACT(MONTH FROM o.order_date) AS month,
                        COUNT(o.order_id) AS order_count,
                        SUM(od.quantity * od.unit_price * (1 - od.discount)) AS total_sales
                     FROM orders o
                     JOIN order_details od ON o.order_id = od.order_id  
                     WHERE EXTRACT(YEAR FROM o.order_date) = 1996
                     GROUP BY EXTRACT(MONTH FROM o.order_date)
                     ORDER BY month;""",
            "explanation": "Date functions with aggregation and filtering",
            "tables_involved": ["orders", "order_details"],
            "complexity": "advanced"
        }
    },
    
    # ============ COMPLEX QUERIES ============
    "complex_queries": {
        "top_customers_by_revenue": {
            "natural_language": "Who are our top 5 customers by total revenue?",
            "sql": """SELECT 
                        c.company_name,
                        COUNT(DISTINCT o.order_id) AS total_orders,
                        SUM(od.quantity * od.unit_price * (1 - od.discount)) AS total_revenue
                     FROM customers c
                     JOIN orders o ON c.customer_id = o.customer_id
                     JOIN order_details od ON o.order_id = od.order_id
                     GROUP BY c.customer_id, c.company_name
                     ORDER BY total_revenue DESC
                     LIMIT 5;""",
            "explanation": "Multiple JOINs with aggregation and ranking",
            "tables_involved": ["customers", "orders", "order_details"],
            "complexity": "advanced"
        },
        
        "employee_performance": {
            "natural_language": "Show me employee sales performance",
            "sql": """SELECT 
                        e.first_name || ' ' || e.last_name AS employee_name,
                        e.title,
                        COUNT(DISTINCT o.order_id) AS orders_processed,
                        SUM(od.quantity * od.unit_price * (1 - od.discount)) AS total_sales
                     FROM employees e
                     JOIN orders o ON e.employee_id = o.employee_id
                     JOIN order_details od ON o.order_id = od.order_id
                     GROUP BY e.employee_id, e.first_name, e.last_name, e.title
                     ORDER BY total_sales DESC;""",
            "explanation": "String concatenation with multiple JOINs and aggregation",
            "tables_involved": ["employees", "orders", "order_details"],
            "complexity": "advanced"
        },
        
        "category_performance": {
            "natural_language": "Which product categories perform best?",
            "sql": """SELECT 
                        c.category_name,
                        COUNT(DISTINCT p.product_id) AS products_in_category,
                        SUM(od.quantity) AS total_quantity_sold,
                        SUM(od.quantity * od.unit_price * (1 - od.discount)) AS total_revenue
                     FROM categories c
                     JOIN products p ON c.category_id = p.category_id
                     JOIN order_details od ON p.product_id = od.product_id
                     GROUP BY c.category_id, c.category_name
                     ORDER BY total_revenue DESC;""",
            "explanation": "Multi-table JOIN with multiple aggregations",
            "tables_involved": ["categories", "products", "order_details"],
            "complexity": "advanced"
        },
        
        "shipping_analysis": {
            "natural_language": "Analyze shipping costs by shipper",
            "sql": """SELECT 
                        s.company_name AS shipper_name,
                        COUNT(o.order_id) AS orders_shipped,
                        AVG(o.freight) AS avg_freight_cost,
                        SUM(o.freight) AS total_freight_cost
                     FROM shippers s
                     JOIN orders o ON s.shipper_id = o.ship_via
                     GROUP BY s.shipper_id, s.company_name
                     ORDER BY total_freight_cost DESC;""",
            "explanation": "Analysis with multiple aggregation functions",
            "tables_involved": ["shippers", "orders"],
            "complexity": "intermediate"
        }
    }
}

# Query categories for easy reference
QUERY_CATEGORIES = {
    "beginner": ["basic_queries", "simple filtered_queries"],
    "intermediate": ["filtered_queries", "join_queries", "basic aggregation"],
    "advanced": ["complex joins", "aggregation_queries", "complex_queries"]
}

# Common query patterns for the agent to learn
QUERY_PATTERNS = {
    "count_pattern": "SELECT COUNT(*) FROM table_name WHERE condition;",
    "sum_pattern": "SELECT SUM(column) FROM table_name GROUP BY category;", 
    "join_pattern": "SELECT columns FROM table1 t1 JOIN table2 t2 ON t1.key = t2.key;",
    "aggregation_pattern": "SELECT category, COUNT(*), AVG(value) FROM table GROUP BY category;",
    "top_n_pattern": "SELECT columns FROM table ORDER BY value DESC LIMIT n;"
}

# Utility function to get queries by complexity
def get_queries_by_complexity(complexity_level):
    """Returns all queries of a specific complexity level"""
    result = {}
    for category, queries in SAMPLE_QUERIES.items():
        for query_name, query_info in queries.items():
            if query_info["complexity"] == complexity_level:
                result[f"{category}.{query_name}"] = query_info
    return result

# Utility function to get queries involving specific tables
def get_queries_with_tables(table_names):
    """Returns queries that involve any of the specified tables"""
    if isinstance(table_names, str):
        table_names = [table_names]
    
    result = {}
    for category, queries in SAMPLE_QUERIES.items():
        for query_name, query_info in queries.items():
            if any(table in query_info["tables_involved"] for table in table_names):
                result[f"{category}.{query_name}"] = query_info
    return result

# Test queries for agent validation
TEST_QUERIES = {
    "simple_tests": [
        {
            "question": "How many customers do we have?",
            "expected_sql_pattern": "SELECT COUNT(*) FROM customers",
            "expected_result_type": "single_number"
        },
        {
            "question": "Show me all product categories",
            "expected_sql_pattern": "SELECT * FROM categories",
            "expected_result_type": "table"
        }
    ],
    
    "intermediate_tests": [
        {
            "question": "Show me orders with customer company names",
            "expected_sql_pattern": "JOIN orders.*customers",
            "expected_result_type": "table"
        },
        {
            "question": "Which products are out of stock?",
            "expected_sql_pattern": "WHERE units_in_stock = 0",
            "expected_result_type": "table"
        }
    ],
    
    "advanced_tests": [
        {
            "question": "Calculate total revenue by product category",
            "expected_sql_pattern": "SUM.*GROUP BY.*JOIN",
            "expected_result_type": "aggregated_table"
        },
        {
            "question": "Who are our top 3 customers by total orders?",
            "expected_sql_pattern": "COUNT.*GROUP BY.*ORDER BY.*LIMIT 3",
            "expected_result_type": "ranked_table"
        }
    ]
}