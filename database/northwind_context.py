# database/northwind_context.py

BUSINESS_CONTEXT = {
    "customers": {
        "description": "Customer information and contact details for companies that purchase products",
        "purpose": "Track who buys from us, their location, and contact information",
        "key_fields": ["customer_id", "company_name", "contact_name", "country", "city"],
        "business_questions": [
            "How many customers do we have?",
            "Which customers are from Germany?",
            "Show me customer contact information",
            "Which countries have the most customers?",
            "Who are our customers in Berlin?"
        ],
        "row_count": 91,
        "business_importance": "High - Primary revenue source"
    },
    
    "orders": {
        "description": "Customer orders with dates, shipping information, and freight costs",
        "purpose": "Track all sales transactions, shipping details, and order fulfillment",
        "key_fields": ["order_id", "customer_id", "employee_id", "order_date", "shipped_date", "freight"],
        "business_questions": [
            "How many orders were placed in 1996?",
            "Which orders are still pending shipment?",
            "Show me orders by customer",
            "What's the total freight cost this month?",
            "Which employee processed the most orders?"
        ],
        "row_count": 830,
        "business_importance": "Critical - Core business transactions"
    },
    
    "order_details": {
        "description": "Individual line items for each order showing products, quantities, prices, and discounts",
        "purpose": "Detail what was ordered, how much, at what price, and any discounts applied",
        "key_fields": ["order_id", "product_id", "unit_price", "quantity", "discount"],
        "business_questions": [
            "What products were ordered the most?",
            "Calculate total revenue by product",
            "Show me orders with discounts",
            "What's the average order quantity?",
            "Which products generate the most revenue?"
        ],
        "row_count": 2155,
        "business_importance": "Critical - Revenue calculation basis"
    },
    
    "products": {
        "description": "Product catalog with pricing, inventory levels, and supplier information",
        "purpose": "Manage product information, pricing, and inventory tracking",
        "key_fields": ["product_id", "product_name", "unit_price", "units_in_stock", "category_id"],
        "business_questions": [
            "Which products are out of stock?",
            "Show me products by category",
            "What's the most expensive product?",
            "Which products need reordering?",
            "What products are discontinued?"
        ],
        "row_count": 77,
        "business_importance": "High - Inventory and pricing management"
    },
    
    "categories": {
        "description": "Product categories for organizing the product catalog",
        "purpose": "Classify products into business-relevant groupings",
        "key_fields": ["category_id", "category_name", "description"],
        "business_questions": [
            "How many product categories do we have?",
            "Show me all beverage products",
            "Which category has the most products?",
            "What are the different product categories?"
        ],
        "row_count": 8,
        "business_importance": "Medium - Product organization"
    },
    
    "suppliers": {
        "description": "Supplier information and contact details for product sourcing",
        "purpose": "Manage relationships with product suppliers and vendors",
        "key_fields": ["supplier_id", "company_name", "contact_name", "country", "phone"],
        "business_questions": [
            "How many suppliers do we have?",
            "Which suppliers are from the USA?",
            "Show me supplier contact information",
            "Which supplier provides the most products?"
        ],
        "row_count": 29,
        "business_importance": "Medium - Supply chain management"
    },
    
    "employees": {
        "description": "Employee information including personal details, job titles, and reporting structure",
        "purpose": "Track employee information and organizational hierarchy",
        "key_fields": ["employee_id", "first_name", "last_name", "title", "hire_date", "reports_to"],
        "business_questions": [
            "How many employees do we have?",
            "Who are the sales representatives?",
            "Show me the organizational hierarchy",
            "Which employees were hired in 1992?"
        ],
        "row_count": 9,
        "business_importance": "Medium - HR and sales tracking"
    },
    
    "shippers": {
        "description": "Shipping companies used for order delivery",
        "purpose": "Manage shipping partner information and contact details",
        "key_fields": ["shipper_id", "company_name", "phone"],
        "business_questions": [
            "Which shipping companies do we use?",
            "Show me shipper contact information",
            "How many shipping partners do we have?"
        ],
        "row_count": 6,
        "business_importance": "Low - Shipping logistics"
    },
    
    "territories": {
        "description": "Sales territories assigned to employees",
        "purpose": "Manage geographic sales territory assignments",
        "key_fields": ["territory_id", "territory_description", "region_id"],
        "business_questions": [
            "How many territories do we have?",
            "Which territories are in the Eastern region?",
            "Show me all sales territories"
        ],
        "row_count": 53,
        "business_importance": "Low - Sales territory management"
    },
    
    "region": {
        "description": "Geographic regions for organizing sales territories",
        "purpose": "Group territories into larger geographic regions",
        "key_fields": ["region_id", "region_description"],
        "business_questions": [
            "How many regions do we operate in?",
            "Show me all regions",
            "Which region has the most territories?"
        ],
        "row_count": 4,
        "business_importance": "Low - Geographic organization"
    }
}

# Business scenario context
BUSINESS_SCENARIO = {
    "company_type": "Food and Beverage Trading Company",
    "business_model": "B2B wholesale distribution of specialty food products",
    "key_metrics": [
        "Total Revenue",
        "Orders per Month", 
        "Average Order Value",
        "Customer Retention",
        "Inventory Turnover",
        "Top Selling Products",
        "Sales by Region/Country"
    ],
    "common_analyses": [
        "Sales performance by employee",
        "Product performance by category", 
        "Customer ordering patterns",
        "Geographic sales distribution",
        "Seasonal sales trends",
        "Inventory management",
        "Freight cost analysis"
    ]
}