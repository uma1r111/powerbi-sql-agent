FOREIGN_KEY_RELATIONSHIPS = {
    "orders": {
        "customer_id": {
            "references": "customers.customer_id",
            "relationship_type": "many_to_one",
            "description": "Each order belongs to one customer"
        },
        "employee_id": {
            "references": "employees.employee_id", 
            "relationship_type": "many_to_one",
            "description": "Each order is processed by one employee"
        },
        "ship_via": {
            "references": "shippers.shipper_id",
            "relationship_type": "many_to_one", 
            "description": "Each order is shipped via one shipper"
        }
    },
    
    "order_details": {
        "order_id": {
            "references": "orders.order_id",
            "relationship_type": "many_to_one",
            "description": "Each order detail belongs to one order"
        },
        "product_id": {
            "references": "products.product_id",
            "relationship_type": "many_to_one",
            "description": "Each order detail references one product"
        }
    },
    
    "products": {
        "supplier_id": {
            "references": "suppliers.supplier_id",
            "relationship_type": "many_to_one",
            "description": "Each product has one supplier"
        },
        "category_id": {
            "references": "categories.category_id",
            "relationship_type": "many_to_one", 
            "description": "Each product belongs to one category"
        }
    },
    
    "employees": {
        "reports_to": {
            "references": "employees.employee_id",
            "relationship_type": "many_to_one",
            "description": "Self-referencing relationship for employee hierarchy"
        }
    },
    
    "employee_territories": {
        "employee_id": {
            "references": "employees.employee_id",
            "relationship_type": "many_to_one",
            "description": "Links employees to their territories"
        },
        "territory_id": {
            "references": "territories.territory_id", 
            "relationship_type": "many_to_one",
            "description": "Links territories to employees"
        }
    },
    
    "territories": {
        "region_id": {
            "references": "region.region_id",
            "relationship_type": "many_to_one",
            "description": "Each territory belongs to one region"
        }
    }
}

# Table Connection Mapping (for JOIN operations)
TABLE_CONNECTIONS = {
    "customers": {
        "direct_connections": ["orders"],
        "indirect_connections": {
            "order_details": "via orders",
            "products": "via orders -> order_details",
            "employees": "via orders", 
            "shippers": "via orders",
            "categories": "via orders -> order_details -> products",
            "suppliers": "via orders -> order_details -> products"
        }
    },
    
    "orders": {
        "direct_connections": ["customers", "employees", "shippers", "order_details"],
        "indirect_connections": {
            "products": "via order_details",
            "categories": "via order_details -> products", 
            "suppliers": "via order_details -> products"
        }
    },
    
    "order_details": {
        "direct_connections": ["orders", "products"],
        "indirect_connections": {
            "customers": "via orders",
            "employees": "via orders",
            "shippers": "via orders",
            "categories": "via products",
            "suppliers": "via products"
        }
    },
    
    "products": {
        "direct_connections": ["categories", "suppliers", "order_details"],
        "indirect_connections": {
            "orders": "via order_details",
            "customers": "via order_details -> orders",
            "employees": "via order_details -> orders"
        }
    },
    
    "categories": {
        "direct_connections": ["products"], 
        "indirect_connections": {
            "order_details": "via products",
            "orders": "via products -> order_details",
            "customers": "via products -> order_details -> orders"
        }
    },
    
    "suppliers": {
        "direct_connections": ["products"],
        "indirect_connections": {
            "order_details": "via products",
            "orders": "via products -> order_details", 
            "customers": "via products -> order_details -> orders"
        }
    },
    
    "employees": {
        "direct_connections": ["orders", "employee_territories", "employees"],  # self-reference
        "indirect_connections": {
            "customers": "via orders",
            "order_details": "via orders",
            "products": "via orders -> order_details",
            "territories": "via employee_territories",
            "region": "via employee_territories -> territories"
        }
    },
    
    "shippers": {
        "direct_connections": ["orders"],
        "indirect_connections": {
            "customers": "via orders",
            "order_details": "via orders", 
            "products": "via orders -> order_details"
        }
    },
    
    "territories": {
        "direct_connections": ["employee_territories", "region"],
        "indirect_connections": {
            "employees": "via employee_territories"
        }
    },
    
    "region": {
        "direct_connections": ["territories"],
        "indirect_connections": {
            "employee_territories": "via territories",
            "employees": "via territories -> employee_territories"
        }
    }
}

# Common JOIN patterns for SQL generation
COMMON_JOIN_PATTERNS = {
    "customer_orders": {
        "tables": ["customers", "orders"],
        "join_condition": "customers.customer_id = orders.customer_id",
        "description": "Get customer information with their orders"
    },
    
    "order_details_with_products": {
        "tables": ["order_details", "products"],
        "join_condition": "order_details.product_id = products.product_id", 
        "description": "Get order line items with product information"
    },
    
    "products_with_categories": {
        "tables": ["products", "categories"],
        "join_condition": "products.category_id = categories.category_id",
        "description": "Get products with their category information"
    },
    
    "products_with_suppliers": {
        "tables": ["products", "suppliers"], 
        "join_condition": "products.supplier_id = suppliers.supplier_id",
        "description": "Get products with their supplier information"
    },
    
    "orders_with_employees": {
        "tables": ["orders", "employees"],
        "join_condition": "orders.employee_id = employees.employee_id",
        "description": "Get orders with the employee who processed them"
    },
    
    "full_order_details": {
        "tables": ["orders", "customers", "order_details", "products"],
        "join_conditions": [
            "orders.customer_id = customers.customer_id",
            "orders.order_id = order_details.order_id", 
            "order_details.product_id = products.product_id"
        ],
        "description": "Complete order information with customer and product details"
    },
    
    "employee_hierarchy": {
        "tables": ["employees e1", "employees e2"],
        "join_condition": "e1.reports_to = e2.employee_id",
        "description": "Self-join to show employee reporting relationships"
    },
    
    "employee_territories_full": {
        "tables": ["employees", "employee_territories", "territories", "region"],
        "join_conditions": [
            "employees.employee_id = employee_territories.employee_id",
            "employee_territories.territory_id = territories.territory_id",
            "territories.region_id = region.region_id"
        ],
        "description": "Employees with their assigned territories and regions"
    }
}

# to get related tables
def get_related_tables(table_name):
    """Returns tables that can be joined with the given table"""
    if table_name in TABLE_CONNECTIONS:
        direct = TABLE_CONNECTIONS[table_name]["direct_connections"]
        indirect = list(TABLE_CONNECTIONS[table_name]["indirect_connections"].keys())
        return {
            "direct": direct,
            "indirect": indirect,
            "all": direct + indirect
        }
    return {"direct": [], "indirect": [], "all": []}

# to get join path between two tables  
def get_join_path(from_table, to_table):
    """Returns the path to join two tables"""
    if from_table in TABLE_CONNECTIONS:
        # Check direct connection
        if to_table in TABLE_CONNECTIONS[from_table]["direct_connections"]:
            return f"Direct: {from_table} -> {to_table}"
        
        # Check indirect connection
        if to_table in TABLE_CONNECTIONS[from_table]["indirect_connections"]:
            path = TABLE_CONNECTIONS[from_table]["indirect_connections"][to_table]
            return f"Indirect: {from_table} -> {path} -> {to_table}"
    
    return "No known relationship"