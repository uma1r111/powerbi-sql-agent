# Test database documentation and tools
from database.northwind_context import BUSINESS_CONTEXT
from database.relationships import get_related_tables, get_join_path
from database.sample_queries import get_queries_by_complexity
from tools.sql_tools import sql_executor
from tools.schema_tools import schema_inspector
from tools.validation_tools import query_validator

# Test 1: Database documentation
print("=== Testing Database Documentation ===")
print(f"Tables available: {len(BUSINESS_CONTEXT)}")
print(f"Customers table context: {BUSINESS_CONTEXT['customers']['description']}")

# Test 2: Relationships
print(f"\n=== Testing Relationships ===")
related = get_related_tables('customers')
print(f"Tables related to customers: {related['direct']}")

# Test 3: Sample queries
print(f"\n=== Testing Sample Queries ===")
basic_queries = get_queries_by_complexity('beginner')
print(f"Found {len(basic_queries)} beginner queries")

# Test 4: Tools
print(f"\n=== Testing Tools ===")
# Test schema inspector
context = schema_inspector.get_table_context('customers')
print(f"Schema inspector works: {context['success']}")

# Test SQL executor with simple query
result = sql_executor.execute_query("SELECT COUNT(*) FROM customers;")
print(f"SQL executor works: {result['success']}")
print(f"Customer count: {result['data'][0] if result['success'] else 'Failed'}")

# Test validator
validation = query_validator.validate_query("SELECT * FROM customers LIMIT 5;")
print(f"Query validator works: {validation['success']}")