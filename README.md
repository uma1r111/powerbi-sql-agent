# NLP to SQL Agent

![NLP to SQL Agent Flow](NLP_to_SQL_Agent_Flow.png)

## Overview

This project implements an **intelligent agent** that translates **natural language questions** into **SQL queries**, validates them against schema and business logic, and optionally executes them safely against a database.  

It is designed with a modular, **graph-based planning architecture**, enabling multi-step reasoning about queries and schema exploration. The system is schema-aware, supports safety checks, and provides actionable suggestions to improve query quality.

---

## Features

- **Natural Language to SQL**: Converts NL questions into SQL queries using planner nodes.
- **Schema Inspection**: Ensures queries refer only to valid tables and columns.
- **Query Validation**:  
  - Syntax and security checks (e.g., SQL injection prevention).  
  - Table and column verification.  
  - Business logic checks (e.g., revenue calculation, date filters).  
  - Query complexity estimation (low, medium, high).
- **Graph-based Planning**: Nodes and edges define dependencies for query generation and execution.
- **Database Safety**: Optional `EXPLAIN` validation before executing queries to ensure safety.
- **Tool Registry**: Modular tools for SQL operations, schema inspection, and validation.
- **State Management**: Tracks agent knowledge and planning state across multi-step reasoning.

---

## Agent Flow

The agent follows this high-level flow:

1) Intent Understanding: Receive natural language question.

2) Schema Inspection: Identify relevant tables and key columns.

3) Query Planning: Generate candidate SQL queries using planner nodes.

4) Validation: Check syntax, security, table/column correctness, business rules, and query complexity.

5) Execution & Recommendations: Return results and actionable suggestions.

## Project Structure

```text
.
├── database/
│   ├── explore_schema.py       # Functions to explore database schema
│   ├── northwind_context.py    # Northwind schema context and table definitions
│   ├── relationships.py        # Table relationships
│   └── sample_queries.py       # Sample SQL queries for testing
├── flow/
│   ├── edge.py                 # Graph edge definition
│   └── graph.py                # Graph structure for planning
├── nodes/
│   ├── planner.py              # Planner node for query generation
│   └── schema_inspector.py     # Schema inspection node
├── state/
│   ├── agent_state.py          # Tracks agent's knowledge and context
│   └── plan_state.py           # Tracks execution of plans and steps
├── tools/
│   ├── schema_tools.py         # Schema-related utility functions
│   ├── sql_tools.py            # SQL execution and formatting tools
│   └── validation_tools.py     # Syntax, security, and business logic validation
├── NLP_to_SQL_Agent_flow.png   # Flow diagram of the agent
└── README.md                   # Project documentation
