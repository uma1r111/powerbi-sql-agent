from state.agent_state import AgentState, StateManager
from nodes.schema_inspector import schema_inspector_node
from nodes.planner import planner_node

# Create initial state
state = StateManager.create_initial_state("Show me customers from Germany")

# Run schema inspector
state = schema_inspector_node.execute(state)
print("Selected tables:", state.selected_tables)
print("Business intent:", state.business_intent)

# Run planner  
state = planner_node.execute(state)
print("Generated SQL:", state.cleaned_sql)