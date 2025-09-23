from state.agent_state import AgentState, StateManager
from state.plan_state import PlanManager

# Test state creation
state = StateManager.create_initial_state("Show me customers from Germany")
plan = PlanManager.create_basic_sql_plan("Show me customers from Germany", ["customers"])

print("State created:", state.user_query)
print("Plan created:", plan.plan_name)
print("Steps:", len(plan.steps))