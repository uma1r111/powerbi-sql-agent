# state/plan_state.py

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class PlanStatus(str, Enum):
    """Enumeration of possible plan statuses"""
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REVISED = "revised"
    CANCELLED = "cancelled"

class StepStatus(str, Enum):
    """Enumeration of possible step statuses"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PlanStep(BaseModel):
    """Individual step in an execution plan"""
    
    step_id: str = Field(..., description="Unique identifier for the step")
    step_number: int = Field(..., description="Order of step in the plan")
    description: str = Field(..., description="Human-readable description of the step")
    node_target: str = Field(..., description="Target node that will execute this step")
    
    # Step metadata
    status: StepStatus = Field(default=StepStatus.PENDING)
    estimated_duration: Optional[float] = Field(default=None, description="Estimated duration in seconds")
    actual_duration: Optional[float] = Field(default=None, description="Actual execution duration")
    
    # Execution details
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    
    # Input/Output tracking
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for this step")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data from this step")
    
    # Error handling
    error_message: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=2)
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Steps that must complete before this one")
    enables: List[str] = Field(default_factory=list, description="Steps that depend on this one")
    
    def start_execution(self) -> None:
        """Mark step as started"""
        self.status = StepStatus.IN_PROGRESS
        self.start_time = datetime.now()
    
    def complete_execution(self, output_data: Optional[Dict[str, Any]] = None) -> None:
        """Mark step as completed"""
        self.status = StepStatus.COMPLETED
        self.end_time = datetime.now()
        if self.start_time:
            self.actual_duration = (self.end_time - self.start_time).total_seconds()
        if output_data:
            self.output_data = output_data
    
    def fail_execution(self, error_message: str) -> None:
        """Mark step as failed"""
        self.status = StepStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error_message
        self.retry_count += 1
    
    def can_retry(self) -> bool:
        """Check if step can be retried"""
        return self.retry_count < self.max_retries and self.status == StepStatus.FAILED

class ExecutionPlan(BaseModel):
    """Complete execution plan for agent processing"""
    
    # Plan identification
    plan_id: str = Field(..., description="Unique identifier for the plan")
    plan_name: str = Field(..., description="Human-readable plan name")
    plan_description: str = Field(..., description="Detailed description of what the plan achieves")
    
    # Plan metadata
    status: PlanStatus = Field(default=PlanStatus.CREATED)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Plan content
    steps: List[PlanStep] = Field(default_factory=list, description="Ordered list of execution steps")
    current_step_index: int = Field(default=0, description="Index of currently executing step")
    
    # Plan context (From YouTube - Enhanced)
    original_query: str = Field(..., description="Original user query")
    business_intent: str = Field(default="", description="Detected business intent")
    complexity_level: Literal["low", "medium", "high"] = Field(default="low")
    estimated_total_duration: Optional[float] = Field(default=None)
    
    # Success criteria
    success_criteria: List[str] = Field(default_factory=list, description="Criteria for successful completion")
    validation_checkpoints: List[str] = Field(default_factory=list, description="Points where validation occurs")
    
    # Revision tracking
    revision_number: int = Field(default=1, description="Number of times plan has been revised")
    revision_history: List[str] = Field(default_factory=list, description="History of plan changes")
    
    # Resource requirements
    required_tables: List[str] = Field(default_factory=list, description="Database tables needed")
    required_tools: List[str] = Field(default_factory=list, description="Tools needed for execution")
    
    def add_step(self, description: str, node_target: str, depends_on: List[str] = None, 
                 estimated_duration: float = None) -> PlanStep:
        """Add a new step to the plan"""
        step_number = len(self.steps) + 1
        step_id = f"{self.plan_id}_step_{step_number}"
        
        step = PlanStep(
            step_id=step_id,
            step_number=step_number,
            description=description,
            node_target=node_target,
            depends_on=depends_on or [],
            estimated_duration=estimated_duration
        )
        
        self.steps.append(step)
        return step
    
    def get_current_step(self) -> Optional[PlanStep]:
        """Get the currently executing step"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next step to execute"""
        next_index = self.current_step_index + 1
        if next_index < len(self.steps):
            return self.steps[next_index]
        return None
    
    def advance_to_next_step(self) -> bool:
        """Move to the next step in the plan"""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            return True
        return False
    
    def get_completed_steps(self) -> List[PlanStep]:
        """Get all completed steps"""
        return [step for step in self.steps if step.status == StepStatus.COMPLETED]
    
    def get_failed_steps(self) -> List[PlanStep]:
        """Get all failed steps"""
        return [step for step in self.steps if step.status == StepStatus.FAILED]
    
    def get_pending_steps(self) -> List[PlanStep]:
        """Get all pending steps"""
        return [step for step in self.steps if step.status == StepStatus.PENDING]
    
    def calculate_progress(self) -> Dict[str, Any]:
        """Calculate plan progress"""
        total_steps = len(self.steps)
        completed_steps = len(self.get_completed_steps())
        failed_steps = len(self.get_failed_steps())
        
        progress_percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "pending_steps": total_steps - completed_steps - failed_steps,
            "progress_percentage": progress_percentage,
            "current_step_description": self.get_current_step().description if self.get_current_step() else "Complete"
        }
    
    def start_execution(self) -> None:
        """Start plan execution"""
        self.status = PlanStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete_execution(self) -> None:
        """Complete plan execution"""
        self.status = PlanStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def fail_execution(self, reason: str) -> None:
        """Mark plan as failed"""
        self.status = PlanStatus.FAILED
        self.completed_at = datetime.now()
        self.revision_history.append(f"Plan failed: {reason} at {datetime.now()}")
    
    def revise_plan(self, revision_note: str) -> None:
        """Revise the current plan"""
        self.revision_number += 1
        self.status = PlanStatus.REVISED
        self.revision_history.append(f"Revision {self.revision_number}: {revision_note} at {datetime.now()}")
    
    def to_system_prompt_format(self) -> str:
        """Format plan for injection into system prompts"""
        current_step = self.get_current_step()
        progress = self.calculate_progress()
        
        plan_text = f"""
CURRENT EXECUTION PLAN: {self.plan_name}
Description: {self.plan_description}
Progress: {progress['completed_steps']}/{progress['total_steps']} steps completed ({progress['progress_percentage']:.1f}%)

PLAN STEPS:
"""
        
        for i, step in enumerate(self.steps):
            status_emoji = {
                StepStatus.COMPLETED: "✅",
                StepStatus.IN_PROGRESS: "🔄", 
                StepStatus.FAILED: "❌",
                StepStatus.PENDING: "⏳",
                StepStatus.SKIPPED: "⏭️"
            }
            
            marker = ">>> " if i == self.current_step_index else "    "
            emoji = status_emoji.get(step.status, "❓")
            
            plan_text += f"{marker}{step.step_number}. {emoji} {step.description}\n"
        
        if current_step:
            plan_text += f"\nCURRENT STEP: {current_step.description}"
            plan_text += f"\nTARGET NODE: {current_step.node_target}"
        
        return plan_text.strip()

class PlanManager:
    """Utility class for managing execution plans"""
    
    @staticmethod
    def create_basic_sql_plan(query: str, selected_tables: List[str]) -> ExecutionPlan:
        """Create a basic plan for SQL query processing"""
        import uuid
        
        plan_id = f"sql_plan_{str(uuid.uuid4())[:8]}"
        plan = ExecutionPlan(
            plan_id=plan_id,
            plan_name="SQL Query Processing Plan",
            plan_description=f"Process natural language query: '{query}' using tables: {', '.join(selected_tables)}",
            original_query=query,
            required_tables=selected_tables,
            required_tools=["SchemaInspectorTool", "QueryValidatorTool", "SQLExecutorTool"]
        )
        
        # Add standard steps
        plan.add_step("Inspect database schema for selected tables", "schema_inspector", estimated_duration=2.0)
        plan.add_step("Generate SQL query based on schema and context", "planner", depends_on=["schema_inspector"], estimated_duration=3.0)
        plan.add_step("Validate generated SQL query", "query_validator", depends_on=["planner"], estimated_duration=1.0)
        plan.add_step("Execute validated SQL query", "sql_executor", depends_on=["query_validator"], estimated_duration=5.0)
        plan.add_step("Format and present results", "output_formatter", depends_on=["sql_executor"], estimated_duration=1.0)
        
        # Set success criteria
        plan.success_criteria = [
            "SQL query generated successfully",
            "Query passes all validation checks",
            "Query executes without errors",
            "Results returned and formatted properly"
        ]
        
        # Set validation checkpoints
        plan.validation_checkpoints = [
            "After SQL generation",
            "After query validation", 
            "After query execution"
        ]
        
        return plan
    
    @staticmethod
    def create_complex_analysis_plan(query: str, selected_tables: List[str], requires_joins: bool = False) -> ExecutionPlan:
        """Create a plan for complex analytical queries"""
        import uuid
        
        plan_id = f"analysis_plan_{str(uuid.uuid4())[:8]}"
        plan = ExecutionPlan(
            plan_id=plan_id,
            plan_name="Complex Analysis Plan",
            plan_description=f"Complex analysis for: '{query}' involving {len(selected_tables)} tables",
            original_query=query,
            complexity_level="high",
            required_tables=selected_tables,
            required_tools=["SchemaInspectorTool", "QueryValidatorTool", "SQLExecutorTool", "StatisticalAnalysisTool"]
        )
        
        # Enhanced steps for complex analysis
        plan.add_step("Analyze table relationships and join requirements", "schema_inspector", estimated_duration=3.0)
        plan.add_step("Select relevant few-shot examples", "example_selector", estimated_duration=2.0)
        plan.add_step("Generate complex SQL with joins and aggregations", "planner", depends_on=["schema_inspector", "example_selector"], estimated_duration=5.0)
        plan.add_step("Validate complex query structure", "query_validator", depends_on=["planner"], estimated_duration=2.0)
        plan.add_step("Execute complex query with monitoring", "sql_executor", depends_on=["query_validator"], estimated_duration=10.0)
        plan.add_step("Analyze results and suggest visualizations", "result_analyzer", depends_on=["sql_executor"], estimated_duration=3.0)
        plan.add_step("Format comprehensive response", "output_formatter", depends_on=["result_analyzer"], estimated_duration=2.0)
        
        return plan

# Export main classes
__all__ = ["PlanStep", "ExecutionPlan", "PlanManager", "PlanStatus", "StepStatus"]