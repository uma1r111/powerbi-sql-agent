"""
FastAPI Backend for IntelliQuery
Provides REST API for the SQL Agent with Dashboard Support
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Import SQL Agent
from flow.graph import agent

# Import preprocessor and classifier
from utils.query_preprocessor import preprocessor
from utils.query_classifier import classifier

# Import dashboard components
from tools.dashboard_manager import dashboard_manager
from tools.chart_recommender import chart_recommender

load_dotenv()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-nowwwwwwwwwwwwwwwwwwww!!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI
app = FastAPI(
    title="IntelliQuery API",
    description="AI-Powered Conversational BI Dashboard",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Session state storage
session_states = {}

# Track last query per session for context
last_query_per_session = {}

# User store
fake_users_db = {
    "sameed@intelliquery.com": {
        "username": "sameed@intelliquery.com",
        "full_name": "Sameed",
        "email": "sameed@intelliquery.com",
        "hashed_password": "1234",
        "disabled": False,
    },
    "izma@intelliquery.com": {
        "username": "izma@intelliquery.com",
        "full_name": "Izma",
        "email": "izma@intelliquery.com",
        "hashed_password": "1234",
        "disabled": False,
    },
    "umair@intelliquery.com": {
        "username": "umair@intelliquery.com",
        "full_name": "Umair",
        "email": "umair@intelliquery.com",
        "hashed_password": "1234",
        "disabled": False,
    }
}

# Pydantic Models
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    token: str
    user: dict

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    success: bool
    sql: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None
    warnings: Optional[List[str]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = datetime.now()
    chart: Optional[Dict[str, Any]] = None  # New: chart config if applicable

class ConversationItem(BaseModel):
    id: int
    question: str
    sql: str
    timestamp: datetime
    result_count: int

# Dashboard Models
class AddChartRequest(BaseModel):
    session_id: str
    query: str
    sql: str
    result: Dict[str, Any]

class CrossFilterRequest(BaseModel):
    session_id: str
    filter_key: str
    filter_value: Any

class RemoveChartRequest(BaseModel):
    session_id: str
    chart_id: str

class UpdatePositionRequest(BaseModel):
    session_id: str
    chart_id: str
    position: Dict[str, int]

# Helper Functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(username: str = Depends(verify_token)):
    user = fake_users_db.get(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return User(**user)

# Routes
@app.get("/")
async def root():
    return {
        "message": "IntelliQuery API",
        "version": "1.0.0",
        "status": "running",
        "agent_ready": True,
        "dashboard_enabled": True
    }

@app.post("/api/login", response_model=LoginResponse)
async def login_json(request: LoginRequest):
    """JSON login endpoint"""
    user = fake_users_db.get(request.email)

    if not user or user["hashed_password"] != request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {
        "token": access_token,
        "user": {
            "email": user["email"],
            "full_name": user["full_name"]
        }
    }

@app.get("/api/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user

@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    """Process natural language query with preprocessing and context awareness"""
    try:
        start_time = datetime.now()
        
        # Get context from last query
        session_context = last_query_per_session.get(request.session_id, {})
        last_query = session_context.get("query", "")
        last_topic = session_context.get("topic", "")
        
        # Check if greeting/casual conversation or follow-up
        needs_sql, response_type = classifier.is_sql_query(
            request.question, 
            last_query=last_query,
            last_topic=last_topic
        )
        
        if not needs_sql:
            # Generate conversational response
            response = classifier.generate_response(request.question, response_type)
            
            print(f"\n💬 Non-SQL query detected: {response_type}")
            print(f"   Query: {request.question}")
            
            return QueryResponse(
                success=True,
                sql=None,
                results=[],
                explanation=response,
                warnings=[],
                error=None,
                execution_time=None,
                chart=None
            )
        
        # Handle follow-up queries
        processed_query = request.question
        if response_type == 'follow_up':
            processed_query = classifier.expand_follow_up_query(
                request.question, 
                last_query, 
                last_topic
            )
            print(f"\n🔗 Context expansion:")
            print(f"   Original: {request.question}")
            print(f"   Expanded: {processed_query}\n")
        
        # Preprocess the SQL query
        preprocessed_query, corrections = preprocessor.preprocess(processed_query)
        
        if corrections:
            print(f"\n📝 Query Preprocessing:")
            print(f"   Original: {processed_query}")
            print(f"   Preprocessed: {preprocessed_query}")
            print(f"   Corrections: {', '.join(corrections)}\n")
        
        existing_state = session_states.get(request.session_id)
        
        # Process SQL query
        print(f"\n🔍 Sending to agent: '{preprocessed_query}'")
        
        result = agent.process_query_sync(
            preprocessed_query,
            session_id=request.session_id,
            existing_state=existing_state
        )
        
        session_states[request.session_id] = result
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Extract topic
        topic = "general"
        if 'product' in preprocessed_query.lower():
            topic = "product"
        elif 'customer' in preprocessed_query.lower():
            topic = "customer"
        elif 'order' in preprocessed_query.lower():
            topic = "order"
        elif 'sales' in preprocessed_query.lower() or 'revenue' in preprocessed_query.lower():
            topic = "sales"
        
        # Store query as context
        last_query_per_session[request.session_id] = {
            "query": preprocessed_query,
            "topic": topic
        }
        
        # Extract response text
        response_text = ""
        if result.messages and len(result.messages) > 0:
            last_message = result.messages[-1]
            if hasattr(last_message, 'content'):
                response_text = last_message.content
            elif isinstance(last_message, dict):
                response_text = last_message.get('content', '')
            else:
                response_text = str(last_message)
        
        # Extract SQL
        sql_query = None
        if hasattr(result, 'cleaned_sql') and result.cleaned_sql:
            sql_query = result.cleaned_sql
        elif hasattr(result, 'generated_sql') and result.generated_sql:
            sql_query = result.generated_sql
        
        # Extract results
        query_results = []
        if hasattr(result, 'execution_results') and result.execution_results:
            if isinstance(result.execution_results, dict):
                query_results = result.execution_results.get('data', [])
            elif isinstance(result.execution_results, list):
                query_results = result.execution_results
        
        # Generate chart if results exist
        chart_config = None
        if query_results and len(query_results) > 0 and result.execution_successful:
            try:
                # Check if query should generate a chart (has visualization keywords)
                should_visualize = any(keyword in preprocessed_query.lower() for keyword in [
                    'show', 'chart', 'graph', 'visualize', 'plot', 'compare', 'trend', 
                    'top', 'best', 'worst', 'distribution', 'breakdown', 'over time'
                ])
                
                if should_visualize or len(query_results) <= 20:
                    # Generate chart automatically
                    chart_config = dashboard_manager.add_chart_from_query(
                        session_id=request.session_id,
                        query=preprocessed_query,
                        sql=sql_query,
                        result={'success': True, 'data': query_results}
                    )
                    
                    if chart_config:
                        print(f"📊 Chart generated: {chart_config.get('type')} - {chart_config.get('title')}")
                
            except Exception as e:
                print(f"⚠️ Could not generate chart: {e}")
        
        # Debug logging
        print(f"\n{'='*70}")
        print(f"📊 SQL QUERY DEBUG:")
        print(f"   User Input: {request.question}")
        print(f"   Preprocessed: {preprocessed_query}")
        print(f"   Topic: {topic}")
        print(f"   SQL Found: {sql_query is not None}")
        if sql_query:
            print(f"   SQL: {sql_query[:100]}...")
        print(f"   Results Count: {len(query_results)}")
        print(f"   Chart Generated: {chart_config is not None}")
        if chart_config:
            print(f"   Chart Type: {chart_config.get('type')}")
        print(f"   AI Response Length: {len(response_text)}")
        print(f"   AI Response Preview: {response_text[:200]}...")
        print(f"   Execution Successful: {result.execution_successful}")
        print(f"   Processing Complete: {result.processing_complete}")
        if result.errors:
            print(f"   Errors: {result.errors}")
        print(f"{'='*70}\n")
        
        return QueryResponse(
            success=True,
            sql=sql_query,
            results=query_results,
            explanation=response_text,
            warnings=[],
            error=None,
            execution_time=execution_time if sql_query else None,
            chart=chart_config
        )

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Query error: {str(e)}\n{error_trace}")
        
        return QueryResponse(
            success=False,
            sql=None,
            results=[],
            explanation=None,
            warnings=[],
            error=str(e),
            execution_time=None,
            chart=None
        )

# ==================== DASHBOARD ENDPOINTS ====================

@app.get("/api/dashboard/initial")
async def get_initial_dashboard(
    session_id: str = "default",
    current_user: User = Depends(get_current_user)
):
    """
    Generate and return initial dashboard on login
    Returns 6 default charts with KPIs and visualizations
    """
    try:
        print(f"\n🎨 Generating initial dashboard for {current_user.email}")
        
        dashboard = dashboard_manager.generate_initial_dashboard(
            user_email=current_user.email,
            session_id=session_id
        )
        
        print(f"✅ Dashboard generated with {len(dashboard.get('charts', []))} charts")
        
        return {
            "success": True,
            "dashboard": dashboard
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Dashboard generation error: {e}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate dashboard: {str(e)}"
        )

@app.get("/api/dashboard/current")
async def get_current_dashboard(
    session_id: str = "default",
    current_user: User = Depends(get_current_user)
):
    """
    Get current dashboard state for a session
    """
    try:
        dashboard = dashboard_manager.get_dashboard(session_id)
        
        if not dashboard:
            # Generate new dashboard if none exists
            dashboard = dashboard_manager.generate_initial_dashboard(
                user_email=current_user.email,
                session_id=session_id
            )
        
        return {
            "success": True,
            "dashboard": dashboard
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dashboard: {str(e)}"
        )

@app.post("/api/dashboard/add-chart")
async def add_chart_to_dashboard(
    request: AddChartRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Manually add a chart to dashboard from query result
    """
    try:
        chart = dashboard_manager.add_chart_from_query(
            session_id=request.session_id,
            query=request.query,
            sql=request.sql,
            result=request.result
        )
        
        if not chart:
            raise HTTPException(
                status_code=400,
                detail="Could not generate chart from query result"
            )
        
        return {
            "success": True,
            "chart": chart
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add chart: {str(e)}"
        )

@app.post("/api/dashboard/cross-filter")
async def apply_cross_filter(
    request: CrossFilterRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Apply cross-filter to dashboard
    Re-queries all charts with the filter applied
    """
    try:
        print(f"\n🔗 Applying cross-filter: {request.filter_key} = {request.filter_value}")
        
        updated_dashboard = dashboard_manager.apply_filter(
            session_id=request.session_id,
            filter_key=request.filter_key,
            filter_value=request.filter_value
        )
        
        return {
            "success": True,
            "dashboard": updated_dashboard
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply filter: {str(e)}"
        )

@app.post("/api/dashboard/clear-filters")
async def clear_all_filters(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Clear all filters from dashboard and refresh
    """
    try:
        updated_dashboard = dashboard_manager.clear_filters(session_id)
        
        return {
            "success": True,
            "dashboard": updated_dashboard
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear filters: {str(e)}"
        )

@app.post("/api/dashboard/remove-chart")
async def remove_chart_from_dashboard(
    request: RemoveChartRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Remove a chart from dashboard
    """
    try:
        success = dashboard_manager.remove_chart(
            session_id=request.session_id,
            chart_id=request.chart_id
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Chart not found"
            )
        
        return {
            "success": True,
            "message": "Chart removed successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove chart: {str(e)}"
        )

@app.post("/api/dashboard/update-position")
async def update_chart_position(
    request: UpdatePositionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update chart position in grid layout
    """
    try:
        success = dashboard_manager.update_chart_position(
            session_id=request.session_id,
            chart_id=request.chart_id,
            position=request.position
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Chart not found"
            )
        
        return {
            "success": True,
            "message": "Chart position updated"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update position: {str(e)}"
        )

@app.delete("/api/dashboard/clear")
async def clear_dashboard(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Clear all charts from dashboard
    """
    try:
        success = dashboard_manager.clear_dashboard(session_id)
        
        return {
            "success": success,
            "message": "Dashboard cleared" if success else "Dashboard not found"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear dashboard: {str(e)}"
        )

# ==================== EXISTING ENDPOINTS ====================

@app.get("/api/history", response_model=List[ConversationItem])
async def get_conversation_history(
    session_id: str = "default",
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    """Get conversation history"""
    return []

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected",
        "ai_model": "ready",
        "dashboard": "enabled"
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 Starting IntelliQuery API Server...")
    print("=" * 70)
    print("📍 API will be available at: http://localhost:8000")
    print("📍 API docs at: http://localhost:8000/docs")
    print("📍 Query Preprocessor: ENABLED ✅")
    print("📍 Context Awareness: ENABLED ✅")
    print("📍 Dashboard Manager: ENABLED ✅")
    print("📍 Chart Recommender: ENABLED ✅")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)