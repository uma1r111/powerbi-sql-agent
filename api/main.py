"""
FastAPI Backend for IntelliQuery
Provides REST API for the SQL Agent
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

class ConversationItem(BaseModel):
    id: int
    question: str
    sql: str
    timestamp: datetime
    result_count: int

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
        "agent_ready": True
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
                execution_time=None  # None instead of 0.0
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
            execution_time=execution_time if sql_query else None
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
            execution_time=None
        )

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
        "ai_model": "ready"
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
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)