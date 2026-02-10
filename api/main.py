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

# Add parent directory to path to import from other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Import your SQL Agent from flow.graph
from flow.graph import agent

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

# Session state storage (in production, use Redis or database)
session_states = {}

# User store with 3 users
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
    """JSON login endpoint for the React frontend"""
    user = fake_users_db.get(request.email)
    
    # 🔍 DEBUG PRINT - This will show up in your terminal
    print(f"\n\n👉 DEBUG: Email='{request.email}'")
    print(f"👉 DEBUG: Password Sent='{request.password}'")
    print(f"👉 DEBUG: Password Expected='{user['hashed_password'] if user else 'NO USER FOUND'}'\n\n")

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
    """Process natural language query"""
    try:
        start_time = datetime.now()

        # Get existing state for this session (for follow-up context)
        existing_state = session_states.get(request.session_id)

        # Process query using the agent
        result = agent.process_query_sync(
            request.question,
            session_id=request.session_id,
            existing_state=existing_state
        )

        # Store state for next query in this session
        session_states[request.session_id] = result

        execution_time = (datetime.now() - start_time).total_seconds()

        # Extract response from result
        response_text = ""
        if result.messages and len(result.messages) > 0:
            response_text = result.messages[-1].content

        # Extract SQL and results from the state
        sql_query = getattr(result, 'sql_query', None)
        query_results = getattr(result, 'query_results', [])

        return QueryResponse(
            success=True,
            sql=sql_query,
            results=query_results if query_results else [],
            explanation=response_text,
            warnings=[],
            error=None,
            execution_time=execution_time
        )

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Query error: {error_detail}")
        
        return QueryResponse(
            success=False,
            sql=None,
            results=[],
            explanation=None,
            warnings=[],
            error=str(e),
            execution_time=0
        )

@app.get("/api/history", response_model=List[ConversationItem])
async def get_conversation_history(
    session_id: str = "default",
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    """Get conversation history"""
    # This would query your state's context window
    # For now, returning empty list
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
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)