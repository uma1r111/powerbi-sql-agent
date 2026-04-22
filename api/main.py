"""
FastAPI Backend for IntelliQuery
Provides REST API for the SQL Agent with Dashboard Support
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
import os
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from flow.graph import agent
from utils.query_preprocessor import preprocessor
from utils.query_classifier import classifier
from tools.dashboard_manager import dashboard_manager
from tools.chart_recommender import chart_recommender
from database.redis_client import RedisClient
from database.session_store import user_session_store
from rag.document_store import document_store
from rag.rag_agent import query_rag_agent

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-nowwwwwwwwwwwwwwwwwwww!!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="IntelliQuery API",
    description="AI-Powered Conversational BI Dashboard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# LangGraph AgentState — kept in memory (not JSON-serializable)
session_states = {}

# Redis singleton
redis_client = RedisClient.get_instance()

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

# ------------------------------------------------------------------ #
#  Pydantic Models                                                     #
# ------------------------------------------------------------------ #

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
    conv_id: Optional[int] = None

class QueryResponse(BaseModel):
    success: bool
    sql: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None
    warnings: Optional[List[str]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = datetime.now()
    chart: Optional[Dict[str, Any]] = None

class ConversationItem(BaseModel):
    id: int
    question: str
    sql: str
    timestamp: datetime
    result_count: int

class SaveSessionRequest(BaseModel):
    conversations: List[Dict[str, Any]]
    active_conv_id: int

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

# ------------------------------------------------------------------ #
#  Auth helpers                                                        #
# ------------------------------------------------------------------ #

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

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

# ------------------------------------------------------------------ #
#  Redis helpers for last_query_per_session                           #
# ------------------------------------------------------------------ #

def _get_session_context(session_id: str) -> dict:
    data = redis_client.get(redis_client.last_query_key(session_id))
    return data if data is not None else {}

def _save_session_context(session_id: str, query: str, topic: str) -> None:
    redis_client.set(
        redis_client.last_query_key(session_id),
        {"query": query, "topic": topic}
    )

# ------------------------------------------------------------------ #
#  Root                                                                #
# ------------------------------------------------------------------ #

@app.get("/")
async def root():
    return {
        "message": "IntelliQuery API",
        "version": "1.0.0",
        "status": "running",
        "agent_ready": True,
        "dashboard_enabled": True,
        "redis": "connected" if redis_client.is_connected else "fallback mode"
    }

# ------------------------------------------------------------------ #
#  Auth routes                                                         #
# ------------------------------------------------------------------ #

@app.post("/api/login", response_model=LoginResponse)
async def login_json(request: LoginRequest):
    user = fake_users_db.get(request.email)
    if not user or user["hashed_password"] != request.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {
        "token": access_token,
        "user": {"email": user["email"], "full_name": user["full_name"]}
    }

@app.get("/api/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# ------------------------------------------------------------------ #
#  Session endpoints                                                   #
# ------------------------------------------------------------------ #

@app.get("/api/session/load")
async def load_user_session(current_user: User = Depends(get_current_user)):
    """
    Called immediately after login.
    Returns the user's full session (conversations + visualization history)
    from Redis so the frontend restores exactly where they left off.
    Returns first_login=True if no session exists yet.
    """
    try:
        session = user_session_store.load_session(current_user.email)

        if not session:
            print(f"👋 First login for {current_user.email} — no session found")
            return {"success": True, "session": None, "first_login": True}

        print(f"✅ Session restored for {current_user.email} — "
              f"{len(session.get('conversations', []))} conversations")

        return {"success": True, "session": session, "first_login": False}

    except Exception as e:
        print(f"❌ Session load error: {e}")
        return {"success": False, "session": None, "first_login": True}


@app.post("/api/session/save")
async def save_user_session(
    request: SaveSessionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Called by the frontend whenever state changes.
    Strips raw result data (too large) — only saves text, SQL, chart configs.
    Chart SQLs are stored so data can be re-fetched fresh on next login.
    """
    try:
        sanitized_convs = []
        for conv in request.conversations:
            sanitized_messages = []
            for msg in conv.get("messages", []):
                sanitized_messages.append({
                    "type": msg.get("type"),
                    "content": msg.get("content"),
                    "sql": msg.get("sql"),
                    "execution_time": msg.get("execution_time"),
                    "timestamp": msg.get("timestamp"),
                    "chart": {
                        "chart_id": msg["chart"].get("chart_id"),
                        "title": msg["chart"].get("title"),
                        "type": msg["chart"].get("type"),
                    } if msg.get("chart") else None,
                    # Strip result rows — re-fetched fresh on load
                    "results": [],
                })

            sanitized_charts = []
            for chart in conv.get("charts", []):
                sanitized_charts.append({
                    "chart_id": chart.get("chart_id"),
                    "title": chart.get("title"),
                    "type": chart.get("type"),
                    "sql": chart.get("sql"),
                    "config": chart.get("config", {}),
                    "query": chart.get("query", ""),
                    "created_at": chart.get("created_at"),
                    "position": chart.get("position", {}),
                })

            sanitized_convs.append({
                "id": conv["id"],
                "title": conv["title"],
                "messages": sanitized_messages,
                "charts": sanitized_charts,
                "lastUpdated": conv.get("lastUpdated"),
            })

        success = user_session_store.save_session(current_user.email, {
            "conversations": sanitized_convs,
            "active_conv_id": request.active_conv_id,
        })

        return {"success": success}

    except Exception as e:
        print(f"❌ Session save error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/session/stats")
async def get_session_stats(current_user: User = Depends(get_current_user)):
    """Debug endpoint — returns session stats for current user."""
    stats = user_session_store.get_session_stats(current_user.email)
    return {"success": True, "stats": stats}


# ------------------------------------------------------------------ #
#  Query routing                                                       #
# ------------------------------------------------------------------ #

# Terms that strongly indicate the question is about YOUR DATABASE DATA.
# If none of these appear and docs are uploaded, we route to RAG instead.
_DB_SIGNALS = [
    # Northwind entities
    "customer", "order", "product", "employee", "supplier", "shipper",
    "category", "territory", "region",
    # Business metrics
    "revenue", "sales", "profit", "quantity", "stock", "inventory",
    "discount", "freight", "price",
    # Explicit data requests
    "show me", "list", "how many", "count", "top ", "bottom ",
    "best selling", "worst", "compare", "trend", "chart", "graph",
    "monthly", "yearly", "quarterly", "by country", "by category",
]

def _is_rag_query(question: str) -> bool:
    """
    Route to RAG when:
      1. Documents have been uploaded, AND
      2. The question has no clear signal that it's about the database.

    This means ANY question that isn't obviously about Northwind data will
    be answered from uploaded PDFs + web — so users never need to know
    which endpoint to call.
    """
    if not document_store.has_documents():
        return False
    q = question.lower()
    # If there's a clear DB signal, let the SQL agent handle it
    return not any(signal in q for signal in _DB_SIGNALS)


# ------------------------------------------------------------------ #
#  Query endpoint                                                      #
# ------------------------------------------------------------------ #

@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        start_time = datetime.now()

        # ── RAG route ─────────────────────────────────────────────────────────
        if _is_rag_query(request.question):
            print(f"\n📄 RAG route detected for: '{request.question}'")
            rag_result = query_rag_agent(request.question)
            sources_tag = " + ".join(rag_result["sources_used"]) or "none"
            explanation = (
                f"{rag_result['answer']}\n\n"
                f"*(Sources: {sources_tag} | {rag_result['steps']} tool call(s))*"
            )
            return QueryResponse(
                success=True, sql=None, results=[], explanation=explanation,
                warnings=[], error=None,
                execution_time=(datetime.now() - start_time).total_seconds(),
                chart=None,
            )

        session_context = _get_session_context(request.session_id)
        last_query = session_context.get("query", "")
        last_topic = session_context.get("topic", "")

        needs_sql, response_type = classifier.is_sql_query(
            request.question,
            last_query=last_query,
            last_topic=last_topic
        )

        if not needs_sql:
            response = classifier.generate_response(request.question, response_type)
            print(f"\n💬 Non-SQL query detected: {response_type}")
            return QueryResponse(
                success=True, sql=None, results=[], explanation=response,
                warnings=[], error=None, execution_time=None, chart=None
            )

        processed_query = request.question
        if response_type == 'follow_up':
            processed_query = classifier.expand_follow_up_query(
                request.question, last_query, last_topic
            )
            print(f"\n🔗 Context expansion: '{request.question}' → '{processed_query}'")

        preprocessed_query, corrections = preprocessor.preprocess(processed_query)

        if corrections:
            print(f"\n📝 Preprocessed: '{preprocessed_query}' corrections: {corrections}")

        existing_state = session_states.get(request.session_id)
        print(f"\n🔍 Sending to agent: '{preprocessed_query}'")

        result = agent.process_query_sync(
            preprocessed_query,
            session_id=request.session_id,
            existing_state=existing_state
        )

        session_states[request.session_id] = result
        execution_time = (datetime.now() - start_time).total_seconds()

        topic = "general"
        if 'product' in preprocessed_query.lower(): topic = "product"
        elif 'customer' in preprocessed_query.lower(): topic = "customer"
        elif 'order' in preprocessed_query.lower(): topic = "order"
        elif 'sales' in preprocessed_query.lower() or 'revenue' in preprocessed_query.lower(): topic = "sales"

        _save_session_context(request.session_id, preprocessed_query, topic)

        response_text = ""
        for msg in reversed(result.messages or []):
            msg_type = getattr(msg, 'type', None) or (msg.get('type') if isinstance(msg, dict) else None)
            if msg_type in ('ai', 'AIMessage') or (hasattr(msg, '__class__') and 'AI' in msg.__class__.__name__):
                response_text = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                break

        if not response_text or not result.processing_complete:
            if result.errors:
                last_err = result.errors[-1]
                if last_err.startswith("Workflow execution failed:") or last_err.startswith("Query validation error:"):
                    response_text = "I was unable to process your query due to an internal error. Please try rephrasing your question."
                else:
                    response_text = last_err
            elif not result.execution_successful:
                response_text = "I was unable to retrieve results for that query. Please try rephrasing or simplifying your question."
            else:
                response_text = "Query processed successfully."

        sql_query = None
        if hasattr(result, 'cleaned_sql') and result.cleaned_sql:
            sql_query = result.cleaned_sql
        elif hasattr(result, 'generated_sql') and result.generated_sql:
            sql_query = result.generated_sql

        query_results = []
        if hasattr(result, 'execution_results') and result.execution_results:
            if isinstance(result.execution_results, dict):
                query_results = result.execution_results.get('data', [])
            elif isinstance(result.execution_results, list):
                query_results = result.execution_results

        chart_config = None
        if query_results and len(query_results) > 0 and result.execution_successful:
            try:
                should_visualize = any(keyword in preprocessed_query.lower() for keyword in [
                    'show', 'chart', 'graph', 'visualize', 'plot', 'compare', 'trend',
                    'top', 'best', 'worst', 'distribution', 'breakdown', 'over time'
                ])
                if should_visualize or len(query_results) <= 20:
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

        print(f"\n{'='*70}")
        print(f"   User Input: {request.question}")
        print(f"   SQL Found: {sql_query is not None} | Results: {len(query_results)} | Chart: {chart_config is not None}")
        print(f"   Execution Successful: {result.execution_successful} | Complete: {result.processing_complete}")
        if result.errors: print(f"   Errors: {result.errors}")
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
        print(f"❌ Query error: {str(e)}\n{traceback.format_exc()}")
        return QueryResponse(
            success=False, sql=None, results=[], explanation=None,
            warnings=[], error=str(e), execution_time=None, chart=None
        )

# ------------------------------------------------------------------ #
#  Dashboard endpoints                                                 #
# ------------------------------------------------------------------ #

@app.get("/api/dashboard/initial")
async def get_initial_dashboard(
    session_id: str = "default",
    current_user: User = Depends(get_current_user)
):
    try:
        dashboard = dashboard_manager.generate_initial_dashboard(
            user_email=current_user.email, session_id=session_id
        )
        return {"success": True, "dashboard": dashboard}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")


@app.get("/api/dashboard/current")
async def get_current_dashboard(
    session_id: str = "default",
    current_user: User = Depends(get_current_user)
):
    try:
        dashboard = dashboard_manager.get_dashboard(session_id)
        if not dashboard:
            dashboard = dashboard_manager.generate_initial_dashboard(
                user_email=current_user.email, session_id=session_id
            )
        return {"success": True, "dashboard": dashboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")


@app.post("/api/dashboard/add-chart")
async def add_chart_to_dashboard(
    request: AddChartRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        chart = dashboard_manager.add_chart_from_query(
            session_id=request.session_id, query=request.query,
            sql=request.sql, result=request.result
        )
        if not chart:
            raise HTTPException(status_code=400, detail="Could not generate chart from query result")
        return {"success": True, "chart": chart}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add chart: {str(e)}")


@app.post("/api/dashboard/cross-filter")
async def apply_cross_filter(
    request: CrossFilterRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        updated_dashboard = dashboard_manager.apply_filter(
            session_id=request.session_id,
            filter_key=request.filter_key,
            filter_value=request.filter_value
        )
        return {"success": True, "dashboard": updated_dashboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply filter: {str(e)}")


@app.post("/api/dashboard/clear-filters")
async def clear_all_filters(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        updated_dashboard = dashboard_manager.clear_filters(session_id)
        return {"success": True, "dashboard": updated_dashboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear filters: {str(e)}")


@app.post("/api/dashboard/remove-chart")
async def remove_chart_from_dashboard(
    request: RemoveChartRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        success = dashboard_manager.remove_chart(
            session_id=request.session_id, chart_id=request.chart_id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Chart not found")
        return {"success": True, "message": "Chart removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove chart: {str(e)}")


@app.post("/api/dashboard/update-position")
async def update_chart_position(
    request: UpdatePositionRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        success = dashboard_manager.update_chart_position(
            session_id=request.session_id,
            chart_id=request.chart_id,
            position=request.position
        )
        if not success:
            raise HTTPException(status_code=404, detail="Chart not found")
        return {"success": True, "message": "Chart position updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update position: {str(e)}")


@app.delete("/api/dashboard/clear")
async def clear_dashboard(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        success = dashboard_manager.clear_dashboard(session_id)
        return {"success": success, "message": "Dashboard cleared" if success else "Dashboard not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear dashboard: {str(e)}")


@app.get("/api/history", response_model=List[ConversationItem])
async def get_conversation_history(
    session_id: str = "default",
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    return []


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "IntelliQuery API",
        "database": "connected",
        "ai_model": "ready",
        "dashboard": "enabled",
        "redis": redis_client.get_stats()
    }


# ── RAG models ────────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    question: str

class RAGQueryResponse(BaseModel):
    answer: str
    sources_used: List[str]
    steps: int

class RAGUploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    total_chunks: int
    message: str

class RAGSourcesResponse(BaseModel):
    sources: List[str]
    total_chunks: int


# ── RAG endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/rag/upload", response_model=RAGUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a PDF to be indexed in the company knowledge base.
    Requires authentication. Supports PDF files only.
    """
    from rag.document_store import SUPPORTED_EXTENSIONS
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Save to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks_added = document_store.add_document(tmp_path, source_label=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return RAGUploadResponse(
        filename=file.filename,
        chunks_indexed=chunks_added,
        total_chunks=document_store.chunk_count,
        message=f"Successfully indexed '{file.filename}' ({chunks_added} chunks).",
    )


@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Ask a question answered from company documents (PDFs) or the web.
    The agent searches company docs first; falls back to web search if needed.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = query_rag_agent(request.question)
    return RAGQueryResponse(**result)


@app.get("/api/rag/sources", response_model=RAGSourcesResponse)
async def list_rag_sources(current_user: dict = Depends(get_current_user)):
    """List all indexed document sources and total chunk count."""
    return RAGSourcesResponse(
        sources=document_store.list_sources(),
        total_chunks=document_store.chunk_count,
    )


@app.delete("/api/rag/sources/{source_name}")
async def delete_rag_source(
    source_name: str,
    current_user: dict = Depends(get_current_user),
):
    """Remove a document from the knowledge base by its filename."""
    deleted = document_store.delete_source(source_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found.")
    return {"deleted_chunks": deleted, "source": source_name}


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 Starting IntelliQuery API Server...")
    print("=" * 70)
    print("📍 API:        http://localhost:8000")
    print("📍 Docs:       http://localhost:8000/docs")
    print("📍 Redis:     ", "ENABLED ✅" if redis_client.is_connected else "FALLBACK MODE ⚠️")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)