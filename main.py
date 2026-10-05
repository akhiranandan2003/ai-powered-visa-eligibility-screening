# -------------------------------
# main.py — SwiftVisa AI Backend (Production Ready)
# -------------------------------

import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

# Load environment variables from .env file
load_dotenv()

# Import configuration and logging
try:
    from config import settings
    from logging_config import setup_logging, get_logger
    
    # Setup logging
    logger = setup_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/swiftvisa.log")
    )
    logger.info("SwiftVisa Backend Starting...")
except ImportError as e:
    # Fallback to basic logging if modules not available
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("swiftvisa")
    logger.warning(f"Could not import config/logging modules: {e}")

# --- LangChain imports (final for your versions) ---
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


# ----------------------------------------
# CONFIG
# ----------------------------------------
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "vectorstore")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional
TOP_K = 5

# Check if we have a valid OpenAI API key (not placeholder)
USE_OPENAI = OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-") and "your-" not in OPENAI_API_KEY.lower() and len(OPENAI_API_KEY) > 20

# Determine which LLM to use
USE_LLM = USE_OPENAI
LLM_PROVIDER = "openai" if USE_OPENAI else "none"

# ----------------------------------------
# FASTAPI SETUP
# ----------------------------------------
app = FastAPI(
    title="SwiftVisa API",
    description="AI-Powered Visa Eligibility Checker",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow React frontend (configure for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    
    logger.info(f"→ {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"← {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.3f}s")
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ {request.method} {request.url.path} - Error: {str(e)} - Duration: {duration:.3f}s")
        raise

# ----------------------------------------
# LOAD VECTORSTORE
# ----------------------------------------
# Do not load a PyTorch/SentenceTransformer model at startup. On Render's
# 512 MiB free instance that can cause an OOM restart. Chroma can use its
# lightweight default embedding function when the persisted store exists.
db = None
retriever = None

if os.path.isdir(CHROMA_DB_DIR):
    try:
        logger.info("🔍 Loading Chroma vectorstore...")
        db = Chroma(persist_directory=CHROMA_DB_DIR)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
        logger.info("✅ Vectorstore loaded successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Vectorstore could not be loaded: {e}")
else:
    logger.warning(f"⚠️ Vectorstore directory not found: {CHROMA_DB_DIR}. Starting with data-file fallback.")


def fallback_search(query: str, k: int = TOP_K):
    """Low-memory keyword fallback when the Chroma store is unavailable."""
    from langchain_core.documents import Document
    import re

    query_terms = set(re.findall(r"\\b[a-zA-Z0-9]+\\b", query.lower()))
    scored = []
    clean_dir = "data/clean"
    if not os.path.isdir(clean_dir):
        return []

    for filename in os.listdir(clean_dir):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(clean_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            continue
        terms = set(re.findall(r"\\b[a-zA-Z0-9]+\\b", content.lower()))
        score = len(query_terms & terms)
        if score:
            scored.append((score, filename, content))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [Document(page_content=item[2], metadata={"source": item[1]}) for item in scored[:k]]

# ----------------------------------------
# DATA MODELS
# ----------------------------------------
class VisaRequest(BaseModel):
    countryOfCitizenship: str
    destinationCountry: str
    purposeOfVisit: str
    lengthOfStay: str
    age: str


class VectorStoreQuery(BaseModel):
    query: str
    k: int = TOP_K


class PolicyDocument(BaseModel):
    country: str
    visa_type: str
    content: str
    metadata: dict = {}


# ----------------------------------------
# RAG + LLM LOGIC
# ----------------------------------------
def run_rag_with_llm(query: str) -> str:
    """Use LLM (OpenAI) + Chroma for reasoning."""
    try:
        if USE_OPENAI:
            logger.info("🧠 Using ChatOpenAI (GPT-3.5) RAG reasoning...")
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        else:
            raise ValueError("No LLM configured")
        
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        result = qa.run(query)
        logger.info("✅ LLM reasoning completed successfully")
        return result
    except Exception as e:
        logger.error(f"❌ LLM reasoning failed: {e}")
        raise


def run_retrieval_only(query: str) -> str:
    """Fallback retrieval if no LLM key is found."""
    logger.info("ℹ️ Running retrieval-only mode (using vectorstore search).")
    try:
        docs = retriever.invoke(query) if retriever is not None else fallback_search(query)
        if not docs:
            logger.warning("No documents retrieved for query")
            return "❌ No relevant visa information found for your query. Please try different search terms or contact support."
        
        # Get only the most relevant document (top result)
        top_doc = docs[0]
        content = top_doc.page_content.strip()
        
        logger.info(f"✅ Retrieved the most relevant document successfully.")
        
        return f"""🔍 **VISA ELIGIBILITY ASSESSMENT**

Based on the most relevant visa policy document:

📋 {content}

---
💡 **Note**: This result is based on official visa policy documents. For the most accurate and up-to-date information, please verify with the official embassy or consulate."""
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        raise



# ----------------------------------------
# API ENDPOINT
# ----------------------------------------
@app.post("/check-eligibility")
async def check_eligibility(data: VisaRequest):
    """Main visa eligibility checking endpoint"""
    query = (
        f"Determine eligibility for a {data.purposeOfVisit} visa to {data.destinationCountry} "
        f"for a citizen of {data.countryOfCitizenship}, aged {data.age}, "
        f"staying {data.lengthOfStay} days. Provide reasoning and reference policy data."
    )
    logger.info(f"📩 Received eligibility check request: {data.destinationCountry} - {data.purposeOfVisit}")

    if USE_LLM:
        try:
            logger.info(f"🔮 Using {LLM_PROVIDER.upper()} for intelligent reasoning...")
            result = run_rag_with_llm(query)
            logger.info("✅ Eligibility check completed successfully via LLM")
            return {
                "eligibility": result, 
                "provider": LLM_PROVIDER,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"⚠️ LLM error: {e}")
            logger.info("⚠️ Falling back to retrieval-only mode")
            result = run_retrieval_only(query)
            return {
                "eligibility": result, 
                "provider": "retrieval-only",
                "timestamp": datetime.now().isoformat()
            }
    else:
        logger.info("ℹ️ No LLM API key configured - using retrieval-only mode")
        result = run_retrieval_only(query)
        return {
            "eligibility": result, 
            "provider": "retrieval-only",
            "timestamp": datetime.now().isoformat()
        }


# ----------------------------------------
# ADDITIONAL API ENDPOINTS
# ----------------------------------------
@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    import os
    
    health_status = {
        "status": "healthy",
        "service": "SwiftVisa Backend",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "vectorstore_loaded": os.path.exists(CHROMA_DB_DIR),
        "llm_available": USE_LLM,
        "llm_provider": LLM_PROVIDER if USE_LLM else None
    }
    return health_status


@app.get("/countries")
async def get_available_countries():
    """Get list of countries with visa data available"""
    import os
    countries = set()
    clean_dir = "data/clean"
    
    if os.path.exists(clean_dir):
        for filename in os.listdir(clean_dir):
            if filename.endswith(".txt"):
                # Extract country from filename (format: Country_Country_VisaType_Eligibility.txt)
                parts = filename.split("_")
                if len(parts) >= 2:
                    countries.add(parts[0])
    
    return {
        "countries": sorted(list(countries)),
        "count": len(countries)
    }


@app.get("/visa-types/{country}")
async def get_visa_types(country: str):
    """Get available visa types for a specific country"""
    import os
    visa_types = set()
    clean_dir = "data/clean"
    
    if os.path.exists(clean_dir):
        for filename in os.listdir(clean_dir):
            if filename.startswith(f"{country}_") and filename.endswith(".txt"):
                # Extract visa type from filename
                parts = filename.replace(".txt", "").split("_")
                if len(parts) >= 3:
                    visa_type = parts[2]
                    visa_types.add(visa_type)
    
    return {
        "country": country,
        "visa_types": sorted(list(visa_types)),
        "count": len(visa_types)
    }


@app.post("/vectorstore/query")
async def query_vectorstore(request: VectorStoreQuery):
    """Direct vectorstore query for custom searches"""
    if not request.query:
        return {"error": "Query parameter is required"}
    
    try:
        if db is None:\n            docs = fallback_search(request.query, request.k)\n        else:\n            docs = db.similarity_search(request.query, k=request.k)
        results = []
        
        for i, doc in enumerate(docs, 1):
            results.append({
                "rank": i,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "full_length": len(doc.page_content)
            })
        
        return {
            "query": request.query,
            "k": request.k,
            "results_count": len(results),
            "results": results,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "query": request.query}


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    import os
    from datetime import datetime
    
    # Count documents
    clean_dir = "data/clean"
    doc_count = 0
    countries = set()
    visa_types = set()
    
    if os.path.exists(clean_dir):
        for f in os.listdir(clean_dir):
            if f.endswith(".txt"):
                doc_count += 1
                parts = f.split("_")
                if len(parts) >= 3:
                    countries.add(parts[0])
                    visa_types.add(parts[2])
    
    # Check vectorstore
    vectorstore_size = 0
    if os.path.exists("vectorstore"):
        for root, dirs, files in os.walk("vectorstore"):
            vectorstore_size += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "documents_loaded": doc_count,
        "countries_available": len(countries),
        "countries": sorted(list(countries)),
        "visa_types_count": len(visa_types),
        "vectorstore_size_mb": round(vectorstore_size / (1024 * 1024), 2),
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_enabled": USE_LLM,
        "llm_provider": LLM_PROVIDER,
        "openai_available": USE_OPENAI,
        "gemini_available": bool(os.getenv("GEMINI_API_KEY")),
        "top_k_retrieval": TOP_K,
        "api_version": "1.0.0"
    }


@app.post("/analyze-profile")
async def analyze_profile(data: VisaRequest):
    """
    Advanced profile analysis endpoint
    Returns detailed eligibility with confidence scores
    """
    query = (
        f"Analyze visa eligibility comprehensively for:\n"
        f"- Citizen of: {data.countryOfCitizenship}\n"
        f"- Destination: {data.destinationCountry}\n"
        f"- Purpose: {data.purposeOfVisit}\n"
        f"- Duration: {data.lengthOfStay} days\n"
        f"- Age: {data.age} years\n\n"
        f"Provide: 1) Eligibility status, 2) Key requirements, 3) Recommendations, 4) Next steps"
    )
    
    try:
        if USE_LLM:
            result = run_rag_with_llm(query)
            return {
                "status": "success",
                "analysis": result,
                "provider": LLM_PROVIDER,
                "profile": data.dict(),
                "timestamp": __import__("datetime").datetime.now().isoformat()
            }
        else:
            result = run_retrieval_only(query)
            return {
                "status": "success",
                "analysis": result,
                "provider": "retrieval-only",
                "profile": data.dict(),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "note": "LLM not available - using document retrieval"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "profile": data.dict()
        }


@app.get("/visa-requirements/{destination}/{visa_type}")
async def get_visa_requirements(destination: str, visa_type: str):
    """Get specific visa requirements for a destination and visa type"""
    query = f"What are the requirements for a {visa_type} visa to {destination}?"
    
    try:
        if db is None:\n            docs = fallback_search(query, 3)\n        else:\n            docs = db.similarity_search(query, k=3)
        
        if not docs:
            return {
                "status": "not_found",
                "message": f"No information found for {visa_type} visa to {destination}",
                "destination": destination,
                "visa_type": visa_type
            }
        
        requirements = []
        for doc in docs:
            requirements.append({
                "content": doc.page_content[:800],
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            })
        
        return {
            "status": "success",
            "destination": destination,
            "visa_type": visa_type,
            "requirements": requirements,
            "total_documents": len(docs)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "destination": destination,
            "visa_type": visa_type
        }


# ----------------------------------------
# HEALTH CHECK
# ----------------------------------------
@app.get("/")
async def root():
    return {"message": "✅ SwiftVisa Backend Running Perfectly!"}


# ----------------------------------------
# RUN SERVER
# ----------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Uvicorn server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
