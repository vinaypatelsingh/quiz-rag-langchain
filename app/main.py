# main.py — The main FastAPI application entry point
# This is where everything is wired together

import os
from fastapi import FastAPI
from app.api import upload, query
from app.core.config import settings

# Create the FastAPI app instance
app = FastAPI(
    title="RAG System API",
    description="Upload documents and ask questions using LangChain + ChromaDB",
    version="1.0.0"
)

# Register the routers with a URL prefix
# upload.router handles: /api/upload
# query.router handles:  /api/query
app.include_router(upload.router, prefix="/api", tags=["Upload & Train"])
app.include_router(query.router,  prefix="/api", tags=["Query"])


@app.get("/")
def root():
    """Health check endpoint — visit http://localhost:8000 to confirm it's running"""
    return {"status": "RAG System is running", "docs": "http://localhost:8000/docs"}


@app.on_event("startup")
def startup_event():
    """Runs once when the server starts — creates necessary folders"""
    os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    print("✅ RAG System started. Folders ready.")
    print(f"   ChromaDB path: {settings.CHROMA_DB_PATH}")
    print(f"   Upload folder: {settings.UPLOAD_FOLDER}")