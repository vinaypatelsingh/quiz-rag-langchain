import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.config import settings
from app.core.vector_store import get_vector_store
from app.services.document_processor import process_and_store_document

# APIRouter is like a mini FastAPI app — we group related routes here
router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    target: str = Form("rag"), # Default to "rag", can be "quiz"
    strategy: str = Form("langchain") # "langchain" or "custom"
):
    # First, make sure the file type is something we can actually read
    allowed_extensions = [".pdf", ".txt"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_extension}' not supported. Use .pdf or .txt"
        )

    # We need to save the file locally first before processing it
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Now the main part: split the text, embed it, and push to MongoDB
    try:
        # Pick the right collection based on what this file is for
        collection_name = "quiz_vectors" if target == "quiz" else settings.MONGODB_COLLECTION_NAME
        
        # Connect to our vector database
        vector_store = get_vector_store(collection_name=collection_name, is_query=False)
        
        # Run the heavy lifting pipeline
        result = process_and_store_document(file_path, vector_store, strategy=strategy)

        return {
            "status": "success",
            "message": f"Document '{file.filename}' processed successfully.",
            "details": result
        }

    except Exception as e:
        # If anything breaks, tell the user why
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
