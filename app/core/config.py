import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Google Gemini API key
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Path where ChromaDB stores its data on disk
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    # Folder where uploaded documents are saved temporarily
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "./uploaded_docs")

    # ChromaDB collection name (kept for compatibility)
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")

    # MongoDB Atlas Settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "rag_db")
    MONGODB_COLLECTION_NAME: str = os.getenv("MONGODB_COLLECTION_NAME", "documents")
    ATLAS_VECTOR_SEARCH_INDEX_NAME: str = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index")

    # Text chunk size
    CHUNK_SIZE: int = 1000

    # Overlap between chunks
    CHUNK_OVERLAP: int = 200

    # How many top matching chunks to retrieve
    TOP_K_RESULTS: int = 4

    # New collection for quizzes
    QUIZ_COLLECTION_NAME: str = os.getenv("QUIZ_COLLECTION_NAME", "quiz_documents")


settings = Settings()