# vector_store.py — Fixed model name for current Gemini API

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import certifi
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings


def get_embeddings():
    """
    HUMAN EXPLANATION: This is for STORING documents.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GOOGLE_API_KEY,
        task_type="retrieval_document"
    )


def get_query_embeddings():
    """
    HUMAN EXPLANATION: This is for SEARCHING.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GOOGLE_API_KEY,
        task_type="retrieval_query"
    )


def get_vector_store(collection_name: str = None, is_query: bool = False):
    """
    Flexible helper to get the MongoDB Atlas vector store.
    """
    if not settings.MONGODB_URI:
        raise ValueError("MONGODB_URI is not set in the environment variables.")

    if collection_name is None:
        collection_name = settings.MONGODB_COLLECTION_NAME

    embeddings = get_query_embeddings() if is_query else get_embeddings()

    # Initialize MongoDB Client with SSL/TLS verification
    client = MongoClient(
        settings.MONGODB_URI,
        tls=True,
        tlsCAFile=certifi.where()
    )
    db_name = settings.MONGODB_DB_NAME
    collection = client[db_name][collection_name]
    
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=settings.ATLAS_VECTOR_SEARCH_INDEX_NAME
    )


# --- HELPERS ---

def get_query_vector_store():
    # Use the main MongoDB collection for searching
    return get_vector_store(collection_name=settings.MONGODB_COLLECTION_NAME, is_query=True)


def get_quiz_vector_store():
    # Use a separate collection for quizzes
    return get_vector_store(collection_name="quiz_vectors", is_query=True)
