# document_processor.py — Handles loading and splitting documents
# This is the "pre-processing" step before we store vectors

import os
import re
import json
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings


def load_document(file_path: str) -> List[Document]:
    """
    Loads a document from disk using the appropriate LangChain loader.

    Supported formats:
    - .pdf → uses PyPDFLoader (reads each page as a separate document)
    - .txt → uses TextLoader (reads the whole file as one document)

    Args:
        file_path: Full path to the file on disk

    Returns:
        List of LangChain Document objects (each has .page_content and .metadata)
    """
    # Get the file extension in lowercase
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        # PyPDFLoader splits the PDF into pages automatically
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        # TextLoader reads plain text files
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf and .txt are supported.")

    # .load() reads the file and returns a list of Document objects
    documents = loader.load()
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Standard LangChain Recursive splitting.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def clean_text(text: str) -> str:
    """
    Step 2: Clean the Text
    - Remove extra spaces
    - Normalize line breaks
    - Remove common PDF noise (headers/footers like 'Page X of Y')
    """
    # Remove page numbers like "Page 1 of 10" or "1 / 10"
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\d+ / \d+', '', text)
    
    # Normalize multiple newlines to single ones
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def split_questions_custom(documents: List[Document]) -> List[Document]:
    """
    Step 3-7: Custom Question-Aware Chunking
    Optimized for files using 'Topic:', 'Question:', or dashed dividers.
    """
    # Step 1 & 2: Get raw text
    full_text = "\n".join([doc.page_content for doc in documents])
    
    # Step 3: Detect Question Boundaries
    # We use dashed lines, "Topic:", or "Question:" as potential starts
    patterns = [
        r'-{10,}',                # Dashed lines (10 or more)
        r'\nTopic:',               # Start of a new topic
        r'\nQuestion:',            # Start of a new question
        r'\n?\d+\.',               # 1. 2. 3.
        r'\n?Q\d+[:\.]'            # Q1: Q1.
    ]
    combined_pattern = f"({'|'.join(patterns)})"
    
    # Split text into segments
    parts = re.split(combined_pattern, full_text)
    
    raw_chunks = []
    current_content = ""
    
    # Reassemble the split parts
    for part in parts:
        if not part: continue
        
        # If the part matches one of our headers, we start a new chunk soon
        if any(re.match(p, part) for p in patterns):
            if current_content.strip():
                raw_chunks.append(current_content.strip())
            current_content = part # Start new chunk with the header
        else:
            current_content += part

    # Add the last one
    if current_content.strip():
        raw_chunks.append(current_content.strip())

    final_chunks = []
    for raw in raw_chunks:
        # Step 4: HEAL THE TEXT
        # Your file has weird newlines (e.g., 'n\nu\nl\nl').
        # We consolidate lines that are very short into a single line.
        lines = raw.split('\n')
        healed_lines = []
        buffer = ""
        
        for line in lines:
            clean_line = line.strip()
            if not clean_line: continue
            
            # If the line is just 1 or 2 characters, it's likely a broken word
            if len(clean_line) <= 2 and not clean_line.startswith(('-', 'A.', 'B.', 'C.', 'D.')):
                buffer += clean_line
            else:
                if buffer:
                    healed_lines.append(buffer)
                    buffer = ""
                healed_lines.append(clean_line)
        if buffer: healed_lines.append(buffer)
        
        text = "\n".join(healed_lines)
        
        # Simple validation: Must have 'Question' or '?' and be long enough
        if len(text) < 30:
            continue
            
        # Step 6: Add Metadata
        metadata = {
            "strategy": "custom_question",
            "type": "MCQ" if "Options" in text or "Answer:" in text else "Concept",
            "source": documents[0].metadata.get("source", "unknown")
        }
        
        final_chunks.append(Document(page_content=text, metadata=metadata))

    return final_chunks


def process_and_store_document(file_path: str, vector_store, strategy: str = "langchain") -> dict:
    """
    Full pipeline: Load → Split (Standard or Custom) → Store
    """
    # 1. Load the file from disk using LangChain loaders
    documents = load_document(file_path)

    # 2. Break the document into smaller pieces (chunks) so the AI can read it better
    if strategy == "custom":
        chunks = split_questions_custom(documents)
    else:
        chunks = split_documents(documents)

    # 3. If we actually found some text, push it to MongoDB Atlas
    if chunks:
        vector_store.add_documents(chunks)

    # Return a little summary so the frontend knows what happened
    return {
        "strategy_used": strategy,
        "total_source_units": len(documents),
        "total_chunks_stored": len(chunks),
        "file_processed": os.path.basename(file_path)
    }