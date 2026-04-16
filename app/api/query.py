# query.py — Updated to use get_query_vector_store for better search accuracy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.core.vector_store import get_query_vector_store , get_quiz_vector_store
from app.services.rag_chain import run_query, run_quiz_generation, generate_quiz_feedback

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    # Basic check to avoid empty strings
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Grab the database connection (optimized for search)
        vector_store = get_query_vector_store()

        # Let the RAG chain find the answer using the retrieved context
        result = run_query(
            question=request.question,
            vector_store=vector_store
        )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        # Return a generic error if something crashes
        raise HTTPException(status_code=500, detail=f"Error running query: {str(e)}")


# --- QUIZ GENERATION MODELS ---
class QuizRequest(BaseModel):
    subject: str
    num_questions: int = 5
    difficulty: str  # "easy", "medium", "hard"


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    explanation: str


class QuizResponse(BaseModel):
    subject: str
    questions: list[QuizQuestion]


# --- QUIZ GENERATION ENDPOINT ---
@router.post("/quiz", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    try:
        # Connect to the quiz-specific vector collection
        vector_store = get_quiz_vector_store()
        
        # Ask Gemini to build a structured quiz based on the dataset
        quiz_data = run_quiz_generation(
            subject=request.subject,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            vector_store=vector_store
        )
        
        return quiz_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")


# --- QUIZ FEEDBACK MODELS ---
class AnswerDetail(BaseModel):
    question: str
    selected: str
    correct: str
    explanationViewed: bool = False


class FeedbackData(BaseModel):
    assessmentId: str
    score: int
    answers: list[AnswerDetail]
    explanationsUsedCount: int = 0


class FeedbackRequest(BaseModel):
    data: FeedbackData


class FeedbackResponse(BaseModel):
    text: str


# --- QUIZ FEEDBACK ENDPOINT ---
@router.post("/quiz/feedback", response_model=FeedbackResponse)
async def quiz_feedback(request: FeedbackRequest):
    try:
        # Send the score and answers to Gemini for a personalized analysis
        feedback_text = generate_quiz_feedback(
            score=request.data.score,
            answers=[ans.model_dump() for ans in request.data.answers],
            explanations_used_count=request.data.explanationsUsedCount
        )

        return FeedbackResponse(text=feedback_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")
