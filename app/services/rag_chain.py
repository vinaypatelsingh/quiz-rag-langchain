# rag_chain.py — Now uses Google Gemini LLM instead of OpenAI GPT

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.core.config import settings


def get_llm(temperature=0):
    """Initializes the Gemini LLM"""
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True
    )


def get_rag_chain(vector_store):
    """
    Builds the RAG chain using Gemini 1.5 Flash as the LLM.
    """
    llm = get_llm()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K_RESULTS}
    )

    prompt_template = """You are a helpful assistant. Use ONLY the context provided below to answer the question.
If the answer is not found in the context, say "I don't have enough information to answer this question."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain


def run_query(question: str, vector_store) -> dict:
    # Set up the retrieval chain (combines search + Gemini)
    chain = get_rag_chain(vector_store)
    result = chain.invoke({"query": question})

    sources = []
    # Loop through the matching documents to show the user where the info came from
    for doc in result.get("source_documents", []):
        source_name = doc.metadata.get("source", "unknown")
        sources.append({
            "source": source_name,
            "page": doc.metadata.get("page", "N/A"),
            "content_preview": doc.page_content[:200]
        })

    return {
        "answer": result["result"],
        "sources": sources
    }


def run_quiz_generation(subject: str, num_questions: int, difficulty: str, vector_store):
    """
    Generates a structured quiz based on context from the vector store.
    Falls back to LLM's own knowledge if no relevant context is found.
    """
    # 1. Grab some context related to the requested subject
    docs = vector_store.similarity_search(subject, k=10)
    context = "\n".join([d.page_content for d in docs])

    has_context = bool(context.strip())

    # 2. Decide if we have enough local info or if Gemini should use its general brain
    if has_context:
        context_section = f"""Use the following context to generate the quiz:

Context:
{context}"""
    else:
        # Fallback: let the LLM use its own knowledge + internet awareness
        context_section = f"""No specific documents were found in the knowledge base for this subject.
Use your own knowledge and understanding of "{subject}" to generate the quiz.
Ensure the questions are accurate, relevant, and appropriate for the {difficulty} difficulty level."""

    prompt = f"""
    You are an expert educator. Generate a {difficulty} level quiz on the subject below.
    
    Subject: {subject}
    Number of questions: {num_questions}
    Difficulty: {difficulty}

    Instructions:
    1. Generate exactly {num_questions} questions.
    2. Each question must have 4 options (A, B, C, D).
    3. Identify the correct answer clearly.
    4. Provide a brief explanation for why the answer is correct.
    5. Return the response ONLY as a valid JSON object. Do not include markdown formatting or extra text.

    EXPECTED JSON STRUCTURE:
    {{
        "subject": "{subject}",
        "questions": [
            {{
                "question": "The question text here?",
                "options": ["A) Choice 1", "B) Choice 2", "C) Choice 3", "D) Choice 4"],
                "correct_answer": "A) Choice 1",
                "explanation": "Because..."
            }}
        ]
    }}

    {context_section}
    """

    # 3. Call Gemini
    llm = get_llm(temperature=0.7)
    response = llm.invoke(prompt)

    # 4. Parse the JSON output
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        quiz_data = json.loads(content)
        return quiz_data

    except json.JSONDecodeError as e:
        print(f"Error parsing quiz JSON: {e}\nRaw response: {content}")
        return {
            "subject": subject,
            "questions": [],
            "error": "Failed to parse quiz response. Please try again."
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            "subject": subject,
            "questions": [],
            "error": "An unexpected error occurred. Please try again."
        }


def generate_quiz_feedback(score: int, answers: list, explanations_used_count: int) -> str:
    """
    Uses Gemini to analyze quiz performance and generate short,
    actionable feedback highlighting weak areas for improvement.
    """
    total_questions = len(answers)

    # Build a readable summary of each answer for the LLM
    answer_details = ""
    for i, ans in enumerate(answers, 1):
        status = "Correct" if ans["selected"] == ans["correct"] else "Wrong"
        answer_details += (
            f"Q{i}: {ans['question']}\n"
            f"   Selected: {ans['selected']}\n"
            f"   Correct:  {ans['correct']}\n"
            f"   Status:   {status}\n"
            f"   Viewed Explanation: {'Yes' if ans.get('explanationViewed') else 'No'}\n\n"
        )

    prompt = f"""You are an expert educational coach. A student just completed a quiz assessment.
Analyze their performance and provide SHORT structured, point-wise feedback.

Performance Summary:
- Score: {score} out of {total_questions}
- Explanation hints used: {explanations_used_count} out of {total_questions}

Detailed Answers:
{answer_details}

Instructions — follow this EXACT format in your response:

1. Start with a one-line score summary (e.g. "Score: 3/5 — Good effort!").

2. Then give POINT-WISE, TOPIC-WISE analysis. For each question, write one point like:
   • [Topic Name]: Correct ✅ — brief praise  OR
   • [Topic Name]: Incorrect ❌ — what went wrong and what to review

3. After the per-question points, add a section titled "Weak Areas to Improve:" listing ONLY the topics the student got wrong, with a short suggestion for each, like:
   • Astronomy — Review planetary sizes and the solar system hierarchy.
   • Biology — Revisit cell division processes.

4. End with one encouraging closing sentence.

5. If the student used explanation hints, mention it briefly (more hints = less confidence in that area).

IMPORTANT: Return ONLY plain text with bullet points (•). No JSON, no markdown headers, no code blocks."""

    llm = get_llm(temperature=0.7)
    response = llm.invoke(prompt)

    return response.content.strip()