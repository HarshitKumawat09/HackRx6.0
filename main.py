import os
import requests
import pypdf
import io
import time
import gc

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

print("INFO: Python script starting...")

# Config: use smallest fast open-source model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

print("INFO: Loading embedding model ...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("INFO: Embedding model loaded.")

MAX_PDF_SIZE = 1024 * 1024         # Max 1MB
MAX_PDF_PAGES = 10                 # Max 10 pages
CHUNK_SIZE = 5                     # Merge 5 paragraphs per chunk
MAX_CHUNKS = 8                     # Max 8 chunks processed
MAX_QUESTIONS = 1                  # Only 1 question per call

class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

app = FastAPI(title="Tiny RAG Bot (Render-Optimized)")

def fake_llm(context: str, question: str) -> str:
    "Simulated QA: picks the chunk containing the most tokens from question."
    qwords = [w.strip(',.?!"\'') for w in question.lower().split() if len(w) > 3]
    for para in context.split('\n\n'):
        if all(word in para.lower() for word in qwords):
            return para.strip()
    # If nothing matches, just return the first para as fallback
    return context.strip().split('\n\n')[0] if context.strip() else 'The answer is not found in the provided context.'

def answer_questions_from_document(document_url: str, questions: List[str]) -> List[str]:
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Only {MAX_QUESTIONS} question per request.")

    try:
        response = requests.get(document_url, timeout=10)
        response.raise_for_status()
        if len(response.content) > MAX_PDF_SIZE:
            raise HTTPException(status_code=413, detail=f"PDF file too large (>{MAX_PDF_SIZE//1024}KB).")
        with io.BytesIO(response.content) as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            pages = reader.pages[:MAX_PDF_PAGES]
            full_text = "".join(page.extract_text() or "" for page in pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not process PDF: {e}")

    # Chunking
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    chunks = ["\n\n".join(paragraphs[i:i+CHUNK_SIZE]) for i in range(0, len(paragraphs), CHUNK_SIZE)]
    chunks = chunks[:MAX_CHUNKS]
    if not chunks:
        raise HTTPException(status_code=500, detail="No valid text extracted from PDF.")

    # Embeddings
    chunk_vecs = embedding_model.encode(chunks)
    q_vecs = embedding_model.encode(questions)

    # Simple max-similarity match (no chromaDB)
    answers = []
    for qi, qv in enumerate(q_vecs):
        sims = np.dot(chunk_vecs, qv)
        top_idx = int(np.argmax(sims))
        # optional: context = top N chunks for fuzzier QA
        relevant_context = chunks[top_idx]
        answer = fake_llm(relevant_context, questions[qi])
        answers.append(answer)

    # Free up RAM
    del response, full_text, paragraphs, chunks, chunk_vecs, q_vecs
    gc.collect()

    return answers

@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_hackathon_request(request_body: HackathonRequest) -> HackathonResponse:
    answers = answer_questions_from_document(
        document_url=request_body.documents,
        questions=request_body.questions
    )
    return HackathonResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Render Tiny RAG Demo is running!"}
