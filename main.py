import os
import requests
import pypdf
import io
import time
import gc

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

print("INFO: Python script starting...")

# Limits for free-tier
MAX_PDF_SIZE = 1024 * 1024      # 1MB
MAX_PDF_PAGES = 10              # First 10 pages only
CHUNK_SIZE = 5                  # merge 5 paragraphs per chunk
MAX_CHUNKS = 8                  # max 8 chunks considered
MAX_QUESTIONS = 1               # only one question per call

class HackathonRequest(BaseModel):
    documents: str   # URL to PDF
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

app = FastAPI(title="Render Free Tier Tiny RAG API")

def extract_text_from_pdf_url(pdf_url: str) -> str:
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        if len(response.content) > MAX_PDF_SIZE:
            raise HTTPException(status_code=413, detail=f"PDF too large (> {MAX_PDF_SIZE//1024} KB)")
        with io.BytesIO(response.content) as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            pages = reader.pages[:MAX_PDF_PAGES]
            full_text = "".join(page.extract_text() or "" for page in pages)
        return full_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download or read PDF: {e}")

def chunk_text(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = ["\n\n".join(paragraphs[i:i+CHUNK_SIZE]) for i in range(0, len(paragraphs), CHUNK_SIZE)]
    return chunks[:MAX_CHUNKS]

def find_most_similar_chunk(chunks: List[str], question: str) -> str:
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    chunk_vecs = vectorizer.transform(chunks)
    question_vec = vectorizer.transform([question])
    sims = (chunk_vecs * question_vec.T).toarray().flatten()
    best_idx = int(np.argmax(sims))
    return chunks[best_idx] if sims[best_idx] > 0 else ""

def simple_qa(context: str, question: str) -> str:
    # Very simple QA: if context contains key question words, return that snippet, else fallback
    # Token match heuristic
    key_words = [w.strip(',.?!";:').lower() for w in question.split() if len(w) > 3]
    sentences = context.split('.')
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if all(word in sentence_lower for word in key_words):
            return sentence.strip()
    # If no good match, fallback
    if context.strip():
        return context.strip().split('\n')[0]  # first line
    return "The answer is not found in the provided context."

@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_hackathon_request(request_body: HackathonRequest) -> HackathonResponse:
    questions = request_body.questions
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Too many questions! Limit is {MAX_QUESTIONS} per request.")

    full_text = extract_text_from_pdf_url(request_body.documents)
    chunks = chunk_text(full_text)
    if not chunks:
        raise HTTPException(status_code=500, detail="No text extracted from PDF.")

    answers = []
    for question in questions:
        context = find_most_similar_chunk(chunks, question)
        if not context:
            answers.append("The answer is not found in the provided context.")
            continue
        answer = simple_qa(context, question)
        answers.append(answer)

    # Cleanup to reduce RAM pressure
    del full_text, chunks
    gc.collect()

    return HackathonResponse(answers=answers)

@app.get("/")
def root():
    return {"status": "Render Free Tier Tiny RAG API is running!"}
