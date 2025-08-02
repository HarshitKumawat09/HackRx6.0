import os
import requests
import io
import time
import gc

import fitz  # PyMuPDF (very fast PDF extraction)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

print("INFO: Python script starting...")

# ---- TUNABLE LIMITS ----
MAX_PDF_SIZE = 2 * 1024 * 1024  # 2MB (for safety; you can adjust)
MAX_PDF_PAGES = 12              # First 12 pages; adjust if needed
CHUNK_SIZE = 1024               # Characters per chunk (fixed for speed)
MAX_CHUNKS = 10                 # Up to 10 chunks per document
MAX_QUESTIONS = 1               # Only 1 question per call

class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

app = FastAPI(title="Render Fast RAG (PyMuPDF + TF-IDF)")

def extract_text_from_pdf_url(pdf_url: str) -> str:
    t0 = time.time()
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        if len(response.content) > MAX_PDF_SIZE:
            raise HTTPException(status_code=413, detail=f"PDF too large (> {MAX_PDF_SIZE//1024} KB)")
        with io.BytesIO(response.content) as pdf_file:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            # Read only up to MAX_PDF_PAGES
            texts = []
            for page in doc.pages(0, min(len(doc), MAX_PDF_PAGES)):
                texts.append(page.get_text())
            full_text = "\n\n".join(texts)
        print(f"PDF extracted in {time.time()-t0:.2f}s")
        return full_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download or read PDF: {e}")

def chunk_text(text: str) -> List[str]:
    # Fixed-size chunking by chars for fast, even splitting
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    return chunks[:MAX_CHUNKS]

def find_most_similar_chunk(chunks: List[str], question: str) -> str:
    if not chunks:
        return ""
    all_text = chunks + [question]
    vectorizer = TfidfVectorizer().fit(all_text)
    tfidf_matrix = vectorizer.transform(all_text)
    chunk_vecs = tfidf_matrix[:-1]
    question_vec = tfidf_matrix[-1]
    sims = (chunk_vecs * question_vec.T).toarray().flatten()
    best_idx = int(np.argmax(sims))
    return chunks[best_idx] if sims[best_idx] > 0.05 else ""

def simple_qa(context: str, question: str) -> str:
    # Simple QA: pick the sentence in the context with most keyword overlap, else fallback
    key_words = [w.strip(',.?!";:').lower() for w in question.split() if len(w) > 3]
    sentences = context.replace('\n', ' ').split('.')
    best_score, best_sent = -1, ""
    for sent in sentences:
        score = sum(1 for word in key_words if word in sent.lower())
        if score > best_score and score > 0:
            best_score, best_sent = score, sent.strip()
    if best_sent:
        return best_sent
    return "The answer is not found in the provided context."

@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_hackathon_request(request_body: HackathonRequest) -> HackathonResponse:
    questions = request_body.questions
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Too many questions! Limit: {MAX_QUESTIONS} per request.")

    t0 = time.time()
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

    # Free memory right after processing
    del full_text, chunks
    gc.collect()

    print(f"Total RAG time: {time.time()-t0:.2f}s")
    return HackathonResponse(answers=answers)

@app.get("/")
def root():
    return {"status": "Render Fast RAG API is running!"}

