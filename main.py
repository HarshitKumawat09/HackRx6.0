import os
import requests
import pypdf
import io
import google.generativeai as genai
import chromadb
import time
import gc

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

print("INFO: Python script starting...")

# --- Configuration ---
try:
    print("INFO: Configuring Generative AI...")
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    print("INFO: Generative AI configured successfully.")
except KeyError:
    print("FATAL: GOOGLE_API_KEY environment variable not set.")
    exit(1)

# --- Load Models (load once on startup) ---
print("INFO: Loading embedding model (this may take a while)...")
embedding_model = SentenceTransformer('paraphrase-albert-small-v2')
print("INFO: Embedding model loaded successfully.")

print("INFO: Loading LLM...")
llm = genai.GenerativeModel('gemini-1.5-flash-latest')
print("INFO: LLM loaded successfully.")

# --- Configurable Limits for low-memory environment ---
MAX_PDF_SIZE = 2 * 1024 * 1024  # 2 MB max PDF size
MAX_PDF_PAGES = 30              # max 30 pages per PDF
CHUNK_PARAGRAPHS = 3            # paragraphs per chunk (larger chunk size)
MAX_CHUNKS = 20                 # max number of chunks to process
MAX_QUESTIONS = 2               # max questions per API call

# --- Pydantic Models ---
class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

# --- FastAPI App ---
app = FastAPI(
    title="HackRx 6.0 Intelligent Bot",
    description="An advanced RAG API to answer questions from documents."
)

# --- Core RAG Logic ---
def answer_questions_from_document(document_url: str, questions: List[str]) -> List[str]:
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Too many questions. Max allowed is {MAX_QUESTIONS} per request.")

    print("INFO: Request received. Starting RAG process...")
    start_time = time.time()

    # 1. Download and limit PDF size
    try:
        print("INFO: [Step 1/5] Downloading document...")
        response = requests.get(document_url, timeout=30)
        response.raise_for_status()
        if len(response.content) > MAX_PDF_SIZE:
            raise HTTPException(status_code=413, detail=f"PDF file too large (>{MAX_PDF_SIZE // (1024*1024)}MB). Please use a smaller file.")
        
        with io.BytesIO(response.content) as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            num_pages_to_read = min(len(reader.pages), MAX_PDF_PAGES)
            full_document_text = "".join(page.extract_text() or "" for page in reader.pages[:num_pages_to_read])
        
        print(f"INFO: [Step 1/5] Document downloaded and text extracted (up to {num_pages_to_read} pages) in {time.time() - start_time:.2f} seconds.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    # 2. Chunk the Text — larger chunks, limit total chunks
    print("INFO: [Step 2/5] Chunking text...")
    raw_paragraphs = [p.strip() for p in full_document_text.split("\n\n") if p.strip()]
    # Merge paragraphs to form larger chunks
    text_chunks = ["\n\n".join(raw_paragraphs[i:i+CHUNK_PARAGRAPHS]) for i in range(0, len(raw_paragraphs), CHUNK_PARAGRAPHS)]
    # Limit the number of chunks
    text_chunks = text_chunks[:MAX_CHUNKS]  

    if not text_chunks:
        raise HTTPException(status_code=500, detail="Could not extract text chunks from document.")
    print(f"INFO: [Step 2/5] Text split into {len(text_chunks)} chunks.")

    # 3. Create Vector Index in Memory (using ChromaDB)
    print("INFO: [Step 3/5] Creating embeddings for chunks (this can be slow)...")
    chunk_embeddings = embedding_model.encode(text_chunks).tolist()

    client = chromadb.Client()
    # Clear existing collection if any to avoid memory issues
    try:
        client.delete_collection(name="document_chunks")
    except Exception:
        pass

    collection = client.create_collection(name="document_chunks")
    collection.add(
        embeddings=chunk_embeddings,
        documents=text_chunks,
        ids=[f"chunk_{i}" for i in range(len(text_chunks))]
    )
    print(f"INFO: [Step 3/5] Vector index created in {time.time() - start_time:.2f} seconds.")

    # 4. Process Each Question (batch all questions to embeddings)
    final_answers = []
    print("INFO: [Step 4/5] Creating embeddings for all questions (batch processing)...")
    question_embeddings = embedding_model.encode(questions).tolist()
    print("INFO: [Step 4/5] Question embeddings created. Processing each question...")

    for i, question in enumerate(questions):
        print(f"INFO: - Processing question {i+1}/{len(questions)}: '{question[:40]}...'")

        query_emb = [question_embeddings[i]]

        results = collection.query(
            query_embeddings=query_emb,
            n_results=3
        )
        relevant_context = "\n\n".join(results['documents'][0])

        prompt = f"""Based only on the following context, please answer the question. Do not use any external knowledge. If the answer is not in the context, say "The answer is not found in the provided context."

--- CONTEXT ---
{relevant_context}
--- END OF CONTEXT ---

QUESTION: {question}

ANSWER:
"""
        print("INFO:   - Sending context to LLM...")
        try:
            response = llm.generate_content(prompt)
            final_answers.append(response.text.strip())
            print(f"INFO:   - Answer received in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"LLM Error: {e}")
            final_answers.append("Error processing the question with the language model.")

    # Aggressively free memory after request
    del reader, full_document_text, chunk_embeddings, collection, client
    gc.collect()

    print(f"INFO: RAG process complete in {time.time() - start_time:.2f} seconds.")
    return final_answers

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_hackathon_request(request_body: HackathonRequest) -> HackathonResponse:
    answers = answer_questions_from_document(
        document_url=request_body.documents,
        questions=request_body.questions
    )
    return HackathonResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "Intelligent RAG API is running!"}
