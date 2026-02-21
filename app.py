import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from ingest import extract_text_by_page, chunk_by_sections
from embed import build_vectorstore
from generate import generate_answer

# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="DRHP RAG API")

# ── request/response models ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int

class UploadResponse(BaseModel):
    message: str
    pages: int
    chunks: int

# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract text, chunk by sections, build ChromaDB vectorstore.
    Accepts any PDF — replaces existing vectorstore if one exists.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # ingest → chunk → embed
        pages  = extract_text_by_page(tmp_path)
        chunks = chunk_by_sections(pages)
        build_vectorstore(chunks)
        return UploadResponse(
            message="PDF processed and vectorstore built successfully.",
            pages=len(pages),
            chunks=len(chunks),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Take a question, retrieve top 5 chunks from ChromaDB, return a grounded
    answer with section + page citations via Claude (OpenRouter).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = generate_answer(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=request.question,
        answer=answer,
        chunks_used=5,
    )


# ── health check ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "endpoints": ["POST /upload", "POST /query"]}
