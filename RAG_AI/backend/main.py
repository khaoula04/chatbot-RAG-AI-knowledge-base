from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rag

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

@app.on_event("startup")
async def startup():
    rag.load_wikipedia()

@app.post("/ask")
async def ask(q: Question):
    answer = rag.ask(q.query)
    return {"answer": answer}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    n = rag.load_pdf(contents, file.filename)
    return {"message": f"{file.filename} indexé — {n} chunks ajoutés"}

@app.get("/stats")
async def stats():
    return {"total_chunks": rag.collection.count()}