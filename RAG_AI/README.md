# RAG Chatbot — Intelligence Artificielle

Chatbot RAG (Retrieval-Augmented Generation) alimenté par Wikipedia et vos PDFs.

## Stack
- **Backend** : FastAPI + sentence-transformers + ChromaDB + PyMuPDF
- **LLM** : Phi-3 via Ollama (local)
- **Frontend** : HTML/CSS/JS pur

## Installation

```bash
pip install -r backend/requirements.txt
ollama pull phi3
```

## Lancement

```bash
# Terminal 1
ollama run phi3

# Terminal 2
cd backend
uvicorn main:app --reload

# Ouvrir frontend/index.html dans le navigateur
```

## Architecture

1. **Indexation** : Wikipedia + PDFs découpés en chunks de 500 caractères
2. **Embeddings** : `all-MiniLM-L6-v2` convertit chaque chunk en vecteur
3. **Retrieval** : recherche sémantique dans ChromaDB (top 5)
4. **Generation** : Phi-3 génère la réponse en français