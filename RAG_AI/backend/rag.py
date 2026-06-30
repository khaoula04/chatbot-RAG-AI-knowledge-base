import requests
from sentence_transformers import SentenceTransformer
import chromadb
import fitz  # PyMuPDF

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection("rag_collection")

WIKI_TITLES = [
    "Artificial intelligence",
    "History of artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network"
]

def fetch_wiki(title):
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "RAG-Chatbot/1.0"}
    params = {
        "action": "query", "prop": "extracts",
        "explaintext": True, "titles": title, "format": "json"
    }
    r = requests.get(url, params=params, headers=headers)
    pages = r.json()["query"]["pages"]
    return list(pages.values())[0].get("extract", "")[:8000]

def chunk_text(text, size=500):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > size and current:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks

def add_to_collection(chunks, source="wikipedia"):
    existing = collection.count()
    embeddings = model.encode(chunks).tolist()
    ids = [f"{source}_{existing + i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids,
                   metadatas=[{"source": source}] * len(chunks))

def load_wikipedia():
    print("Chargement Wikipedia...")
    for title in WIKI_TITLES:
        text = fetch_wiki(title)
        chunks = chunk_text(text)
        add_to_collection(chunks, source=f"wiki_{title[:20]}")
        print(f"  ✅ {title} — {len(chunks)} chunks")
    print(f"Total : {collection.count()} chunks indexés")

def load_pdf(file_bytes, filename):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    chunks = chunk_text(text)
    add_to_collection(chunks, source=f"pdf_{filename[:20]}")
    return len(chunks)

def retrieve(query, top_k=5):
    embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    return results["documents"][0]

def generate(query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question based solely on the provided context.
If the information is not in the context, say so honestly.
Answer in the same language as the question.

Context:
{context}

Question: {query}
Answer:"""

    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "phi3",
        "prompt": prompt,
        "stream": False
    })
    return r.json()["response"]

def ask(query):
    chunks = retrieve(query)
    return generate(query, chunks)