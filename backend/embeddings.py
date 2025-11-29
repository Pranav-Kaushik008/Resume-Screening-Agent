# backend/embeddings.py
import os
import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# sentence-transformers
from sentence_transformers import SentenceTransformer

# chroma
import chromadb
from chromadb.config import System  # not used directly, just import to ensure version is present

# configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data")
CACHE_DIR = Path("cache_embeddings")
CACHE_DIR.mkdir(exist_ok=True)

# load the local embedding model
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# initialize chroma persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name="resumes", metadata={"hnsw:space": "cosine"})

def _get_cache_path(text: str) -> Path:
    safe = str(abs(hash(text)))
    return CACHE_DIR / f"{safe}.json"

def embed_text(text: str):
    """Return numpy array embedding for text, saving to cache if possible."""
    cache_path = _get_cache_path(text)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                arr = json.load(f)
            return np.array(arr, dtype="float32")
        except Exception:
            pass

    emb = embed_model.encode(text, show_progress_bar=False)
    emb = np.array(emb, dtype="float32")
    # save
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(emb.tolist(), f)
    return emb

def add_resumes_to_store(resumes: list):
    """resumes: list of dicts with keys 'file_name', 'text', 'email', 'phone'"""
    # collect ids, embeddings, metadatas, documents
    ids, embs, metadatas, docs = [], [], [], []
    for r in resumes:
        text = r["text"][:2000]  # limit to reasonable chunk for embedding
        emb = embed_text(text).tolist()
        ids.append(r["file_name"])
        embs.append(emb)
        metadatas.append({"email": r.get("email"), "phone": r.get("phone")})
        docs.append(text)
    # add to chroma (duplicate ids handled by chroma; if already exists, it will append — ensure unique ids)
    try:
        collection.add(ids=ids, embeddings=embs, metadatas=metadatas, documents=docs)
    except Exception:
        # if items already exist with same ids, replace them: remove then add
        for _id in ids:
            try:
                collection.delete(ids=[_id])
            except Exception:
                pass
        collection.add(ids=ids, embeddings=embs, metadatas=metadatas, documents=docs)

def query(job_description: str, n_results: int = 5):
    """Return query results from chroma for a job description."""
    q_emb = embed_text(job_description).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=n_results)
    # results is dict with keys: ids, distances, documents, metadatas
    return results
