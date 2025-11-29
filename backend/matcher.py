# backend/matcher.py
from .embeddings import add_resumes_to_store, query
from .llm_utils import summarize_candidate, create_questions

def process_matching(resumes: list, job_description: str, top_k: int = 5):
    """
    resumes: list of dicts {file_name, text, email, phone}
    job_description: string
    returns: list of result dicts
    """
    # 1) add resumes (embeddings) to chroma
    add_resumes_to_store(resumes)

    # 2) query chroma
    q = query(job_description, n_results=top_k)

    results = []
    ids = q.get("ids", [[]])[0]
    docs = q.get("documents", [[]])[0]
    distances = q.get("distances", [[]])[0]
    metadatas = q.get("metadatas", [[]])[0] if "metadatas" in q else [None]*len(ids)

    for i, _id in enumerate(ids):
        doc_text = docs[i] if i < len(docs) else ""
        score = distances[i] if i < len(distances) else None
        metadata = metadatas[i] if i < len(metadatas) else {}

        # call LLM for summary and questions (safe try/except)
        try:
            summary = summarize_candidate(doc_text, job_description)
        except Exception as e:
            summary = f"Summary unavailable: {e}"

        try:
            questions = create_questions(doc_text, job_description)
        except Exception as e:
            questions = f"Questions unavailable: {e}"

        results.append({
            "name": _id,
            "file_name": _id,
            "score": float(score) if score is not None else None,
            "snippet": doc_text[:800],
            "summary": summary,
            "questions": questions,
            "metadata": metadata
        })
    return results
