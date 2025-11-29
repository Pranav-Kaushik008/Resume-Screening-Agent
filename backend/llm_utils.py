# backend/llm_utils.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")  # change if needed
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "Resume Screening Agent",
    "Content-Type": "application/json"
}

def _call_openrouter(prompt: str, system: str = "You are a helpful assistant.", max_tokens: int = 400):
    """Post a chat completion to OpenRouter and return text. Basic retry + error handling."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment")

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }

    resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
    if resp.status_code != 200:
        # bubble up a helpful message
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")

    rj = resp.json()
    # try common response shapes
    try:
        # OpenAI-style: rj['choices'][0]['message']['content']
        return rj["choices"][0]["message"]["content"]
    except Exception:
        try:
            # sometimes rj has .get('output_text') or .get('output')
            if "output_text" in rj:
                return rj["output_text"]
            if "output" in rj:
                return rj["output"]
        except Exception:
            pass
    # fallback: return full json as str
    return str(rj)

def summarize_candidate(resume_text: str, job_description: str):
    prompt = (
        "You are a concise recruiter assistant. Given the resume excerpt and the job description, produce:\n"
        "1) a 2-3 sentence summary of the candidate highlighting top skills and fit.\n"
        "2) a fit score out of 100 with a one-line rationale.\n\n"
        f"Resume excerpt:\n{resume_text[:1200]}\n\nJob description:\n{job_description[:1200]}\n\n"
        "Return the summary and score in plain text."
    )
    return _call_openrouter(prompt, max_tokens=250)

def create_questions(resume_text: str, job_description: str, n: int = 5):
    prompt = (
        f"Given the resume excerpt and job description, generate {n} targeted interview questions to evaluate the candidate "
        "for the role. For each question add a short note 'What to listen for'.\n\n"
        f"Resume excerpt:\n{resume_text[:1200]}\n\nJob description:\n{job_description[:1200]}\n\n"
        "Return as numbered items with a short 'what to listen for' note."
    )
    return _call_openrouter(prompt, max_tokens=400)
