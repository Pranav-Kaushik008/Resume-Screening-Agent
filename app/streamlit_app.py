# app/streamlit_app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from backend.parse_resume import parse_resume
from backend.matcher import process_matching

st.set_page_config(page_title="Resume Screening Agent", layout="wide")
st.title("Resume Screening Agent (OpenRouter + Local Embeddings)")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top K candidates to show", min_value=1, max_value=50, value=5)

st.header("Job Description")
job_description = st.text_area("Paste job description here")

st.header("Upload Resumes (PDF / DOCX)")
uploaded_files = st.file_uploader("Choose resume files", accept_multiple_files=True, type=['pdf','docx'])

if st.button("Run Screening"):
    if not job_description:
        st.warning("Please paste the job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner("Parsing resumes..."):
            resumes = []
            for f in uploaded_files:
                try:
                    parsed = parse_resume(f.name, f.read())
                    resumes.append(parsed)
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")

        with st.spinner("Indexing and ranking... (first run may be a bit slow)"):
            results = process_matching(resumes, job_description, top_k)

        st.success(f"Found top {len(results)} candidates")
        for i, r in enumerate(results):
            st.subheader(f"{i+1}. {r.get('name')} — score: {r.get('score')}")
            st.markdown("**Contact:**")
            meta = r.get("metadata", {})
            st.write("Email:", meta.get("email") or "N/A", " | Phone:", meta.get("phone") or "N/A")
            st.markdown("**Snippet:**")
            st.code(r.get("snippet", "")[:800])
            st.markdown("**Summary (LLM):**")
            st.write(r.get("summary"))
            st.markdown("**Interview Questions (LLM):**")
            st.write(r.get("questions"))
            st.markdown("---")
