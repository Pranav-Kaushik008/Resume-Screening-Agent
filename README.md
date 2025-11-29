  AI Resume Screening Agent

An AI-powered Resume Screening Agent that automatically evaluates multiple resumes against a job description, ranks candidates using semantic similarity, and generates candidate summaries and interview questions using OpenRouter AI models.


  What the Agent Does

- Upload multiple resumes (PDF / DOCX)
- Parse and extract text, email, phone
- Generate embeddings and store them in a vector database
- Match resumes against job descriptions using semantic similarity
- Rank candidates based on relevance
- Generate candidate summaries and interview questions using LLM
- Display results in an interactive and user-friendly Streamlit UI


  Features

-  Multi-resume upload
-  AI-based scoring & ranking
-  Automated summary generation
-  Interview question generation
-  Persistent Chroma vector database
-  Local embeddings (no cost)
-  Deployable to Streamlit Cloud
-  Modern clean UI


  Tools, APIs & Models Used

 Category                Tools / Models   

 Frontend          :     Streamlit 
 Vector DB         :     ChromaDB (PersistentClient) 
 Embeddings Model  :     `all-MiniLM-L6-v2` (Sentence Transformers) 
 LLM API           :     OpenRouter 
 Example Models    :     `gpt-4o-mini`, `deepseek-chat` 
 Parsing           :     pdfminer.six, python-docx 
 ML libs           :     numpy, torch, sentence-transformers 


  Installation / Setup

 Clone repository
bash:
git clone https://github.com/YOUR-USERNAME/resume-screening-agent.git
cd resume-screening-agent

Create virtual environment
bash:
     python -m venv venv
     venv\Scripts\activate   # Windows

Install dependencies
bash:
     pip install --upgrade pip
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     pip install -r requirements.txt

Create .env file
     OPENROUTER_API_KEY=your_key_here
     OPENROUTER_MODEL=gpt-4o-mini
     EMBEDDING_MODEL=all-MiniLM-L6-v2
     CHROMA_PERSIST_DIR=data

Running the App Locally
streamlit run app/streamlit_app.py
