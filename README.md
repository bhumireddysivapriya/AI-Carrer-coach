# Traditional rag
This project is an end-to-end **traditional RAG application** 
It uses
- Langchain
- Groq chat model
- HuggingFace Embeddings
- streamlit UI
- Resume + Job Description analysis

## Features
- Upload Resume as '.txt', '.pdf', or '.docx'

- upload Job Description as '.txt', '.pdf', or '.docx'
- Build a RAG index from both documents
- Retrieve relevent context from Resume and JD
- Generate:
- Resume match summary
- Skill Gap analysis
- Resume improvement suggestions
- project recommendations
- Interview preparation questions

## RAG stages covered
1. Document LOading
2. Chunking
3. Embeddings
4. Vector Database storages
5. QUery Embedding
6. context Retrieval
7. LLM Answer Generation
## setup
### 1. Create virtual environment
'''bash
python -m venv venv
'''
### 2. Activate environment
windows Powershell:
'''bash
venv\Scripts\activate
'''
### 3. Install requirements
''' bash
pip install -r requirements.txt
'''
### 4. Add Groq API key
Create a '.env' file:
''' env
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
''' 
### 5. Run app
''' bash
streamlit run app.py
'''
