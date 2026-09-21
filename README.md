# Work Visa Consultancy RAG
This project is an end-to-end **work visa consultancy RAG application**.
It uses
- Langchain
- Groq chat model
- HuggingFace Embeddings
- streamlit UI
- Persistent work visa reference knowledge base

## Features

- Admin-only upload of visa policies, checklists, and country guides as '.txt', '.pdf', or '.docx'
- Persistent backend Chroma index for uploaded reference documents
- User-facing chat that answers only from the indexed documents
- Grounded reference accuracy score for each answer

## RAG stages covered
1. Document LOading
2. Chunking
3. Embeddings
4. Vector Database storages
5. QUery Embedding
6. context Retrieval
7. LLM Answer Generation in the chat
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
### 4. Add API and admin settings
Create a '.env' file:
''' env
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
ADMIN_PASSWORD=choose-a-strong-admin-password
''' 
The `ADMIN_PASSWORD` protects the backend document uploader. End users only see the chat.

### 5. Run app
''' bash
streamlit run app.py
'''
