# Prepare-Interview-with-LangChain

## PHASE-1: Build ATS System and get feedback from GPT-4o
A smart Applicant Tracking System (ATS) that uses NLP, Cosine Similarity, and OpenAI GPT-4 to evaluate resumes against job descriptions. Features include:

- ✅ Resume parsing (PDF, DOCX, Images)
- 🎯 Skill extraction & ATS scoring
- 🧾 Cover letter generation using GPT-4
- 🤖 Chatbot for querying resume content
- 📊 Streamlit UI for interactive usage

**Tech Stack:** Python, NLP, Scikit-learn, Spacy, Streamlit, OpenAI, Tesseract OCR

## PHASE-2 : Chat with your Resume using LangChain + GPT + Chromadb 
#### LangChain Componenents
1. Prompts
2. Model
3. Indexes
4. Chains
5. Agents

#### Retrieval Augmented Generation

1. Store data in Vector Database:
Document loading -> Splitting text into chunks -> Store into Vector Store

2. Retrieval(Semantic search):
Query/Question passed to Vector Storage -> Retrieve relevant splits -> from Prompt and LLM -> generate Answer

