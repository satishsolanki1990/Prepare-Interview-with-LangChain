# Prepare-Interview-with-LangChain

## PHASE-1 : Build ATS System and get feedback from GPT-4o
- Basic UI that can handle uploading the resume in pdf or docs form
- Take Job descrition as text input
- Compare the resume against Job description provided
- Provide Resume Match score
- Provide feedback response from gpt-4o model 

## PHASE-2 : Chat with your Resume using LangChain + GPT + Chromadb 
#### LangChain Componenents
1. Prompts
2. Model
3. Indexes
4. Chains
5. Agents

#### Retrieval Augmented Generation

1. Store data to Vector Database:
Document loading -> Splitting text into Chuncks -> Store into Vector Store

2. Retrieval(Semantic search):
Query/Question passed to Vector Storage -> Retrive relevant splits -> from Prompt and LLM -> generate Answer

