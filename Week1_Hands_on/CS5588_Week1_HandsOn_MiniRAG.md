# 🧪 CS 5588 — Week 1: Hands-On Lab
## Embeddings → Retrieval → Generation (Mini-RAG Pipeline)

## Visual Flow
Raw Data / Documents (PDFs, Notes, Web Pages)  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
Embedding Model (Sentence Transformer / CLIP / Instructor-XL)  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
Vector Store / Index (FAISS / Chroma / Generic Vector DB)  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
Retrieved Context (Top-k Semantic Matches)  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
LLM (Reasoning + Prompt Augmentation)  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
Final Answer / System Output

## Lab Objectives
- Set up a **Python + Jupyter/Colab RAG environment**
- Generate and visualize **semantic embeddings**
- Build a **vector index** for fast similarity search
- Implement **top-k retrieval**
- Inject retrieved context into an **LLM prompt**

## Hands-On Tasks
1. **Data Ingestion**
   - Load PDFs / Markdown / Web text
   - Chunk documents for embedding
2. **Embedding Generation**
   - Encode text into vectors
   - Plot embedding similarity (cosine / Euclidean)
3. **Vector Store Construction**
   - Build FAISS or Chroma index
   - Save and reload the index
4. **Retrieval**
   - Query with natural language
   - Inspect top-k retrieved chunks
5. **Generation**
   - Augment prompt with retrieved context
   - Compare with and without retrieval

## Learning Outcomes
- Understand how **semantic search drives LLM grounding**
- Measure **retrieval quality vs. hallucination risk**
- Build a **modular RAG backbone** for later graph and multimodal extensions

## Course Connection
- **Week 1:** Embeddings & Retrieval (System Backbone)  
- **Week 6–9:** Graph Intelligence & Knowledge Reasoning  
- **Week 10–12:** Full RAG & GenAI System Design
