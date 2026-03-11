# CS 5588 — Week 1: Hands-On
**Mini-RAG System: Embeddings, Vector Retrieval & Grounded Generation**  
GitHub → Colab → Hugging Face → Vector Store (FAISS / Chroma)

## Objectives
By the end of this lab, you will:
- Use GitHub for collaborative workflows (repo, commit, PR)
- Run a Jupyter notebook in Google Colab
- Load a Transformer-based embedding model and dataset from Hugging Face Hub
- Build an embedding-based **vector retrieval system** (core backbone of RAG)
- Connect retrieved context to an **LLM for grounded generation**

## GenAI Systems Context (Why this matters)
This lab implements the **retrieval and grounding layer** used in modern GenAI systems:
- **Embeddings** map documents into vectors (semantic meaning space)
- **Vector retrieval** selects relevant context for a query
- **LLMs** generate answers using retrieved context (reducing hallucination and improving trustworthiness)

---

## Checklist

### GitHub
- [ ] Fork the CS 5588 course repository
- [ ] Edit `README.md` with your name, major, and project interest
- [ ] Commit changes
- [ ] Open a Pull Request

### Google Colab
- [ ] Open `week1_mini_rag.ipynb` from GitHub
- [ ] Install required libraries
- [ ] Run all cells successfully

### Hugging Face
- [ ] Find an embedding model (e.g., `all-MiniLM-L6-v2` or equivalent)
- [ ] Find a text dataset (e.g., `ag_news` or domain-specific corpus)

### Retrieval Task
- [ ] Run `search("artificial intelligence in healthcare")`
- [ ] Inspect top-k retrieved documents
- [ ] Compare LLM output **with vs. without retrieval context**

---

## Exit Ticket
Post a GitHub Issue titled:
> **Reflection — Grounded Generation with Mini-RAG (CS 5588)**

Write 2–3 sentences explaining:
1. How embeddings shape retrieval quality
2. Why retrieved context improves reliability and reduces hallucination in GenAI systems
