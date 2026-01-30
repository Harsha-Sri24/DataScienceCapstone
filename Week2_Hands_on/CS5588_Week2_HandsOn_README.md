# CS 5588 — Week 2 Hands-On
## Applied RAG for Product & Venture Development

> **Course:** CS 5588 — Data Science Capstone / GenAI Product Leadership  
> **Student:Harsha Sri Neeriganti**  
> **Project / Product Name:AI-Powered Weather & Climate Intelligence System for Personalized Decision Support**  
> **GitHub:**  
> **Date:**  

---

## 1. Product Overview
### Problem Statement
Describe the real-world problem your product is solving. What is broken, slow, risky, or expensive today?
People need timely and accurate weather information to plan their day, stay safe during severe weather, and analyze climate trends. Current solutions often provide raw data without context, fail to highlight critical alerts, or make it difficult to compare historical patterns. Our product aims to combine precise retrieval of relevant weather data with generative explanations so users can quickly understand current conditions, receive actionable alerts, and explore historical trends in one trustworthy interface.

### Target Users
Who will use this system in practice? (e.g., clinicians, analysts, compliance officers, educators, founders, NGOs, etc.)
•	General users planning daily activities or travel
•	Travelers evaluating weather suitability for destinations and dates
•	Students, researchers, and analysts interested in climate trends


### Value Proposition
Why would someone choose your AI system over existing tools, search engines, or manual workflows?
This goes beyond standard weather apps or search engines by combining precise retrieval of relevant data with context-aware explanations. Users don’t just see raw numbers—they get actionable insights, timely alerts, and clear comparisons to historical trends, all grounded in evidence. This saves time, reduces risk (especially for travelers), and helps researchers or planners make informed decisions without digging through multiple sources

---

## 2. Dataset Reality
### Data Source & Ownership
- Source: Publicly available weather and climate data from agencies and APIs such as NOAA, NASA, city open data portals, and OpenWeather API.

Owner: Public data; no confidential or proprietary information is involved.

### Sensitivity & Ethics
- Sensitivity level (public / internal / regulated / confidential):Public, low-risk data.

Privacy / compliance concerns: Minimal, since the system only uses aggregated weather data and does not store personal user information. Ethical considerations focus on accuracy of alerts and avoiding misinformation in high-stakes scenarios (e.g., severe weather warnings).


### Document Types
- Examples: policies, manuals, research, reports, SOPs, meeting notes, etc.
Weather reports, hourly forecasts, historical climate trends, severe weather alerts, and summary dashboards.
### Expected Scale in Production
- How many documents would this system realistically manage?
The system could manage hundreds of thousands of daily updates, covering multiple cities, weather stations, and regions globally, with each document containing real-time readings, alerts, and historical data for grounding.
---

## 3. User Stories & Risk Awareness

### U1 — Normal Use Case
> As a general user, I want to check the current weather and hourly forecast for my city so that I can plan my day effectively.

**Acceptable Evidence:**  
Real-time weather and hourly forecasts displayed for the selected city
Temperature, precipitation, wind, and humidity clearly visible
**Correct Answer Criteria:**  
Information is accurate, up-to-date, and easily readable
Forecast covers at least the next 24 hours

### U2 — High-Stakes Case
> As a traveler, I want to receive alerts for severe weather events so that I can avoid dangerous conditions and plan trips safely.

**Why This Is High Risk:**  
Missing or delayed alerts could put users in harm’s way (floods, storms, hurricanes)
**Acceptable Evidence:**  
Notifications or dashboard alerts for severe events
Details of the alert and recommended safety precautions
**Correct Answer Criteria:**  
Alerts are timely, actionable, and clearly communicated
Advice is based directly on the alert evidence
### U3 — Ambiguous / Failure Case
> As a researcher, I want to compare current weather with historical trends so that I can understand anomalies or patterns over time.

**What Could Go Wrong:**  
Historical trends might be misaligned or not included in top chunks
LLM might generate conclusions without grounding evidence
**Safeguard Needed:**  
Ensure historical data is correctly chunked and retrievable
Use citations and allow the system to abstain when evidence is insufficient

---

## 4. System Architecture (Product View)

### Chunking Strategy
Fixed or Semantic: Semantic
Chunk size / overlap: ~1000 characters per chunk, overlapping 150 chars
Why this fits your product users: Preserves full alerts, forecasts, and historical trends in a single chunk, so high-stakes travelers (U2) and researchers (U3) get contextually complete evidence.

### Retrieval Design
Keyword layer: BM25 for exact term matches (ensures critical alerts are found)
Vector layer: SentenceTransformer embeddings + FAISS index for semantic similarity (finds relevant content even with different wording)
Hybrid α value(s): 0.5 (balances precision-first and discovery-first behavior)

### Governance Layer
- Re-ranking method: Cross-Encoder reranker (ms-marco-MiniLM-L6-v2)
- What risk this layer reduces: Prevents low-confidence or irrelevant chunks from reaching generation, especially important for U2 high-stakes queries

### Generation Layer
- Model used: Flan-T5-base (seq2seq)
- Grounding & citation strategy: Includes top 3 chunks in context; citations like [Chunk 1], [Chunk 2] indicate source of evidence
- Abstention policy: Returns “Not enough evidence” when chunks don’t contain sufficient relevant information, avoiding hallucinations

---

## 5. Results

| User Story | Method (Keyword / Vector / Hybrid) | Precision@5 | Recall@10 | Trust Score (1–5) | Confidence Score (1–5) |
|------------|-----------------------------------|-------------|-----------|-------------------|-------------------------|
| U1         |Hybrid + Rerank                    |   0.6       | 0.8       |  4                |         4                |
| U2         |Hybrid + Rerank                    |   0.0             | 0.0          |  1                 | 1                        |
| U3         |vector(semantic) + hybrid                                   |    0.3         |    0.5       |      3             |   3                      |

---

## 6. Failure Case & Venture Fix

### Observed Failure
Describe one real failure you observed in your system.
Severe weather alerts for U2_high-stakes were not retrieved; the system returned “Not enough evidence” even though alerts existed in the dataset.

### Real-World Consequence
What could happen if this system were deployed as-is? (legal, financial, ethical, safety, trust, etc.)
Travelers could miss critical warnings, leading to safety risks or legal/ethical concerns for the product provider.

### Proposed System-Level Fix
Data: Move critical alerts to the top of each document

Chunking: Use semantic chunking to preserve complete alerts

Hybrid α: Increase weight of keyword search for high-stakes queries

Re-ranking: Ensure cross-encoder prioritizes chunks with alert keywords

Human-in-the-loop review: Optionally validate high-risk queries before final answer delivery

---

## 7. Evidence of Grounding

Paste one **RAG-grounded answer** below with citations.

> **Answer:**  
>  The current temperature in Station_1 is 12°C with 2mm precipitation, wind 5 km/h, and humidity 68%. Hourly forecasts are displayed for the next 24 hours, including expected changes in temperature and precipitation [Chunk 1]. The forecast dashboard is clear and easy to read [Chunk 2].”
> **Citations:** [Chunk 1], [Chunk 2]

---

## 8. Reflection (3–5 Sentences)
What did you learn about the difference between **building a model** and **building a trustworthy product**?
Building a model is about optimizing algorithms and metrics, but building a trustworthy product requires thinking about user risk, grounding, and failure modes. In this project, we saw that even strong retrieval or generation models can fail if evidence is missing or poorly chunked. Measures like hybrid search, re-ranking, and abstention policies are essential to ensure safety and reliability for different users. Ultimately, product design must balance accuracy, explainability, and risk management, not just raw performance metrics.
---

## Reproducibility Checklist
- [ ] Project dataset included or linked
- [ ] Notebook runs end-to-end
- [ ] User stories + rubric completed
- [ ] Results table filled
- [ ] Screenshots or logs included

---

> *CS 5588 — UMKC School of Science & Engineering*  
> *“We are not building models. We are building products people can trust.”*
