# Retrieval-Augmented Generation System for iOS 18 🚀

This project implements a Retrieval-Augmented Generation (RAG) pipeline that enables up-to-date Q/A on iOS 18 — information that large language models struggle with due to knowledge cutoff limitations.

✅ Combines **LangChain** + **ChromaDB Vector Store** with **Phi-3 Mini 4K Instruct LLM**  
✅ Evaluates retrieval performance under different **chunk sizes** and **overlaps**  
✅ Generates JSON report comparing answer completeness and accuracy

---

## 📌 Motivation
LLMs often fail to answer questions about events after their training cutoff.  
For example:  
> "What new features were added in iOS 18 (2024)?"

➡️ The base model confidently gives incorrect answers.

This project demonstrates how a **custom knowledge base** can improve factual responses using RAG.

> 🎓 This mini-project was developed as part of the **Natural Language Processing course (2024–2025)** in the MSc Data Science program at the University of Padova.

---

## 🧠 Architecture

PDF Knowledge Base → Text Chunking → Embeddings → ChromaDB → Retriever → Phi-3 LLM

---

## 📊 Experiments

Three questions × Three chunking setups:

| Chunk Size | Overlap | Best For |
|-----------|----------|---------|
| 200 | 40 | Precise small facts |
| 400 | 80 | Balanced quality |
| 800 | 160 | Long contextual answers |

✅ Results stored in `rag_report_hosseini_sobhan.json`

---

## 🔧 Technologies Used

- Python
- LangChain, LangChain-Chroma
- HuggingFace Sentence Transformers
- Microsoft Phi-3 Mini 4K Instruct
- BitsAndBytes (4-bit quantization)
- Chroma Vector Store
- PyPDFLoader

---
