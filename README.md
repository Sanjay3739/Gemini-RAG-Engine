# 🧠 Gemini-RAG-Engine:

Gemini-RAG-Engine is an intelligent document QA system that combines Google's Gemini API with the Hypothetical Document Embedding (HyDE) technique for enhanced context-aware retrieval. Upload any PDF, ask a question, and get accurate, AI-generated answers — powered by RAG, Gemini, ChromaDB, and LangChain.
---

## ✅ Features

- 📄 PDF Text Extraction
Seamlessly extract and preprocess text from any PDF using PyPDF2, making complex documents ready for AI-based understanding.

- 🧩 Intelligent Text Chunking
Splits long documents into overlapping, context-preserving chunks using LangChain's RecursiveCharacterTextSplitter for more effective semantic retrieval.

- 🧠 Gemini-Powered Embeddings
Leverages Google Gemini’s text-embedding-004 model to convert document chunks into high-dimensional vectors for precise contextual similarity.

- 📦 Vector Storage with ChromaDB
Stores and retrieves embeddings using ChromaDB, an efficient and scalable vector store for fast retrieval tasks.

- 🪄 HyDE: Hypothetical Document Embedding
Implements the HyDE technique to generate a synthetic, AI-authored document based on your query — improving retrieval relevance dramatically.

- 🔍 Query Rewriting with LLM
Uses Gemini's generative capabilities to intelligently rewrite vague queries into more specific, context-aware questions.

- 💬 Contextual Answer Generation
Generates natural language answers grounded in the retrieved content using Gemini’s generative models — like a chat with your documents.

- 🛠️ Configurable & Extendable
Easily adjustable chunk sizes, embedding models, and retrieval depth (k) for advanced experimentation and customization.

- 📜 Logging & Debugging
Integrated Python logging provides detailed step-by-step execution logs for easy debugging and performance tracking.

---

## 🚀 File Structure

Gemini-RAG-Engine/
│
├── main.py                  # Entry point for PDF-to-Answer pipeline
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not committed)
└── README.md  
---

## ⚙️ Setup Instructions

Follow the steps below to set up and run the project in a clean Python environment.

1. Create & Activate Virtual Environment (Recommended):

   Windows:
   ```
    python -m venv venv
    venv\Scripts\activate
   ```

   macOS / Linux:
   ```
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install Dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the Project
    ```
    python main.py
    ```
---
