## **AI Healthcare Knowledge Assistant using Hybrid RAG**

Finding reliable information from large medical research documents can be challenging. To address this, I built an **AI Healthcare Knowledge Assistant** using **Hybrid Retrieval-Augmented Generation (RAG)** to provide grounded answers from a medical research knowledge base.

### How it works:

**1. Document Processing**
Loaded medical research PDFs and split them into meaningful chunks using **Recursive Character Text Splitter**.

**2. Embeddings & Vector Database**
Used `all-MiniLM-L6-v2` to generate embeddings and stored them in **ChromaDB** for semantic search.

**3. Hybrid Retrieval**
Combined **BM25 keyword retrieval + MMR semantic retrieval + Ensemble Retrieval** to improve both keyword matching and contextual relevance.

**4. Contextual Compression**
Used **Embeddings Filter** to remove less relevant retrieved content before passing the context to the LLM.

**5. HyDE Query Enhancement**
Implemented **HyDE (Hypothetical Document Embeddings)** to transform the user's question into a hypothetical medical paragraph, helping improve semantic retrieval.

**6. Grounded LLM Generation**
Used **`openai/gpt-oss-20b` through Groq** to generate answers strictly from the retrieved medical context, with instructions to avoid guessing or hallucinating information.

### Tech Stack

**Python | LangChain | RAG | Hybrid Retrieval | BM25 | MMR | HyDE | ChromaDB | Hugging Face | Groq | GPT-OSS-20B | Streamlit**

### Key Takeaway

This project helped me understand and implement an advanced RAG pipeline:

**Medical PDFs → Chunking → Embeddings → ChromaDB → BM25 + MMR → Hybrid Retrieval → Compression → HyDE → LLM → Grounded Answer**

The main focus was **improving retrieval quality and reducing hallucinations** by ensuring that answers are based only on the available medical research documents.

⚠️ *This project is for educational and informational purposes only and is not a replacement for professional medical advice.*
