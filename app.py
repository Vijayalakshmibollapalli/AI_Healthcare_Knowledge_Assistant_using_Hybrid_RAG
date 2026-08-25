import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (EnsembleRetriever, ContextualCompressionRetriever)
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# CONFIGURATION
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI Healthcare Knowledge Assistant", layout="wide")

st.image("OIP.webp")

# HEADER
st.markdown("""<h2 style='text-align:left;' >AI Healthcare Knowledge Assistant Using Hybrid RAG</h2> """, unsafe_allow_html=True)


st.markdown(
"""
Ask healthcare questions related to:

- Diabetes
- Hypertension
- Cardiovascular Disease
- Asthma
- Chronic Kidney Disease
"""
)

# EMBEDDINGS
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = load_embeddings()

# VECTOR DATABASE
@st.cache_resource
def load_vectorstore():
    return Chroma(persist_directory="MEDICAL_RAG", embedding_function=embeddings)
vectorstore = load_vectorstore()

# BM25 RETRIEVER
@st.cache_resource
def load_bm25():
    from langchain_community.document_loaders import (PyPDFDirectoryLoader)
    from langchain_text_splitters import (RecursiveCharacterTextSplitter)

    loader = PyPDFDirectoryLoader("Medical_DataSet")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    chunks = splitter.split_documents(documents)
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 5
    return retriever

bm25 = load_bm25()

# MMR RETRIEVER
mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":5, "fetch_k":25, "lambda_mult":0.7})

# HYBRID RETRIEVER
ensemble = EnsembleRetriever(retrievers=[bm25, mmr], weights=[0.4, 0.6])

# CONTEXT COMPRESSION
compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.60)

retriever = ContextualCompressionRetriever(base_retriever=ensemble, base_compressor=compressor)

# LLM
@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY, temperature=0, timeout=300, max_retries=5)
llm = load_llm()

# HYDE QUERY EXPANSION

hyde_prompt = ChatPromptTemplate.from_template(
    """
    Write a short factual paragraph that could appear
    inside a medical review article answering the question.
    Use clinical terminology.
    Do not answer conversationally.

Question:

{question}

Paragraph:
""")

hyde_chain = (hyde_prompt|llm|StrOutputParser())

# QUESTION INPUT

query = st.text_input("", placeholder="Example: What are the symptoms of asthma?")

search = st.button("Get Answer")

if search and query.strip():
    with st.spinner("Searching medical documents..."):

        # HyDE Query Expansion
        hypothetical_document = hyde_chain.invoke({"question": query})

        # Retrieval
        docs = retriever.invoke(hypothetical_document)
        docs = docs[:5]

        # Build Context
        context = "\n\n".join(doc.page_content for doc in docs)

        if len(context.strip()) < 200:

            answer = ("The provided healthcare documents do not contain enough information to answer this question.")
            sources = []

        else:
            prompt = f"""
            
            You are an AI Healthcare Assistant.
            Answer ONLY using the provided context.
            STRICT RULES:
            
            1. Use only retrieved healthcare documents.
            2. Do not use your own medical knowledge.
            3. Do not guess.
            4. Do not infer missing information.
            5. If information is unavailable reply exactly:
            
            The provided healthcare documents do not contain enough information to answer this question.
            
            6. Provide a concise evidence-based answer.
            7. Structure answers using headings and bullet points.
            8. Avoid repeating the same information multiple times.
            9. Ignore references, citations, author names, and bibliography sections.
            10. Do not include journal names or citation numbers in the answer.
            11. Explain medical terms briefly when needed.
            
            Context:
            {context}
            
            Question:
            {query}
            
            Answer:
            """

            response = llm.invoke(prompt)
            answer = response.content

            sources = list(set(
                os.path.basename(doc.metadata.get("source"))
                for doc in docs))

    # Display Answer
    st.subheader("Answer")

    st.write(answer)

    # Display Sources
    if sources:
        with st.expander("Retrieved Sources"):
            for source in sources:
                st.write(source)

# SIDEBAR
st.sidebar.title("RAG Pipeline Configuration")

st.sidebar.markdown(
"""
**Knowledge Base**
- 40 Medical Research PDFs
- 664 Documents Loaded
- 5396 Text Chunks

**Text Processing**
- Chunk Size: 700
- Chunk Overlap: 80

**Embedding Model**
- all-MiniLM-L6-v2

**Vector Database**
- Chroma

**Retrieval Strategy**
- BM25 + MMR Hybrid Retrieval

**Context Optimization**
- Embeddings Filter Compression

**Query Enhancement**
- HyDE

**LLM Generation**
- Groq Llama-3.1-8B
""")

st.sidebar.info("""The assistant retrieves and generates answers only from medical research documents.""")

if st.sidebar.button("Clear"):
    st.rerun()

st.caption("""Educational healthcare assistant. Not a replacement for medical professionals.""")