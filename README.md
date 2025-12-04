
## 🚀 MongoRAG-Intelligence-Hub
**VectorMind RAG Engine**  

This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline using MongoDB Vector Search and OpenAI models.  
It converts PDFs into searchable vector embeddings, retrieves the most relevant context, and generates precise AI responses.  
A single, streamlined Python class handles ingestion, indexing, semantic search, and LLM reasoning.


---
## 🏗️ Project Architecture

![Project Architecture](https://github.com/SushilkumarBarai/MongoRAG-Intelligence-Hub/blob/main/Screenshot.png)


## 📘 Overview

This project demonstrates a complete, production-ready **Retrieval-Augmented Generation (RAG)** pipeline using:

- **MongoDB Vector Search**
- **OpenAI Embeddings**
- **OpenAI GPT Models**
- **PDF Document ingestion**
- **Text chunking and embedding**
- **Semantic search**
- **Context-aware answer generation**

A single Python class handles the entire flow:
PDF → Text Split → Embedding → MongoDB Insert → Vector Index → Vector Search → LLM Answer.

This project is ideal for:
- AI-assisted knowledge bases  
- Document question-answering systems  
- Enterprise RAG systems  
- AI copilots  
- Smart search engines  

---

## 🧠 Features

- Load PDFs from URL or local file  
- Automatically split documents into clean text chunks  
- Convert text into vector embeddings using OpenAI  
- Store vectors in MongoDB for semantic search  
- Create MongoDB Vector Index  
- Fast and accurate vector search  
- Complete RAG response generation using GPT  
- One-class, easy-to-understand, flexible architecture  

---

## 🏗 System Architecture

PDF → Text Split → Embeddings → MongoDB Vector DB → Semantic Search → LLM Answer

---

## 🤖 LLM Response Generation

The retrieved chunks become context, which is passed to GPT-4o for a grounded, accurate answer.

This prevents hallucination and improves reliability.

## 📈 Advantages of MongoDB Vector Search

- Extremely fast ANN performance
- Automated sharding and scaling
- Easy integration with AI pipelines
- Native semantic + keyword search
- Perfect for enterprise-grade RAG systems

## 🚀 Usage Example

Below is the exact code snippet to run the full RAG workflow:

```python
rag = RAGSystem(openai_key=OPENAI_API_KEY, mongo_uri=MONGO_URI)

# Load & split PDF
chunks = rag.process_pdf("https://investors.mongodb.com/node/12236/pdf")

# Insert into MongoDB
rag.ingest_docs(chunks)

# Create vector index
rag.create_vector_index()

# Query the RAG system
answer, docs = rag.answer_query("What is MongoDB AI Applications Program?")
print(answer)
```

## 🛠 Tech Stack

- **Python > 3.9**
- **MongoDB Atlas**
- **OpenAI GPT & Embeddings**
- **LangChain for PDF & Text processing**

## 🧪 Future Enhancements

- Add FastAPI endpoints
- Add Streamlit UI
- Add caching with Redis
- Multi-PDF ingestion
- Real-time vector updates
-Evaluation metrics (MRR, Recall@K, Accuracy)

## 📝 License

This project is open-source. Modify and use freely.
