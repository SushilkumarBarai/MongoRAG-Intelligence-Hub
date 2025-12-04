import os
import time
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGSystem:
    """
    One-Class RAG System:
    - Load PDF
    - Split text into chunks
    - Generate embeddings
    - Store in MongoDB
    - Create vector index
    - Perform vector search
    - Answer using LLM
    """

    def __init__(
        self,
        openai_key: str,
        mongo_uri: str,
        db_name="mongo_rag_db",
        collection_name="vector_test_search",
        embedding_model="text-embedding-3-large",
        chat_model="gpt-4o"
    ):
        # Set API Key
        os.environ["OPENAI_API_KEY"] = openai_key
        
        # Initialize OpenAI client
        self.openai = OpenAI()

        # Model names
        self.embedding_model = embedding_model
        self.chat_model = chat_model

        # MongoDB setup
        self.mongo_client = MongoClient(mongo_uri)
        self.collection = self.mongo_client[db_name][collection_name]

    # -----------------------------
    #  Generate Embedding
    # -----------------------------
    def get_embedding(self, text: str):
        """Generate vector embedding from text."""
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    # -----------------------------
    #  Load & Split PDF
    # -----------------------------
    def process_pdf(self, pdf_url: str, chunk_size=400, chunk_overlap=20):
        """Load PDF and split into text chunks."""
        loader = PyPDFLoader(pdf_url)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        return chunks

    # -----------------------------
    #  Insert into MongoDB
    # -----------------------------
    def ingest_docs(self, chunks):
        """Insert documents with embeddings into MongoDB."""
        payload = []
        for doc in chunks:
            emb = self.get_embedding(doc.page_content)
            payload.append({
                "text": doc.page_content,
                "embedding": emb
            })

        self.collection.insert_many(payload)
        print("📌 Documents inserted successfully!")

    # -----------------------------
    #  Create MongoDB Vector Index
    # -----------------------------
    def create_vector_index(self, index_name="vector_index", dimensions=3072):
        """Create vector search index in MongoDB."""

        index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": dimensions,
                        "path": "embedding",
                        "similarity": "cosine"
                    }
                ]
            },
            name=index_name,
            type="vectorSearch"
        )

        self.collection.create_search_index(model=index_model)

        print("⏳ Waiting for index to become queryable...")

        # Poll until index is ready
        while True:
            indexes = list(self.collection.list_search_indexes(index_name))
            if indexes and indexes[0].get("queryable") is True:
                break
            time.sleep(5)

        print(f"✅ Vector index '{index_name}' is READY!")

    # -----------------------------
    #  Vector Search
    # -----------------------------
    def vector_search(self, query: str, index_name="vector_index", limit=5):
        """Perform vector search in MongoDB using query text."""
        query_vector = self.get_embedding(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "queryVector": query_vector,
                    "path": "embedding",
                    "numCandidates": 3072,
                    "limit": limit
                }
            },
            {"$project": {"_id": 0, "text": 1}}
        ]

        results = list(self.collection.aggregate(pipeline))
        return results

    # -----------------------------
    #  Generate Final LLM Answer
    # -----------------------------
    def answer_query(self, query: str):
        """Get vector results + LLM answer."""
        results = self.vector_search(query)

        context = " ".join(doc["text"] for doc in results)

        prompt = f"""
        Use the following context to answer the question:
        {context}

        Question: {query}
        """

        completion = self.openai.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content, results


# ============================================================
#  SAMPLE EXECUTION
# ============================================================
if __name__ == "__main__":

    OPENAI_KEY = "<YOUR_OPENAI_KEY>"
    MONGO_URI = "mongodb+srv://mydbuser:helloworld@myvector.v9qidxa.mongodb.net/?appName=myvectorsearch"

    rag = RAGSystem(openai_key=OPENAI_KEY, mongo_uri=MONGO_URI)

    # ---- STEP 1: Load & split PDF ----
    chunks = rag.process_pdf("https://investors.mongodb.com/node/12236/pdf")

    # ---- STEP 2: Ingest into MongoDB ----
    # rag.ingest_docs(chunks)

    # ---- STEP 3: Create vector index ----
    # rag.create_vector_index()

    # ---- STEP 4: Ask Question ----
    answer, docs = rag.answer_query("What is MongoDB AI Applications Program?")
    print("\n===== FINAL ANSWER =====\n")
    print(answer)
