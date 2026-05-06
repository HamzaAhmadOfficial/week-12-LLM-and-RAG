# Simple RAG implementation using ChromaDB + sentence-transformers + GPT-2

from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline


class RAGSystem:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="docs")

        self.generator = pipeline("text-generation", model="gpt2")

    def load_documents(self, file_path):
        with open(file_path, "r") as f:
            docs = f.readlines()

        chunks = [doc.strip() for doc in docs if doc.strip()]

        embeddings = self.embed_model.encode(chunks)

        for i, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk],
                embeddings=[embeddings[i]],
                ids=[str(i)]
            )

    def retrieve(self, query, k=3):
        q_emb = self.embed_model.encode([query])

        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=k
        )

        return results['documents'][0]

    def generate_answer(self, query):
        context = self.retrieve(query)

        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        response = self.generator(prompt, max_length=150)
        return response[0]['generated_text']


if __name__ == "__main__":
    rag = RAGSystem()
    rag.load_documents("data/documents.txt")

    query = input("Ask a question: ")
    answer = rag.generate_answer(query)

    print("\nRAG Answer:\n", answer)