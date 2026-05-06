# semantic_search.py
# Implements semantic search using embeddings + FAISS + TF-IDF comparison

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load sample documents
def load_documents():
    with open("data/documents.txt", "r") as f:
        docs = f.readlines()
    return [d.strip() for d in docs]


# Semantic Search
class SemanticSearch:
    def __init__(self, docs):
        self.docs = docs
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(docs)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    def search(self, query, k=3):
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb), k)

        return [self.docs[i] for i in I[0]]

    def save_index(self):
        faiss.write_index(self.index, "embeddings/faiss_index.bin")


# TF-IDF Search
class KeywordSearch:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query, k=3):
        q_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ q_vec.T).toarray().flatten()

        top_idx = np.argsort(scores)[::-1][:k]
        return [self.docs[i] for i in top_idx]


if __name__ == "__main__":
    docs = load_documents()

    semantic = SemanticSearch(docs)
    keyword = KeywordSearch(docs)

    query = input("Enter query: ")

    print("\nSemantic Search Results:")
    print(semantic.search(query))

    print("\nKeyword Search Results:")
    print(keyword.search(query))

    semantic.save_index()