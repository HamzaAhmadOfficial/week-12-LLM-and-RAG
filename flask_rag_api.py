# Flask API for RAG system with upload + QA

from flask import Flask, request, jsonify, render_template
from rag_system import RAGSystem
import os

app = Flask(__name__)
rag = RAGSystem()

UPLOAD_FOLDER = "data/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

conversation_history = []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    rag.load_documents(filepath)

    return jsonify({"message": "File uploaded and processed"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")

    answer = rag.generate_answer(question)

    conversation_history.append({
        "question": question,
        "answer": answer
    })

    return jsonify({
        "answer": answer,
        "history": conversation_history
    })


if __name__ == "__main__":
    app.run(debug=True)