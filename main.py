import ollama
import chromadb


# -----------------------------
# 1. Create ChromaDB
# -----------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


# -----------------------------
# 2. Embedding function
# -----------------------------

def get_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return response["embeddings"][0]


# -----------------------------
# 3. Document chunks
# -----------------------------

chunks = [
    "TCP uses acknowledgements and retransmission to provide reliable delivery.",
    "HTTP is an application layer protocol used for communication on the web.",
    "TCP uses sequence numbers to ensure data arrives in the correct order."
]


# -----------------------------
# 4. Store chunks
# -----------------------------

for i, chunk in enumerate(chunks):

    embedding = get_embedding(chunk)

    collection.upsert(
        ids=[f"chunk_{i + 1}"],
        documents=[chunk],
        embeddings=[embedding]
    )


# -----------------------------
# 5. User question
# -----------------------------

question = input("\nAsk a question: ")


# -----------------------------
# 6. Retrieve relevant chunks
# -----------------------------

question_embedding = get_embedding(question)

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

retrieved_chunks = results["documents"][0]


# -----------------------------
# 7. Build context
# -----------------------------

context = "\n\n".join(retrieved_chunks)


# -----------------------------
# 8. Create prompt
# -----------------------------

prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""
# -----------------------------
# 9. Generate answer using Qwen
# -----------------------------

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# -----------------------------
# 10. Display answer
# -----------------------------

print("\nAnswer:")
print(response["message"]["content"])