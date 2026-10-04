import ollama
import chromadb


# Connect to existing ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


# Generate embedding for a question
def get_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return response["embeddings"][0]


# Get user's question
question = input("\nAsk a question: ")


# Convert question into an embedding
question_embedding = get_embedding(question)


# Retrieve relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

retrieved_chunks = results["documents"][0]


# Combine retrieved chunks into context
context = "\n\n".join(retrieved_chunks)


# Create prompt
prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""


# Generate answer
response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


print("\nAnswer:")
print(response["message"]["content"])