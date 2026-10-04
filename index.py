import ollama
import chromadb


# Create ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create/get collection
collection = client.get_or_create_collection(
    name="documents"
)


# Generate embedding using Ollama
def get_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return response["embeddings"][0]


# Document chunks
chunks = [
    "TCP uses acknowledgements and retransmission to provide reliable delivery.",
    "HTTP is an application layer protocol used for communication on the web.",
    "TCP uses sequence numbers to ensure data arrives in the correct order."
]


# Store chunks and embeddings
for i, chunk in enumerate(chunks):

    embedding = get_embedding(chunk)

    collection.upsert(
        ids=[f"chunk_{i + 1}"],
        documents=[chunk],
        embeddings=[embedding]
    )

print("Documents indexed successfully!")