import ollama
import chromadb


# Create ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create/get collection
collection = client.get_or_create_collection(
    name="documents"
)


# Function to generate embeddings using Ollama
def get_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )

    return response["embeddings"][0]


# Our document chunks
chunks = [
    "TCP uses acknowledgements and retransmission to provide reliable delivery.",
    "HTTP is an application layer protocol used for communication on the web.",
    "TCP uses sequence numbers to ensure data arrives in the correct order."
]


# Generate embeddings and store chunks
for i, chunk in enumerate(chunks):

    embedding = get_embedding(chunk)

    collection.add(
        ids=[f"chunk_{i + 1}"],
        documents=[chunk],
        embeddings=[embedding]
    )


print("Chunks stored in ChromaDB!")


# Embed the user's question
question = "How does TCP provide reliable communication?"

question_embedding = get_embedding(question)


# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)


print("\nRetrieved chunks:")

for document in results["documents"][0]:
    print("-", document)