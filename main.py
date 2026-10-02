import ollama

text = "TCP provides reliable communication using acknowledgements and retransmission."

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])