import ollama

context = """
TCP is a connection-oriented transport-layer protocol.
It provides reliable data delivery using acknowledgements,
sequence numbers, and retransmission of lost packets.
"""

question = "Why is TCP reliable?"

prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}
"""

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response["message"]["content"])