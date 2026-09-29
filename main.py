import ollama

# Load the document
with open("sample.txt", "r", encoding="utf-8") as file:
    context = file.read()

question = "Why is TCP reliable?"

prompt = f"""
Answer the question using only the information provided in the context.

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