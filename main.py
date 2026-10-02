from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the document
with open("sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

# Split document
chunks = splitter.split_text(text)

# Display chunks
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)