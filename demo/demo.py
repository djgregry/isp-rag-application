import os
import time
import pymupdf
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# Initialize Pinecone client with API key
pc = Pinecone(api_key=PINECONE_API_KEY)


# Initializing Pinecone Database Index
def get_index(index_name: str) -> Pinecone.Index: 
    # Create a new index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024, # Dimension of the vector embeddings
            metric="cosine", # Similarity metric
            spec=ServerlessSpec(cloud="aws", region="us-east-1") 
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    
    # Retrieve and return the index from Pinecone
    return pc.Index(index_name)
    

# Loading Documents (PDFs) directly into a Vector Database

# Build our own knowledge base to provide context to the LLM
def pdf_to_text(pdf: str) -> str:
    doc = pymupdf.open(pdf)
    content = ""
    for page in doc:
        text = page.get_text()
        content += text + "\n\n"

    return content


# Load text into the vector database
def load_text(text: str, chunk_size):
    # Target Pinecone index where to store vector embeddings
    index = get_index("isp")

    # Break original text into chunks of manageable size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-100)]
    labeled_chunks = [{"id": f"vec_{i+1}", "text": chunk} for i, chunk in enumerate(chunks)]

    # Generate vector embeddings for each of chunk
    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[chunk for chunk in chunks],
        parameters={"input_type": "passage"}
    ) # creates EmbeddingsList object

    # Prepare embeddings for insert
    records = []
    for c, e in zip(labeled_chunks, embeddings):
        records.append({
            "id": c["id"],
            "values": e["values"],
            "metadata": {"text": c["text"]}
        })
    # Upsert records into the index
    index.upsert(
        vectors=records,
        namespace="example"
    )
    
        
# Search vector embeddings with respect to a query
def get_context(query: str):
    # Target Pinecone index where to find vector embeddings
    index = get_index("isp")
    # Generate vector from query to search vector db
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    # Search index for k most similar vectors and return them
    results = index.query(
        namespace="example",
        vector=query_embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    return results


def generate_response(query, context):
    client = Groq(api_key=GROQ_API_KEY)
    
    context_text = [c['metadata']['text'] for c in context['matches']]
    system_prompt = f"""You are a helpful assistant. If relevant, use the context to answer the user's query.
    
    Context: {" ".join(context_text)}"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion


def main():
    # Load example PDF into vector database
    # text = pdf_to_text('data/GoodpastureSyndrome.pdf')
    # load_text(text, chunk_size=500)

    # Generate a response based on user query
    query = input("Say something: ")
    context = get_context(query)
    response = generate_response(query, context)
    print(response.choices[0].message.content)
    

if __name__ == '__main__':
    main()
