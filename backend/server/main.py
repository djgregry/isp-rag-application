import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from weaviate.connect import ConnectionParams
from pydantic import BaseModel
from groq import Groq
import weaviate

load_dotenv()

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"X-HuggingFace-Api-Key": HUGGINGFACE_API_KEY}

# Connect to Weaviate Database
def create_weaviate_client(max_retries=5):
    for attempt in range(max_retries):
        try:
            client = weaviate.WeaviateClient(
                connection_params=ConnectionParams.from_params(
                    http_host="weaviate",
                    http_port="8080",
                    http_secure=False,
                    grpc_host="weaviate",
                    grpc_port="50051",
                    grpc_secure=False,
                ),
                additional_headers=HEADERS
            )
            client.connect()

            # # Verify connection
            if client.is_ready():
                print("Weaviate connection successful!")
                return client
            else:
                print(f"Weaviate not ready. Attempt {attempt + 1}")
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(5)  
    
    raise Exception("Could not connect to Weaviate after multiple attempts")


# Initialize FastAPI application
app = FastAPI()

# Configure CORS to allow requests from specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data model for the request body
class QueryRequest(BaseModel):
    query: str = None


# Define API endpoint for handling queries
@app.post("/query")
async def handle_query(request: QueryRequest):
    response = generate_response(request.query)
    return response


def get_context(query: str) -> str:
    """
    Fetch relevant context from the Weaviate database based on the query.

    Args:
        query (str): The search query
    
    Returns:
        str: Combined abstract texts from the database.
    """
    try:
        client = create_weaviate_client()
        
        if not client.collections.exists("Articles"):
            return ""
        
        collection = client.collections.get("Articles")
        response = collection.query.near_text(query = query, limit = 2)
        context = "\n".join(obj.properties["abstract"] for obj in response.objects) 

        client.close()
        return context
    
    except Exception as e:
        print(f"Error fetching context: {e}")
        return "No relevant context found"



def generate_response(query):
    """
    Generate a response based on the query and relevant context.

    Args:
        query (str): The user query.
    
    Returns:
        str: Generated reponse from the language model.
    """
    client = Groq(api_key=GROQ_API_KEY)
    context_text = get_context(query)
    system_prompt = f"""You are a helpful assistant. If relevant, use the context to answer the user's query.
    
    Context: {context_text}"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        model="llama3-8b-8192",
    )
    return {
        "context" : context_text, 
        "response": chat_completion.choices[0].message.content
    }