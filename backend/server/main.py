import os
import time
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from weaviate.connect import ConnectionParams
from pydantic import BaseModel
from groq import Groq
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

load_dotenv()

from scraper import PudmedArticleScrapper

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

            # Verify connection
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data model for the request body
class ChatMessage(BaseModel):
    role: str = None
    content: str = None

class GenerationRequest(BaseModel):
    model: str = "llama-3.3-70b-versatile"
    chats: List[ChatMessage]
    collections: List[str]
    distance: float = 0
    limit: int = 1

class Article(BaseModel):
    url: str = None


# Define API endpoint for handling chat generation
@app.post("/generate-chat")
async def handle_query(request: GenerationRequest):
    response = generate_response(
        model = request.model, 
        chats = request.chats, 
        collections = request.collections,
        limit = request.limit,
        distance = request.distance
    )
    return response


# Define API endpoint for loading documents 
@app.post("/load-content")
async def handle_content_load(article: Article):
    response = load_content(url = article.url)
    return response


# Define API endpoint for retrieving content from a collection in the Weaviate database
@app.get("/retrieve-content")
async def handle_retrieval(query: str="", limit: int=1, distance: float=0, collection: str=""):
    response = get_context(query, [collection], limit, distance)
    return response["context"]


def get_context(query: str, collections: List[str], limit: int, distance: float) -> str:
    """
    Fetch relevant context from the Weaviate database based on the query.

    Args:
        query (str): The search query.
        collections (List[str]): Existing Weaviate collections from which to pull relevant context.
        limit (int): Number of relevant texts to include in the context.
        distance (int): Minimum distance from query.
    
    Returns:
        dict: Dictionary containing the URL, and the content.
                Returns None values if no relevant context is found.
    """
    if not query:
        return ""
    
    try:
        client = create_weaviate_client()
        retrieved_context = []

        for collection in collections:
            if not client.collections.exists(collection):
                raise ValueError(f'Collection {collection} not found.')
                
            # Initialize sentence transformer model for embedding
            model = SentenceTransformer("neuml/pubmedbert-base-embeddings")
            embedding = model.encode(query)
            
            # Query collection for relevant abstracts
            collection = client.collections.get(collection)
            response = collection.query.near_vector(
                near_vector=embedding.tolist(),
                limit=limit, 
                distance = distance,
                include_vector=True
            )

            # Return abstract if found
            if response.objects:
                retrieved_context.append({
                    "url" : response.objects[0].properties["url"],
                    "content": response.objects[0].properties.get("abstract")# or response.objects[0].properties.get("chunk") or "")
                })

            else:
                retrieved_context.append({ "url" : None, "content": None })
        
        return { "context" : retrieved_context }
            
    except Exception as e:
        print(f"Error fetching context: {e}")
        return "No relevant context found"
    
    finally:
        client.close()


def generate_response(model: str, chats: List[ChatMessage], collections: List[str], limit: int, distance: float):
    """
    Generate a response based on the given chat history and relevant context.

    Args:
        chat (List[ChatMessage]): The chat history with user and assistant messages.
        collections (List[str]): Existing Weaviate collections from which to pull relevant context.
        limit (int): Number of relevant texts to include in the context.
        distance (int): Minimum distance from query.
    
    Returns:
        dict: Dictionary containing the source URL(s), generated response, and classification.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)

        # Get latest query from chat history
        query = chats[-1].content

        # Retrieve relevant context from the database
        contexts = get_context(query, collections, limit, distance)
        
        if isinstance(contexts, str):
            context_text = ""
        else:
            context_text = "\n\n".join(
                context["content"] for context in contexts["context"] if context.get("content") is not None
            )

        # Define system prompt injected with relevant content
        system_msg = """
        You are an information extraction system that answers questions about Alzheimer's disease using only the information provided in the context.
        Your task is to generate a complete, detailed, and informative response by synthesizing as much relevant information as possible from the context. You may combine facts from different parts of the context, but do not add anything that is not explicitly stated.
        Always answer in your own words using clear and precise language. Do not copy text verbatim unless necessary for accuracy. Avoid speculation.
        If the answer cannot be derived from the context, try to generate an answer without it.
        Do not mention the context or refer to it in your answer.
        """

        user_prompt = f"""
        Context:
        {context_text}

        Question:
        {query}
        """

        # Construct message list starting with system prompt
        messages = [{
            "role": "system", 
            "content": system_msg
        }]

        # Add chat history
        messages.extend([{"role": chat.role, "content": chat.content} for chat in chats[:-1]])
        messages.append({"role": "user", "content": user_prompt})

        # Generate chat completion
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
        )

        return {
            "urls" : [context["url"] for context in contexts["context"]], 
            "response": chat_completion.choices[0].message.content,
        }
    
    except Exception as e:
        return {
            "urls": None,
            "response": f"An error occured while generating the response: {e}",
        }


def load_content(url: str):
    """
    Load full text article into a Weaviate collection in chunks.

    Args:
        url (str): URL of the article to be loaded

    Returns:
        dict: Dictionary containing a response message indicating success or failure.
    """
    try:
        client = create_weaviate_client()
        
        if not client.collections.exists("Articles"):
            client.collections.create(
                name="Articles",
                vectorizer_config = Configure.Vectorizer.none(),
                properties=[
                    Property(name="abstract", data_type=DataType.TEXT),
                    Property(name="url", data_type=DataType.TEXT, skip_vectorization=True)
                ]
            )
        collection = client.collections.get("LoadedAArticlesrticles")

        # Check that content from this URL has not yet been loaded into the database
        if url_exists(collection, url):
            return {
                "response": "Content already exists in Weaviate Database."
            }
        
        # Scrape and chunk the article content
        scraper = PudmedArticleScrapper(url)
        chunks = scraper.get_content_chunks()

        # Initialize sentence transformer model for embedding
        model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

        # Add extracts from the article to Articles collection in batches
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                embedding = model.encode(chunk) 
                batch.add_object(
                    properties={"chunk": chunk, "url": url},
                    vector=embedding.tolist()
                )
        client.close()

        return {
            "response": f"Success loading content into Weaviate database."
        }

    except Exception as e:
        return {
            "response": f"An error occurred while loading content: {e}"
        }
    

def url_exists(collection: weaviate.WeaviateClient, url: str) -> bool:
    """Check if URL already exists in the Weaviate collection."""

    # Query collection for objects with the specified URL
    result = collection.query.fetch_objects(
        filters=Filter.by_property("url").equal(url), 
        limit=1
    )
    return len(result.objects) > 0