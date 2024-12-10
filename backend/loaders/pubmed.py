import os
import time
import argparse
from typing import Dict, List
import weaviate
from dotenv import load_dotenv
from Bio import Entrez
from weaviate.classes.config import Configure, Property, DataType

# Load environment variables
load_dotenv()

# Environment variables
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_API_KEY")
PUBMED_BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/"
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
HEADERS = {"X-HuggingFace-Api-Key": HUGGINGFACE_KEY}

# Configure Entrez
Entrez.email = NCBI_EMAIL
Entrez.api_key = NCBI_API_KEY


def fetch_pubmed_articles(query: str, retmax:int=100) -> List[Dict[str,str]]:
    """
    Fetch PubMed articles based on a query.

    Args:
        query (str): The search query for PubMed.
        retmax (int): Maximum number of articles to return.
    
    Returns:
        List[Dict[str,str]]: A list of articles with abstract and URL.
    """
    try:
        search_result = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
        record = Entrez.read(search_result)
        ids = record.get("IdList", [])
        
        articles = list()
        for article_id in ids:
            # Fetch details of an article given its ID
            fetch_id = Entrez.efetch(db="pubmed", id=article_id, rettype="gb", retmode="xml")
            records = Entrez.read(fetch_id)

            for record in records.get("PubmedArticle", []):
                article = record['MedlineCitation']['Article']

                # Extract article details
                title = article.get("ArticleTitle", "Title Not Available")
                abstract_sections = article.get("Abstract", {}).get("AbstractText", [])
                abstract = "\n\n".join(
                    f"## {section.attributes.get('Label', 'SECTION')}\n\n{section}"
                    for section in abstract_sections
                )

                # Add article detials to the list
                articles.append({
                    "abstract": f"## TITLE\n\n{title}\n\n{abstract}",
                    "url": f"{PUBMED_BASE_URL}{article_id}"
                })

                # Avoid API rate limits
                time.sleep(0.1)
        
        return articles
    
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return []


def setup_weaviate(client: weaviate.Client):
    """
    Set up the Weaviate database.

    Args:
        client (weaviate.Client): Weaviate client instance.
    """
    if not client.collections.exists("Articles"):
        client.collections.create(
            "Articles",
            vectorizer_config = Configure.Vectorizer.text2vec_huggingface(
                model="sentence-transformers/all-MiniLM-L6-v2",
            ),
            properties=[
                Property(name="abstract", data_type=DataType.TEXT),
                Property(name="url", data_type=DataType.TEXT, skip_vectorization=True)
            ]
        )
        print("Collection created successfully.")


def add_data(client: weaviate.Client, collection: str):
    """
    Add data to the Weaviate collection.
    """
    setup_weaviate(client)
    articles = fetch_pubmed_articles("Alzheimer's Disease", retmax=100000)
    collection = client.collections.get(collection)
    with collection.batch.dynamic() as batch:
        for article in articles:
            batch.add_object(properties=article)
    print("Articles added successfully to Weaviate.")


def delete_collection(client: weaviate.Client, collection_name: str):
    """
    Delete a specific collection.
    """
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    else:
        print(f"Collection '{collection_name}' does not exist.")


def delete_all_collections(client: weaviate.Client):
    """
    Delete all collections.
    """
    client.collections.delete_all()
    print("All collections deleted successfully.")


def main():
    parser = argparse.ArgumentParser(description="Weaviate Operations Script")
    parser.add_argument("--add-data", type=str, help="Add data to the specified collection")
    parser.add_argument("--delete-collection", type=str, help="Delete a specific collection")
    parser.add_argument("--delete-all", action="store_true", help="Delete all collections")

    args = parser.parse_args()

    try:
        client = weaviate.connect_to_local(grpc_port=50052, headers=HEADERS)

        if args.add_data:
            add_data(client, args.add_data)
        elif args.delete_collection:
            delete_collection(client, args.delete_collection)
        elif args.delete_all:
            delete_all_collections(client)
        else:
            print("No operation specified. Use --help for options.")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        client.close()


if __name__ == '__main__':
    main()