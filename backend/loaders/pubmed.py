import os
import argparse
from typing import Dict, List
import weaviate
from dotenv import load_dotenv
from Bio import Entrez
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
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


# Solution to limited results on Entrex esearch
def split_date_ranges(start_date: str, end_date: str, days_per_range: int) -> List[Dict[str, str]]:
    date_ranges = []
    current_start = datetime.strptime(start_date, "%Y/%m/%d")
    current_end = min(current_start+timedelta(days=days_per_range), datetime.strptime(end_date,"%Y/%m/%d"))

    # Iterate through the date range, creating smaller chunks
    while current_start < datetime.strptime(end_date, "%Y/%m/%d"):
        date_ranges.append({
            "start": current_start.strftime("%Y/%m/%d"),
            "end": current_end.strftime("%Y/%m/%d")
        })
        current_start = current_end + timedelta(days=1)
        current_end = min(current_start+timedelta(days=days_per_range), datetime.strptime(end_date,"%Y/%m/%d"))

    return date_ranges


def search_pubmed_in_range(query:str, start_date:str = "", end_date:str = ""):
    # Split date ranges for pagination
    try:
        ids = []
        date_ranges = split_date_ranges(start_date, end_date, days_per_range=100)

        for date_range in date_ranges:
            print(f"Fetching articles published {date_range['start']}-{date_range['end']}")
            with Entrez.esearch(
                db="pubmed", 
                term=query, 
                retmax=9999,
                mindate=date_range['start'],
                maxdate=date_range['end']
            ) as search_result:
                record =  Entrez.read(search_result)
                ids.extend(record.get("IdList", []))
        
        return ids
    
    except Exception as e:
        print(f"Error searching articles: {e}")
        return []


def fetch_pubmed_articles(ids: List[str]) -> List[Dict[str, str]]:
    """
    Fetch and structuring PubMed articles given a list of Article IDs

    Args:
        ids (List[str]): List of Articles IDs given from PubMed Database

    Returns:
        List[Dict[str,str]]: A list of articles with abstract and URL.
    """
    with Entrez.efetch(db="pubmed", id=",".join(ids), rettype="gb", retmode="xml") as fetch_id:
        articles = list()
        records = Entrez.read(fetch_id)

        for record in records.get("PubmedArticle", []):
            article = record['MedlineCitation']['Article']
            article_id = record['MedlineCitation']['PMID']

            # Extract article details
            title = article.get("ArticleTitle", "Title Not Available")
            abstract_sections = article.get("Abstract", {}).get("AbstractText", [])
            abstract = "\n\n".join(
                f"## {section.attributes.get('Label', 'SECTION')}\n\n{section}"
                for section in abstract_sections
            )

            # Add article details to the list
            articles.append({
                "abstract": f"## TITLE\n\n{title}\n\n{abstract}",
                "url": f"{PUBMED_BASE_URL}{article_id}"
            })
        
        return articles


def setup_weaviate(client: weaviate.Client):
    """
    Set up the Weaviate database.

    Args:
        client (weaviate.Client): Weaviate client instance.
    """
    if not client.collections.exists("Articles"):
        client.collections.create(
            "Articles",
            vectorizer_config = Configure.Vectorizer.none(),
            properties=[
                Property(name="abstract", data_type=DataType.TEXT),
                Property(name="url", data_type=DataType.TEXT, skip_vectorization=True)
            ]
        )
        print("Collection created successfully.")


def add_data(client: weaviate.Client, query: str):
    """
    Add data to the Weaviate collection.
    """
    setup_weaviate(client)
    collection = client.collections.get("Abstracts")
    
    # Query articles which have and abstract and have a full free text available at PubMed Central
    refined_query = f"{query} AND hasabstract[text] AND \"pubmed pmc\"[sb]"
    ids = search_pubmed_in_range(refined_query, "2000/01/01", "2025/03/01")
    model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

    # Use batch processing for efficient insertion of articles into the Weaviate collection
    with collection.batch.dynamic() as batch:
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            fetch_ids = ids[i:i+batch_size]
            articles = fetch_pubmed_articles(fetch_ids)

            for article in articles:
                embedding = model.encode(article["abstract"]) 
                batch.add_object(
                    properties=article,
                    vector=embedding.tolist()
                )

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
    parser.add_argument("--add-data", type=str, help="Add data from PubMed with respect to a query")
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