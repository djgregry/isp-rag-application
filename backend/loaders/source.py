import os
import requests
from bs4 import BeautifulSoup
import weaviate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from vector_db import *
import argparse
from pathlib import Path
import json

from backend.loaders.vector_db import *

# Load environment variables
load_dotenv()

# Environment variables
HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"X-HuggingFace-Api-Key": HUGGINGFACE_KEY}

MIN_SIZE = 500
MAX_SIZE = 10500

MD_HEADERS = {
    "h1": "#",
    "h2": "##",
    "h3": "###",
    "h4": "####",
    "h5": "#####",
    "h6": "######",
}

class HTMLChunker:

    def __init__(self, min_length=1000, max_length=10500):
        self.min_length = min_length
        self.max_length = max_length


    def confirm(self, prompt):
        while True:
            ans = input(f"\nInclude this section? (y/n/q)\n\n{prompt[:1000]}...\n\n> ").strip().lower()
            if ans == "y":
                return True
            elif ans == "n":
                return False
            elif ans == "q":
                raise KeyboardInterrupt("User quit the filtering process.")
            else:
                print("Please answer with 'y', 'n', or 'q'.")


    def _get_elements(self, soup):
        elements = []
        current_sect = None

        SKIP_CONTAINERS = ["nav", "footer", "aside", "script", "style", "noscript", "button"]

        for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):

            if any(elem.find_parent(tag) for tag in SKIP_CONTAINERS): # Avoid text in useless html elements
                continue

            if elem.name in MD_HEADERS:
                if current_sect and '\n\n' in current_sect[1] and len(current_sect[1]) > 100:
                    if self.confirm(current_sect[1]):
                        elements.append(current_sect)
                    
                current_sect = (elem.name, f"{MD_HEADERS[elem.name]} {elem.text.strip()}")

            else:
                content = elem.text.strip()
                if content:
                    if current_sect:
                        current_sect = (current_sect[0], current_sect[1] + "\n\n" + content)

        if current_sect and '\n\n' in current_sect[1] and len(current_sect[1]) > 100:
            if self.confirm(current_sect[1]):
                elements.append(current_sect)
        return elements


    def _split_long_text(self, text: str) -> list:
        parts = []
        start = 0
        overlap = self.max_length / 5

        while start < len(text):
            end = min(start + self.max_length, len(text))
            parts.append(text[start:end].strip())
            start += int(self.max_length - overlap)
        return parts


    def html_to_md_chunks(self, html: str) -> list:
        soup = BeautifulSoup(html, features="html.parser")
        elements = self._get_elements(soup)

        full_text = "\n\n".join(text for _, text in elements).strip()
        if self.min_length <= len(full_text.replace(' ','')) <= self.max_length:
            return [full_text]

        header_levels = list(MD_HEADERS.keys())
        current_content = {h: "" for h in header_levels}   
        current_tag = "h6" # Default lowest tag     
        chunks = []


        def load_chunk(content: dict) -> list:
            nonlocal current_tag
            chunk = "".join(content[h] for h in header_levels if content[h])
            chunk = chunk.strip()

            if len(chunk) > self.max_length:
                current_level = int(current_tag[1])
                for h in header_levels:
                    if int(h[1]) < current_level and content[h]:
                        content[h] = ""
                        return load_chunk(content) # try with greatest ancestor gone
                
                # All ancestors gone and still too long to split, fallback to text split
                return self._split_long_text(chunk)
            
            elif len(chunk) >= self.min_length:
                return [chunk]
        
            return []
        

        def update_hierarchy(tag, text):
            nonlocal current_tag, current_content
            current_tag = tag
            level = int(tag[1])
            current_content[tag] = text + "\n\n"

            # Clear all tags lower on the hierarchy than current_tag
            for h in header_levels:
                if int(h[1]) > level:
                    current_content[h] = ""


        for tag, text in elements:
            chunks.extend(load_chunk(current_content.copy()))
            update_hierarchy(tag, text.strip())
        
        chunks.extend(load_chunk(current_content.copy()))
        return chunks



def chunk_web_data(client: weaviate.Client, url: str, collection_name: str):
    """
    Add webpage data from URL or local file to the Weaviate collection.
    """ 
    setup_weaviate(client, collection_name)
    collection = client.collections.get(collection_name)
        
    model = SentenceTransformer("neuml/pubmedbert-base-embeddings")
    
    chunker = HTMLChunker(MIN_SIZE, MAX_SIZE)

    if url.startswith("http"):
        response = requests.get(url)
        if response.status_code != 200:
            print("Unable to load source.")
            return 
        html = response.text

    elif url.startswith("file://"):
        path = Path(url.replace("file://", ""))
        html = path.read_text(encoding="utf-8")
    else:
        raise ValueError("Unsupported source type.")
    
    chunks = chunker.html_to_md_chunks(html)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(json.dumps(chunk))
        embedding = model.encode(chunk)
        collection.data.insert(
            properties={"abstract": chunk, "url": url},
            vector=embedding.tolist()
        )
        print("Successfully loaded into Weaviate collection.")

    print("All chunks loaded successfully!")


def add_sources(client: weaviate.Client):
    try:
        setup_weaviate(client, 'Articles')
        setup_weaviate(client, 'Sources')

        source_collection = client.collections.get('Sources')
        pubmed_collection = client.collections.get('Articles')

        with pubmed_collection.batch.fixed_size(batch_size=200) as batch:
            for item in source_collection.iterator(
                include_vector=True
            ):
                batch.add_object(
                    properties=item.properties,
                    vector=item.vector['default']
                )

    except Exception as e:
        print(f"Error occured loading 'Sources' collection into 'Articles' collection: {e}")
    
    


def main():
    parser = argparse.ArgumentParser(description="Medical Page Operations Script")
    parser.add_argument("--load-chunks", type=str, help="Generate chunks from given URL or HTML file. Load into Weaviate collection.")
    parser.add_argument("--add-sources", action="store_true", help="Add 'Sources' to main 'Articles' collection.")
    parser.add_argument("--delete-collection", action="store_true", help="Delete the 'Sources' collection.")

    args = parser.parse_args()

    try:
        client = weaviate.connect_to_local(grpc_port=50052, headers=HEADERS)

        if args.load_chunks:
            chunk_web_data(client, args.load_chunks, "Sources")

        elif args.add_sources:
            add_sources(client)

        elif args.delete_collection:
            delete_collection(client, "Sources")
    
        else:
            print("No operation specified. Use --help for options.")

    except Exception as e:
        print("Error in main execution:", e)

    finally:
        client.close()


if __name__ == "__main__":
    main()