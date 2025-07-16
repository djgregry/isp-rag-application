import weaviate
from weaviate.classes.config import Configure, Property, DataType

def setup_weaviate(client: weaviate.Client, collection: str):
    """
    Set up the Weaviate database.

    Args:
        client (weaviate.Client): Weaviate client instance.
        collection (str): Weaviate collection name.
    """
    if not client.collections.exists(collection):
        client.collections.create(
            collection,
            vectorizer_config = Configure.Vectorizer.none(),
            properties=[
                Property(name="abstract", data_type=DataType.TEXT),
                Property(name="url", data_type=DataType.TEXT, skip_vectorization=True)
            ]
        )
        print("Collection created successfully.")


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