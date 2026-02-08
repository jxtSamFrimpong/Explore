import os
from dotenv import load_dotenv
#import voyageai
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions


openai_key = os.getenv("GITHUB_TOKEN")
gemini_key = os.getenv("GEMINI_API_KEY")
voyageai_key = os.getenv("VOYAGE_API_KEY")
endpoint = "https://models.github.ai/inference"
#endpoint = "http://52.31.193.32:7006"

# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=openai_key, model_name="text-embedding-3-small"
# )
# # Initialize the Chroma client with persistence
# chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
# collection_name = "document_qa_collection"
# collection = chroma_client.get_or_create_collection(
#     name=collection_name, embedding_function=openai_ef
# )


# client = OpenAI(
#     base_url=endpoint,
#     api_key=openai_key,
# )

from make_chunks import chunked_documents
print(chunked_documents[0])


# Function to generate embeddings using OpenAI API
def get_claude_embeddings(chunks_list, collection):
    # 3. Add chunks (Chroma calls Voyage AI automatically)
    collection.add(
        documents=[i["text"] for i in chunked_documents],
        ids=[i["id"] for i in chunked_documents]
    )

voyage_ef = embedding_functions.VoyageAIEmbeddingFunction(
        api_key=voyageai_key,
        model_name="voyage-3"  # Latest general-purpose model
)
client = chromadb.PersistentClient(path="./my_local_db")
collection = client.get_or_create_collection(
    name="my_claude_chunks",
    embedding_function=voyage_ef
)
get_claude_embeddings(chunked_documents, collection)
