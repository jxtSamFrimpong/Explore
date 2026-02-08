import os
from dotenv import load_dotenv
load_dotenv()


#import chromadb
#from openai import OpenAI
#from chromadb.utils import embedding_functions

# token = os.environ["GITHUB_TOKEN"]
# endpoint = "https://models.github.ai/inference"
# model_name = "openai/gpt-4o"

# client = OpenAI(
#     base_url=endpoint,
#     api_key=token,
# )
#
#
# openai_ef = embedding_functions

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    try:
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath) and filename.endswith(".txt"):
                with open(
                        os.path.join(directory_path, filename), "r", encoding="utf-8"
                ) as file:
                    documents.append({"id": filename, "text": file.read()})
    # except PermissionError as e:
    #     print(e)
    # except UnicodeDecodeError as e:
    #     print(e)
    # except IOError as e:
    #     print (e)
    except Exception as e:
        print(e)
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Now, lets Load documents from the directory
directory_path = "./parsed_and_pprocessed_docs"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")


#And then lets break the docs into chunks
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

print(f"Split documents into {len(chunked_documents)} chunks")