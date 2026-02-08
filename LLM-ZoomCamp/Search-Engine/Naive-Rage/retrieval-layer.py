import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

voyageai_key = os.getenv("VOYAGE_API_KEY")
# anthropic_client = anthropic.Anthropic(api_key="ANTHROPIC_API_KEY")

voyage_ef = embedding_functions.VoyageAIEmbeddingFunction(
    api_key=voyageai_key,
    model_name="voyage-3" # Latest general-purpose model
)


client = chromadb.PersistentClient(path="./my_local_db")
collection = client.get_or_create_collection(
    name="my_claude_chunks",
    embedding_function=voyage_ef
)




# results = collection.query(
#     query_texts=["What was Conover concerns about writers jobs"],
#     n_results=1
# )

# 2. Retrieve context from Chroma
query_text = "Throw light on saint Michaels prayer in a way I can understand to make my life better"
results = results = collection.query(
    query_texts=[query_text],
    n_results=1
)

# Flatten the chunks into one context block
context = "\n\n".join(results["documents"][0])

print(context)

# 3. Build the "Thoughtful" Prompt
prompt = f"""
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say you don't know.

<context>
{context}
</context>
"""

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4o"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": context,
        },
        {
            "role": "user",
            "content": query_text,
        }
    ],
    model=model_name,
    stream=True,
    stream_options={'include_usage': True}
)

usage = None
print("\n" +  "#"*20 + "AI assistants answer" + "#"*20 + "\n" )
for update in response:
    if update.choices and update.choices[0].delta:
        print(update.choices[0].delta.content or "", end="")
    if update.usage:
        usage = update.usage

if usage:
    print("\n")
    for k, v in usage.model_dump().items():
        print(f"{k} = {v}")