from pinecone import Pinecone
import os
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import json
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "cohere-test"
index = pc.Index(index_name)

embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

results = vector_store.similarity_search(
    "What is the name of the service that enables employees of federal entities to obtain approval for their sick leaves?",
    k=10,
)
print("results length", len(results))
for result in results:
    print(f"id : {result.id}, \nsource : {result.metadata["source"]}\nContent : {result.page_content}")
