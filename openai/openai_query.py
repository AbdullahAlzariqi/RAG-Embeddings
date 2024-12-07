from pinecone import Pinecone, ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import json
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "openai-test"
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

results = vector_store.similarity_search(
    "What are the required documents to apply for the service the enables employees to apply and approve their applications but for private entity employees? ",
    k=10,
)
print("results length", len(results))
for result in results:
    print(f"id : {result.id}, \nsource : {result.metadata["source"]}\nContent : {result.page_content}")
