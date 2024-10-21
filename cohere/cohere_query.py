from pinecone import Pinecone, ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "cohere-test"
index = pc.Index(index_name)

embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

results = vector_store.similarity_search(
    "What are the fees for medical attestation. Is there any name for this service?",
    k=5,
)
print("results", results)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")