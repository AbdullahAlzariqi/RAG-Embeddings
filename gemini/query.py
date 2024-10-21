from pinecone import Pinecone, ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "gemini-test-full"
index = pc.Index(index_name)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_document")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

results = vector_store.similarity_search(
    "What are the fees for medical attestation. Is there any name for this service?",
    k=5,
)
print("results", results)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")