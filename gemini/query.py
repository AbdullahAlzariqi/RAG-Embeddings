from pinecone import Pinecone
import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import json
load_dotenv()

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
print("results length", len(results))
data_json = []
for res in results:
    data_json.append({"text":res.page_content,"metadata":""})
    print(f"* {res.page_content} [{res.metadata}]")

json_object = json.dumps(data_json, indent=2)
 
# Writing to sample.json
with open("./results/results.json", "w") as outfile:
    outfile.write(json_object)
