from pinecone import Pinecone
import os
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
import json
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "voyageai-test"
index = pc.Index(index_name)

embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.getenv("VOYAGEAI_API_KEY"), model="voyage-multilingual-2"
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

results = vector_store.similarity_search(
    "What are the fees for medical attestation. Is there any name for this service?",
    k=5,
)
print("results", results)
data_json = []
for res in results:
    data_json.append({"text":res.page_content,"metadata":""})
    print(f"* {res.page_content} [{res.metadata}]")

json_object = json.dumps(data_json, indent=2)
 
# Writing to sample.json
with open("./results/voyage_ai_results.json", "w") as outfile:
    outfile.write(json_object)
