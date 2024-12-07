from pinecone import Pinecone
import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import json
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "gemini-test"
index = pc.Index(index_name)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_document")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

results = vector_store.similarity_search(
    " What are the required documents to apply for the service the enables employees to apply and approve their applications but for private entity employees? ",
    k=10,
)
print("results length", len(results))
for result in results:
    print(f"id : {result.id}, \nsource : {result.metadata["source"]}\nContent : {result.page_content}")

# data_json = []
# for res in results:
#     data_json.append({"text":res.page_content,"metadata":""})
#     print(f"* {res.page_content} [{res.metadata}]")

# json_object = json.dumps(data_json, indent=2)
 
# # Writing to sample.json
# with open("./results/results.json", "w") as outfile:
#     outfile.write(json_object)


