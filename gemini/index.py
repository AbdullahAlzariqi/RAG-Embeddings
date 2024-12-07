from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from collections import Counter
from langchain_core.documents import Document
import json
from pathlib import Path

path = Path("url_content_mapping.json")



embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=os.getenv("GOOGLE_API_KEY"),
                                           task_type="QUESTION_ANSWERING")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    is_separator_regex=False,
)



with path.open('r', encoding='utf-8') as file:
  data = json.load(file)

""" data is a list of dictionaries, each dictionary has the following keys:"""

docs=[]

for item in data:
  chunks = text_splitter.split_text(item['content'])
  for chunk in chunks:
    doc = Document(page_content=chunk, metadata={"source": item['url']})
    docs.append(doc)

print(len(docs))
print(docs[0])

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "gemini-test"  

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)






vector_store.add_documents(documents=docs, ids=[str(id) for id in range(len(docs))])