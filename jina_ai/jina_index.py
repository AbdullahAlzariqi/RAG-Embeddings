from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import json
from pathlib import Path



load_dotenv()

embeddings = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_AI_API_KEY"), model_name="jina-embeddings-v3"
)

# Load example document
with open("Cleaned_MOHAP.txt", encoding='utf-8') as f:
    MOHAP_data = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=528,
    chunk_overlap=102,
    is_separator_regex=False,
)
path = Path("url_content_mapping.json")

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

index_name = "jina-test"  

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)





vector_store.add_documents(documents=docs, ids=[str(id) for id in range(len(docs))])
