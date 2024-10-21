from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore



load_dotenv()

embeddings = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_AI_API_KEY"), model_name="jina-embeddings-v3"
)

# Load example document
with open("Cleaned_MOHAP.txt", encoding='utf-8') as f:
    MOHAP_data = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=128,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([MOHAP_data])


text_content = []
urls = []
for text in texts:
    if text.page_content == "":
        texts.remove(text)
    elif text.page_content.startswith("URL: https:"):
        urls.append(text.page_content)
        texts.remove(text)
    else :
        text_content.append(text)




pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "jina-test"  

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
print(len(text_content))





vector_store.add_documents(documents=text_content, ids=[str(id) for id in range(len(text_content))])
