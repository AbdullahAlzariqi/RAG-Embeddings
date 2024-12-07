from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from collections import Counter




load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=os.getenv("GOOGLE_API_KEY"),
                                           task_type="QUESTION_ANSWERING")



# Load example document
with open("/Users/abdullah/Documents/Embedding-Test/RAG-Embeddings/Cleaned_MOHAP.txt", encoding='utf-8') as f:
    MOHAP_data = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=102,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([MOHAP_data])
# print(texts[0].page_content)
# print(texts[1].page_content)
# print(texts[2])

text_content = []
urls = []
seenContent= set()
print(len(texts))

for text in texts:
    if text.page_content == "":
        texts.remove(text)
    # elif text.page_content.startswith("URL: https:"):
    #     urls.append(text.page_content)
    #     texts.remove(text)
    elif text.page_content in seenContent:
        texts.remove(text)
        if len(texts) != len(seenContent):
            print(len(texts))
            print(len(seenContent))
            print("\n")
    else :
        text_content.append(text.page_content)
        seenContent.add(text.page_content)
        if len(texts) != len(seenContent):
            print(len(texts))
            print(len(seenContent))
            print("\n")


counts = Counter(text_content)
duplicates = [string for string, count in counts.items() if count > 1]

print(len(texts))
print(len(text_content))
print(len(seenContent))







# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# index_name = "gemini-test"  

# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)






# vector_store.add_documents(documents=texts, ids=[str(id) for id in range(len(text_content))])
