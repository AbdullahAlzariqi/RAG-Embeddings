from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("Cleaned_MOHAP.txt", encoding='utf-8') as f:
    MOHAP_data = f.read()
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=528,
    chunk_overlap=102,
    is_separator_regex=False,
)
texts = text_splitter.split_text(MOHAP_data)

sum = 0
for i, chunk in enumerate(texts):
    sum = len(chunk)+sum

print(sum/len(texts))