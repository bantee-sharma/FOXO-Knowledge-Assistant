from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
loader = PyMuPDFLoader("SQL Revision Notes.pdf")

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)

chunks = text_splitter.split_documents(document)
embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(search_type="similarity",kwargs={"k":3})

question = "what is sql"
retrieve_docs = retriever.invoke(question)

context = " ".join([i.page_content for i in retrieve_docs])
print(context)
