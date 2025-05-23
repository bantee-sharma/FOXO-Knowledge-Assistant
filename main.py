from langchain_community.document_loaders import TextLoader,PyMuPDFLoader,UnstructuredWordDocumentLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import


def my_docs(folder_path):
    all_documents = []
    for file in Path(folder_path).glob("*"):
        if file.suffix == ".pdf":
            loader = PyMuPDFLoader(str(file))
        elif file.suffix == ".txt":
            loader = TextLoader(str(file),encoding="utf-8")
        elif file.suffix == ".docs":
            loader = UnstructuredWordDocumentLoader(str(file))
        else:
            continue

        documents = loader.load()
        all_documents.extend(documents)

    return all_documents

doc = my_docs("docs")

text_spiltter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunks = text_spiltter.split_documents(doc)

print(len(chunks))