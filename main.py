from langchain.document_loaders import PyMuPDFLoader,PyPDFLoader



loader = PyMuPDFLoader("SQL Revision Notes.pdf")

docs = loader.load()

print(docs)

