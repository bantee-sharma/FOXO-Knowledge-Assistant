from langchain_community.document_loaders import PyMuPDFLoader,TextLoader,UnstructuredWordDocumentLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

folder_path = "docs"

def my_docs(folder_path):
    all_documents = []

    for file in Path(folder_path).glob("*"):
        if file.suffix == ".pdf":
            loader = PyMuPDFLoader(str(file))
        elif file.suffix == ".txt":
            loader = TextLoader(str(file),encoding="utf-8")
        elif file.suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(str(file))
        else:
            continue

        document = loader.load()
        all_documents.extend(document)
    return all_documents

doc = my_docs("docs")

text_splitters = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap=100
)

chunks = text_splitters.split_documents(doc)

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(search_type="similarity",kwargs={"k":1})

prompt = PromptTemplate(
    template='''You are a helpfull assistant.
                Answer the question from the following context.
                if context is insufficient just say, I don;t Know.
                Context: {context}
                Question:{question}
                Answer:''',
                input_variables=["context","question"]
)


while True:
    question = input("Ask Question: ").strip()
    if question.lower() in ["exit","quit"]:
        print("Exiting...")
        break
    else:
        retrieve_docs = retriever.invoke(question)

        context = ([i.page_content for i in retrieve_docs])

        final_prompt = prompt.invoke({"context":context,"question":question})

        result = llm.invoke(final_prompt)

        print(result.content)