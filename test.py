from langchain_community.document_loaders import PyMuPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


folder_path = "docs"

def my_docs(folder_path):
    all_docs = []

    for file in Path(folder_path).glob("*"):
        if file.suffix == ".pdf":
            loader = PyMuPDFLoader(str(file))
        elif file.suffix == ".txt":
            loader = TextLoader(str(file),encoding="utf-8")
        else:
            continue

        documents = loader.load()
        all_docs.extend(documents)
    return all_docs

docs = my_docs(folder_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,chunk_overlap=100
)

chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(search_type="similarity",kwargs={"k":3})

prompt = PromptTemplate(
    template='''You are a helpfull AI Assistant.
    Answer the question from the following context. If context is insufficient just say, I don't know.
    {context}
    Question:{question}
    Answer:''',
    input_variables=["context","question"]
)

print("Knowledge Assistant ready! Type 'exit' to quit.")

while True:
    question = input("Ask Question: ").strip()
    if question.lower() in ["exit","quit"]:
        print("Exiting...")
        break
    else:
        
        retrieve_docs = retriever.invoke(question)

# context = " ".join([i.page_content for i in retrieve_docs])

        context = ""
        for doc in retrieve_docs:
            source = doc.metadata.get("source","Unknown Source")
            content = doc.page_content
            context += f"\n[Source: {source}\n{content}\n]"


        final_prompt = prompt.invoke({"context":context,"question":question})

        result = llm.invoke(final_prompt)
        print(result.content)