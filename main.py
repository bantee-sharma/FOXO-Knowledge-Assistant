from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

loader = PyMuPDFLoader("SQL Revision Notes.pdf")

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)

chunks = text_splitter.split_documents(document)
embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(search_type="similarity",kwargs={"k":3})

prompt = PromptTemplate(
    template='''You are a helpfull AI Assistant.
    Answer the question from the following context. If Context if insufficient just say, I don't know.
    {context}
    Question:{question}
    Answer:''',
    input_variables=["context","question"]
)



question = "what is sql"
retrieve_docs = retriever.invoke(question)

context = " ".join([i.page_content for i in retrieve_docs])

final_prompt = prompt.invoke({"context":context,"question":question})

result = llm.invoke(final_prompt)
print(result.content)

