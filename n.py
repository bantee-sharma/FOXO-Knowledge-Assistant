from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun,tool
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
from langchain.memory import ConversationBufferMemory

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
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

    documents = loader.load()
    all_documents.extend(documents)

    return all_documents

doc = my_docs("docs")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunks = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(search_type="similarity",kwargs={"k":3})

@tool
def doc_qa_tool(question:str)->str:
    '''Answer question from the following context'''
    retrieve_docs = retriever.invoke(question)

    context = "".join([i.page_content for i in retrieve_docs])
    qa_prompt = PromptTemplate(
    template='''Yor are a helpfull AI assistant. Answer the question from the following context.
    If the answer is not present in the context, respond with: "The answer is not available in the provided context.
    Context: {context}
    Question: {question}
    Answer: ''',
    input_variables=["context","question"])

    finla_prompt = qa_prompt.invoke({"context":context,"question":question})

    response = llm.invoke(finla_prompt)
    return response.content

@tool
def weather(city:str)->str:
    "Provides weather information for the given city."
    return f"The current temperature in {city} is 25°C."

search_tool = DuckDuckGoSearchRun()

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[doc_qa_tool,weather,search_tool],
    prompt=prompt
)

agent_executer = AgentExecutor(
    agent=agent,
    tools=[doc_qa_tool,weather,search_tool],
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)





print("Knowledge Assistant ready! Type 'exit' to quit.")


while True:
    question = input("Ask Question: ").strip().lower()
    
    if question in ["exit","quit"]:
        print("Exiting the session. Thank you for using the assistant!")
        break
    
    else:
        res = agent_executer.invoke({"input" :question})
        
        print(res["output"])


        