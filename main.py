from langchain_community.document_loaders import TextLoader,PyMuPDFLoader,UnstructuredWordDocumentLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")


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

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks,embeddings)

retriever = vector_store.as_retriever(search_type="similarity",kwargs={"k":3})

@tool
def doc_qa_tool(question:str)->str:
    '''Answer question from the following context'''
    retriever_docs = retriever.invoke(question)
    context = " ".join([i.page_content for i in retriever_docs])

    qa_prompt = PromptTemplate(
    template='''You are a helpfull AI assistant. Answer the question from the following context.
                if context is insufficient just say, I don;t Know.
                Context:{context}
                Question:{question}
                Answer: ''',
                input_variables=["context","question"])

    final_prompt = qa_prompt.invoke({"context":context,"question":question})

    result = llm.invoke(final_prompt)
    return result.content

@tool
def weather(city:str)->str:
    "Weather of the city"
    return f"The current temperature in {city} is 25°C."

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[doc_qa_tool,weather],
    prompt=prompt
)

agent_executer = AgentExecutor(
    agent=agent,
    tools=[weather,doc_qa_tool],
    verbose=True
)

print("Knowledge Assistant ready! Type 'exit' to quit.")

while True:
    question = input("Ask question: ").strip()
    if question.lower() in ["exit","quit"]:
        print("Exiting...")
        break
    else:
        response = agent_executer.invoke({"input":question})
        print("AI: ",response)
   
        
        



