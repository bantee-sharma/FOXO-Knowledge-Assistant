# Import necessary libraries
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
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment
load_dotenv()

# LLM Model
llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

# Load Documents
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

# Load documents from the "docs" directory
doc = my_docs("docs")

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunks = text_splitter.split_documents(doc)

# Embeddings and Vector store creation
embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks,embeddings)

#Retriever
retriever = vector_store.as_retriever(search_type="similarity",kwargs={"k":3})

# Tool Document based Question Answering
@tool
def doc_qa_tool(question:str)->str:
    '''Answer question from the following context'''

    # Retrieve relevant document
    retriever_docs = retriever.invoke(question)

    # Join all retrieved
    context = " ".join([i.page_content for i in retriever_docs])

    # prompt template for answering questions using the context
    qa_prompt = PromptTemplate(
    template='''Yor are a helpfull AI assistant. Answer the question from the following context.
    If the answer is not present in the context, respond with: "The answer is not available in the provided context.
            
    Context:{context}
    Question:{question}
    Answer: ''',
    input_variables=["context","question"])

    # Format the prompt 
    final_prompt = qa_prompt.invoke({"context":context,"question":question})

    # LLM response
    result = llm.invoke(final_prompt)
    return result.content

# Tool Weather Info
@tool
def weather(city:str)->str:
    "Provides weather information for the given city."
    return f"The current temperature in {city} is 25°C."

# Tool Web Search
search_tool = DuckDuckGoSearchRun()

# Load a React-style prompt template for agent reasoning from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create a ReAct agent with the language model and tools
agent = create_react_agent(
    llm=llm,
    tools=[doc_qa_tool,weather,search_tool],
    prompt=prompt
)

# Set up an executor to run the agent
agent_executer = AgentExecutor(
    agent=agent,
    tools=[weather,doc_qa_tool,search_tool],
    verbose=True
)

# CLI
print("Knowledge Assistant ready! Type 'exit' to quit.")

while True:
    question = input("Ask question: ").strip()
    if question.lower() in ["exit","quit"]:
        print("Exiting...")
        break
    else:
        # Invoke the agent executor to get a response to the question
        response = agent_executer.invoke({"input":question})
        print("AI: ",response)
   
        
        

