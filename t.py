from langchain_community.document_loaders import PyMuPDFLoader,TextLoader,UnstructuredWordDocumentLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import tool
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub

# Load environment variables
load_dotenv()

# load llm
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

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

        document = loader.load()
        all_documents.extend(document)
    return all_documents

doc = my_docs("docs")

# Text splitting
text_splitters = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)
chunks = text_splitters.split_documents(doc)

# Embeddings and Vector store creation
embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks,embeddings)

#Retriever
retriever = vector_store.as_retriever(search_type="similarity",kwargs={"k":1})

@tool
def doc_qa_tool(question:str)->str:
    "Answers questions using document context."
    retrieve_docs = retriever.invoke(question)
    context = " ".join([i.page_content for i in retrieve_docs])
    
    qa_prompt = PromptTemplate(
    template='''You are a helpfull assistant.
                Answer the question from the following context.
                if context is insufficient just say, I don;t Know.
                Context: {context}
                Question:{question}
                Answer:''',
                input_variables=["context","question"])
    
    final_prompt = qa_prompt.invoke({"context":context,"question":question})
        
        # LLM response
    result = llm.invoke(final_prompt)
    return result.content
    

@tool
def weather(city:str)->str:
    "Weather of city"
    return  f"The Weather of {city} is 25°C"

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[weather,doc_qa_tool],
    prompt=prompt
)

agent_exe = AgentExecutor(
    agent=agent,
    tools=[weather,doc_qa_tool],
    verbose=True
)

# CLI
print("Knowledge Assistant ready! Type 'exit' to quit.")

while True:
    question = input("Ask Question: ").strip()
    if question.lower() in ["exit","quit"]:
        print("Exiting...")
        break
    else:
        response = agent_exe.invoke({"input":question})
        print(response)