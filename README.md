# 🤖 AI-Powered Knowledge Assistant

An intelligent assistant that answers questions based on your **local documents** using **Retrieval-Augmented Generation (RAG)**, with tool integrations for **web search** and **weather information**.

---

## 🔍 Features

- 📄 Load and process documents in **PDF, TXT, and DOCX** formats
- ✂️ Split documents into manageable chunks for better semantic search
- 🧠 Use **FAISS vector store** and **HuggingFace embeddings** for document retrieval
- ⚡ Generate answers using **Google Gemini 2.0 Flash** via LangChain's Generative AI interface
- 🛠️ Built with **LangChain Agents** using a ReAct-based tool architecture

### 🚀 Tools Included

- **Document QA Tool** – Answers questions using your uploaded documents
- **Weather Tool** – Provides current temperature for a given city
- **DuckDuckGo Search Tool** – Performs web search if the local context is insufficient
- 🧠 Intelligent tool selection based on question context using LangChain’s ReAct agent framework

### 💬 Interface

- CLI (Command Line Interface) for interactive Q&A sessions

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/bantee-sharma/knowledge-assistant.git
cd knowledge-assistant
```


## 2.Create and activate a virtual environment (optional but recommended):
**On Linux/macOS**
```bash
python3 -m venv venv

source venv/bin/activate
```


**On Windows**
```bash
python -m venv venv

venv\Scripts\activate
```

## 4.Install the required Python packages:
```bash
pip install -r requirements.txt
```

## 5.Set up environment variables:
   **Create a .env file in the root directory and add your API keys or configuration, for example:**
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## 6.Run the assistant:
```bash
python main.py
```
## 7.How to Use
**Ask your question directly in the command line after launching the app**

**Type exit or quit to end the session**

## 📁 Example Folder Structure
```bash
knowledge-assistant/
├── docs/                  # Folder to store your local documents (PDF, TXT, DOCX)
├── main.py                # Main script file
├── .env                   # Environment variable file
├── requirements.txt       # Python dependencies
└── README.md              # This file
```
## Sample Questions and Responses
```bash
Knowledge Assistant ready! Type 'exit' to quit.
Ask question: hii


> Entering new AgentExecutor chain...
This is not a question that requires the use of any tools.
Final Answer: Hello!

> Finished chain.
AI:  {'input': 'hii', 'output': 'Hello!'}
Ask question: what is primary key


> Entering new AgentExecutor chain...
This is a database concept question. I should use the doc_qa_tool to answer.
Action: doc_qa_tool
Action Input: what is a primary key in a database?A primary key is a field in a table that uniquely identifies each row/record in a database table. Primary keys must contain unique values and cannot have NULL values.I now know the final answer
Final Answer: A primary key is a field in a table that uniquely identifies each row/record in a database table. Primary keys must contain unique values and cannot have NULL values.

> Finished chain.
AI:  {'input': 'what is primary key', 'output': 'A primary key is a field in a table that uniquely identifies each row/record in a database table. Primary keys must contain unique values and cannot have NULL values.'}
Ask question: exit
Exiting...
```
## Limitations and Next Steps

Currently, the weather tool returns a static temperature; integrating a real weather API would improve accuracy.

UI could be enhanced beyond CLI for better user experience.

Add more tools like calculator or calendar for expanded capabilities.
