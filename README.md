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


2. **Create and activate a virtual environment (optional but recommended):**

    python3 -m venv venv

   
    source venv/bin/activate   # On Linux/macOS

   
    venv\Scripts\activate      # On Windows
   

4. **Install the required Python packages:**
    pip install -r requirements.txt

5. **Set up environment variables:**
   Create a .env file in the root directory and add your API keys or configuration, for example:

   GOOGLE_API_KEY=your_google_api_key_here

6. **Run the assistant:**
   python main.py

7. **Interact with the assistant in the command line. Type your question and press Enter. Type exit or quit to stop.**
   



