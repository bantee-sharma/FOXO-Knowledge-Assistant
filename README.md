# AI-Powered Knowledge Assistant

An intelligent AI assistant that answers questions based on your local documents using Retrieval-Augmented Generation (RAG), with additional tool integrations for web search and weather information.

---

## Features

- Load and process documents in **PDF**, **TXT**, and **DOCX** formats
- Split documents into manageable chunks for better semantic search
- Use **FAISS** vector store and **HuggingFace embeddings** for document retrieval
- Generate responses using **Google Gemini 2.0 Flash** (via LangChain Google Generative AI)
- Tool-based architecture with LangChain agents:
  - **Document QA tool:** Answers questions using uploaded documents
  - **Weather tool:** Provides current temperature for a given city
  - **DuckDuckGo Search tool:** Performs web searches when local context is insufficient
- Intelligent tool selection based on question context using LangChain React agent framework
- Command-line interface (CLI) for interactive question and answer sessions

---

## Setup Instructions

1. **Clone the repository:**

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
   



