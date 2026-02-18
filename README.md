# Local RAG Chatbot

This project is a simple command-line Question & Answering (Q&A) system built as part of an intern task. It uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on a provided text document.

The entire system runs 100% locally, requires no API keys, and uses open-source tools.

## Tech Stack

* **Python 3.8+**
* **Framework:** [LangChain](https://www.langchain.com/)
* **LLM:** [Ollama](https://ollama.com/) with **Mistral 7B** (100% local and free)
* **Embeddings:** [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (local, in-memory/on-disk)

## Setup and Installation

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

Before you begin, you **must** have the following installed:

* **Python 3.8** or newer.
* **Git** (for cloning the repository).
* **Ollama**: This is the most important dependency. It runs the Mistral 7B model locally.
    1.  Go to [ollama.com](https://ollama.com/) and download the application for your OS (macOS, Linux, or Windows).
    2.  Install it.
    3.  After installation, open your terminal and pull the Mistral model:
    ```bash
   ollama pull mistral
    ```
    4.  You can verify it's working by running `ollama list` in your terminal. You should see `mistral` in the list. The Ollama application must be **running** in the background for the Python script to work.

### 2. Clone the Repository
Clone this public repository to your local machine.
```bash
git clone https://github.com/ashankgupta/local-rag-chatbot.git
cd local-rag-chatbot
```

### 3. Set up a Virtual Environment
It's best practice to use a virtual environment to manage dependencies.
```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate
```
### 4. Install Dependencies
Install all the required Python libraries from the requirements.txt file.

```bash
pip install -r requirements.txt
```
## How to Run

Once you have completed the setup:
- Ensure Ollama is running: Make sure the Ollama desktop application is open, or the service is running.
- Run the Python script:

```bash
python main.py
```
The script will first:
- Load the speech.txt file.
- Load the HuggingFace embedding model (this may take a moment on first run).
- Create and save a local vector database in a new directory named chroma_db/

After the one-time setup, you will see a prompt. You can now ask questions!
```bash
Initializing RAG pipeline...
Loading document from speech.txt...
Document split into 1 chunks.
Loading embedding model 'sentence-transformers/all-MiniLM-L6-v2'...
Creating new vector store...
Vector store created and saved to chroma_db.
Creating RAG chain...
RAG chain created successfully.

--- Local Q&A System ---
Setup complete. You can now ask questions about the speech.
Type 'exit' or 'quit' to stop.

Question: What is the real remedy?

Retrieving and generating answer...

Answer:
Based on the provided context, the real remedy is to destroy the belief in the sanctity of the shastras.

Question: What is the capital of France?

Retrieving and generating answer...

Answer:
I don't know.

(This is the correct behavior, as the answer is not in the provided text.)
```
To stop the program, type exit or quit.

## Project Structure
```bash
local-rag-chatbot/
├── .git/
├── chroma_db/        # (This will be created on first run to store vectors)
├── main.py           # (The main Python script)
├── speech.txt        # (The source text data)
├── requirements.txt  # (List of Python dependencies)
└── README.md         # (Info about Project)
```
