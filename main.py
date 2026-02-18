import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---

# Path to the source text file
TEXT_FILE_PATH = "speech.txt"

# Path to the local vector store
VECTOR_DB_PATH = "chroma_db" 

# Embedding model (local, no API key needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model (local, using Ollama)
OLLAMA_MODEL = "mistral"

# --- RAG Pipeline Components ---

def load_and_split_documents(file_path):
    """
    Step 1 & 2: Load the text file and split it into chunks.
    """
    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Document split into {len(chunks)} chunks.")
    return chunks

def get_or_create_vectorstore(chunks, embeddings):
    """
    Step 3: Create embeddings and store them in a Chroma vector store.
    """
    if os.path.exists(VECTOR_DB_PATH):
        print(f"Loading existing vector store from {VECTOR_DB_PATH}...")
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=VECTOR_DB_PATH
        )
        vectorstore.persist()
        print(f"Vector store created and saved to {VECTOR_DB_PATH}.")
        
    return vectorstore

def format_docs(docs):
    """Helper function to format retrieved documents."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(vectorstore):
    """
    Step 4 & 5: Create the RAG chain using LCEL (LangChain Expression Language).
    """
    print("Creating RAG chain...")
    
    # Initialize the Retriever
    retriever = vectorstore.as_retriever()
    
    # Initialize the LLM
    llm = ChatOllama(model=OLLAMA_MODEL)
    
    # Create a Prompt Template
    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Base your answer *only* on the provided context.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    # Create the RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain created successfully.")
    return rag_chain, retriever

# --- Main Execution ---

def main():
    """
    Main function to run the Q&A system.
    """
    try:
        # --- Setup ---
        print("Initializing RAG pipeline...")
        
        # Load and split documents
        chunks = load_and_split_documents(TEXT_FILE_PATH)
        
        # Initialize embeddings model
        print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Create or load the vector store
        vectorstore = get_or_create_vectorstore(chunks, embeddings)
        
        # Create the QA chain
        qa_chain, retriever = create_qa_chain(vectorstore)

        print("\n--- RAG Q&A System ---")
        print("Setup complete. You can now ask questions about the speech.")
        print("Type 'exit' or 'quit' to stop.")
        
        # --- Q&A Loop ---
        while True:
            query = input("\nQuestion: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            if not query.strip():
                continue
                
            # --- Run the RAG chain ---
            try:
                print("Retrieving and generating answer...")
                
                # Get the answer
                answer = qa_chain.invoke(query)
                
                print("\nAnswer:")
                print(answer)
                
            except Exception as e:
                print(f"An error occurred while processing your question: {e}")

    except FileNotFoundError:
        print(f"Error: The file '{TEXT_FILE_PATH}' was not found.")
        print("Please make sure it is in the same directory as main.py.")
    except Exception as e:
        print(f"An unexpected error occurred during setup: {e}")
        print("Please ensure Ollama is running and 'mistral' is pulled.")
        print("Run 'ollama list' in your terminal to check.")

if __name__ == "__main__":
    main()