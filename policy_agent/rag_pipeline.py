import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Remove: from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (API Keys)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_policy_index")

def build_vector_database(pdf_directory: str = os.path.join(BASE_DIR, "policies")):
    """
    Reads all PDFs in the directory, chunks them, generates embeddings, 
    and saves the FAISS vector database locally.
    Run this ONLY ONCE when you add new policy PDFs.
    """
    print(f"Loading PDFs from '{pdf_directory}'...")
    documents = []
    
    # Iterate through all PDFs in the folder
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created '{pdf_directory}' folder. Please add PDFs and run again.")
        return

    for file in os.listdir(pdf_directory):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_directory, file))
            documents.extend(loader.load())
            
    if not documents:
        print("No PDFs found. Please drop an insurance policy PDF in the folder.")
        return

    print(f"Loaded {len(documents)} pages. Chunking text...")
    
    # We chunk by paragraphs/sections to keep the context intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print("Generating embeddings in batches to avoid API rate limits...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = None
    batch_size = 15 # Send 30 chunks at a time
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        
        # If it's the first batch, create the database. Otherwise, add to it.
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
            
        print(f"Processed {i + len(batch)} / {len(chunks)} chunks...")
        
        # Pause for 10 seconds to respect Google's free tier limits
        if i + batch_size < len(chunks):
            print("Pausing for 10 seconds for API limits...")
            time.sleep(15) 
            
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Success! Vector database saved locally to '{FAISS_INDEX_PATH}'.")


def retrieve_policy_clauses(query: str, top_k: int = 3) -> list[str]:
    """
    Takes a search query (e.g., 'Cardiac Arrest coverage limit') and returns 
    the exact policy text chunks from the vector database.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Run build_vector_database() first.")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True # Required for local FAISS loads in Langchain
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    
    # Return just the text content for the Agent to read
    return [doc.page_content for doc in docs]

# ==========================================
# 3. TEST THE RAG SYSTEM
# ==========================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        # Run this in terminal to build DB: python rag_pipeline.py --build
        build_vector_database()
    else:
        # Run this in terminal to test search: python rag_pipeline.py
        try:
            print("--- Testing Policy Retrieval ---")
            test_query = "What is the policy on Cardiac Arrest and Heart Disease?"
            print(f"Query: {test_query}\n")
            
            results = retrieve_policy_clauses(test_query)
            
            for i, result in enumerate(results, 1):
                print(f"--- Retrieved Clause {i} ---")
                print(result)
                print("-" * 40)
        except Exception as e:
            print(f"Error: {e}")