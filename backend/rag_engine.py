import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialize environment variables and secure keys
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Define where our local vector database will save its files
DB_FAISS_PATH = "vectorstore/db_faiss"

def process_pdf(file_path: str):
    """
    Step A: Read the PDF, break it into smaller chunks, 
    convert to vectors, and save them locally.
    """
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split text into chunks of 500 characters with 50-character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Initialize the Google Gemini Embedding Model with the updated active model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    # Convert chunks to embeddings and save to a local folder using FAISS
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    print(f"🟩 Success: Processed {len(docs)} text chunks into the database.")
    return len(docs)


def ask_teacher(question: str):
    """
    Step B: Search the local FAISS database for relevant text chunks,
    feed them to Gemini, and get an answer structured like an AI Teacher.
    """
    # 1. Reload the local embedding model and vector database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    # Load the vector store from our disk (allow dangerous deserialization since it's our own local file)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Turn the vector store into a search retriever (fetches top 3 closest chunks matching the question)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 2. Setup our Chat Large Language Model with the required conversion flag
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash", 
        google_api_key=api_key, 
        temperature=0.3,
        convert_system_message_to_human=True  # 🛠️ FIX APPLIED HERE
    )
    
    # 3. Craft the Advanced Hybrid Teacher Personality Prompt
    system_prompt = (
        "You are 'Texteach AI', a patient, friendly, and brilliant school teacher.\n"
        "Your ultimate goal is to break down complex concepts into the simplest possible explanations "
        "so that even a complete beginner can understand and retain them.\n\n"
        
        "CRITICAL INSTRUCTIONS:\n"
        "1. CORE SOURCE: Use the provided document context below as your primary source of facts and guidelines.\n"
        "2. COMPREHENSIVE TEACHING: If the context is too dense, technical, or lacks foundational definitions, "
        "you ARE explicitly allowed to use your general knowledge to explain terms, build out vivid real-world analogies, "
        "and provide step-by-step breakdowns.\n"
        "3. GROUNDING: Never contradict the provided context. If a user asks about something completely unrelated to the "
        "document's overall subject matter, politely guide them back to the study topic.\n"
        "4. STYLE: Use short punchy sentences, formatting like bold text, and bullet points to maximize readability.\n\n"
        
        "Document Context:\n{context}"
    )

    
    # 4. Package everything into a LangChain retrieval chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 5. Invoke the chain to get our answer
    response = rag_chain.invoke({"input": question})
    return response["answer"]
