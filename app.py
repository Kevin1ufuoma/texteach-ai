import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================================================
# 🎓 TEXTEACH AI - STREAMLIT PRODUCTION APPLICATION
# ==========================================================================

st.set_page_config(page_title="Texteach AI App", page_icon="🎓", layout="wide")

# Directory pathways
DB_FAISS_PATH = "vectorstore/db_faiss"
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Fetch API Key securely from Streamlit Secrets Environment
api_key = st.secrets.get("GEMINI_API_KEY")

# App header layout
st.title("🎓 Texteach AI App")
st.caption("Your patient, smart knowledge management companion breaking down complex topics step-by-step.")

# Maintain persistent conversation states in Streamlit memory loops
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Teacher. Upload a study document on the left, and I will break down its contents using simple analogies. What are we studying today?"}
    ]
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# --------------------------------------------------------------------------
# 📁 SIDEBAR PANEL - KNOWLEDGE INGESTION & CONTROL PANEL
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Step 1: Upload Knowledge")
    uploaded_file = st.file_uploader("Upload a PDF study chapter or article to begin", type=["pdf"])
    
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        # Guard clause to check if this file has already been vectorized
        if not st.session_state.db_ready:
            with st.status("Reading and processing document into vectors...", expanded=True) as status:
                try:
                    # Write file safely to temporary disk workspace
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Parse and slice content
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    docs = text_splitter.split_documents(documents)
                    
                    # Compute vector positions via Google active cloud paths
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
                    db = FAISS.from_documents(docs, embeddings)
                    db.save_local(DB_FAISS_PATH)
                    
                    st.session_state.db_ready = True
                    status.update(label=f"🟩 Successfully processed {len(docs)} text chunks!", state="complete")
                    st.success("Memory updated successfully!")
                except Exception as e:
                    status.update(label="❌ Ingestion framework error.", state="error")
                    st.error(f"Error details: {str(e)}")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        
    st.markdown("---")
    st.subheader("💡 Learning Tips")
    st.info("Ask the teacher to break dense concepts down using analogies or request a short quiz to test your retention!")
    
    # 🔄 CLEAR SESSION FEATURE
    if st.button("🔄 Clear Session / New File", use_container_width=True):
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        st.session_state.messages = [
            {"role": "assistant", "content": "Memory wiped successfully! Ready for your next document. Upload it on the left whenever you are ready!"}
        ]
        st.session_state.db_ready = False
        st.rerun()

# --------------------------------------------------------------------------
# 💬 MAIN AREA - ACTIVE TEACHER CHAT VIEWPORT
# --------------------------------------------------------------------------
st.subheader("💬 Step 2: Ask Your Teacher")

# Render active chat timeline out of session state registers
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Lock dialogue input box if memory vectors haven't finished compiling
if not st.session_state.db_ready:
    st.chat_input("Please upload a PDF file to activate the teacher conversation...", disabled=True)
else:
    if user_query := st.chat_input("Ask a targeted question about your document..."):
        # Append student message bubble immediately
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # RAG pipeline retrieval and response loop
        with st.chat_message("assistant"):
            with st.spinner("Thinking and searching memory data..."):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
                    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
                    retriever = db.as_retriever(search_kwargs={"k": 3})
                    
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.8-flash", 
                        google_api_key=api_key, 
                        temperature=0.3,
                        convert_system_message_to_human=True
                    )
                    
                    system_prompt = (
                        "You are 'Texteach AI', a patient, friendly, and brilliant school teacher.\n"
                        "Your ultimate goal is to break down complex concepts into the simplest possible explanations "
                        "so that even a complete beginner can understand and retain them.\n\n"
                        "CRITICAL INSTRUCTIONS:\n"
                        "1. CORE SOURCE: Use the provided document context below as your primary source of facts.\n"
                        "2. COMPREHENSIVE TEACHING: If the context is too dense, technical, or lacks foundational definitions, "
                        "you ARE explicitly allowed to use your general knowledge to explain terms, build out vivid real-world analogies, "
                        "and provide step-by-step breakdowns.\n"
                        "3. GROUNDING: Never contradict the provided context.\n"
                        "4. STYLE: Use short punchy sentences, formatting like bold text, and bullet points to maximize readability.\n\n"
                        "Document Context:\n{context}"
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    response = rag_chain.invoke({"input": user_query})
                    teacher_answer = response["answer"]
                    
                    st.markdown(teacher_answer)
                    st.session_state.messages.append({"role": "assistant", "content": teacher_answer})
                except Exception as e:
                    st.error(f"Teacher engine error: {str(e)}")
