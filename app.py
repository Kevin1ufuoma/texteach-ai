import os
import shutil
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================================================
# 🎓 TEXTEACH AI - STREAMLIT PRODUCTION APPLICATION (WITH QUIZ EXTENSION)
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

# 🔒 SAFE QUIZ MEMORY REGISTERS (Isolated from the chat history loop)
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

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
                    st.success("Teacher memory updated successfully!")
                except Exception as e:
                    status.update(label="❌ Ingestion framework error.", state="error")
                    st.error(f"Error details: {str(e)}")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        
    st.markdown("---")
    st.subheader("💡 Learning Tips")
    st.info("Ask the teacher to break dense concepts down using analogies or request a short quiz to test your retention!")
    
    # 🔄 CLEAR SESSION FEATURE (Wipes both Chat and Quiz registers safely)
    if st.button("🔄 Clear Session / New File", use_container_width=True):
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        st.session_state.messages = [
            {"role": "assistant", "content": "Memory wiped successfully! Ready for your next document. Upload it on the left whenever you are ready!"}
        ]
        st.session_state.db_ready = False
        st.session_state.quiz_questions = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.rerun()

# Create clean side-by-side main panels using columns
col_chat, col_quiz = st.columns([1, 1])

# --------------------------------------------------------------------------
# 💬 PANEL A - ACTIVE TEACHER CHAT VIEWPORT
# --------------------------------------------------------------------------
with col_chat:
    st.subheader("💬 Step 2: Ask Your Teacher")
    
    # Box layout with set height constraint for clean visual flow
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Lock dialogue input box if memory vectors haven't finished compiling
    if not st.session_state.db_ready:
        st.chat_input("Please upload a PDF file to activate the teacher conversation...", disabled=True, key="chat_disabled_input")
    else:
        if user_query := st.chat_input("Ask a targeted question about your document...", key="chat_active_input"):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)
                
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
                                "Your ultimate goal is to break down complex concepts into the simplest possible explanations.\n\n"
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
            st.rerun()

# --------------------------------------------------------------------------
# 📝 PANEL B - AUTOMATED MULTIPLE-CHOICE QUIZ GENERATOR
# --------------------------------------------------------------------------
with col_quiz:
    st.subheader("📝 Step 3: Test Your Retention")
    
    if not st.session_state.db_ready:
        st.info("Once a document finishes vectorizing on the left sidebar, the quiz generator tool will activate here.")
    else:
        # Action button to trigger the quiz generation
        if st.button("🎯 Generate Custom 3-Question Quiz from Document", use_container_width=True):
            with st.spinner("Analyzing document context to draft test questions..."):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
                    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
                    
                    # Pull random, diverse text segments to assemble different questions
                    retriever = db.as_retriever(search_kwargs={"k": 4})
                    relevant_docs = retriever.get_relevant_documents("important concepts definitions facts")
                    context_chunk = "\n".join([d.page_content for d in relevant_docs])
                    
                    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key, temperature=0.5)
                    
                    quiz_prompt = (
                        "You are an expert examiner. Generate exactly 3 unique multiple-choice questions based ONLY on the context below.\n"
                        "Each question must have exactly 4 choices (A, B, C, D) and a single correct answer.\n"
                        "Format your output EXACTLY like this layout so it can be parsed cleanly:\n\n"
                        "Q1: What is...? \n"
                        "A) choice 1\n"
                        "B) choice 2\n"
                        "C) choice 3\n"
                        "D) choice 4\n"
                        "Correct: A\n\n"
                        "Context:\n" + context_chunk
                    )

                    raw_quiz = llm.predict(quiz_prompt)
                    
                    # Parse block using structured Regex configurations
                    q_blocks = raw_quiz.strip().split("Q")
                    parsed_questions = []
                    
                    for block in q_blocks:
                        if not block.strip():
                            continue
                        lines = [line.strip() for line in block.split("\n") if line.strip()]
                        if len(lines) >= 6:
                            # Reconstruct the question headline text safely
                            question_text = "Q" + lines[0]
                            options = {}
                            correct_answer = "A"
                            for line in lines[1:]:
                                if line.startswith("A)"): options["A"] = line
                                elif line.startswith("B)"): options["B"] = line
                                elif line.startswith("C)"): options["C"] = line
                                elif line.startswith("D)"): options["D"] = line
                                elif "Correct:" in line:
                                    correct_answer = line.split(":")[-1].strip()
                            
                            if options and question_text:
                                parsed_questions.append({
                                    "question": question_text,
                                    "options": options,
                                    "correct": correct_answer
                                })
                    
                    if len(parsed_questions) >= 1:
                        st.session_state.quiz_questions = parsed_questions[:3]
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.rerun()
                    else:
                        st.error("Failed to parse formatting constraints cleanly. Please tap the generate button again.")
                except Exception as e:
                    st.error(f"Quiz production failure: {str(e)}")

        # Render Active Quiz Form UI if questions reside inside local session registers
        if st.session_state.quiz_questions:
            st.markdown("---")
            st.markdown("### ✏️ Answer the Questions Below:")
            
            # Map elements into individual radio button modules
            for i, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"**{q['question']}**")
                options_list = list(q['options'].values())
                
                # Check for historical user selection markers
                current_selection = st.session_state.quiz_answers.get(i, None)
                try:
                    default_idx = options_list.index(current_selection) if current_selection else 0
                except ValueError:
                    default_idx = 0
                    
                selected_option = st.radio(
                    f"Select option for question {i+1}:", 
                    options_list, 
                    index=default_idx,
                    key=f"q_radio_{i}",
                    label_visibility="collapsed"
                )
                
                # Reverse lookup to capture just the key indicator string character ('A','B',etc)
                for key, val in q['options'].items():
                    if val == selected_option:
                        st.session_state.quiz_answers[i] = selected_option
                st.markdown("")

            if st.button("🏁 Submit Answers for Grading", use_container_width=True):
                st.session_state.quiz_submitted = True
                st.rerun()

            # Process Grading Display Output Card Actions
            if st.session_state.quiz_submitted:
                st.markdown("---")
                st.markdown("### 📊 Your Results:")
                score = 0
                
                for i, q in enumerate(st.session_state.quiz_questions):
                    user_ans = st.session_state.quiz_answers.get(i, "")
                    correct_key = q['correct']
                    correct_text = q['options'].get(correct_key, f"{correct_key})")
                    
                    if user_ans.startswith(correct_key):
                        score += 1
                        st.success(f"**Question {i+1}**: Correct! 🎉\n- Your answer: {user_ans}")
                    else:
                        st.error(f"**Question {i+1}**: Incorrect ❌\n- Your answer: {user_ans}\n- Correct answer: {correct_text}")
                
                percentage = int((score / len(st.session_state.quiz_questions)) * 100)
                if percentage == 100:
                    st.balloons()
                    st.success(f"🏆 Final Score: **{score}/{len(st.session_state.quiz_questions)} ({percentage}%)** - Masterful retention! Excellent work.")
                else:
                    st.info(f"📋 Final Score: **{score}/{len(st.session_state.quiz_questions)} ({percentage}%)** - Review the explanations in the chat panel to master these points!")

