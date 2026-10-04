import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import process_pdf, ask_teacher, DB_FAISS_PATH

app = FastAPI(title="Texteach AI API")

# 🛠️ SECURITY & CONNECTIVITY: Allow our JavaScript frontend to talk to this Python backend safely
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary directory to save uploaded files before reading them
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Texteach AI Backend Server is Running! 🎓"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives the student's question and returns the teacher-persona response"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported at this moment.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file chunks to our local temp folder
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Pass the local file path directly into our RAG process pipeline
        chunks_created = process_pdf(file_path)
        return {
            "status": "success",
            "message": f"Successfully parsed document into {chunks_created} knowledge fragments."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        # Clean up by removing the temporary raw PDF file to save space
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/ask")
def ask_question(body: QuestionRequest):
    """Receives the student's question and returns the teacher-persona response"""
    try:
        teacher_response = ask_teacher(body.question)
        return {"answer": teacher_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Teacher engine error: {str(e)}")


@app.post("/clear")
def clear_session():
    """Wipes out the locally stored vector database files and temporary files"""
    try:
        # Check if our local vector database exists and delete it entirely
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree("vectorstore") # Removes the entire database directory
            
        # Clean up any leftover temp files inside the upload folder
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True) # Recreate empty directory
            
        return {"status": "success", "message": "Memory cleared! Ready for a new document."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to wipe session data: {str(e)}")
