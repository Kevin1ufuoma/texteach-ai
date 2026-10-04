from rag_engine import ask_teacher

print("💬 Querying the AI Teacher...")
try:
    # Replace the question text below with a question about whatever PDF you uploaded!
    question = "Give me a quick summary of this document." 

    answer = ask_teacher(question)
    print("\n🤖 TEACHER RESPONSE:\n" + "="*40)
    print(answer)
    print("="*40)
    print("🎉 RAG Query Test Passed Successfully!")
except Exception as e:
    print(f"❌ Query failed with error: {e}")
