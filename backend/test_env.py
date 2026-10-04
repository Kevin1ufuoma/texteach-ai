import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key and api_key != "your_actual_api_key_here":
    print("✅ Success! Your .env file was read correctly and your API key is secure.")
else:
    print("❌ Error: Could not read the GEMINI_API_KEY. Check your .env file name and location.")
