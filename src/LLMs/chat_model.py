import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(dotenv_path="C:/Users/Mr. Daanyal/Desktop/Agentic_Bootcamp/CorrectiveRAG/.env")

def get_chat_model():
    return ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
